"""Arm controller with camera-leveling visual servoing.

2-DOF planar arm with DH convention:
  Joint 1 (base):  Z out of servo, X along link 1
  Joint 2 (elbow): Z inverted (180° Y-rot), X along link 2 to camera
  Camera:          fixed normal to link 2, faces along joint-1 Z

Camera-leveling constraint:
  θ2 = θ1 − π/2  →  link 2 always vertical, camera always level

Fully event-driven (NO timer):
  One detection received → compute error → one P-step → publish command.
  No detection → arm holds position.  No stale data, no independent rate.

Controller (one step per detection):
  error_x  = face_pixel_x − image_centre_x
  Δθ1      = clamp(Kp · error_x, −MAX_STEP, +MAX_STEP)
  θ1_new   = clamp(θ1 + Δθ1, LIMIT_MIN, LIMIT_MAX)
  servo1   = θ1_new
  servo2   = θ1_new   (leveling: θ2 + π/2 = θ1)
"""

from typing import Any, Optional

import math
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.node import Publisher, Subscription
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2D

from face_lock.constants import (
    ARM_DEADBAND_PX,
    ARM_KP,
    ARM_MAX_STEP_RAD,
    ARM_SERVO_LIMIT_MAX_RAD,
    ARM_SERVO_LIMIT_MIN_RAD,
    DEADLOCK_HOME_RAD,
    DEADLOCK_JOINT_NAME,
    LOWER_ARM_HOME_RAD,
    LOWER_ARM_JOINT_NAME,
    UPPER_ARM_HOME_RAD,
    UPPER_ARM_JOINT_NAME,
)


# ── Kinematics helpers ──────────────────────────────────────────────────


def _theta2_for_leveling(theta1: float) -> float:
    """Camera leveling: θ2 = θ1 − π/2 keeps link 2 vertical."""
    return theta1 - math.pi / 2


def _servo_to_theta1(servo1: float) -> float:
    """Servo1 radians → joint angle θ1 (identity mapping)."""
    return servo1


# ── Node ────────────────────────────────────────────────────────────────


class ArmController(LifecycleNode):
    """Event-driven P-controller arm controller with camera leveling.

    Control fires ONLY on receipt of a detection message — no independent
    timer, no stale data, no drift between detection rate and control rate.
    """

    joint_pub: Publisher
    joint_states_sub: Subscription
    detection_sub: Subscription
    image_sub: Subscription
    reset_arm_srv: Any

    def __init__(self) -> None:
        super().__init__("arm_controller")
        self._active: bool = False

        # Current θ1 (updated from servo feedback)
        self._theta1: float = LOWER_ARM_HOME_RAD
        self._deadlock_rad: float = DEADLOCK_HOME_RAD

        # Latest face position from detection (raw, no filtering)
        self._face_x: Optional[float] = None
        self._face_y: Optional[float] = None

        # Pending debug annotation (set in detection_cb, drawn in image_cb)
        self._pending_debug: bool = False
        self._pending_error_x: float = 0.0
        self._pending_theta1: float = LOWER_ARM_HOME_RAD

        # Debug image saving
        self._bridge = CvBridge()
        self._debug_image_dir = "/tmp/arm_debug"
        self._debug_count: int = 0
        os.makedirs(self._debug_image_dir, exist_ok=True)

        # Parameters
        self.declare_parameter("image_width", 640.0)
        self.declare_parameter("image_height", 480.0)
        self.declare_parameter("kp", ARM_KP)
        self.declare_parameter("deadband_px", ARM_DEADBAND_PX)
        self.declare_parameter("servo_limit_min_rad", ARM_SERVO_LIMIT_MIN_RAD)
        self.declare_parameter("servo_limit_max_rad", ARM_SERVO_LIMIT_MAX_RAD)
        self.declare_parameter("max_step_rad", ARM_MAX_STEP_RAD)

    # ── helpers ──────────────────────────────────────────────────────

    def _pf(self, name: str, default: float) -> float:
        v = self.get_parameter(name).value
        return float(v) if v is not None else default

    @staticmethod
    def _clamp_servo(rad: float) -> float:
        return max(0.0, min(math.pi, rad))

    # ── lifecycle ────────────────────────────────────────────────────

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring arm controller (event-driven)")

        self._image_width = self._pf("image_width", 640.0)
        self._image_height = self._pf("image_height", 480.0)
        self._kp = self._pf("kp", ARM_KP)
        self._deadband_px = self._pf("deadband_px", ARM_DEADBAND_PX)
        self._limit_min = self._pf("servo_limit_min_rad", ARM_SERVO_LIMIT_MIN_RAD)
        self._limit_max = self._pf("servo_limit_max_rad", ARM_SERVO_LIMIT_MAX_RAD)
        self._max_step = self._pf("max_step_rad", ARM_MAX_STEP_RAD)

        self._theta1 = LOWER_ARM_HOME_RAD
        self.get_logger().info(
            f"Home θ1={math.degrees(self._theta1):.1f}°  "
            f"limits=[{math.degrees(self._limit_min):.0f}°, "
            f"{math.degrees(self._limit_max):.0f}°]  "
            f"Kp={self._kp}  deadband={self._deadband_px}px  "
            f"max_step={math.degrees(self._max_step):.1f}°"
        )

        self.joint_pub = self.create_lifecycle_publisher(
            JointState, "/arm/joint_commands", 10
        )
        self.joint_states_sub = self.create_subscription(
            JointState, "/arm/joint_states", self._joint_states_cb, 10
        )
        self.detection_sub = self.create_subscription(
            Detection2D, "/face_recognition/detection", self._detection_cb, 10
        )
        # debug_landmarks is published by face_recognition just AFTER detection.
        # Saving the debug image here (rather than in detection_cb) guarantees
        # the annotated frame is the one that matches the detection coordinates.
        self.image_sub = self.create_subscription(
            Image, "/face_recognition/debug_landmarks", self._image_cb, 1
        )
        self.reset_arm_srv = self.create_service(
            Trigger, "reset_arm", self._reset_arm_cb
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating arm controller")
        self._active = True
        self._face_x = None
        self._face_y = None
        self._pending_debug = False
        result = super().on_activate(state)
        self._theta1 = LOWER_ARM_HOME_RAD
        self._deadlock_rad = DEADLOCK_HOME_RAD
        self._publish_joint_command()
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating arm controller — returning to home")
        self._active = False
        self._face_x = None
        self._face_y = None
        self._pending_debug = False
        self._theta1 = LOWER_ARM_HOME_RAD
        self._deadlock_rad = DEADLOCK_HOME_RAD
        self._publish_joint_command()
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up arm controller")
        return TransitionCallbackReturn.SUCCESS

    # ── callbacks ────────────────────────────────────────────────────

    def _joint_states_cb(self, msg: JointState) -> None:
        """Sync internal θ1 with actual servo feedback."""
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                break
            if name == LOWER_ARM_JOINT_NAME:
                self._theta1 = _servo_to_theta1(float(msg.position[idx]))
            elif name == DEADLOCK_JOINT_NAME:
                self._deadlock_rad = self._clamp_servo(msg.position[idx])

    def _detection_cb(self, msg: Detection2D) -> None:
        """Receive detection → apply one P-step → publish.

        This is the ONLY place the arm moves.  No timer, no repeat.
        The next move happens only when the next detection arrives.
        """
        if not self._active:
            return

        self._face_x = float(msg.bbox.center.position.x)
        self._face_y = float(msg.bbox.center.position.y)

        error_x = self._face_x - self._image_width / 2.0

        direction = (
            "RIGHT" if error_x > self._deadband_px
            else "LEFT" if error_x < -self._deadband_px
            else "CENTRE"
        )
        self.get_logger().info(
            f"[ARM] {direction}  err_x={error_x:+.1f}px  θ1={math.degrees(self._theta1):.1f}°"
        )

        if abs(error_x) <= self._deadband_px:
            # Face is centred — no movement needed; still flag debug image.
            self._pending_debug = True
            self._pending_error_x = error_x
            self._pending_theta1 = self._theta1
            return

        # ── P-step (one step per detection, capped) ───────────────────
        delta = self._kp * error_x
        delta = max(-self._max_step, min(self._max_step, delta))
        new_theta1 = self._theta1 + delta
        new_theta1 = max(self._limit_min, min(self._limit_max, new_theta1))

        self._theta1 = new_theta1
        self._publish_joint_command()

        # Flag: save debug image when the matching annotated frame arrives
        self._pending_debug = True
        self._pending_error_x = error_x
        self._pending_theta1 = self._theta1

    def _image_cb(self, msg: Image) -> None:
        """Receive annotated frame from face_recognition and save debug image.

        face_recognition publishes detection then debug_landmarks in the same
        callback, so this message corresponds to the most recent detection.
        """
        if not self._pending_debug:
            return
        self._pending_debug = False

        try:
            # debug_landmarks is now bgr8 (raw frame + detection box)
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        self._debug_count += 1
        h, w = frame.shape[:2]
        cx_img = int(w / 2)
        cy_img = int(h / 2)

        # Green crosshair at image centre (goal)
        cv2.drawMarker(frame, (cx_img, cy_img), (0, 255, 0),
                       cv2.MARKER_CROSS, 40, 2)
        cv2.putText(frame, "CENTRE", (cx_img + 5, cy_img - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if self._face_x is not None and self._face_y is not None:
            fx = int(self._face_x)
            fy = int(self._face_y)
            # Red dot at face detection
            cv2.circle(frame, (fx, fy), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"FACE ({fx},{fy})", (fx + 12, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Orange error line
            cv2.line(frame, (cx_img, cy_img), (fx, fy), (0, 140, 255), 2)

        info = (f"err_x={self._pending_error_x:+.1f}px  "
                f"theta1={math.degrees(self._pending_theta1):.1f}deg  "
                f"#{self._debug_count}")
        cv2.putText(frame, info, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        path = os.path.join(self._debug_image_dir,
                            f"arm_debug_{self._debug_count:05d}.jpg")
        cv2.imwrite(path, frame)

    # ── publishing ───────────────────────────────────────────────────

    def _publish_joint_command(self) -> None:
        theta2 = _theta2_for_leveling(self._theta1)
        # servo1 = θ1 (direct), servo2 = θ2 + π/2 = θ1 (leveling)
        servo1 = self._theta1
        servo2 = theta2 + math.pi / 2  # = θ1

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [LOWER_ARM_JOINT_NAME, UPPER_ARM_JOINT_NAME, DEADLOCK_JOINT_NAME]
        msg.position = [
            self._clamp_servo(servo1),
            self._clamp_servo(servo2),
            self._clamp_servo(self._deadlock_rad),
        ]
        self.joint_pub.publish(msg)

    # ── services ─────────────────────────────────────────────────────

    def _reset_arm_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        del req
        self._theta1 = LOWER_ARM_HOME_RAD
        self._deadlock_rad = DEADLOCK_HOME_RAD
        self._face_x = None
        self._face_y = None
        self._pending_debug = False
        self._publish_joint_command()
        self._active = False
        res.success = True
        res.message = "Arm reset to home (tracking frozen)"
        return res


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ArmController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
