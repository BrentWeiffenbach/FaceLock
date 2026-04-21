"""Arm controller with camera-leveling visual servoing.

2-DOF planar arm with DH convention:
  Joint 1 (base):  Z out of servo, X along link 1
  Joint 2 (elbow): Z inverted (180° Y-rot), X along link 2 to camera
  Camera:          fixed normal to link 2, faces along joint-1 Z

Camera-leveling constraint keeps link 2 vertical (θ1 − θ2 = π/2),
reducing the arm to 1-DOF control.  A P-controller drives θ1 based
on horizontal pixel error to centre the face in the camera frame.

Controller (matches example.html with sign flip for physical servo):
  error_x = face_pixel_x − image_centre_x
  desired_θ1 = θ1 + Kp · error_x      (positive: face-right → increase θ1)
  θ1 += (desired_θ1 − θ1) · α          (exponential smoothing, α = 0.15)
  θ2 = θ1 − π/2                        (camera leveling)
  servo1 = θ1,  servo2 = θ2 + π/2 = θ1
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
    ARM_CONTROL_RATE_HZ,
    ARM_DEADBAND_PX,
    ARM_KP,
    ARM_MAX_STEP_RAD,
    ARM_SERVO_LIMIT_MAX_RAD,
    ARM_SERVO_LIMIT_MIN_RAD,
    ARM_SMOOTHING_ALPHA,
    ARM_TRACK_EMA_ALPHA,
    ARM_TRACK_TIMEOUT_SEC,
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


def _theta1_to_servo(theta1: float) -> float:
    """Joint angle θ1 → servo1 radians (identity mapping)."""
    return theta1


def _theta2_to_servo(theta2: float) -> float:
    """Joint angle θ2 → servo2 radians (offset by π/2)."""
    return theta2 + math.pi / 2


def _servo_to_theta1(servo1: float) -> float:
    """Servo1 radians → joint angle θ1 (identity mapping)."""
    return servo1


# ── Node ────────────────────────────────────────────────────────────────


class ArmController(LifecycleNode):
    """P-controller visual-servoing arm controller with camera leveling."""

    joint_pub: Publisher
    joint_states_sub: Subscription
    detection_sub: Subscription
    reset_arm_srv: Any
    control_timer: Any

    def __init__(self) -> None:
        super().__init__("arm_controller")
        self._active: bool = False

        # Current base servo angle = θ1 (updated from servo feedback)
        self._theta1: float = LOWER_ARM_HOME_RAD
        self._deadlock_rad: float = DEADLOCK_HOME_RAD

        # Detection EMA state
        self._last_detection_time: float = 0.0
        self._x_filtered: Optional[float] = None
        self._y_filtered: Optional[float] = None
        self._last_debug_log_time: float = 0.0

        # Debug image saving
        self._bridge = CvBridge()
        self._latest_frame: Optional[np.ndarray] = None
        self._command_count: int = 0
        self._debug_image_dir = "/tmp/arm_debug"
        os.makedirs(self._debug_image_dir, exist_ok=True)

        # Parameters
        self.declare_parameter("image_width", 640.0)
        self.declare_parameter("image_height", 480.0)
        self.declare_parameter("kp", ARM_KP)
        self.declare_parameter("deadband_px", ARM_DEADBAND_PX)
        self.declare_parameter("servo_limit_min_rad", ARM_SERVO_LIMIT_MIN_RAD)
        self.declare_parameter("servo_limit_max_rad", ARM_SERVO_LIMIT_MAX_RAD)
        self.declare_parameter("control_rate_hz", ARM_CONTROL_RATE_HZ)
        self.declare_parameter("track_timeout_sec", ARM_TRACK_TIMEOUT_SEC)
        self.declare_parameter("ema_alpha", ARM_TRACK_EMA_ALPHA)
        self.declare_parameter("smoothing_alpha", ARM_SMOOTHING_ALPHA)
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
        self.get_logger().info("Configuring arm controller")

        self._image_width = self._pf("image_width", 640.0)
        self._image_height = self._pf("image_height", 480.0)
        self._kp = self._pf("kp", ARM_KP)
        self._deadband_px = self._pf("deadband_px", ARM_DEADBAND_PX)
        self._limit_min = self._pf("servo_limit_min_rad", ARM_SERVO_LIMIT_MIN_RAD)
        self._limit_max = self._pf("servo_limit_max_rad", ARM_SERVO_LIMIT_MAX_RAD)
        self._track_timeout = self._pf("track_timeout_sec", ARM_TRACK_TIMEOUT_SEC)
        self._ema_alpha = self._pf("ema_alpha", ARM_TRACK_EMA_ALPHA)
        self._smoothing_alpha = self._pf("smoothing_alpha", ARM_SMOOTHING_ALPHA)
        self._max_step_rad = self._pf("max_step_rad", ARM_MAX_STEP_RAD)
        control_hz = self._pf("control_rate_hz", ARM_CONTROL_RATE_HZ)

        # Initialise θ1 from home position
        self._theta1 = LOWER_ARM_HOME_RAD
        self.get_logger().info(
            f"Home θ1={math.degrees(self._theta1):.1f}°  "
            f"limits=[{math.degrees(self._limit_min):.0f}°, "
            f"{math.degrees(self._limit_max):.0f}°]"
        )

        # Pub / sub / srv
        self.joint_pub = self.create_lifecycle_publisher(
            JointState, "/arm/joint_commands", 10
        )
        self.joint_states_sub = self.create_subscription(
            JointState, "/arm/joint_states", self._joint_states_cb, 10
        )
        self.detection_sub = self.create_subscription(
            Detection2D, "/face_recognition/detection", self._detection_cb, 10
        )
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_cb, 1
        )
        self.reset_arm_srv = self.create_service(
            Trigger, "reset_arm", self._reset_arm_cb
        )

        # Fixed-rate control loop
        period = 1.0 / max(1.0, control_hz)
        self.control_timer = self.create_timer(period, self._control_loop)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating arm controller")
        self._active = True
        self._last_detection_time = time.monotonic()
        self._x_filtered = None
        self._y_filtered = None
        # Activate lifecycle publishers, then send home
        result = super().on_activate(state)
        self._theta1 = LOWER_ARM_HOME_RAD
        self._deadlock_rad = DEADLOCK_HOME_RAD
        self._publish_joint_command()
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating arm controller — returning to home")
        self._active = False
        # Send home before super() deactivates the lifecycle publisher
        self._theta1 = LOWER_ARM_HOME_RAD
        self._deadlock_rad = DEADLOCK_HOME_RAD
        self._x_filtered = None
        self._y_filtered = None
        self._publish_joint_command()
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up arm controller")
        return TransitionCallbackReturn.SUCCESS

    # ── callbacks ────────────────────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        """Store latest camera frame for debug overlay."""
        try:
            self._latest_frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

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
        """EMA-filter the face detection centre."""
        if not self._active:
            return

        cx = float(msg.bbox.center.position.x)
        cy = float(msg.bbox.center.position.y)

        if self._x_filtered is None or self._y_filtered is None:
            self._x_filtered = cx
            self._y_filtered = cy
        else:
            a = self._ema_alpha
            self._x_filtered = a * cx + (1.0 - a) * self._x_filtered
            self._y_filtered = a * cy + (1.0 - a) * self._y_filtered

        self._last_detection_time = time.monotonic()

    def _control_loop(self) -> None:
        """P-control: horizontal pixel error → θ1 adjustment."""
        if not self._active:
            return

        # Hold position when face is lost
        if time.monotonic() - self._last_detection_time > self._track_timeout:
            return

        if self._x_filtered is None:
            return

        # ── horizontal pixel error from image centre ─────────────────
        error_x = self._x_filtered - self._image_width / 2.0

        # ── debug logging (throttled to 1 Hz) ────────────────────────
        now = time.monotonic()
        if now - self._last_debug_log_time >= 1.0:
            self._last_debug_log_time = now
            direction = (
                "RIGHT" if error_x > self._deadband_px
                else "LEFT" if error_x < -self._deadband_px
                else "CENTRE"
            )
            self.get_logger().info(
                f"[ARM] {direction}  err_x={error_x:+.1f}px  "
                f"θ1={math.degrees(self._theta1):.1f}°"
            )

        # ── deadband ─────────────────────────────────────────────────
        if abs(error_x) <= self._deadband_px:
            return

        # ── P-controller ─────────────────────────────────────────────
        # Physical servo direction: increasing servo1 → arm pans right.
        # face-right (positive error_x) → increase θ1 → arm pans right.
        delta_theta1 = self._kp * error_x

        # ── compute desired and clamp to servo limits ────────────────
        desired_theta1 = self._theta1 + delta_theta1
        desired_theta1 = max(self._limit_min, min(self._limit_max, desired_theta1))

        # ── exponential smoothing (prevents instant jumps) ───────────
        step = (desired_theta1 - self._theta1) * self._smoothing_alpha

        # ── hard cap on step size per tick ────────────────────────────
        step = max(-self._max_step_rad, min(self._max_step_rad, step))

        self._theta1 = max(
            self._limit_min, min(self._limit_max, self._theta1 + step)
        )
        self._publish_joint_command()
        self._maybe_save_debug_image(error_x)

    # ── debug image saving ────────────────────────────────────────

    def _maybe_save_debug_image(self, error_x: float) -> None:
        """Save annotated frame every 5 control commands."""
        self._command_count += 1
        if self._command_count % 5 != 0:
            return
        if self._latest_frame is None:
            return
        if self._x_filtered is None or self._y_filtered is None:
            return

        frame = self._latest_frame.copy()
        h, w = frame.shape[:2]
        cx_img = int(w / 2)
        cy_img = int(h / 2)
        fx = int(self._x_filtered)
        fy = int(self._y_filtered)

        # Green crosshair at image centre
        cv2.drawMarker(frame, (cx_img, cy_img), (0, 255, 0),
                        cv2.MARKER_CROSS, 30, 2)
        cv2.putText(frame, "IMG_CENTER", (cx_img + 5, cy_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Red dot at filtered face detection
        cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"FACE ({fx},{fy})", (fx + 10, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Orange line between them
        cv2.line(frame, (cx_img, cy_img), (fx, fy), (0, 165, 255), 2)

        # Info text
        info = (f"err_x={error_x:+.1f}px  "
                f"theta1={math.degrees(self._theta1):.1f}deg  "
                f"cmd#{self._command_count}")
        cv2.putText(frame, info, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        path = os.path.join(self._debug_image_dir,
                            f"arm_debug_{self._command_count:05d}.jpg")
        cv2.imwrite(path, frame)

    # ── publishing ───────────────────────────────────────────────────

    def _publish_joint_command(self) -> None:
        theta2 = _theta2_for_leveling(self._theta1)
        servo1 = _theta1_to_servo(self._theta1)
        servo2 = _theta2_to_servo(theta2)

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
        self._x_filtered = None
        self._y_filtered = None
        self._publish_joint_command()
        # Freeze tracking so stale detections cannot move the arm.
        # on_activate() re-enables when CHECKING restarts next cycle.
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
