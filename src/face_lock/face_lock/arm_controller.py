from typing import Any, Optional

import math
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.node import Publisher, Subscription
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2D

from face_lock.constants import (
    ARM_TRACK_DEADBAND_PX,
    ARM_TRACK_EMA_ALPHA,
    ARM_TRACK_KP_LOWER_US_PX,
    ARM_TRACK_KP_UPPER_US_PX,
    ARM_TRACK_MAX_SLEW_US,
    ARM_TRACK_TIMEOUT_SEC,
    DEADLOCK_HOME_RAD,
    DEADLOCK_JOINT_NAME,
    LOWER_ARM_HOME_RAD,
    LOWER_ARM_JOINT_NAME,
    SERVO_PULSE_MAX_US,
    SERVO_PULSE_MIN_US,
    UPPER_ARM_HOME_RAD,
    UPPER_ARM_JOINT_NAME,
)


class ArmController(LifecycleNode):
    joint_pub: Publisher
    joint_states_sub: Subscription
    detection_sub: Subscription
    reset_arm_srv: Any
    tracking_timer: Any

    def __init__(self) -> None:
        super().__init__("arm_controller")
        self._active: bool = False
        self._last_joint_state: Optional[JointState] = None
        self._last_detection_time: float = 0.0

        # Control state in PWM microseconds
        self._lower_pwm_us: int = self._rad_to_pwm(LOWER_ARM_HOME_RAD)
        self._upper_pwm_us: int = self._rad_to_pwm(UPPER_ARM_HOME_RAD)
        self._deadlock_cmd_rad: float = DEADLOCK_HOME_RAD

        # EMA filter state — initialised on first detection
        self._x_filtered: Optional[float] = None
        self._y_filtered: Optional[float] = None

        self.declare_parameter("image_width", 640.0)
        self.declare_parameter("image_height", 480.0)
        self.declare_parameter("ema_alpha", ARM_TRACK_EMA_ALPHA)
        self.declare_parameter("kp_lower_us_px", ARM_TRACK_KP_LOWER_US_PX)
        self.declare_parameter("kp_upper_us_px", ARM_TRACK_KP_UPPER_US_PX)
        self.declare_parameter("max_slew_us", float(ARM_TRACK_MAX_SLEW_US))
        self.declare_parameter("deadband_px", ARM_TRACK_DEADBAND_PX)
        self.declare_parameter("track_timeout_sec", ARM_TRACK_TIMEOUT_SEC)
        self.declare_parameter("invert_lower", False)
        self.declare_parameter("invert_upper", True)

        self._image_width: float = 640.0
        self._image_height: float = 480.0
        self._ema_alpha: float = ARM_TRACK_EMA_ALPHA
        self._kp_lower: float = ARM_TRACK_KP_LOWER_US_PX
        self._kp_upper: float = ARM_TRACK_KP_UPPER_US_PX
        self._max_slew_us: float = float(ARM_TRACK_MAX_SLEW_US)
        self._deadband_px: float = ARM_TRACK_DEADBAND_PX
        self._track_timeout_sec: float = ARM_TRACK_TIMEOUT_SEC
        self._invert_lower: bool = False
        self._invert_upper: bool = True

    # ------------------------------------------------------------------
    # PWM / radian helpers (must match pi_hardware conversions)
    # ------------------------------------------------------------------

    @staticmethod
    def _rad_to_pwm(rad: float) -> int:
        """Map radians [0, π] → PWM [500, 2500] µs."""
        clamped = max(0.0, min(math.pi, rad))
        pulse = SERVO_PULSE_MIN_US + (clamped / math.pi) * (
            SERVO_PULSE_MAX_US - SERVO_PULSE_MIN_US
        )
        return int(round(pulse))

    @staticmethod
    def _pwm_to_rad(pulse_us: int) -> float:
        """Map PWM [500, 2500] µs → radians [0, π]."""
        clamped = max(SERVO_PULSE_MIN_US, min(SERVO_PULSE_MAX_US, pulse_us))
        return (clamped - SERVO_PULSE_MIN_US) / (
            SERVO_PULSE_MAX_US - SERVO_PULSE_MIN_US
        ) * math.pi

    @staticmethod
    def _clamp_pwm(pulse_us: float) -> int:
        return int(round(max(SERVO_PULSE_MIN_US, min(SERVO_PULSE_MAX_US, pulse_us))))

    def _param_float(self, name: str, default: float) -> float:
        value = self.get_parameter(name).value
        if value is None:
            return default
        return float(value)

    def _param_bool(self, name: str, default: bool) -> bool:
        value = self.get_parameter(name).value
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring arm controller")
        self._image_width = self._param_float("image_width", 640.0)
        self._image_height = self._param_float("image_height", 480.0)
        self._ema_alpha = self._param_float("ema_alpha", ARM_TRACK_EMA_ALPHA)
        self._kp_lower = self._param_float("kp_lower_us_px", ARM_TRACK_KP_LOWER_US_PX)
        self._kp_upper = self._param_float("kp_upper_us_px", ARM_TRACK_KP_UPPER_US_PX)
        self._max_slew_us = self._param_float(
            "max_slew_us", float(ARM_TRACK_MAX_SLEW_US)
        )
        self._deadband_px = self._param_float("deadband_px", ARM_TRACK_DEADBAND_PX)
        self._track_timeout_sec = self._param_float(
            "track_timeout_sec", ARM_TRACK_TIMEOUT_SEC
        )
        self._invert_lower = self._param_bool("invert_lower", False)
        self._invert_upper = self._param_bool("invert_upper", True)

        # Publishers
        self.joint_pub: Publisher = self.create_lifecycle_publisher(
            JointState, "/arm/joint_commands", 10
        )

        # Subscribers
        self.joint_states_sub: Subscription = self.create_subscription(
            JointState, "/arm/joint_states", self.joint_states_cb, 10
        )

        self.detection_sub: Subscription = self.create_subscription(
            Detection2D, "/face_recognition/detection", self.detection_cb, 10
        )

        # Services
        self.reset_arm_srv = self.create_service(
            Trigger, "reset_arm", self.reset_arm_cb
        )

        self.tracking_timer = self.create_timer(0.1, self._tracking_watchdog_cb)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating arm controller")
        self._active = True
        self._last_detection_time = time.monotonic()
        # Reset EMA state so stale coordinates don't bias first update
        self._x_filtered = None
        self._y_filtered = None
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating arm controller")
        self._active = False
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up arm controller")
        return TransitionCallbackReturn.SUCCESS

    @staticmethod
    def _clamp_rad(value: float) -> float:
        return max(0.0, min(math.pi, value))

    def _publish_joint_command(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [LOWER_ARM_JOINT_NAME, UPPER_ARM_JOINT_NAME, DEADLOCK_JOINT_NAME]
        msg.position = [
            self._pwm_to_rad(self._lower_pwm_us),
            self._pwm_to_rad(self._upper_pwm_us),
            self._clamp_rad(self._deadlock_cmd_rad),
        ]
        self.joint_pub.publish(msg)

    def joint_states_cb(self, msg: JointState) -> None:
        self._last_joint_state = msg
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                break
            if name == LOWER_ARM_JOINT_NAME:
                self._lower_pwm_us = self._rad_to_pwm(msg.position[idx])
            elif name == UPPER_ARM_JOINT_NAME:
                self._upper_pwm_us = self._rad_to_pwm(msg.position[idx])
            elif name == DEADLOCK_JOINT_NAME:
                self._deadlock_cmd_rad = self._clamp_rad(msg.position[idx])

    def detection_cb(self, msg: Detection2D) -> None:
        if not self._active:
            return

        cx = float(msg.bbox.center.position.x)
        cy = float(msg.bbox.center.position.y)

        # ── Step 1: EMA low-pass filter ──────────────────────────────────
        # x_filtered = α·x_new + (1−α)·x_old   (α = 0.2 → "heavy" target)
        x_prev = self._x_filtered
        y_prev = self._y_filtered
        if x_prev is None or y_prev is None:
            x_filt: float = cx
            y_filt: float = cy
        else:
            x_filt = self._ema_alpha * cx + (1.0 - self._ema_alpha) * x_prev
            y_filt = self._ema_alpha * cy + (1.0 - self._ema_alpha) * y_prev
        self._x_filtered = x_filt
        self._y_filtered = y_filt

        # ── Step 2: pixel error relative to image centre ─────────────────
        image_cx = self._image_width / 2.0
        image_cy = self._image_height / 2.0
        err_x = x_filt - image_cx   # +ve → face is right of centre
        err_y = y_filt - image_cy   # +ve → face is below centre

        # Deadband: ignore tiny errors caused by residual detection noise
        if abs(err_x) < self._deadband_px:
            err_x = 0.0
        if abs(err_y) < self._deadband_px:
            err_y = 0.0

        # ── Step 3: P-controller → desired PWM delta (µs) ────────────────
        # Joint 1 (Base):  higher PWM moves camera RIGHT
        #   face on right (err_x > 0) → increase lower PWM
        delta_lower = self._kp_lower * err_x
        if self._invert_lower:
            delta_lower = -delta_lower

        # Joint 2 (Elbow): higher PWM moves camera LEFT (direction inverted)
        #   Default invert_upper=True flips the response so the elbow tracks Y
        delta_upper = self._kp_upper * err_y
        if self._invert_upper:
            delta_upper = -delta_upper

        # ── Step 4: slew-rate limiting ────────────────────────────────────
        # Cap the maximum PWM change per update to suppress jerk
        delta_lower = max(-self._max_slew_us, min(self._max_slew_us, delta_lower))
        delta_upper = max(-self._max_slew_us, min(self._max_slew_us, delta_upper))

        # ── Step 5: integrate and clamp to [500, 2500] µs ────────────────
        self._lower_pwm_us = self._clamp_pwm(self._lower_pwm_us + delta_lower)
        self._upper_pwm_us = self._clamp_pwm(self._upper_pwm_us + delta_upper)

        self._last_detection_time = time.monotonic()
        self._publish_joint_command()

    def _tracking_watchdog_cb(self) -> None:
        if not self._active:
            return
        # Hold current PWM when face is lost (no auto-snap to home)
        if time.monotonic() - self._last_detection_time > self._track_timeout_sec:
            return

    def reset_arm_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        del req
        self._lower_pwm_us = self._rad_to_pwm(LOWER_ARM_HOME_RAD)
        self._upper_pwm_us = self._rad_to_pwm(UPPER_ARM_HOME_RAD)
        self._deadlock_cmd_rad = DEADLOCK_HOME_RAD
        self._x_filtered = None
        self._y_filtered = None
        self._publish_joint_command()
        res.success = True
        res.message = "Arm reset command published"
        return res


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: ArmController = ArmController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
