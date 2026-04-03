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
    ARM_TRACK_MAX_STEP_RAD,
    ARM_TRACK_TIMEOUT_SEC,
    DEADLOCK_HOME_RAD,
    DEADLOCK_JOINT_NAME,
    LOWER_ARM_HOME_RAD,
    LOWER_ARM_JOINT_NAME,
    LOWER_ARM_TRACK_GAIN,
    UPPER_ARM_HOME_RAD,
    UPPER_ARM_JOINT_NAME,
    UPPER_ARM_TRACK_GAIN,
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
        self._last_cmd_time: float = 0.0

        self._lower_cmd_rad: float = LOWER_ARM_HOME_RAD
        self._upper_cmd_rad: float = UPPER_ARM_HOME_RAD
        self._deadlock_cmd_rad: float = DEADLOCK_HOME_RAD

        self.declare_parameter("image_width", 640.0)
        self.declare_parameter("image_height", 480.0)
        self.declare_parameter("lower_gain", LOWER_ARM_TRACK_GAIN)
        self.declare_parameter("upper_gain", UPPER_ARM_TRACK_GAIN)
        self.declare_parameter("deadband_px", ARM_TRACK_DEADBAND_PX)
        self.declare_parameter("max_step_rad", ARM_TRACK_MAX_STEP_RAD)
        self.declare_parameter("track_timeout_sec", ARM_TRACK_TIMEOUT_SEC)
        self.declare_parameter("invert_lower", False)
        self.declare_parameter("invert_upper", True)

        self._image_width: float = 640.0
        self._image_height: float = 480.0
        self._lower_gain: float = LOWER_ARM_TRACK_GAIN
        self._upper_gain: float = UPPER_ARM_TRACK_GAIN
        self._deadband_px: float = ARM_TRACK_DEADBAND_PX
        self._max_step_rad: float = ARM_TRACK_MAX_STEP_RAD
        self._track_timeout_sec: float = ARM_TRACK_TIMEOUT_SEC
        self._invert_lower: bool = False
        self._invert_upper: bool = True

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
        self._lower_gain = self._param_float("lower_gain", LOWER_ARM_TRACK_GAIN)
        self._upper_gain = self._param_float("upper_gain", UPPER_ARM_TRACK_GAIN)
        self._deadband_px = self._param_float("deadband_px", ARM_TRACK_DEADBAND_PX)
        self._max_step_rad = self._param_float("max_step_rad", ARM_TRACK_MAX_STEP_RAD)
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
        self._last_cmd_time = 0.0
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
            self._clamp_rad(self._lower_cmd_rad),
            self._clamp_rad(self._upper_cmd_rad),
            self._clamp_rad(self._deadlock_cmd_rad),
        ]
        self.joint_pub.publish(msg)
        self._last_cmd_time = time.monotonic()

    def joint_states_cb(self, msg: JointState) -> None:
        self._last_joint_state = msg
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                break
            if name == LOWER_ARM_JOINT_NAME:
                self._lower_cmd_rad = self._clamp_rad(msg.position[idx])
            elif name == UPPER_ARM_JOINT_NAME:
                self._upper_cmd_rad = self._clamp_rad(msg.position[idx])
            elif name == DEADLOCK_JOINT_NAME:
                self._deadlock_cmd_rad = self._clamp_rad(msg.position[idx])

    def detection_cb(self, msg: Detection2D) -> None:
        if not self._active:
            return

        cx = float(msg.bbox.center.position.x)
        cy = float(msg.bbox.center.position.y)
        image_cx = self._image_width / 2.0
        image_cy = self._image_height / 2.0

        err_x = cx - image_cx
        err_y = cy - image_cy

        if abs(err_x) < self._deadband_px:
            err_x = 0.0
        if abs(err_y) < self._deadband_px:
            err_y = 0.0

        norm_x = err_x / image_cx if image_cx > 0.0 else 0.0
        norm_y = err_y / image_cy if image_cy > 0.0 else 0.0

        lower_delta = self._lower_gain * norm_x
        upper_delta = self._upper_gain * norm_y
        lower_delta = max(-self._max_step_rad, min(self._max_step_rad, lower_delta))
        upper_delta = max(-self._max_step_rad, min(self._max_step_rad, upper_delta))

        if self._invert_lower:
            lower_delta *= -1.0
        if self._invert_upper:
            upper_delta *= -1.0

        self._lower_cmd_rad = self._clamp_rad(self._lower_cmd_rad + lower_delta)
        self._upper_cmd_rad = self._clamp_rad(self._upper_cmd_rad + upper_delta)
        self._last_detection_time = time.monotonic()
        self._publish_joint_command()

    def _tracking_watchdog_cb(self) -> None:
        if not self._active:
            return

        now = time.monotonic()
        if now - self._last_detection_time > self._track_timeout_sec:
            # When face is lost, hold current command; no auto-snap.
            return

    def reset_arm_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        del req
        self._lower_cmd_rad = LOWER_ARM_HOME_RAD
        self._upper_cmd_rad = UPPER_ARM_HOME_RAD
        self._deadlock_cmd_rad = DEADLOCK_HOME_RAD
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
