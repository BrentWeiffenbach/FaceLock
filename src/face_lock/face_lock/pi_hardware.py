import math
from typing import Any, Optional

try:
    import pigpio
except ImportError:
    pigpio = None  # type: ignore[assignment]

import rclpy
from rclpy.node import Node, Publisher, Subscription
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from face_lock import constants as c


class PiHardware(Node):
    pir_pub: Publisher
    btn_pub: Publisher
    joint_state_pub: Publisher
    joint_sub: Subscription
    lock_srv: Any
    unlock_srv: Any
    io_timer: Any

    def __init__(self) -> None:
        super().__init__("pi_hardware")
        self.declare_parameter("use_mock", True)
        use_mock_val = self.get_parameter("use_mock").value
        self.use_mock: bool = self._coerce_bool(use_mock_val)

        self.pi: Optional[Any] = None
        self._magnet_locked: bool = False
        self._last_pir: Optional[bool] = None
        self._last_button: Optional[bool] = None
        self._deadlock_pulse_us: int = c.DEADLOCK_LOCK_PULSE_US
        self._lower_arm_pulse_us: int = self._radians_to_pulse_us(c.LOWER_ARM_HOME_RAD)
        self._upper_arm_pulse_us: int = self._radians_to_pulse_us(c.UPPER_ARM_HOME_RAD)

        # Publishers
        self.pir_pub: Publisher = self.create_publisher(Bool, "/io/pir", 10)
        self.btn_pub: Publisher = self.create_publisher(Bool, "/io/button", 10)
        self.joint_state_pub: Publisher = self.create_publisher(
            JointState, "/arm/joint_states", 10
        )

        # Subscribers
        self.joint_sub: Subscription = self.create_subscription(
            JointState, "/arm/joint_commands", self.joint_cb, 10
        )

        # Services
        self.lock_srv = self.create_service(Trigger, "lock_door", self.lock_cb)
        self.unlock_srv = self.create_service(Trigger, "unlock_door", self.unlock_cb)

        self._init_hardware()
        self.io_timer = self.create_timer(0.05, self._poll_inputs)

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def _init_hardware(self) -> None:
        if self.use_mock:
            self.get_logger().warn("pi_hardware running in mock mode (no GPIO writes)")
            return

        if pigpio is None:
            raise RuntimeError("pigpio module is not installed")

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Failed to connect to pigpio daemon. Start pigpiod first.")

        self.pi.set_mode(c.PIR_SENSOR_GPIO, pigpio.INPUT)
        self.pi.set_mode(c.CLOSED_DOOR_BUTTON_GPIO, pigpio.INPUT)
        self.pi.set_pull_up_down(c.CLOSED_DOOR_BUTTON_GPIO, pigpio.PUD_UP)

        self.pi.set_mode(c.ELECTROMAGNET_GPIO, pigpio.OUTPUT)
        self.pi.write(c.ELECTROMAGNET_GPIO, 0)

        self.pi.set_mode(c.LOWER_ARM_SERVO_GPIO, pigpio.OUTPUT)
        self.pi.set_mode(c.UPPER_ARM_SERVO_GPIO, pigpio.OUTPUT)
        self.pi.set_mode(c.DEADLOCK_MICROSERVO_GPIO, pigpio.OUTPUT)

        self._set_arm_pulse(c.LOWER_ARM_JOINT_NAME, self._lower_arm_pulse_us)
        self._set_arm_pulse(c.UPPER_ARM_JOINT_NAME, self._upper_arm_pulse_us)
        self._set_deadlock_pulse(c.DEADLOCK_LOCK_PULSE_US)

        self.get_logger().info("Connected to pigpio daemon and configured GPIO")

    def _poll_inputs(self) -> None:
        if self.use_mock or self.pi is None:
            pir = False
            button_pressed = False
        else:
            pir = bool(self.pi.read(c.PIR_SENSOR_GPIO))
            # Pull-up logic: low means pressed/closed.
            button_pressed = self.pi.read(c.CLOSED_DOOR_BUTTON_GPIO) == 0

        if self._last_pir is None or pir != self._last_pir:
            self.pir_pub.publish(Bool(data=pir))
            self._last_pir = pir

        if self._last_button is None or button_pressed != self._last_button:
            self.btn_pub.publish(Bool(data=button_pressed))
            self._last_button = button_pressed

    @staticmethod
    def _radians_to_pulse_us(position_rad: float) -> int:
        clamped = max(0.0, min(math.pi, position_rad))
        ratio = clamped / math.pi
        pulse = c.SERVO_PULSE_MIN_US + ratio * (
            c.SERVO_PULSE_MAX_US - c.SERVO_PULSE_MIN_US
        )
        return int(round(pulse))

    @staticmethod
    def _pulse_us_to_radians(pulse_us: int) -> float:
        clamped = max(c.SERVO_PULSE_MIN_US, min(c.SERVO_PULSE_MAX_US, pulse_us))
        ratio = (clamped - c.SERVO_PULSE_MIN_US) / (
            c.SERVO_PULSE_MAX_US - c.SERVO_PULSE_MIN_US
        )
        return ratio * math.pi

    def _set_deadlock_pulse(self, pulse_us: int) -> None:
        pulse = max(c.SERVO_PULSE_MIN_US, min(c.SERVO_PULSE_MAX_US, pulse_us))
        self._deadlock_pulse_us = pulse
        if not self.use_mock and self.pi is not None:
            self.pi.set_servo_pulsewidth(c.DEADLOCK_MICROSERVO_GPIO, pulse)
        self._publish_joint_state()

    def _set_arm_pulse(self, joint_name: str, pulse_us: int) -> None:
        pulse = max(c.SERVO_PULSE_MIN_US, min(c.SERVO_PULSE_MAX_US, pulse_us))
        if joint_name == c.LOWER_ARM_JOINT_NAME:
            self._lower_arm_pulse_us = pulse
            if not self.use_mock and self.pi is not None:
                self.pi.set_servo_pulsewidth(c.LOWER_ARM_SERVO_GPIO, pulse)
        elif joint_name == c.UPPER_ARM_JOINT_NAME:
            self._upper_arm_pulse_us = pulse
            if not self.use_mock and self.pi is not None:
                self.pi.set_servo_pulsewidth(c.UPPER_ARM_SERVO_GPIO, pulse)
        self._publish_joint_state()

    def _publish_joint_state(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            c.LOWER_ARM_JOINT_NAME,
            c.UPPER_ARM_JOINT_NAME,
            c.DEADLOCK_JOINT_NAME,
        ]
        msg.position = [
            self._pulse_us_to_radians(self._lower_arm_pulse_us),
            self._pulse_us_to_radians(self._upper_arm_pulse_us),
            self._pulse_us_to_radians(self._deadlock_pulse_us),
        ]
        self.joint_state_pub.publish(msg)

    def joint_cb(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        for idx, joint_name in enumerate(msg.name):
            if idx >= len(msg.position):
                break
            if joint_name == c.DEADLOCK_JOINT_NAME:
                self._set_deadlock_pulse(self._radians_to_pulse_us(msg.position[idx]))
                self.get_logger().debug(
                    f"Commanded {c.DEADLOCK_JOINT_NAME} to {msg.position[idx]:.3f} rad"
                )
            elif joint_name in (c.LOWER_ARM_JOINT_NAME, c.UPPER_ARM_JOINT_NAME):
                self._set_arm_pulse(
                    joint_name, self._radians_to_pulse_us(msg.position[idx])
                )
                self.get_logger().debug(
                    f"Commanded {joint_name} to {msg.position[idx]:.3f} rad"
                )

    def lock_cb(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        del req
        self._magnet_locked = True
        if not self.use_mock and self.pi is not None:
            self.pi.write(c.ELECTROMAGNET_GPIO, 1)
        self._set_deadlock_pulse(c.DEADLOCK_LOCK_PULSE_US)
        res.success = True
        res.message = "Door locked: magnet HIGH, deadlock engaged"
        return res

    def unlock_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        del req
        self._magnet_locked = False
        if not self.use_mock and self.pi is not None:
            self.pi.write(c.ELECTROMAGNET_GPIO, 0)
        self._set_deadlock_pulse(c.DEADLOCK_UNLOCK_PULSE_US)
        res.success = True
        res.message = "Door unlocked: magnet LOW, deadlock released"
        return res

    def destroy_node(self):
        if not self.use_mock and self.pi is not None:
            self.pi.write(c.ELECTROMAGNET_GPIO, 0)
            self.pi.set_servo_pulsewidth(c.LOWER_ARM_SERVO_GPIO, 0)
            self.pi.set_servo_pulsewidth(c.UPPER_ARM_SERVO_GPIO, 0)
            self.pi.set_servo_pulsewidth(c.DEADLOCK_MICROSERVO_GPIO, 0)
            self.pi.stop()
            self.pi = None
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: PiHardware = PiHardware()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
