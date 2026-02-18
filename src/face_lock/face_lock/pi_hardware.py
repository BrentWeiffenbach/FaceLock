from typing import Any, Optional

import rclpy
from rclpy.node import Node, Publisher, Subscription
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class PiHardware(Node):
    pir_pub: Publisher
    btn_pub: Publisher
    joint_state_pub: Publisher
    joint_sub: Subscription
    lock_srv: Any
    unlock_srv: Any

    def __init__(self) -> None:
        super().__init__("pi_hardware")
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

    def joint_cb(self, msg: JointState) -> None:
        pass

    def lock_cb(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        return res

    def unlock_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        return res


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: PiHardware = PiHardware()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
