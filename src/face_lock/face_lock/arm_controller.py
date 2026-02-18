from typing import Any, Optional

import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.node import Publisher, Subscription
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2D


class ArmController(LifecycleNode):
    joint_pub: Publisher
    joint_states_sub: Subscription
    detection_sub: Subscription
    reset_arm_srv: Any

    def __init__(self) -> None:
        super().__init__("arm_controller")

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring arm controller")
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

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating arm controller")
        # TODO: Start processing camera frames
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating arm controller")
        # TODO: Stop processing
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up arm controller")
        return TransitionCallbackReturn.SUCCESS

    def joint_states_cb(self, msg: JointState) -> None:
        pass

    def detection_cb(self, msg: Detection2D) -> None:
        pass

    def reset_arm_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        return res


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: ArmController = ArmController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
