from typing import Optional

import rclpy
from rclpy.node import Node, Publisher, Subscription
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2D

from robot_interfaces.msg import FaceBlendshapes


class FaceLockManager(Node):
    pir_sub: Subscription
    button_sub: Subscription
    blendshape_sub: Subscription
    detection_sub: Subscription
    state_pub: Publisher

    def __init__(self) -> None:
        super().__init__("face_lock_manager")
        # Subscriptions
        self.pir_sub: Subscription = self.create_subscription(
            Bool, "/io/pir", self.pir_cb, 10
        )
        self.button_sub: Subscription = self.create_subscription(
            Bool, "/io/button", self.button_cb, 10
        )
        self.blendshape_sub: Subscription = self.create_subscription(
            FaceBlendshapes, "/face_recognition/blendshapes", self.password_cb, 10
        )
        self.detection_sub: Subscription = self.create_subscription(
            Detection2D, "/face_recognition/detection", self.detection_cb, 10
        )

        # Publishers
        self.state_pub: Publisher = self.create_publisher(Bool, "/robot_state", 10)

    def pir_cb(self, msg: Bool) -> None:
        pass

    def detection_cb(self, msg: Detection2D) -> None:
        pass

    def password_cb(self, msg: FaceBlendshapes) -> None:
        pass

    def button_cb(self, msg: Bool) -> None:
        pass


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: FaceLockManager = FaceLockManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
