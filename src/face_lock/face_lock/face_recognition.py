from typing import Optional

import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.node import Publisher, Subscription
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D

from robot_interfaces.msg import FaceBlendshapes


class FaceRecognitionNode(LifecycleNode):
    image_sub: Subscription
    blendshapes_pub: Publisher
    detection_pub: Publisher

    def __init__(self) -> None:
        super().__init__("face_recognition")

    def image_cb(self, msg: Image) -> None:
        pass

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring face recognition")

        # Subscribers
        self.image_sub: Subscription = self.create_subscription(
            Image, "/camera/image_raw", self.image_cb, 10
        )

        # Publishers
        # TODO: Only send a blendshape if identity for that frame is verified
        self.blendshapes_pub: Publisher = self.create_lifecycle_publisher(
            FaceBlendshapes, "/face_recognition/blendshapes", 10
        )

        self.detection_pub: Publisher = self.create_lifecycle_publisher(
            Detection2D, "/face_recognition/detection", 10
        )

        # TODO: Load face recognition model
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating face recognition")
        # TODO: Start processing camera frames
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating face recognition")
        # TODO: Stop processing
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up face recognition")
        # TODO: Unload model
        return TransitionCallbackReturn.SUCCESS


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: FaceRecognitionNode = FaceRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
