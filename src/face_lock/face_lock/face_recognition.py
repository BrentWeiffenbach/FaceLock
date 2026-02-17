import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D

from robot_interfaces.msg import FaceBlendshapes


class FaceRecognitionNode(LifecycleNode):
    def __init__(self):
        super().__init__("face_recognition")

    def on_configure(self, state):
        self.get_logger().info("Configuring face recognition")

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_cb, 10
        )

        # Publishers
        self.blendshapes_pub = self.create_lifecycle_publisher(
            FaceBlendshapes, "/face_recognition/blendshapes", 10
        )

        self.detection_pub = self.create_lifecycle_publisher(
            Detection2D, "/face_recognition/detection", 10
        )

        # TODO: Load face recognition model
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info("Activating face recognition")
        # TODO: Start processing camera frames
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info("Deactivating face recognition")
        # TODO: Stop processing
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info("Cleaning up face recognition")
        # TODO: Unload model
        return TransitionCallbackReturn.SUCCESS


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
