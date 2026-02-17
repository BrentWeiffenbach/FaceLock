import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import Image


class CameraNode(LifecycleNode):
    def __init__(self):
        super().__init__("camera")
        self.cap = None
        self.bridge = CvBridge()
        self.timer = None

        # Declare parameters
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)

    def on_configure(self, state):
        self.get_logger().info("Configuring camera")

        # Get parameters
        camera_index = self.get_parameter("camera_index").value
        self.fps = self.get_parameter("fps").value
        width = self.get_parameter("width").value
        height = self.get_parameter("height").value

        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera with index {camera_index}")
            return TransitionCallbackReturn.FAILURE

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Get actual camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create publisher
        self.image_pub = self.create_lifecycle_publisher(Image, "~/image_raw", 10)

        self.get_logger().info(
            f"Camera configured: {self.width}x{self.height} @ {self.fps} fps"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info("Activating camera")
        # Create timer to capture and publish frames
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.capture_and_publish)
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.get_logger().info("Deactivating camera")
        # Stop timer
        if self.timer:
            self.timer.cancel()
            self.timer = None
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        self.get_logger().info("Cleaning up camera")
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        return TransitionCallbackReturn.SUCCESS

    def capture_and_publish(self):
        """Capture frame and publish image"""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        # Publish image
        try:
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(image_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
