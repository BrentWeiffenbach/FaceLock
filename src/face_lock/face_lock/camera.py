from typing import Any, Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.node import Publisher, Timer
from sensor_msgs.msg import Image


class CameraNode(LifecycleNode):
    cap: Optional[cv2.VideoCapture]
    bridge: CvBridge
    timer: Optional[Timer]
    fps: float
    width: int
    height: int
    image_pub: Optional[Publisher]

    def __init__(self) -> None:
        super().__init__("camera")
        self.cap: Optional[cv2.VideoCapture] = None
        self.bridge: CvBridge = CvBridge()
        self.timer: Optional[Timer] = None
        self.fps: float = 60.0
        self.width: int = 1920
        self.height: int = 1080
        self.image_pub: Optional[Publisher] = None

        # Declare parameters
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("fps", 60.0)
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring camera")

        # Get parameters
        camera_index_param: Any = self.get_parameter("camera_index").value
        fps_param: Any = self.get_parameter("fps").value
        width_param: Any = self.get_parameter("width").value
        height_param: Any = self.get_parameter("height").value

        camera_index: int = (
            int(camera_index_param) if camera_index_param is not None else 0
        )
        self.fps = float(fps_param) if fps_param is not None else 60.0
        width: int = int(width_param) if width_param is not None else 1920
        height: int = int(height_param) if height_param is not None else 1080

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

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating camera")
        # Create timer to capture and publish frames
        timer_period: float = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.capture_and_publish)
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating camera")
        # Stop timer
        if self.timer:
            self.timer.cancel()
            self.timer = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up camera")
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        return TransitionCallbackReturn.SUCCESS

    def capture_and_publish(self) -> None:
        """Capture frame and publish image"""
        if self.cap is None:
            return
        ret: bool
        frame: Any
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        # Publish image
        try:
            image_msg: Image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            if self.image_pub:
                self.image_pub.publish(image_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: CameraNode = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
