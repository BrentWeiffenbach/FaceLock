import math
from typing import Any, Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.node import Publisher, Timer
from sensor_msgs.msg import Image, JointState

from face_lock.constants import (
    LOWER_ARM_IK_DIRECTION,
    LOWER_ARM_IK_ZERO_OFFSET,
    LOWER_ARM_JOINT_NAME,
    UPPER_ARM_IK_DIRECTION,
    UPPER_ARM_IK_ZERO_OFFSET,
    UPPER_ARM_JOINT_NAME,
)


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
        self.camera_index: int = 0

        # Current servo angles for image derotation (default = home position)
        self._s1_servo: float = math.pi / 2
        self._s2_servo: float = math.pi / 2
        self._rotation_offset_deg: float = 0.0

        # Declare parameters
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("fps", 60.0)
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)
        self.declare_parameter("rotation_offset_deg", 0.0)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring camera")

        # Get parameters
        camera_index_param: Any = self.get_parameter("camera_index").value
        fps_param: Any = self.get_parameter("fps").value
        width_param: Any = self.get_parameter("width").value
        height_param: Any = self.get_parameter("height").value

        self.camera_index = (
            int(camera_index_param) if camera_index_param is not None else 0
        )
        self.fps = float(fps_param) if fps_param is not None else 30.0
        self.width = int(width_param) if width_param is not None else 1920
        self.height = int(height_param) if height_param is not None else 1080

        rot_param: Any = self.get_parameter("rotation_offset_deg").value
        self._rotation_offset_deg = float(rot_param) if rot_param is not None else 0.0

        # Create publisher
        self.image_pub = self.create_lifecycle_publisher(Image, "~/image_raw", 10)

        # Subscribe to arm joint states so we can derotate the published image
        # based on the camera's current physical orientation.
        # Regular subscription (not lifecycle) so it persists across activations.
        self.create_subscription(
            JointState, "/arm/joint_states", self._joint_states_cb, 10
        )

        self.get_logger().info(
            f"Camera parameters set: {self.width}x{self.height} @ {self.fps} fps, "
            f"rotation_offset={self._rotation_offset_deg:.1f}°"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating camera")

        # Open camera with explicit V4L2 backend
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(
                f"Failed to open camera with index {self.camera_index}"
            )
            return TransitionCallbackReturn.FAILURE

        # Limit internal buffer to 1 frame to avoid stale-frame buildup
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read back actual camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Verify the stream actually starts — read() triggers VIDIOC_STREAMON
        stream_ok = False
        for _ in range(30):
            ret, _ = self.cap.read()
            if ret:
                stream_ok = True
                break

        if not stream_ok:
            self.get_logger().error(
                "Camera stream failed to start — check resolution/bandwidth"
            )
            self.cap.release()
            self.cap = None
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info(
            f"Camera opened: {self.width}x{self.height} @ {self.fps} fps"
        )

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

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up camera")
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        return TransitionCallbackReturn.SUCCESS

    def _joint_states_cb(self, msg: JointState) -> None:
        """Track current servo angles so capture_and_publish can derotate the image."""
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                break
            if name == LOWER_ARM_JOINT_NAME:
                self._s1_servo = float(msg.position[idx])
            elif name == UPPER_ARM_JOINT_NAME:
                self._s2_servo = float(msg.position[idx])

    def capture_and_publish(self) -> None:
        """Capture frame, derotate based on arm orientation, and publish."""
        if self.cap is None:
            return
        ret: bool
        frame: Any
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        # ── Image derotation ─────────────────────────────────────────
        # The camera is fixed to the tip of linkage-2, so it rolls (rotates
        # around its optical axis = world +Z) as the arm sweeps.  We compute
        # the camera's world-frame "up" direction and rotate the image the
        # opposite way so that image-up always means world-up.
        #
        # Physical frame geometry (per hardware description):
        #   Base servo:  rotation axis = +Z_world
        #   Elbow servo: flipped 180° around X from base → rotation axis = -Z_world
        #
        # Camera-up world angle = q1 + q2_base + q2_elbow
        #   q2_base = (q1 - q1_home) contributes +1 per radian of q1
        #   q2_elbow: elbow rotates around -Z, so it contributes -1 per radian of q2
        #
        # camera_up_angle_from_+X = q1 - q2
        # Correction = -(camera_up_angle - home_angle) = -(q1 - q2 - π/2)
        # BUT: when camera rolled CCW by φ, content appears CW; correct by applying -φ.
        # correction_deg = (q1 - q2 - π/2) × (180/π)
        # Verified: home→0°, arm-right→-90°(CW), arm-left→+90°(CCW) ✓
        q1_cam = LOWER_ARM_IK_DIRECTION * (self._s1_servo - LOWER_ARM_IK_ZERO_OFFSET)
        q2_cam = UPPER_ARM_IK_DIRECTION * (self._s2_servo - UPPER_ARM_IK_ZERO_OFFSET)
        correction_deg = (
            (q1_cam - q2_cam - math.pi / 2) * (180.0 / math.pi)
            + self._rotation_offset_deg
        )
        if abs(correction_deg) > 0.5:
            h_f, w_f = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w_f / 2.0, h_f / 2.0), correction_deg, 1.0)
            frame = cv2.warpAffine(frame, M, (w_f, h_f))

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
