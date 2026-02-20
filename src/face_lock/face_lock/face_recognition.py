import os
from typing import Optional

import mediapipe as mp
import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.node import Publisher, Subscription
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D

from robot_interfaces.msg import FaceBlendshapes

package_share_dir = get_package_share_directory("face_lock")
model_path = "/workspaces/FaceLock/src/face_lock/tasks/face_landmarker.task"

# TODO: use python face_recognition to verify identity before blendshapes


class FaceRecognitionNode(LifecycleNode):
    image_sub: Subscription
    blendshapes_pub: Publisher
    detection_pub: Publisher
    cur_image: Optional[Image] = None

    def __init__(self) -> None:
        super().__init__("face_recognition")
        if not os.path.isfile(model_path):
            self.get_logger().warn(
                f"FaceLandmarker model file does not exist: {model_path}"
            )

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring face recognition")

        # Subscribers
        self.image_sub: Subscription = self.create_subscription(
            Image, "/camera/image_raw", self.image_cb, 10
        )

        # Publishers
        self.blendshapes_pub: Publisher = self.create_lifecycle_publisher(
            FaceBlendshapes, "/face_recognition/blendshapes", 10
        )

        self.detection_pub: Publisher = self.create_lifecycle_publisher(
            Detection2D, "/face_recognition/detection", 10
        )

        self.debug_image_pub: Publisher = self.create_lifecycle_publisher(
            Image, "/face_recognition/debug_landmarks", 10
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating face recognition")

        if not os.path.isfile(model_path):
            self.get_logger().error(
                f"FaceLandmarker model file does not exist: {model_path}"
            )
            return TransitionCallbackReturn.FAILURE

        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=VisionRunningMode.IMAGE,
        )

        try:
            self.get_logger().debug(
                f"Creating FaceLandmarker with model path: {model_path}"
            )
            self.landmarker = FaceLandmarker.create_from_options(options)
            self.get_logger().info("FaceLandmarker successfully created and ready.")
        except Exception as e:
            self.get_logger().error(f"Failed to create FaceLandmarker: {e}")
            return TransitionCallbackReturn.FAILURE

        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating face recognition")
        self.landmarker.close()
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up face recognition")
        return TransitionCallbackReturn.SUCCESS

    def image_cb(self, msg: Image) -> None:
        if not msg:
            return

        if not hasattr(self, "landmarker"):
            self.get_logger().warn("Landmarker not initialized yet; skipping image.")
            return

        try:
            np_image = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
                (msg.height, msg.width, -1)
            )

            if msg.encoding.lower() in ("bgr8", "bgr"):
                self.get_logger().debug("Converting BGR to RGB")
                np_image = np_image[:, :, ::-1].copy()

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            # Detect faces and blendshapes
            detection_result = self.landmarker.detect(mp_image)

            if not detection_result.face_blendshapes:
                self.get_logger().debug("No face blendshapes detected.")
                return

            blendshapes = detection_result.face_blendshapes[0]
            self.get_logger().debug(f"Detected {len(blendshapes)} blendshapes.")

            blend_msg = FaceBlendshapes()
            blend_msg.header.stamp = self.get_clock().now().to_msg()
            blend_msg.header.frame_id = msg.header.frame_id
            blend_msg.coefficients = [float(cat.score) for cat in blendshapes]
            blend_msg.shape_names = [cat.category_name for cat in blendshapes]

            self.get_logger().debug("Publishing blendshapes message.")
            self.blendshapes_pub.publish(blend_msg)

            # Debug Image
            annotated = FaceRecognitionNode.draw_landmarks_on_image(
                np_image, detection_result
            )
            debug_msg = Image()
            debug_msg.header.stamp = blend_msg.header.stamp
            debug_msg.header.frame_id = msg.header.frame_id
            debug_msg.height = annotated.shape[0]
            debug_msg.width = annotated.shape[1]
            debug_msg.encoding = "rgb8"
            debug_msg.is_bigendian = False
            debug_msg.step = annotated.shape[1] * 3
            debug_msg.data = annotated.astype(np.uint8).tobytes()
            self.get_logger().debug("Publishing debug image.")
            self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.

            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return annotated_image


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: FaceRecognitionNode = FaceRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
