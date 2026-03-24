"""Face recognition lifecycle node."""

import json
import os
import time
from typing import List, Optional

import face_recognition as face_recog_lib  # type: ignore
import mediapipe as mp
import numpy as np
import rclpy
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as RunningMode,
)
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    FaceLandmarksConnections,
)
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.node import Publisher
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from face_lock.constants import (
    BLENDSHAPE_THRESHOLD,
    IDENTITIES_DIR,
    IGNORED_BLENDSHAPES,
    PASSWORDS_DIR,
)
from robot_interfaces.msg import FaceBlendshapes
from robot_interfaces.srv import SetName

MODEL_PATH = "/workspaces/FaceLock/src/face_lock/tasks/face_landmarker.task"
ID_CHECK_INTERVAL = 1.0  # seconds between full face_recognition checks
ID_REC_SCALE = 2  # downsample factor: half-resolution balances speed and accuracy


class FaceRecognitionNode(LifecycleNode):
    """Lifecycle node for face recognition and blendshape detection."""

    def __init__(self) -> None:
        super().__init__("face_recognition")
        self._face_encodings: List[np.ndarray] = []
        self._active_encoding: Optional[np.ndarray] = None
        self._active_identity_name: str = ""
        self._known_encodings: List[np.ndarray] = []  # all registered identities
        self._known_names: List[str] = []
        self._id_verified: bool = False  # cached identity result
        self._id_matched_name: str = ""
        self._id_last_check: float = 0.0
        self._last_log_time: float = 0.0
        self._all_passwords: List[List[List[str]]] = []
        self._pw_progress: List[int] = []
        self._pw_waiting_neutral: List[bool] = []
        self._password: List[List[str]] = []
        self._active_password_name: str = ""
        self._cur_raw_image: Optional[np.ndarray] = None
        self.landmarker: Optional[FaceLandmarker] = None
        self._frame_count: int = 0

        if not os.path.isfile(MODEL_PATH):
            self.get_logger().warn(f"Model missing: {MODEL_PATH}")

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring")
        self.create_subscription(Image, "/camera/image_raw", self.image_cb, 1)
        self.blendshapes_pub: Publisher = self.create_lifecycle_publisher(
            FaceBlendshapes, "/face_recognition/blendshapes", 10
        )
        self.debug_image_pub: Publisher = self.create_lifecycle_publisher(
            Image, "/face_recognition/debug_landmarks", 10
        )
        self.password_matched_pub: Publisher = self.create_lifecycle_publisher(
            Bool, "/face_recognition/password_matched", 10
        )

        for srv, cb in [
            ("/face_recognition/record_person", self._record_person_cb),
            ("/face_recognition/save_identity", self._save_identity_cb),
            ("/face_recognition/record_action", self._record_action_cb),
            ("/face_recognition/save_password", self._save_password_cb),
            ("/face_recognition/password_state", self._password_state_cb),
            ("/face_recognition/list_identities", self._list_identities_cb),
            ("/face_recognition/list_passwords", self._list_passwords_cb),
            ("/face_recognition/load_identity", self._load_identity_cb),
            ("/face_recognition/load_password", self._load_password_cb),
            ("/face_recognition/delete_identity", self._delete_identity_cb),
            ("/face_recognition/delete_password", self._delete_password_cb),
        ]:
            self.create_service(Trigger, srv, cb)

        for srv, cb in [
            ("/face_recognition/set_identity_name", self._set_identity_name_cb),
            ("/face_recognition/set_password_name", self._set_password_name_cb),
            ("/face_recognition/remove_password_step", self._remove_password_step_cb),
        ]:
            self.create_service(SetName, srv, cb)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating")
        if not os.path.isfile(MODEL_PATH):
            self.get_logger().error(f"Model file not found: {MODEL_PATH}")
            return TransitionCallbackReturn.FAILURE

        try:
            base_opts = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
            opts = FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                running_mode=RunningMode.IMAGE,
            )
            self.landmarker = FaceLandmarker.create_from_options(opts)
        except Exception as e:
            self.get_logger().error(f"Failed to load landmarker: {e}")
            return TransitionCallbackReturn.FAILURE

        count = self._load_all_identities()
        if count:
            self.get_logger().info(
                f"Loaded {count} identit{'y' if count == 1 else 'ies'}: {self._known_names}"
            )
        else:
            self.get_logger().warn(
                "No saved identities found — blendshapes will publish unverified"
            )
        pw_count = self._load_all_passwords()
        if pw_count:
            self.get_logger().info(f"Loaded {pw_count} password(s)")
        else:
            self.get_logger().warn("No saved passwords found")

        self._frame_count = 0
        self._id_last_check = 0.0
        self._last_log_time = 0.0
        self.get_logger().info("Landmarker ready, processing images")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating")
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
        self._cur_raw_image = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._face_encodings.clear()
        self._active_encoding = None
        self._active_identity_name = ""
        self._known_encodings.clear()
        self._known_names.clear()
        self._all_passwords.clear()
        self._pw_progress.clear()
        self._pw_waiting_neutral.clear()
        self._password.clear()
        self._active_password_name = ""
        self._cur_raw_image = None
        return TransitionCallbackReturn.SUCCESS

    def _load_all_identities(self) -> int:
        """Reload all saved identity encodings from disk into _known_encodings."""
        self._known_encodings.clear()
        self._known_names.clear()
        for name in self._list_npy(IDENTITIES_DIR):
            try:
                enc = np.load(os.path.join(IDENTITIES_DIR, f"{name}.npy"))
                self._known_encodings.append(enc)
                self._known_names.append(name)
            except Exception as e:
                self.get_logger().warn(f"Could not load identity '{name}': {e}")
        return len(self._known_encodings)

    def _load_all_passwords(self) -> int:
        """Reload all saved password sequences from disk into _all_passwords."""
        self._all_passwords.clear()
        self._pw_progress.clear()
        self._pw_waiting_neutral.clear()
        for name in self._list_npy(PASSWORDS_DIR):
            try:
                data = np.load(
                    os.path.join(PASSWORDS_DIR, f"{name}.npy"), allow_pickle=True
                )
                self._all_passwords.append([list(step) for step in data])
                self._pw_progress.append(0)
                self._pw_waiting_neutral.append(False)
            except Exception as e:
                self.get_logger().warn(f"Could not load password '{name}': {e}")
        return len(self._all_passwords)

    def image_cb(self, msg: Image) -> None:
        if not self.landmarker:
            return

        self._frame_count += 1
        now = time.monotonic()
        log_now = now - self._last_log_time >= 1.0

        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, -1)
            )
            rgb = (
                raw[:, :, ::-1].copy() if "bgr" in msg.encoding.lower() else raw.copy()
            )
            self._cur_raw_image = rgb

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)

            if not res.face_landmarks:
                if log_now:
                    self.get_logger().info("Checking | no face detected")
                    self._last_log_time = now
                return

            annotated = self.draw_landmarks_on_image(rgb, res)
            debug_msg = Image()
            debug_msg.header = msg.header
            debug_msg.height, debug_msg.width = annotated.shape[:2]
            debug_msg.encoding = "rgb8"
            debug_msg.step = annotated.shape[1] * 3
            debug_msg.data = annotated.tobytes()
            self.debug_image_pub.publish(debug_msg)

            if not res.face_blendshapes:
                return

            cats = res.face_blendshapes[0]
            active = sorted(
                c.category_name
                for c in cats
                if c.score > BLENDSHAPE_THRESHOLD
                and c.category_name not in IGNORED_BLENDSHAPES
            )

            # Always publish blendshapes for any detected face (display + timer reset)
            blend_msg = FaceBlendshapes()
            blend_msg.header = msg.header
            blend_msg.coefficients = [float(c.score) for c in cats]
            blend_msg.shape_names = [c.category_name for c in cats]
            self.blendshapes_pub.publish(blend_msg)

            # --- Identity verification cache (runs at most once per ID_CHECK_INTERVAL) ---
            if self._known_encodings:
                now = time.monotonic()
                if now - self._id_last_check >= ID_CHECK_INTERVAL:
                    small = np.ascontiguousarray(rgb[::ID_REC_SCALE, ::ID_REC_SCALE])
                    locs = face_recog_lib.face_locations(small)
                    prev_verified = self._id_verified
                    if locs:
                        encs = face_recog_lib.face_encodings(small, locs)
                        if encs:
                            matches = face_recog_lib.compare_faces(
                                self._known_encodings, encs[0]
                            )
                            matched = [
                                self._known_names[i] for i, m in enumerate(matches) if m
                            ]
                            self._id_verified = bool(matched)
                            self._id_matched_name = matched[0] if matched else ""
                        else:
                            self._id_verified = False
                            self._id_matched_name = ""
                    else:
                        self._id_verified = False
                        self._id_matched_name = ""
                    self._id_last_check = now
                    if self._id_verified != prev_verified:
                        if self._id_verified:
                            self.get_logger().info(
                                f"Identity verified: {self._id_matched_name}"
                            )
                        else:
                            self.get_logger().warn(
                                "Identity lost: no registered face matched"
                            )

            # Periodic status line — always visible so you can see what's happening
            if log_now:
                self._last_log_time = now
                id_str = self._id_matched_name if self._id_verified else "UNVERIFIED"
                # Use a lower threshold for logging so neutral faces show activity
                visible = sorted(
                    f"{c.category_name}:{c.score:.2f}"
                    for c in cats
                    if c.score > 0.15 and c.category_name not in IGNORED_BLENDSHAPES
                )
                bs_str = "  ".join(visible) if visible else "(neutral)"
                if self._all_passwords:
                    pw_str = "  ".join(
                        f"pw[{i}]: {p}/{len(pw)} steps"
                        for i, (p, pw) in enumerate(
                            zip(self._pw_progress, self._all_passwords)
                        )
                    )
                else:
                    pw_str = "no passwords loaded"
                self.get_logger().info(
                    f"Checking | id={id_str} | {pw_str}\n  shapes: {bs_str}"
                )

            # Password check: only when identity is confirmed (or no identities registered)
            if self._all_passwords and (self._id_verified or not self._known_encodings):
                self._check_password(set(active))

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def _check_password(self, active: set) -> None:
        """Advance per-password step counters; publish when a full sequence matches."""
        for i, pw in enumerate(self._all_passwords):
            if not pw:
                continue
            if not active:
                # Neutral face clears the wait-for-neutral flag so next step can register
                self._pw_waiting_neutral[i] = False
                continue
            if self._pw_waiting_neutral[i]:
                continue
            expected = set(pw[self._pw_progress[i]])
            if expected and expected.issubset(active):
                self._pw_progress[i] += 1
                self.get_logger().info(
                    f"Password step {self._pw_progress[i]}/{len(pw)} matched"
                )
                if self._pw_progress[i] >= len(pw):
                    self.get_logger().info("Password matched — publishing signal")
                    self.password_matched_pub.publish(Bool(data=True))
                    self._pw_progress[i] = 0
                else:
                    self._pw_waiting_neutral[i] = True
            elif self._pw_progress[i] > 0:
                self.get_logger().info(
                    f"Password mismatch at step {self._pw_progress[i] + 1} — resetting"
                )
                self._pw_progress[i] = 0
                self._pw_waiting_neutral[i] = False

    def _wait_for_image(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        start = time.time()
        while self._cur_raw_image is None and (time.time() - start) < timeout:
            time.sleep(0.05)
        return self._cur_raw_image

    def _record_person_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        rgb = self._wait_for_image()
        if rgb is None:
            res.success, res.message = False, "No image received."
            return res

        locs = face_recog_lib.face_locations(rgb)
        if not locs:
            res.success, res.message = False, "No face detected."
            return res

        encs = face_recog_lib.face_encodings(rgb, locs)
        if not encs:
            res.success, res.message = False, "Could not encode face."
            return res

        self._face_encodings.append(encs[0])
        res.success = True
        res.message = f"Recorded sample {len(self._face_encodings)}."
        return res

    def _remove_password_step_cb(
        self, req: SetName.Request, res: SetName.Response
    ) -> SetName.Response:
        try:
            idx = int(req.name)
        except ValueError:
            res.success, res.message = False, f"Invalid index '{req.name}'."
            return res
        if idx < 0 or idx >= len(self._password):
            res.success, res.message = (
                False,
                (f"Index {idx} out of range (0–{len(self._password) - 1})."),
            )
            return res
        removed = self._password.pop(idx)
        res.success = True
        res.message = f"Step {idx + 1} removed: {removed}"
        return res

    def _set_identity_name_cb(
        self, req: SetName.Request, res: SetName.Response
    ) -> SetName.Response:
        self._active_identity_name = req.name.strip()
        res.success = True
        res.message = f"Identity name set to '{self._active_identity_name}'."
        return res

    def _set_password_name_cb(
        self, req: SetName.Request, res: SetName.Response
    ) -> SetName.Response:
        self._active_password_name = req.name.strip()
        res.success = True
        res.message = f"Password name set to '{self._active_password_name}'."
        return res

    def _save_identity_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        name = self._active_identity_name
        if not self._face_encodings:
            res.success, res.message = False, "No samples to save."
            return res
        if not name:
            res.success, res.message = False, "No identity name set."
            return res
        os.makedirs(IDENTITIES_DIR, exist_ok=True)
        avg = np.mean(self._face_encodings, axis=0)
        np.save(os.path.join(IDENTITIES_DIR, f"{name}.npy"), avg)
        self._active_encoding = avg
        count = len(self._face_encodings)
        self._face_encodings.clear()
        self._load_all_identities()  # refresh known encodings
        res.success = True
        res.message = f"Identity '{name}' saved ({count} samples)."
        return res

    def _record_action_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        if self._active_encoding is None:
            res.success, res.message = False, "No identity loaded."
            return res
        rgb = self._wait_for_image()
        if rgb is None:
            res.success, res.message = False, "No image received."
            return res

        locs = face_recog_lib.face_locations(rgb)
        if not locs:
            res.success, res.message = False, "No face detected."
            return res

        encs = face_recog_lib.face_encodings(rgb, locs)
        if not encs:
            res.success, res.message = False, "Could not encode face."
            return res

        matches = face_recog_lib.compare_faces([self._active_encoding], encs[0])
        if not matches[0]:
            res.success, res.message = False, "Identity mismatch."
            return res

        if not self.landmarker:
            res.success, res.message = False, "Landmarker inactive."
            return res

        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_res = self.landmarker.detect(mp_img)
        except Exception as e:
            res.success, res.message = False, f"Detection error: {e}"
            return res

        if not mp_res.face_blendshapes:
            res.success, res.message = False, "No blendshapes detected."
            return res

        active_shapes = sorted(
            [
                c.category_name
                for c in mp_res.face_blendshapes[0]
                if c.score > BLENDSHAPE_THRESHOLD
                and c.category_name not in IGNORED_BLENDSHAPES
            ]
        )
        self._password.append(active_shapes)

        res.success = True
        res.message = f"Action recorded step {len(self._password)}: {active_shapes}"
        return res

    def _save_password_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        name = self._active_password_name
        if not self._password:
            res.success, res.message = False, "No steps recorded."
            return res
        if not name:
            res.success, res.message = False, "No password name set."
            return res
        os.makedirs(PASSWORDS_DIR, exist_ok=True)
        np.save(
            os.path.join(PASSWORDS_DIR, f"{name}.npy"),
            np.array(self._password, dtype=object),
        )
        count = len(self._password)
        self._password.clear()
        self._load_all_passwords()
        res.success = True
        res.message = f"Password '{name}' saved ({count} steps)."
        return res

    def _password_state_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        res.success = True
        res.message = json.dumps(self._password)
        return res

    def _list_identities_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        names = self._list_npy(IDENTITIES_DIR)
        res.success = True
        res.message = json.dumps(names)
        return res

    def _list_passwords_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        names = self._list_npy(PASSWORDS_DIR)
        res.success = True
        res.message = json.dumps(names)
        return res

    def _load_identity_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        name = self._active_identity_name
        if not name:
            res.success, res.message = False, "No identity name set."
            return res
        path = os.path.join(IDENTITIES_DIR, f"{name}.npy")
        if not os.path.isfile(path):
            res.success, res.message = False, f"Identity '{name}' not found."
            return res
        self._active_encoding = np.load(path)
        res.success = True
        res.message = f"Identity '{name}' loaded."
        return res

    def _load_password_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        name = self._active_password_name
        if not name:
            res.success, res.message = False, "No password name set."
            return res
        path = os.path.join(PASSWORDS_DIR, f"{name}.npy")
        if not os.path.isfile(path):
            res.success, res.message = False, f"Password '{name}' not found."
            return res
        self._password = [list(step) for step in np.load(path, allow_pickle=True)]
        res.success = True
        res.message = f"Password '{name}' loaded ({len(self._password)} steps)."
        return res

    def _delete_identity_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        name = self._active_identity_name
        if not name:
            res.success, res.message = False, "No identity name set."
            return res
        path = os.path.join(IDENTITIES_DIR, f"{name}.npy")
        if not os.path.isfile(path):
            res.success, res.message = False, f"Identity '{name}' not found."
            return res
        os.remove(path)
        self._active_encoding = None
        self._active_identity_name = ""
        self._load_all_identities()  # refresh known encodings
        res.success = True
        res.message = f"Identity '{name}' deleted."
        return res

    def _delete_password_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        name = self._active_password_name
        if not name:
            res.success, res.message = False, "No password name set."
            return res
        path = os.path.join(PASSWORDS_DIR, f"{name}.npy")
        if not os.path.isfile(path):
            res.success, res.message = False, f"Password '{name}' not found."
            return res
        os.remove(path)
        self._password.clear()
        self._active_password_name = ""
        self._load_all_passwords()
        res.success = True
        res.message = f"Password '{name}' deleted."
        return res

    @staticmethod
    def _list_npy(directory: str) -> List[str]:
        if not os.path.isdir(directory):
            return []
        return sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".npy"))

    @staticmethod
    def draw_landmarks_on_image(
        rgb_image: np.ndarray, detection_result: FaceLandmarkerResult
    ) -> np.ndarray:
        """Draws landmarks on the given image.

        Args:
            rgb_image: A NumPy array representing the RGB image.
            detection_result: The result object from FaceLandmarker.

        Returns:
            A new NumPy array with the landmarks drawn.
        """
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return annotated_image


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = FaceRecognitionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
