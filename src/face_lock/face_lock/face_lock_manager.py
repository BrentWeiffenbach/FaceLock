import os
from enum import Enum, auto
from typing import Optional

import rclpy
from lifecycle_msgs.srv import ChangeState
from rclpy.client import Client
from rclpy.node import Node, Publisher, Subscription
from rclpy.task import Future
from rclpy.timer import Timer
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from face_lock.constants import IDENTITIES_DIR, PASSWORDS_DIR
from robot_interfaces.msg import FaceBlendshapes

# region Init

DEACTIVATE_TIME = 15.0


class RobotState(Enum):
    DISABLED = auto()
    SETUP = auto()
    CHECKING = auto()
    UNLOCKING = auto()
    OPEN = auto()
    LOCKING = auto()


class FaceLockManager(Node):
    pir_sub: Subscription
    button_sub: Subscription
    blendshape_sub: Subscription
    detection_sub: Subscription
    state_pub: Publisher
    camera_state: Client
    face_recognition_state: Client
    arm_controller_state: Client
    configured: dict[str, bool]
    activated: dict[str, bool]
    config_timer: Timer
    deactivate_timer: Timer
    _robot_state: RobotState
    _password_set: bool
    _all_clients: list[tuple[Client, str]]

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
        self.password_matched_sub: Subscription = self.create_subscription(
            Bool, "/face_recognition/password_matched", self.password_matched_cb, 10
        )

        # Services
        self.setup_srv = self.create_service(
            Trigger, "setup_password", self.setup_password_cb
        )
        self.complete_setup_srv = self.create_service(
            Trigger,
            "/face_lock_manager/complete_setup",
            self.complete_setup_cb,
        )

        # Service Clients
        self.lock_door_srv = self.create_client(Trigger, "lock_door")
        self.unlock_door_srv = self.create_client(Trigger, "unlock_door")
        self.reset_arm_srv = self.create_client(Trigger, "reset_arm")

        # Lifecycle node management services
        self.camera_state = self.create_client(ChangeState, "/camera/change_state")
        self.face_recognition_state = self.create_client(
            ChangeState, "/face_recognition/change_state"
        )
        self.arm_controller_state = self.create_client(
            ChangeState, "/arm_controller/change_state"
        )
        self.password_gui_state = self.create_client(
            ChangeState, "/password_gui/change_state"
        )

        # Track configuration and activation state
        self.configured = {
            "camera": False,
            "face_recognition": False,
            "arm_controller": False,
            "password_gui": False,
        }
        self.configuring = {
            "camera": False,
            "face_recognition": False,
            "arm_controller": False,
            "password_gui": False,
        }
        self.activated = {
            "camera": False,
            "face_recognition": False,
            "arm_controller": False,
            "password_gui": False,
        }

        self._all_clients = [
            (self.camera_state, "camera"),
            (self.face_recognition_state, "face_recognition"),
            (self.arm_controller_state, "arm_controller"),
            (self.password_gui_state, "password_gui"),
        ]

        # Start timer to configure nodes after startup
        self.config_timer = self.create_timer(0.5, self.configure_nodes)

        # Timer to deactivate system if PIR is not triggered for a certain duration
        self.deactivate_timer = self.create_timer(
            DEACTIVATE_TIME, self.deactivate_system
        )
        self.deactivate_timer.cancel()

        # Password management — auto-detect if setup was completed in a previous run
        self._password_set = self._has_saved_data()
        if self._password_set:
            self.get_logger().info(
                "Saved identity/password data found — system ready for CHECKING"
            )
        else:
            self.get_logger().warn("No saved data found — please run Setup first")
        self._robot_state: RobotState = RobotState.DISABLED

    # region State
    @property
    def robot_state(self) -> RobotState:
        return self._robot_state

    @robot_state.setter
    def robot_state(self, new_state: RobotState) -> None:
        if new_state == self._robot_state:
            return
        if self._password_set:
            self.get_logger().info(
                f"State transition: {self._robot_state.name} -> {new_state.name}"
            )
            self._robot_state = new_state
            self._on_state_enter(new_state)
        elif new_state == RobotState.SETUP:
            self.get_logger().info("Entering SETUP state")
            self._robot_state = new_state
            self._on_state_enter(new_state)

    @staticmethod
    def _has_saved_data() -> bool:
        """Return True if at least one identity file exists on disk."""
        if not os.path.isdir(IDENTITIES_DIR):
            return False
        return any(f.endswith(".npy") for f in os.listdir(IDENTITIES_DIR))

    def _on_state_enter(self, state: RobotState) -> None:
        _handlers = {
            RobotState.DISABLED: self._enter_disabled,
            RobotState.SETUP: self._enter_setup,
            RobotState.CHECKING: self._enter_checking,
            RobotState.UNLOCKING: self._enter_unlocking,
            RobotState.OPEN: self._enter_open,
            RobotState.LOCKING: self._enter_locking,
        }
        handler = _handlers.get(state)
        if handler:
            handler()

    def _enter_disabled(self) -> None:
        self.get_logger().info("Entering DISABLED state, deactivating all nodes")
        self.deactivate_timer.cancel()
        for client, name in self._all_clients:
            self.deactivate_node(client, name)

    def _enter_setup(self) -> None:
        self.get_logger().info(
            "Entering SETUP state, activating camera, face_recognition and password_gui nodes"
        )
        clients = [
            (self.camera_state, "camera"),
            (self.face_recognition_state, "face_recognition"),
            (self.password_gui_state, "password_gui"),
        ]
        for client, name in clients:
            self.activate_node(client, name)

    def _enter_checking(self) -> None:
        self.get_logger().info(
            "Entering CHECKING state, activating camera, face_recognition and arm_controller nodes"
        )
        clients = [
            (self.camera_state, "camera"),
            (self.face_recognition_state, "face_recognition"),
            (self.arm_controller_state, "arm_controller"),
        ]
        for client, name in clients:
            self.activate_node(client, name)
        self.deactivate_timer.reset()

    def _enter_unlocking(self) -> None:
        self.get_logger().info(
            "Entering UNLOCKING state, unlocking door and resetting arm"
        )
        self.deactivate_timer.cancel()
        self.unlock_door_srv.wait_for_service()
        unlock_req = Trigger.Request()
        unlock_future = self.unlock_door_srv.call_async(unlock_req)

        self.reset_arm_srv.wait_for_service()
        reset_req = Trigger.Request()
        reset_future = self.reset_arm_srv.call_async(reset_req)

        def both_done_cb(_):
            if unlock_future.done() and reset_future.done():
                self.unlock_done_cb(unlock_future, reset_future)
            else:
                if not unlock_future.done():
                    unlock_future.add_done_callback(both_done_cb)
                if not reset_future.done():
                    reset_future.add_done_callback(both_done_cb)

        unlock_future.add_done_callback(both_done_cb)
        reset_future.add_done_callback(both_done_cb)

    def _enter_open(self) -> None:
        self.get_logger().info("Entering OPEN state, waiting for door to close")
        self.deactivate_timer.cancel()
        clients = [
            (self.camera_state, "camera"),
            (self.face_recognition_state, "face_recognition"),
        ]
        for client, name in clients:
            self.deactivate_node(client, name)

    def _enter_locking(self) -> None:
        self.get_logger().info("Entering LOCKING state, locking door")
        self.lock_door_srv.wait_for_service()
        lock_req = Trigger.Request()
        future = self.lock_door_srv.call_async(lock_req)
        future.add_done_callback(self.lock_done_cb)

    # region Sub Callbacks
    def pir_cb(self, msg: Bool) -> None:
        if msg.data and self.robot_state in (RobotState.DISABLED, RobotState.CHECKING):
            self.get_logger().info(
                f"PIR sensor triggered: {msg.data} | Robot State: {self.robot_state.name}"
            )
            if self.robot_state == RobotState.DISABLED:
                self.robot_state = RobotState.CHECKING
            else:
                self.deactivate_timer.reset()

    def password_cb(self, msg: FaceBlendshapes) -> None:
        if self.robot_state == RobotState.CHECKING:
            self.deactivate_timer.reset()

    def password_matched_cb(self, msg: Bool) -> None:
        if msg.data and self.robot_state == RobotState.CHECKING:
            self.get_logger().info("Password matched — unlocking door")
            self.robot_state = RobotState.UNLOCKING

    def button_cb(self, msg: Bool) -> None:
        if self.robot_state == RobotState.OPEN and msg.data:
            self.get_logger().info(
                "Door button indicates door has closed, locking door"
            )
            self.robot_state = RobotState.LOCKING

    # region Services
    def setup_password_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        self._password_set = False
        self.get_logger().info("Setting up new password, entering SETUP state")
        self.robot_state = RobotState.SETUP
        res.success = True
        res.message = "Password setup initiated"
        return res

    def complete_setup_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        self._password_set = self._has_saved_data()
        self.get_logger().info("Setup complete, returning to DISABLED state")
        self.robot_state = RobotState.DISABLED
        res.success = True
        res.message = "Setup complete; system returning to DISABLED state."
        return res

    # region async cbs
    def unlock_done_cb(self, unlock_fut: Future, reset_fut: Future) -> None:
        unlock_result = unlock_fut.result()
        reset_result = reset_fut.result()
        if (
            unlock_result
            and unlock_result.success
            and reset_result
            and reset_result.success
        ):
            self.get_logger().info("Door unlocked and arm reset successfully")
            self.robot_state = RobotState.OPEN
        else:
            self.get_logger().error(
                "Failed to unlock door or reset arm, trying again..."
            )
            self._enter_unlocking()

    def lock_done_cb(self, fut: Future) -> None:
        result = fut.result()
        if result and result.success:
            self.get_logger().info("Door locked successfully")
            self.robot_state = RobotState.DISABLED
        elif result and result.success is False:
            self.get_logger().error("Failed to lock door, trying again...")
            self._enter_locking()

    # region Lifecycle
    def configure_nodes(self) -> None:
        all_configured = True
        for client, name in self._all_clients:
            if not self.configured[name]:
                all_configured = False
                if self.configuring[name]:
                    continue  # already waiting on a response
                if client.wait_for_service(timeout_sec=1.0):
                    self.configure_node(client, name)
                else:
                    self.get_logger().warn(
                        f"Service not available for configuration: {client.srv_name}"
                    )
        if all_configured:
            self.config_timer.cancel()

    def configure_node(self, client: Client, name: str) -> None:
        self.get_logger().info(f"Configuring node: {client.srv_name}")
        self.configuring[name] = True
        req = ChangeState.Request()
        req.transition.id = 1  # Transition ID for "configure"
        future = client.call_async(req)
        future.add_done_callback(
            lambda f: self.handle_configure_result(f, client, name)
        )

    def handle_configure_result(
        self, future: Future, client: Client, name: str
    ) -> None:
        self.configuring[name] = False
        result = future.result()
        if result is not None and result.success:
            self.get_logger().info(f"{name} node configured successfully")
            self.configured[name] = True
        else:
            self.get_logger().error(f"Failed to configure {name} node")

    def activate_node(self, client: Client, name: str) -> None:
        if self.activated[name]:
            self.get_logger().info(f"{name} already activated")
            return
        req = ChangeState.Request()
        req.transition.id = 3  # Transition ID for "activate"
        future = client.call_async(req)
        future.add_done_callback(lambda f: self.handle_activate_result(f, name))

    def handle_activate_result(self, future: Future, name: str) -> None:
        result = future.result()
        if result is not None:
            self.get_logger().info(f"{name} activated successfully")
            self.activated[name] = True
        else:
            self.get_logger().error(f"Failed to activate {name}")

    def deactivate_node(self, client: Client, name: str) -> None:
        if not self.activated[name]:
            self.get_logger().info(f"{name} already deactivated")
            return
        req = ChangeState.Request()
        req.transition.id = 4  # Transition ID for "deactivate"
        future = client.call_async(req)
        future.add_done_callback(lambda f: self.handle_deactivate_result(f, name))

    def handle_deactivate_result(self, future: Future, name: str) -> None:
        result = future.result()
        if result is not None:
            self.get_logger().info(f"{name} deactivated successfully")
            self.activated[name] = False
        else:
            self.get_logger().error(f"Failed to deactivate {name}")

    def deactivate_system(self) -> None:
        self.get_logger().info("Inactivity timeout, disabling system")
        self.robot_state = RobotState.DISABLED


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: FaceLockManager = FaceLockManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
