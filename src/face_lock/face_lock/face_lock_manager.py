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

from robot_interfaces.msg import FaceBlendshapes

DEACTIVATE_TIME = 15.0


class RobotState(Enum):
    DISABLED = auto()
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

        # Services
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

        # Track configuration and activation state
        self.configured = {
            "camera": False,
            "face_recognition": False,
            "arm_controller": False,
        }
        self.configuring = {
            "camera": False,
            "face_recognition": False,
            "arm_controller": False,
        }
        self.activated = {
            "camera": False,
            "face_recognition": False,
            "arm_controller": False,
        }

        # Start timer to configure nodes after startup
        self.config_timer = self.create_timer(0.5, self.configure_nodes)

        # Timer to deactivate system if PIR is not triggered for a certain duration
        self.deactivate_timer = self.create_timer(
            DEACTIVATE_TIME, self.deactivate_system
        )
        self.deactivate_timer.cancel()

        # Password management
        self.password: list[FaceBlendshapes] = []
        # TODO: Load password blendshapes from file or parameter
        self.entered_password: list[FaceBlendshapes] = []

        self._robot_state: RobotState = RobotState.DISABLED

    @property
    def robot_state(self) -> RobotState:
        return self._robot_state

    @robot_state.setter
    def robot_state(self, new_state: RobotState) -> None:
        if new_state == self._robot_state:
            return
        self.get_logger().info(
            f"State transition: {self._robot_state.name} -> {new_state.name}"
        )
        self._robot_state = new_state
        self._on_state_enter(new_state)

    def _on_state_enter(self, state: RobotState) -> None:
        if state == RobotState.DISABLED:
            self.deactivate_timer.cancel()
            self.entered_password.clear()
            clients: list[tuple[Client, str]] = [
                (self.camera_state, "camera"),
                (self.face_recognition_state, "face_recognition"),
                (self.arm_controller_state, "arm_controller"),
            ]
            for client, name in clients:
                self.deactivate_node(client, name)
        elif state == RobotState.CHECKING:
            clients = [
                (self.camera_state, "camera"),
                (self.face_recognition_state, "face_recognition"),
                (self.arm_controller_state, "arm_controller"),
            ]
            for client, name in clients:
                self.activate_node(client, name)
            self.deactivate_timer.reset()

    def configure_nodes(self) -> None:
        clients = [
            (self.camera_state, "camera"),
            (self.face_recognition_state, "face_recognition"),
            (self.arm_controller_state, "arm_controller"),
        ]
        all_configured = True
        for client, name in clients:
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

    def pir_cb(self, msg: Bool) -> None:
        if msg.data and self.robot_state in (RobotState.DISABLED, RobotState.CHECKING):
            self.get_logger().info(f"PIR sensor triggered: {msg.data}")
            self.deactivate_timer.reset()
            if self.robot_state == RobotState.DISABLED:
                self.robot_state = RobotState.CHECKING

    def password_cb(self, msg: FaceBlendshapes) -> None:
        if self.robot_state != RobotState.CHECKING:
            return
        self.get_logger().info("Blendshapes received, resetting deactivate timer")
        self.deactivate_timer.reset()

        # face_recognition node will only publish blendshapes if the identity is verified
        # TODO: Figure out how to compare the blendshapes to a password
        self.entered_password.append(msg)
        for i in range(min(len(self.entered_password), len(self.password))):
            if self.entered_password[i] != self.password[i]:
                self.get_logger().info("Incorrect password attempt")
                self.entered_password.clear()
                return
        if len(self.entered_password) == len(self.password):
            self.get_logger().info("Correct password entered, unlocking door")

            self.unlock_door_srv.wait_for_service()
            unlock_req = Trigger.Request()
            unlock_future = self.unlock_door_srv.call_async(unlock_req)

            self.reset_arm_srv.wait_for_service()
            reset_req = Trigger.Request()
            reset_future = self.reset_arm_srv.call_async(reset_req)

            # Wait for both futures to complete before calling unlock_done_cb
            def both_done_cb(_):
                if unlock_future.done() and reset_future.done():
                    self.unlock_done_cb(unlock_future, reset_future)
                else:
                    # If not both done, re-add the callback to the other future
                    if not unlock_future.done():
                        unlock_future.add_done_callback(both_done_cb)
                    if not reset_future.done():
                        reset_future.add_done_callback(both_done_cb)

            unlock_future.add_done_callback(both_done_cb)
            reset_future.add_done_callback(both_done_cb)

            self.robot_state = RobotState.UNLOCKING

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
            self.unlock_door_srv.wait_for_service()
            unlock_req = Trigger.Request()
            unlock_future = self.unlock_door_srv.call_async(unlock_req)
            self.reset_arm_srv.wait_for_service()
            reset_req = Trigger.Request()
            reset_future = self.reset_arm_srv.call_async(reset_req)
            unlock_future.add_done_callback(
                lambda f: self.unlock_done_cb(unlock_future, reset_future)
            )
            reset_future.add_done_callback(
                lambda f: self.unlock_done_cb(unlock_future, reset_future)
            )

    def button_cb(self, msg: Bool) -> None:
        if self.robot_state == RobotState.OPEN and msg.data:
            self.get_logger().info(
                "Door button indicates door has closed, locking door"
            )
            self.lock_door_srv.wait_for_service()
            lock_req = Trigger.Request()
            future = self.lock_door_srv.call_async(lock_req)
            future.add_done_callback(self.lock_done_cb)
            self.robot_state = RobotState.LOCKING

    def lock_done_cb(self, fut: Future) -> None:
        result = fut.result()
        if result and result.success:
            self.get_logger().info("Door locked successfully")
            self.robot_state = RobotState.DISABLED
        elif result and result.success is False:
            self.get_logger().error("Failed to lock door, trying again...")
            self.lock_door_srv.wait_for_service()
            lock_req = Trigger.Request()
            future = self.lock_door_srv.call_async(lock_req)
            future.add_done_callback(self.lock_done_cb)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: FaceLockManager = FaceLockManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
