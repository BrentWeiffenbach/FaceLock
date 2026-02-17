import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2D


class ArmController(LifecycleNode):
    def __init__(self):
        super().__init__("arm_controller")

    def on_configure(self, state):
        # Publishers
        self.joint_pub = self.create_lifecycle_publisher(
            JointState, "/arm/joint_commands", 10
        )

        # Subscribers
        self.joint_states_sub = self.create_subscription(
            JointState, "/arm/joint_states", self.joint_states_cb, 10
        )

        self.detection_sub = self.create_subscription(
            Detection2D, "/face_recognition/detection", self.detection_cb, 10
        )

        # Services
        self.reset_arm_srv = self.create_service(
            Trigger, "reset_arm", self.reset_arm_cb
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        return super().on_activate(state)

    def on_deactivate(self, state):
        return super().on_deactivate(state)

    def joint_states_cb(self, msg):
        pass

    def detection_cb(self, msg):
        pass

    def reset_arm_cb(self, req, res):
        return res


def main(args=None):
    rclpy.init(args=args)
    node = ArmController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
