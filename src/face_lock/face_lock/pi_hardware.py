import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class PiHardware(Node):
    def __init__(self):
        super().__init__("pi_hardware")
        # Publishers
        self.pir_pub = self.create_publisher(Bool, "/io/pir", 10)
        self.btn_pub = self.create_publisher(Bool, "/io/button", 10)
        self.joint_state_pub = self.create_publisher(
            JointState, "/arm/joint_states", 10
        )

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, "/arm/joint_commands", self.joint_cb, 10
        )

        # Services
        self.lock_srv = self.create_service(Trigger, "lock_door", self.lock_cb)
        self.unlock_srv = self.create_service(Trigger, "unlock_door", self.unlock_cb)

    def joint_cb(self, msg):
        pass  # Write PWM to servos

    def lock_cb(self, req, res):
        return res  # Control Electromagnet

    def unlock_cb(self, req, res):
        return res  # Control DC Motor


def main(args=None):
    rclpy.init(args=args)
    node = PiHardware()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
