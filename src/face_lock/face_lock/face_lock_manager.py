import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from robot_interfaces.msg import FaceBlendshapes


class FaceLockManager(Node):
    def __init__(self):
        super().__init__("face_lock_manager")
        # Subscriptions
        self.pir_sub = self.create_subscription(Bool, "/io/pir", self.pir_cb, 10)
        self.button_sub = self.create_subscription(
            Bool, "/io/button", self.button_cb, 10
        )
        self.blendshape_sub = self.create_subscription(
            FaceBlendshapes, "/face_recognition/blendshapes", self.password_cb, 10
        )

        # Publishers
        self.state_pub = self.create_publisher(Bool, "/robot_state", 10)

    def pir_cb(self, msg):
        if msg.data:
            pass

    def password_cb(self, msg):
        if msg.data:
            pass

    def button_cb(self, msg):
        if msg.data:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = FaceLockManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
