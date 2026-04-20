"""Shared constants for the face_lock package."""

import math

IDENTITIES_DIR = "/workspaces/FaceLock/src/face_lock/data/identities"
PASSWORDS_DIR = "/workspaces/FaceLock/src/face_lock/data/passwords"

# GPIO mapping
ELECTROMAGNET_GPIO = 17
CLOSED_DOOR_BUTTON_GPIO = 16
PIR_SENSOR_GPIO = 24
LOWER_ARM_SERVO_GPIO = 27
UPPER_ARM_SERVO_GPIO = 22
DEADLOCK_MICROSERVO_GPIO = 23

# Servo pulse widths (microseconds)
SERVO_PULSE_MIN_US = 500
SERVO_PULSE_CENTER_US = 1500
SERVO_PULSE_MAX_US = 2500

# Joint names used on /arm/joint_commands and /arm/joint_states
LOWER_ARM_JOINT_NAME = "lower_arm_joint"
UPPER_ARM_JOINT_NAME = "upper_arm_joint"
DEADLOCK_JOINT_NAME = "deadlock_joint"

# Servo home angles (radians)
LOWER_ARM_HOME_RAD = 1.57
UPPER_ARM_HOME_RAD = 1.57
DEADLOCK_HOME_RAD = 1.57

# Arm linkage lengths (inches)
LINKAGE_1_LENGTH = 5.2   # base servo → elbow servo
LINKAGE_2_LENGTH = 6.5   # elbow servo → camera lens

# IK servo-to-joint angle mapping
#   joint_angle = DIRECTION * (servo_rad - ZERO_OFFSET)
#   servo_rad   = DIRECTION * joint_angle + ZERO_OFFSET
# Tune these with test_arm.py until the arm moves correctly in XY.
LOWER_ARM_IK_DIRECTION = 1
LOWER_ARM_IK_ZERO_OFFSET = 0.0            # servo rad when q1 = 0 (link 1 horizontal right)
UPPER_ARM_IK_DIRECTION = 1
UPPER_ARM_IK_ZERO_OFFSET = math.pi / 2    # servo rad when q2 = 0 (elbow straight)
ARM_ELBOW_UP = True                        # IK solution preference

# Visual-servoing IK P-controller
# Sign convention: image derotation (camera.py) aligns image-right = world-right.
# A face to the RIGHT (err_x > 0) needs the arm to move in +x, so kp_x is positive.
# A face BELOW centre (err_y > 0) needs the camera lowered, so kp_y is negative
# because image +Y is downward while workspace +Y is upward.
ARM_IK_KP_X = 0.005          # workspace inches per pixel error per update
ARM_IK_KP_Y = -0.005         # negative: image Y down vs workspace Y up
ARM_IK_MAX_JOINT_STEP = 0.04 # max joint angle change per update (rad); 0.04×50 Hz ≈ 2 rad/s
ARM_CONTROL_RATE_HZ = 50.0   # arm controller timer frequency (Hz)

# Arm tracking control defaults (legacy, kept for reference)
ARM_TRACK_MAX_STEP_RAD = 0.08
ARM_TRACK_DEADBAND_PX = 15.0
ARM_TRACK_TIMEOUT_SEC = 1.5
LOWER_ARM_TRACK_GAIN = 0.55
UPPER_ARM_TRACK_GAIN = 0.45

# Visual-servoing controller parameters (PWM-space)
ARM_TRACK_EMA_ALPHA = 0.3        # Low-pass filter coefficient (0 < α ≤ 1); 0.3@50 Hz ≈ 67 ms lag
ARM_TRACK_KP_LOWER_US_PX = 2.0  # P-gain for Joint 1 (Base)  [µs / px]
ARM_TRACK_KP_UPPER_US_PX = 1.5  # P-gain for Joint 2 (Elbow) [µs / px]
ARM_TRACK_MAX_SLEW_US = 20      # Max PWM change per update  [µs]

# Deadlock servo positions
DEADLOCK_LOCK_PULSE_US = SERVO_PULSE_MIN_US
DEADLOCK_UNLOCK_PULSE_US = SERVO_PULSE_MAX_US
DEADLOCK_HOME_PULSE_US = SERVO_PULSE_CENTER_US

BLENDSHAPE_THRESHOLD = 0.45

IGNORED_BLENDSHAPES = {
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "browDownLeft",
    "browDownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "browInnerUp",
}
