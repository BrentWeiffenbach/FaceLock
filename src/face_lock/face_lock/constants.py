"""Shared constants for the face_lock package."""

IDENTITIES_DIR = "/workspaces/FaceLock/src/face_lock/data/identities"
PASSWORDS_DIR = "/workspaces/FaceLock/src/face_lock/data/passwords"

# GPIO mapping
ELECTROMAGNET_GPIO = 17
CLOSED_DOOR_BUTTON_GPIO = 16
PIR_SENSOR_GPIO = 23
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

# Arm tracking control defaults
ARM_TRACK_MAX_STEP_RAD = 0.08
ARM_TRACK_DEADBAND_PX = 15.0
ARM_TRACK_TIMEOUT_SEC = 1.5
LOWER_ARM_TRACK_GAIN = 0.55
UPPER_ARM_TRACK_GAIN = 0.45

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
