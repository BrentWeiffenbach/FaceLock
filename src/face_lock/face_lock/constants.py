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

# Servo home angles (radians) — centre of [0, π] servo range.
# Attach servo horns so that π/2 (1500 µs) is the centre of the
# reachable workspace.  At home both links point straight up.
LOWER_ARM_HOME_RAD = math.pi / 2
UPPER_ARM_HOME_RAD = math.pi / 2
DEADLOCK_HOME_RAD = math.pi / 2

# Arm linkage lengths (inches)
LINKAGE_1_LENGTH = 5.2   # base servo → elbow servo
LINKAGE_2_LENGTH = 6.5   # elbow servo → camera lens

# ── 2-DOF planar arm DH convention ──────────────────────────────────
#
#   Joint 1 (base):  Z out of servo rotation, X along link 1
#   Joint 2 (elbow): Z inverted (180° Y rotation), X along link 2
#   Camera:          fixed normal to link 2, faces along joint-1 Z
#
# Forward kinematics:
#   elbow_x = L1 · cos(θ1)
#   elbow_y = L1 · sin(θ1)
#   camera_x = elbow_x + L2 · cos(θ1 − θ2)
#   camera_y = elbow_y + L2 · sin(θ1 − θ2)
#
# Camera-leveling constraint (link 2 always vertical → camera level):
#   θ1 − θ2 = π/2   ⟹   θ2 = θ1 − π/2
#
# With leveling applied the camera traces a circle of radius L1:
#   camera_x = L1 · cos(θ1)
#   camera_y = L1 · sin(θ1) + L2
#
# Joint ↔ servo mapping:
#   servo1 = θ1                (direct — 0 rad = link 1 horizontal right)
#   servo2 = θ2 + π/2         (offset — servo centre = elbow straight)
# With leveling: servo2 = (θ1 − π/2) + π/2 = θ1  (mirrors servo1)
# ─────────────────────────────────────────────────────────────────────

# Visual-servoing P-controller
ARM_KP = 0.005                                # proportional gain (rad / px)
ARM_DEADBAND_PX = 5.0                         # pixels from image centre to ignore
ARM_SERVO_LIMIT_MIN_RAD = math.radians(10)    # right-side servo limit (10°)
ARM_SERVO_LIMIT_MAX_RAD = math.radians(170)   # left-side servo limit (170°)
ARM_CONTROL_RATE_HZ = 20.0                    # control loop frequency (Hz)
ARM_TRACK_TIMEOUT_SEC = 1.5                   # seconds before "face lost"
ARM_TRACK_EMA_ALPHA = 0.15                    # EMA smoothing for detection pos

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
