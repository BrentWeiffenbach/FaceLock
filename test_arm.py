#!/usr/bin/env python3
"""Interactive 2-DOF arm IK tester.

Move the end-effector in XY with keyboard controls and verify that the
arm tracks correctly through its workspace.  Uses pigpio directly (no ROS).

Controls:
    W / S          move +Y / -Y
    A / D          move -X / +X
    Q / E          decrease / increase step size
    H              return to home position
    1 / 2          select elbow-up / elbow-down solution
    R              re-solve IK at current XY (useful after changing elbow mode)
    ESC / Ctrl-C   quit

Usage:
    python3 test_arm.py            # real hardware (needs pigpiod)
    python3 test_arm.py --mock     # no GPIO, IK math only
"""

import curses
import math
import sys
import time

try:
    import pigpio
except ImportError:
    pigpio = None  # type: ignore[assignment]

# ── Arm geometry (must match constants.py) ──────────────────────────────

L1 = 5.2   # link 1 length (inches): base servo -> elbow
L2 = 6.5   # link 2 length (inches): elbow -> camera lens

# Servo PWM range
PWM_MIN = 500
PWM_MAX = 2500

# Servo-to-IK-angle mapping  (servo_rad = DIR * ik_angle + OFFSET)
LOWER_DIR = 1
LOWER_ZERO = 0.0
UPPER_DIR = 1
UPPER_ZERO = math.pi / 2

# GPIO pins
LOWER_GPIO = 27
UPPER_GPIO = 22

# Defaults
DEFAULT_STEP = 0.25
MIN_STEP = 0.05
MAX_STEP = 2.0

REACH_MAX = L1 + L2
REACH_MIN = abs(L1 - L2)


# ── Math helpers ────────────────────────────────────────────────────────

def forward_kinematics(q1: float, q2: float) -> tuple:
    """Return (x, y) in inches for IK joint angles q1, q2."""
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    y = L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
    return x, y


def inverse_kinematics(x: float, y: float, elbow_up: bool = True):
    """Solve for (q1, q2) given desired (x, y).  Returns None if unreachable."""
    dist_sq = x * x + y * y
    dist = math.sqrt(dist_sq)
    if dist > REACH_MAX or dist < REACH_MIN:
        return None

    cos_q2 = (dist_sq - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    sin_q2 = math.sqrt(1.0 - cos_q2 * cos_q2)
    if not elbow_up:
        sin_q2 = -sin_q2

    q2 = math.atan2(sin_q2, cos_q2)
    q1 = math.atan2(y, x) - math.atan2(L2 * sin_q2, L1 + L2 * cos_q2)
    return q1, q2


def ik_to_servo(q1: float, q2: float) -> tuple:
    """Convert IK joint angles to servo angles (radians)."""
    s1 = LOWER_DIR * q1 + LOWER_ZERO
    s2 = UPPER_DIR * q2 + UPPER_ZERO
    return s1, s2


def servo_to_ik(s1: float, s2: float) -> tuple:
    """Convert servo radians to IK joint angles."""
    q1 = LOWER_DIR * (s1 - LOWER_ZERO)
    q2 = UPPER_DIR * (s2 - UPPER_ZERO)
    return q1, q2


def rad_to_pwm(rad: float) -> int:
    """Servo radians [0, pi] -> PWM [500, 2500] us."""
    clamped = max(0.0, min(math.pi, rad))
    return int(round(PWM_MIN + (clamped / math.pi) * (PWM_MAX - PWM_MIN)))


def pwm_to_rad(pwm: int) -> float:
    """PWM [500, 2500] us -> servo radians [0, pi]."""
    clamped = max(PWM_MIN, min(PWM_MAX, pwm))
    return (clamped - PWM_MIN) / (PWM_MAX - PWM_MIN) * math.pi


def servo_valid(s1: float, s2: float) -> bool:
    """Check servo angles are within [0, pi] (PWM range)."""
    return 0.0 <= s1 <= math.pi and 0.0 <= s2 <= math.pi


# ── Hardware ────────────────────────────────────────────────────────────

class ArmHardware:
    def __init__(self, mock: bool = False):
        self.mock = mock
        self.pi = None
        if not mock:
            if pigpio is None:
                raise RuntimeError("pigpio module is not installed")
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise RuntimeError(
                    "Failed to connect to pigpio daemon. Start pigpiod first."
                )
            self.pi.set_mode(LOWER_GPIO, pigpio.OUTPUT)
            self.pi.set_mode(UPPER_GPIO, pigpio.OUTPUT)

    def set_servos(self, lower_pwm: int, upper_pwm: int) -> None:
        if self.pi:
            self.pi.set_servo_pulsewidth(LOWER_GPIO, lower_pwm)
            self.pi.set_servo_pulsewidth(UPPER_GPIO, upper_pwm)

    def stop(self) -> None:
        if self.pi:
            self.pi.set_servo_pulsewidth(LOWER_GPIO, 0)
            self.pi.set_servo_pulsewidth(UPPER_GPIO, 0)
            self.pi.stop()


# ── Main ────────────────────────────────────────────────────────────────

def main(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    mock = "--mock" in sys.argv
    hw = ArmHardware(mock=mock)

    # Start from home: both servos at centre (pi/2 rad)
    home_s1 = math.pi / 2
    home_s2 = math.pi / 2
    home_q1, home_q2 = servo_to_ik(home_s1, home_s2)
    x, y = forward_kinematics(home_q1, home_q2)

    step = DEFAULT_STEP
    elbow_up = True

    q1, q2 = home_q1, home_q2
    s1, s2 = home_s1, home_s2
    lower_pwm = rad_to_pwm(s1)
    upper_pwm = rad_to_pwm(s2)
    hw.set_servos(lower_pwm, upper_pwm)

    status = "Ready  (use --mock for no-GPIO mode)" if not mock else "Ready  (MOCK mode)"

    def try_move(new_x: float, new_y: float) -> str:
        """Attempt to move to (new_x, new_y).  Returns status string."""
        nonlocal x, y, q1, q2, s1, s2, lower_pwm, upper_pwm
        result = inverse_kinematics(new_x, new_y, elbow_up)
        if result is None:
            return f"UNREACHABLE ({new_x:+.2f}, {new_y:+.2f})"
        new_q1, new_q2 = result
        new_s1, new_s2 = ik_to_servo(new_q1, new_q2)
        if not servo_valid(new_s1, new_s2):
            return (
                f"SERVO LIMIT  s1={math.degrees(new_s1):.0f} "
                f"s2={math.degrees(new_s2):.0f}"
            )
        q1, q2 = new_q1, new_q2
        s1, s2 = new_s1, new_s2
        x, y = new_x, new_y
        lower_pwm = rad_to_pwm(s1)
        upper_pwm = rad_to_pwm(s2)
        hw.set_servos(lower_pwm, upper_pwm)
        return f"OK -> ({x:+.2f}, {y:+.2f})"

    while True:
        # ── Draw ────────────────────────────────────────────────────
        stdscr.erase()
        reach = math.sqrt(x * x + y * y)
        mode_str = "MOCK" if mock else "LIVE"

        stdscr.addstr(0, 0, f" 2-DOF Arm IK Tester  [{mode_str}]", curses.A_BOLD)
        stdscr.addstr(1, 0, "-" * 50)
        stdscr.addstr(3, 0, f"  End-effector   X = {x:+7.2f}\"    Y = {y:+7.2f}\"")
        stdscr.addstr(4, 0, f"  IK angles     q1 = {math.degrees(q1):+7.1f} deg"
                             f"   q2 = {math.degrees(q2):+7.1f} deg")
        stdscr.addstr(5, 0, f"  Servo angles  s1 = {math.degrees(s1):+7.1f} deg"
                             f"   s2 = {math.degrees(s2):+7.1f} deg")
        stdscr.addstr(6, 0, f"  PWM          low = {lower_pwm:5d} us"
                             f"   upp = {upper_pwm:5d} us")
        stdscr.addstr(7, 0, f"  Step size    {step:.2f}\"       "
                             f"  Elbow: {'UP' if elbow_up else 'DOWN'}")
        stdscr.addstr(8, 0, f"  Reach        {reach:.2f}\"       "
                             f"  Range: {REACH_MIN:.1f}\" - {REACH_MAX:.1f}\"")
        stdscr.addstr(9, 0, "-" * 50)
        stdscr.addstr(10, 0, f"  {status}")
        stdscr.addstr(12, 0, "  Controls:")
        stdscr.addstr(13, 0, "    W/S   +Y / -Y          A/D   -X / +X")
        stdscr.addstr(14, 0, "    Q/E   step -/+          H     home")
        stdscr.addstr(15, 0, "    1/2   elbow up/down     R     re-solve IK")
        stdscr.addstr(16, 0, "    ESC   quit")
        stdscr.refresh()

        # ── Input ───────────────────────────────────────────────────
        key = stdscr.getch()
        if key < 0:
            continue

        if key == 27:  # ESC
            break
        elif key in (ord("w"), ord("W"), curses.KEY_UP):
            status = try_move(x, y + step)
        elif key in (ord("s"), ord("S"), curses.KEY_DOWN):
            status = try_move(x, y - step)
        elif key in (ord("a"), ord("A"), curses.KEY_LEFT):
            status = try_move(x - step, y)
        elif key in (ord("d"), ord("D"), curses.KEY_RIGHT):
            status = try_move(x + step, y)
        elif key in (ord("q"), ord("Q")):
            step = max(MIN_STEP, round(step - 0.05, 2))
            status = f"Step -> {step:.2f}\""
        elif key in (ord("e"), ord("E")):
            step = min(MAX_STEP, round(step + 0.05, 2))
            status = f"Step -> {step:.2f}\""
        elif key in (ord("h"), ord("H")):
            hx, hy = forward_kinematics(home_q1, home_q2)
            status = try_move(hx, hy)
            if status.startswith("OK"):
                status = "Homed"
        elif key == ord("1"):
            elbow_up = True
            status = try_move(x, y)
            if status.startswith("OK"):
                status = "Elbow UP"
        elif key == ord("2"):
            elbow_up = False
            status = try_move(x, y)
            if status.startswith("OK"):
                status = "Elbow DOWN"
        elif key in (ord("r"), ord("R")):
            status = try_move(x, y)
            if status.startswith("OK"):
                status = "IK re-solved"

    hw.stop()


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    print("Arm test complete.")
