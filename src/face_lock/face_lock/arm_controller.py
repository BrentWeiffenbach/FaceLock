"""Arm controller with 2-DOF inverse-kinematics visual servoing.

Replaces the old PWM-space P-controller with a proper IK pipeline:
  1. EMA-filter the face detection centre
  2. Compute pixel error from image centre
  3. Scale pixel error to workspace delta (inches)
  4. Add delta to current end-effector position (from FK)
  5. Solve 2-DOF IK for the new target
  6. Clamp joint-angle step, convert to servo radians, publish
"""

from typing import Any, Optional

import math
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.node import Publisher, Subscription
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2D

from face_lock.constants import (
    ARM_IK_ACQUIRE_STEP_SCALE,
    ARM_CONTROL_RATE_HZ,
    ARM_ELBOW_UP,
    ARM_IK_BOUNDARY_PULL_IN,
    ARM_IK_KP_X,
    ARM_IK_KP_Y,
    ARM_IK_MAX_JOINT_VEL,
    ARM_IK_MAX_JOINT_STEP,
    ARM_IK_MAX_WORKSPACE_STEP,
    ARM_IK_TIP_ANGLE_LIMIT_DEG,
    ARM_TRACK_DEADBAND_PX,
    ARM_TRACK_EMA_ALPHA,
    ARM_TRACK_EMA_TAU_SEC,
    ARM_TRACK_OUTLIER_REJECT_PX,
    ARM_TRACK_REACQUIRE_RAMP_SEC,
    ARM_TRACK_TIMEOUT_SEC,
    DEADLOCK_HOME_RAD,
    DEADLOCK_JOINT_NAME,
    LINKAGE_1_LENGTH,
    LINKAGE_2_LENGTH,
    LOWER_ARM_HOME_RAD,
    LOWER_ARM_IK_DIRECTION,
    LOWER_ARM_IK_ZERO_OFFSET,
    LOWER_ARM_JOINT_NAME,
    UPPER_ARM_HOME_RAD,
    UPPER_ARM_IK_DIRECTION,
    UPPER_ARM_IK_ZERO_OFFSET,
    UPPER_ARM_JOINT_NAME,
)

# ── Kinematics helpers ──────────────────────────────────────────────────

# Reachable workspace: annulus between REACH_MIN and REACH_MAX inches from origin.
# A small margin keeps cos_q2 away from ±1 (avoids numerical singularity).
_REACH_MAX = LINKAGE_1_LENGTH + LINKAGE_2_LENGTH          # 11.7"
_REACH_MIN = abs(LINKAGE_1_LENGTH - LINKAGE_2_LENGTH)     # 1.3"
_REACH_MARGIN = 0.05                                       # inches


def _forward_kinematics(q1: float, q2: float) -> tuple:
    """FK: IK joint angles (q1, q2) → workspace (x, y) in inches."""
    x = LINKAGE_1_LENGTH * math.cos(q1) + LINKAGE_2_LENGTH * math.cos(q1 + q2)
    y = LINKAGE_1_LENGTH * math.sin(q1) + LINKAGE_2_LENGTH * math.sin(q1 + q2)
    return x, y


def _inverse_kinematics(
    x: float, y: float, elbow_up: bool = ARM_ELBOW_UP
) -> Optional[tuple]:
    """IK: workspace (x, y) → (q1, q2) or None if unreachable."""
    dist_sq = x * x + y * y
    reach_max = LINKAGE_1_LENGTH + LINKAGE_2_LENGTH
    reach_min = abs(LINKAGE_1_LENGTH - LINKAGE_2_LENGTH)
    dist = math.sqrt(dist_sq)
    if dist > reach_max or dist < reach_min:
        return None

    cos_q2 = (dist_sq - LINKAGE_1_LENGTH ** 2 - LINKAGE_2_LENGTH ** 2) / (
        2 * LINKAGE_1_LENGTH * LINKAGE_2_LENGTH
    )
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    sin_q2 = math.sqrt(1.0 - cos_q2 * cos_q2)
    if not elbow_up:
        sin_q2 = -sin_q2

    q2 = math.atan2(sin_q2, cos_q2)
    q1 = math.atan2(y, x) - math.atan2(
        LINKAGE_2_LENGTH * sin_q2,
        LINKAGE_1_LENGTH + LINKAGE_2_LENGTH * cos_q2,
    )
    return q1, q2


def _ik_to_servo(q1: float, q2: float) -> tuple:
    """Convert IK joint angles → servo angles (radians in [0, π])."""
    s1 = LOWER_ARM_IK_DIRECTION * q1 + LOWER_ARM_IK_ZERO_OFFSET
    s2 = UPPER_ARM_IK_DIRECTION * q2 + UPPER_ARM_IK_ZERO_OFFSET
    return s1, s2


def _servo_to_ik(s1: float, s2: float) -> tuple:
    """Convert servo radians → IK joint angles."""
    q1 = LOWER_ARM_IK_DIRECTION * (s1 - LOWER_ARM_IK_ZERO_OFFSET)
    q2 = UPPER_ARM_IK_DIRECTION * (s2 - UPPER_ARM_IK_ZERO_OFFSET)
    return q1, q2


# ── Node ────────────────────────────────────────────────────────────────


class ArmController(LifecycleNode):
    """IK-based visual-servoing arm controller."""

    joint_pub: Publisher
    joint_states_sub: Subscription
    detection_sub: Subscription
    reset_arm_srv: Any
    control_timer: Any

    def __init__(self) -> None:
        super().__init__("arm_controller")
        self._active: bool = False

        # Current IK joint angles (updated from servo feedback)
        self._q1: float = 0.0
        self._q2: float = 0.0
        self._deadlock_rad: float = DEADLOCK_HOME_RAD

        # Detection EMA state
        self._last_detection_time: float = 0.0
        self._last_control_time: float = 0.0
        self._acquired_time: float = 0.0
        self._had_recent_detection: bool = False
        self._x_filtered: Optional[float] = None
        self._y_filtered: Optional[float] = None
        self._last_debug_log_time: float = 0.0

        # Parameters
        self.declare_parameter("image_width", 640.0)
        self.declare_parameter("image_height", 480.0)
        self.declare_parameter("ema_alpha", ARM_TRACK_EMA_ALPHA)
        self.declare_parameter("ema_tau_sec", ARM_TRACK_EMA_TAU_SEC)
        self.declare_parameter("kp_x", ARM_IK_KP_X)
        self.declare_parameter("kp_y", ARM_IK_KP_Y)
        self.declare_parameter("max_joint_step", ARM_IK_MAX_JOINT_STEP)
        self.declare_parameter("max_joint_vel", ARM_IK_MAX_JOINT_VEL)
        self.declare_parameter("deadband_px", ARM_TRACK_DEADBAND_PX)
        self.declare_parameter("track_timeout_sec", ARM_TRACK_TIMEOUT_SEC)
        self.declare_parameter("elbow_up", ARM_ELBOW_UP)
        self.declare_parameter("control_rate_hz", ARM_CONTROL_RATE_HZ)
        self.declare_parameter("max_workspace_step", ARM_IK_MAX_WORKSPACE_STEP)
        self.declare_parameter("boundary_pull_in", ARM_IK_BOUNDARY_PULL_IN)
        self.declare_parameter("tip_angle_limit_deg", ARM_IK_TIP_ANGLE_LIMIT_DEG)
        self.declare_parameter("outlier_reject_px", ARM_TRACK_OUTLIER_REJECT_PX)
        self.declare_parameter("reacquire_ramp_sec", ARM_TRACK_REACQUIRE_RAMP_SEC)
        self.declare_parameter("acquire_step_scale", ARM_IK_ACQUIRE_STEP_SCALE)
        self.declare_parameter("control_on_detection", True)

    # ── helpers ──────────────────────────────────────────────────────

    def _pf(self, name: str, default: float) -> float:
        v = self.get_parameter(name).value
        return float(v) if v is not None else default

    def _pb(self, name: str, default: bool) -> bool:
        v = self.get_parameter(name).value
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes"}
        return bool(v)

    @staticmethod
    def _clamp_servo(rad: float) -> float:
        return max(0.0, min(math.pi, rad))

    # ── lifecycle ────────────────────────────────────────────────────

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring arm controller (IK mode)")

        self._image_width = self._pf("image_width", 640.0)
        self._image_height = self._pf("image_height", 480.0)
        self._ema_alpha = self._pf("ema_alpha", ARM_TRACK_EMA_ALPHA)
        self._ema_tau = self._pf("ema_tau_sec", ARM_TRACK_EMA_TAU_SEC)
        self._kp_x = self._pf("kp_x", ARM_IK_KP_X)
        self._kp_y = self._pf("kp_y", ARM_IK_KP_Y)
        self._max_joint_step = self._pf("max_joint_step", ARM_IK_MAX_JOINT_STEP)
        self._max_joint_vel = self._pf("max_joint_vel", ARM_IK_MAX_JOINT_VEL)
        self._deadband_px = self._pf("deadband_px", ARM_TRACK_DEADBAND_PX)
        self._track_timeout = self._pf("track_timeout_sec", ARM_TRACK_TIMEOUT_SEC)
        self._elbow_up = self._pb("elbow_up", ARM_ELBOW_UP)
        control_hz = self._pf("control_rate_hz", ARM_CONTROL_RATE_HZ)
        self._max_workspace_step = self._pf(
            "max_workspace_step", ARM_IK_MAX_WORKSPACE_STEP
        )
        self._boundary_pull_in = self._pf("boundary_pull_in", ARM_IK_BOUNDARY_PULL_IN)
        tip_limit_deg = self._pf("tip_angle_limit_deg", ARM_IK_TIP_ANGLE_LIMIT_DEG)
        self._tip_angle_limit = math.radians(max(0.0, tip_limit_deg))
        self._outlier_reject_px = self._pf(
            "outlier_reject_px", ARM_TRACK_OUTLIER_REJECT_PX
        )
        self._reacquire_ramp_sec = self._pf(
            "reacquire_ramp_sec", ARM_TRACK_REACQUIRE_RAMP_SEC
        )
        self._acquire_step_scale = max(
            0.0, min(1.0, self._pf("acquire_step_scale", ARM_IK_ACQUIRE_STEP_SCALE))
        )
        self._control_on_detection = self._pb("control_on_detection", True)
        self._control_period = 1.0 / max(1.0, control_hz)

        # Initialise IK angles from home servo position
        self._q1, self._q2 = _servo_to_ik(LOWER_ARM_HOME_RAD, UPPER_ARM_HOME_RAD)
        hx, hy = _forward_kinematics(self._q1, self._q2)
        self.get_logger().info(
            f"Home: ({hx:.2f}, {hy:.2f})\" | "
            f"q1={math.degrees(self._q1):.1f} q2={math.degrees(self._q2):.1f}"
        )

        # Pub / sub / srv
        self.joint_pub = self.create_lifecycle_publisher(
            JointState, "/arm/joint_commands", 10
        )
        self.joint_states_sub = self.create_subscription(
            JointState, "/arm/joint_states", self._joint_states_cb, 10
        )
        self.detection_sub = self.create_subscription(
            Detection2D, "/face_recognition/detection", self._detection_cb, 10
        )
        self.reset_arm_srv = self.create_service(
            Trigger, "reset_arm", self._reset_arm_cb
        )

        # Fixed-rate control loop (decoupled from detection rate)
        period = 1.0 / max(1.0, control_hz)
        self.control_timer = self.create_timer(period, self._control_loop)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating arm controller (IK mode)")
        self._active = True
        self._last_detection_time = time.monotonic()
        self._last_control_time = 0.0
        self._acquired_time = 0.0
        self._had_recent_detection = False
        self._x_filtered = None
        self._y_filtered = None
        # Activate lifecycle publishers first, then send home so the arm
        # always starts from a known position when CHECKING begins.
        result = super().on_activate(state)
        self._q1, self._q2 = _servo_to_ik(LOWER_ARM_HOME_RAD, UPPER_ARM_HOME_RAD)
        self._deadlock_rad = DEADLOCK_HOME_RAD
        s1, s2 = _ik_to_servo(self._q1, self._q2)
        self._publish_joint_command(s1, s2)
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating arm controller — returning to home")
        self._active = False
        # Send home BEFORE super() deactivates the lifecycle publisher so
        # the arm always returns to its rest position (DISABLED / SETUP).
        self._q1, self._q2 = _servo_to_ik(LOWER_ARM_HOME_RAD, UPPER_ARM_HOME_RAD)
        self._deadlock_rad = DEADLOCK_HOME_RAD
        self._x_filtered = None
        self._y_filtered = None
        self._acquired_time = 0.0
        self._had_recent_detection = False
        s1, s2 = _ik_to_servo(self._q1, self._q2)
        self._publish_joint_command(s1, s2)
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up arm controller")
        return TransitionCallbackReturn.SUCCESS

    # ── callbacks ────────────────────────────────────────────────────

    def _joint_states_cb(self, msg: JointState) -> None:
        """Sync internal IK angles with actual servo positions."""
        s1: Optional[float] = None
        s2: Optional[float] = None
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                break
            if name == LOWER_ARM_JOINT_NAME:
                s1 = msg.position[idx]
            elif name == UPPER_ARM_JOINT_NAME:
                s2 = msg.position[idx]
            elif name == DEADLOCK_JOINT_NAME:
                self._deadlock_rad = self._clamp_servo(msg.position[idx])
        if s1 is not None and s2 is not None:
            self._q1, self._q2 = _servo_to_ik(s1, s2)

    def _detection_cb(self, msg: Detection2D) -> None:
        """Store latest detection with time-based EMA low-pass filter.

        Using time-based EMA (alpha = 1 - exp(-dt/tau)) rather than a fixed
        per-frame alpha means the filter responds consistently regardless of
        the detection rate.  face_recognition runs at ~1 Hz on the Pi; a
        fixed alpha=0.15 would need 20+ detections (~20 s) to converge,
        whereas a tau=0.4 s gives alpha≈0.92 at 1 Hz (near-instant snap)
        and alpha≈0.033 at 30 Hz (smooth).
        """
        if not self._active:
            return

        cx = float(msg.bbox.center.position.x)
        cy = float(msg.bbox.center.position.y)

        now = time.monotonic()
        is_first_detection = self._x_filtered is None or self._y_filtered is None
        if is_first_detection:
            self._x_filtered = cx
            self._y_filtered = cy
        else:
            if self._x_filtered is None or self._y_filtered is None:
                return
            xf = float(self._x_filtered)
            yf = float(self._y_filtered)
            dist_px = math.hypot(cx - xf, cy - yf)
            if dist_px > self._outlier_reject_px:
                self.get_logger().debug(
                    f"[ARM] outlier detection ignored: dist={dist_px:.1f}px "
                    f"(threshold={self._outlier_reject_px:.0f}px)"
                )
                return
            dt = now - self._last_detection_time
            a = 1.0 - math.exp(-dt / max(self._ema_tau, 1e-6))
            self._x_filtered = a * cx + (1.0 - a) * xf
            self._y_filtered = a * cy + (1.0 - a) * yf

        self._last_detection_time = now
        if not self._had_recent_detection:
            self._acquired_time = now
            self._had_recent_detection = True

        if self._control_on_detection and not is_first_detection:
            # React immediately to new detections instead of waiting for the timer tick.
            self._control_loop()

    def _control_loop(self) -> None:
        """Fixed-rate control: pixel error → workspace Δ → IK → joints."""
        if not self._active:
            return

        now = time.monotonic()
        prev_control_time = self._last_control_time
        # Prevent stacked detection+timer calls from creating back-to-back jumps.
        if prev_control_time > 0.0 and (now - prev_control_time) < (0.5 * self._control_period):
            return
        control_dt = (
            self._control_period
            if prev_control_time <= 0.0
            else max(1e-3, now - prev_control_time)
        )
        self._last_control_time = now

        # Hold position when face is lost
        if now - self._last_detection_time > self._track_timeout:
            self._had_recent_detection = False
            return

        if self._x_filtered is None or self._y_filtered is None:
            return

        # ── pixel error from image centre ────────────────────────────
        err_x = self._x_filtered - self._image_width / 2.0
        err_y = self._y_filtered - self._image_height / 2.0

        # ── debug logging (throttled to 1 Hz) ────────────────────────
        now_dbg = time.monotonic()
        if now_dbg - self._last_debug_log_time >= 1.0:
            self._last_debug_log_time = now_dbg
            dir_x = (
                "RIGHT" if err_x > self._deadband_px
                else "LEFT" if err_x < -self._deadband_px
                else "CENTER-X"
            )
            dir_y = (
                "DOWN" if err_y > self._deadband_px
                else "UP" if err_y < -self._deadband_px
                else "CENTER-Y"
            )
            s1_dbg, s2_dbg = _ik_to_servo(self._q1, self._q2)
            cur_x_dbg, cur_y_dbg = _forward_kinematics(self._q1, self._q2)
            ee_dist_dbg = math.sqrt(cur_x_dbg ** 2 + cur_y_dbg ** 2)
            self.get_logger().info(
                f"[ARM] face={dir_x}/{dir_y}  "
                f"err=({err_x:+.1f},{err_y:+.1f})px  "
                f"EE=({cur_x_dbg:.2f}\",{cur_y_dbg:.2f}\") dist={ee_dist_dbg:.2f}\"/{_REACH_MAX:.2f}\"  "
                f"q1={math.degrees(self._q1):.1f}\u00b0 q2={math.degrees(self._q2):.1f}\u00b0  "
                f"s1={math.degrees(s1_dbg):.1f}\u00b0 s2={math.degrees(s2_dbg):.1f}\u00b0"
            )

        if abs(err_x) < self._deadband_px:
            err_x = 0.0
        if abs(err_y) < self._deadband_px:
            err_y = 0.0

        if err_x == 0.0 and err_y == 0.0:
            return

        # ── workspace delta (inches) ─────────────────────────────────
        # Camera is kept level (q1+q2 = π/2) so image axes stay
        # aligned with the world frame at all arm configurations.
        dx = self._kp_x * err_x
        dy = self._kp_y * err_y
        workspace_step = math.hypot(dx, dy)
        if workspace_step > self._max_workspace_step:
            scale = self._max_workspace_step / workspace_step
            dx *= scale
            dy *= scale

        # ── current position via FK ──────────────────────────────────
        cur_x, cur_y = _forward_kinematics(self._q1, self._q2)

        # ── Project target to reachable workspace ─────────────────────
        # The home position points straight up at (0", 11.7") which is
        # exactly max extension.  Any lateral dx makes dist > reach_max
        # and IK returns None, freezing the arm for as long as the face
        # stays near the vertical axis.  Instead, clamp the target
        # radially onto the workspace boundary so the arm always rotates
        # toward the face even from a fully-extended starting position.
        tgt_x = cur_x + dx
        tgt_y = cur_y + dy
        tgt_dist = math.sqrt(tgt_x * tgt_x + tgt_y * tgt_y)
        _clamped = False
        max_target_radius = max(
            _REACH_MIN + _REACH_MARGIN + 0.05,
            _REACH_MAX - _REACH_MARGIN - max(0.0, self._boundary_pull_in),
        )
        if tgt_dist > max_target_radius:
            scale = max_target_radius / tgt_dist
            tgt_x, tgt_y = tgt_x * scale, tgt_y * scale
            _clamped = True
        elif 0.0 < tgt_dist < _REACH_MIN + _REACH_MARGIN:
            scale = (_REACH_MIN + _REACH_MARGIN) / tgt_dist
            tgt_x, tgt_y = tgt_x * scale, tgt_y * scale
            _clamped = True
        elif tgt_dist == 0.0:
            # Degenerate: arm is at origin — move to inner boundary
            tgt_x, tgt_y = _REACH_MIN + _REACH_MARGIN, 0.0
            _clamped = True

        # ── IK for new target ────────────────────────────────────────
        result = _inverse_kinematics(tgt_x, tgt_y, self._elbow_up)
        if result is None:
            self.get_logger().debug(
                f"[ARM] IK failed (post-clamp) tgt=({tgt_x:.2f}\",{tgt_y:.2f}\")"
            )
            return  # should not happen after clamping, but guard anyway
        if _clamped:
            self.get_logger().debug(
                f"[ARM] target clamped to ({tgt_x:.2f}\",{tgt_y:.2f}\") "
                f"(raw dist={tgt_dist:.2f}\")"
            )

        desired_q1, desired_q2 = result

        # Hard safety clamp on desired IK target: keep linkage-2 tip angle
        # (q1+q2) within +/- limit before joint-step limiting.
        desired_tip = desired_q1 + desired_q2
        if desired_tip > self._tip_angle_limit:
            desired_q2 = self._tip_angle_limit - desired_q1
        elif desired_tip < -self._tip_angle_limit:
            desired_q2 = -self._tip_angle_limit - desired_q1

        # ── proportional joint-step (preserves direction ratio) ──────
        # Independent per-joint clamping is wrong near max extension:
        # when q2 needs +20° and q1 needs -6°, both get clamped to ±ms
        # (equal/opposite), keeping q1+q2 constant and causing the arm
        # to sweep along a fixed elbow arc instead of converging.
        # Proportional scaling gives the full ms budget to the largest
        # change and scales the other joint down accordingly.
        ms = self._max_joint_step
        if self._max_joint_vel > 0.0:
            ms = min(ms, self._max_joint_vel * control_dt)
        if self._had_recent_detection and self._reacquire_ramp_sec > 0.0:
            age = max(0.0, now - self._acquired_time)
            if age < self._reacquire_ramp_sec:
                t = age / self._reacquire_ramp_sec
                ramp_scale = self._acquire_step_scale + (1.0 - self._acquire_step_scale) * t
                ms *= ramp_scale

        needed_q1 = desired_q1 - self._q1
        needed_q2 = desired_q2 - self._q2
        max_needed = max(abs(needed_q1), abs(needed_q2))
        if max_needed > ms:
            scale = ms / max_needed
            dq1 = needed_q1 * scale
            dq2 = needed_q2 * scale
        else:
            dq1 = needed_q1
            dq2 = needed_q2
        new_q1 = self._q1 + dq1

        # ── camera levelling: keep tip angle at π/2 ──────────────────
        # Counter-rotate servo 2 so the camera always faces the same
        # direction regardless of q1.  This eliminates the eye-in-hand
        # rotation problem and makes simple kp_x/kp_y gains stable.
        new_q2 = (math.pi / 2.0) - new_q1

        # ── convert to servo angles and validate ─────────────────────
        s1, s2 = _ik_to_servo(new_q1, new_q2)
        if not (0.0 <= s1 <= math.pi and 0.0 <= s2 <= math.pi):
            self.get_logger().debug(
                f"[ARM] servo limit hit s1={math.degrees(s1):.1f}° s2={math.degrees(s2):.1f}°"
            )
            return  # servo limits — hold position

        self._q1 = new_q1
        self._q2 = new_q2
        self._publish_joint_command(s1, s2)

    # ── publishing ───────────────────────────────────────────────────

    def _publish_joint_command(self, s1: float, s2: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [LOWER_ARM_JOINT_NAME, UPPER_ARM_JOINT_NAME, DEADLOCK_JOINT_NAME]
        msg.position = [
            self._clamp_servo(s1),
            self._clamp_servo(s2),
            self._clamp_servo(self._deadlock_rad),
        ]
        self.joint_pub.publish(msg)

    # ── services ─────────────────────────────────────────────────────


    def _reset_arm_cb(
        self, req: Trigger.Request, res: Trigger.Response
    ) -> Trigger.Response:
        del req
        self._q1, self._q2 = _servo_to_ik(LOWER_ARM_HOME_RAD, UPPER_ARM_HOME_RAD)
        self._deadlock_rad = DEADLOCK_HOME_RAD
        self._x_filtered = None
        self._y_filtered = None
        s1, s2 = _ik_to_servo(self._q1, self._q2)
        self._publish_joint_command(s1, s2)
        # Freeze tracking so stale detections after the password match
        # (face_recognition deactivates ~5 s later) cannot move the arm.
        # on_activate() re-enables this when CHECKING restarts next cycle.
        self._active = False
        res.success = True
        res.message = "Arm reset to home via IK (tracking frozen)"
        return res


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ArmController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
