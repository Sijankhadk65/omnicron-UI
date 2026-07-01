"""Blocking robot logic — a thin, Qt-free wrapper around the Fairino SDK.

Every method here is BLOCKING and is meant to run on the robot worker thread (see
``service.py``), never on the Qt main thread. Keeping this class free of any Qt
imports means it can be unit-tested and reused headless, and it makes the
threading boundary explicit: ``RobotService`` owns the thread; ``RobotController``
owns the connection and the SDK calls.

Logging is done through an injected ``log(msg, level)`` callable so the service
can forward messages to the UI as Qt signals without this class knowing about Qt.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from omnicron_ui.robot import geometry
from omnicron_ui.robot.sdk import Robot

# Log levels the UI understands (info/success/warn/error). Matches the color
# coding used by the log panel.
LogFn = Callable[[str, str], None]


def _default_log(msg: str, level: str = "info") -> None:
    print(f"[{level}] {msg}")


def _ret_level(ret: int) -> str:
    """Map an SDK return code to a log level (0 = success, else error)."""
    return "success" if ret == 0 else "error"


def _fmt(vals) -> str:
    """Compact [a, b, c] formatter for poses / joint vectors in logs."""
    return "[" + ", ".join(f"{v:.1f}" for v in vals) + "]"


# Default PTP speed as a percentage of max (kept low for safe testing).
DEFAULT_VEL = 20.0

# Default linear acceleration for physical-speed MoveL (mm/s^2).
DEFAULT_ACCEL = 200.0

# The "home" joint configuration the program parks the arm at between tasks.
# Same pose used by farino_app; joint values in degrees [j1..j6].
HOME_JOINTS = [-90.0, -90.0, 85.0, -85.0, -90.0, 0.0]

# Arc strike/extinguish timeout for ARCStart/ARCEnd (ms).
ARC_TIMEOUT_MS = 10000


@dataclass(frozen=True)
class WeldParams:
    """Parameters for one weld pass. SAFE BY DEFAULT: live=False is a DRY WELD —
    the motion is identical to a real pass but nothing is energized (no arc, no
    gas, no current), matching the weld-test ladder (prove the path first).

    Current/voltage normally come from the WebApp welding process picked by
    ``arc_num`` (Welder → Welding process parameters), output to the welder over
    the control-box analog outputs (current=AO0, voltage=AO1; the welder runs in
    external/analog mode, front-panel knobs bypassed). Set ``current``/
    ``voltage`` here only to OVERRIDE that process from code.

    ``gas`` defaults False to match the gasless flux-cored setup.
    """

    live: bool = False          # False = dry run: log the steps, energize nothing
    io: int = 0                 # ioType: 0 = controller IO, 1 = extended IO
    arc_num: int = 1            # WebApp welding process number (ARCStart arcNum)
    current: float | None = None   # A, AO0 override (None = use the process)
    voltage: float | None = None   # V, AO1 override (None = use the process)
    gas: bool = False           # open gas around the pass (off for flux-cored)
    timeout_ms: int = ARC_TIMEOUT_MS


class RobotError(RuntimeError):
    """Raised when an SDK call returns a non-zero error code."""


class RobotController:
    """Holds the live RPC connection and exposes blocking robot actions."""

    def __init__(self, log: LogFn = _default_log) -> None:
        self.robot: Robot.RPC | None = None
        self.log = log
        self.tool = 0
        self.user = 0

    @property
    def connected(self) -> bool:
        return self.robot is not None

    # --- connection ---------------------------------------------------------

    def connect(self, ip: str) -> tuple[int, int]:
        """Connect, switch to auto mode, enable, clear faults, read active frames.

        The Mode(0) -> RobotEnable(1) -> ResetAllError sequence is required before
        any motion command, otherwise motions fail with error 154.
        """
        self.log(f"Connecting to {ip} …", "info")
        robot = Robot.RPC(ip)
        time.sleep(0.5)
        robot.Mode(0)
        time.sleep(0.5)
        robot.RobotEnable(1)
        time.sleep(1.0)
        robot.ResetAllError()
        time.sleep(0.5)

        self.tool = robot.GetActualTCPNum()[1]
        self.user = robot.GetActualWObjNum()[1]
        self.robot = robot
        self.log(f"Connected. Active tool={self.tool}, workpiece={self.user}.", "success")
        return self.tool, self.user

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                self.robot.CloseRPC()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
        self.robot = None
        self.log("Disconnected.", "info")

    def get_state(self) -> dict | None:
        """Snapshot of TCP pose, joints, and fault code, or None if disconnected.

        Reads the SDK's realtime state (kept current by its background socket
        thread), so it is cheap to poll. Returns
        ``{"tcp": [...], "joints": [...], "fault": [...]}``.
        """
        if self.robot is None:
            return None
        return {
            "tcp": self.robot.GetActualTCPPose()[1],           # [x,y,z,rx,ry,rz]
            "joints": self.robot.GetActualJointPosDegree()[1],  # [j1..j6] deg
            "fault": self.robot.GetRobotErrorCode()[1],         # [main, sub]
        }

    def reset_error(self) -> int:
        self._require_connection()
        ret = self.robot.ResetAllError()
        self.log(f"ResetAllError returned {ret}", _ret_level(ret))
        return ret

    # --- PTP motion ---------------------------------------------------------
    #
    # These are INTERNAL orchestration primitives, not user-facing controls.
    # The UI never exposes direct-movement widgets; motion is sequenced by
    # program logic:
    #   * move_ptp_joints — reach known joint configurations (e.g. home).
    #   * move_ptp_pose   — reach base-frame targets produced by the vision
    #     pipeline (camera coords put through the camera→base transform).

    def move_home(self, vel: float = DEFAULT_VEL) -> int:
        """Park the arm at the program's home configuration (PTP in joint space).

        Joint-space PTP is used deliberately: home is a known-good joint
        configuration, so going straight to the joints avoids IK and any
        reachability/elbow-flip surprises.
        """
        self.log("Moving to home position…", "info")
        return self.move_ptp_joints(HOME_JOINTS, vel)

    def move_ptp_joints(self, joints, vel: float = DEFAULT_VEL) -> int:
        """PTP (MoveJ) straight to an explicit joint configuration.

        Internal primitive — called by program logic to reach named joint
        configurations (see ``move_home``), never wired to a UI control.
        ``joints`` is [j1..j6] in degrees. MoveJ moves point-to-point in JOINT
        space, so this is the most direct/robust PTP — no IK, no reachability
        surprises (the joints are the target).
        """
        self._require_connection()
        joints = [float(j) for j in joints]
        if len(joints) != 6:
            raise RobotError(f"Expected 6 joint values, got {len(joints)}.")
        self.log(f"PTP → joints {_fmt(joints)} @ vel={vel}%", "info")
        ret = self.robot.MoveJ(joints, self.tool, self.user, vel=float(vel))
        self.log(f"MoveJ returned {ret}", _ret_level(ret))
        if ret != 0:
            raise RobotError(f"MoveJ failed with error code {ret}.")
        return ret

    def move_ptp_pose(self, x, y, z, rx=None, ry=None, rz=None,
                      vel: float = DEFAULT_VEL) -> int:
        """PTP to a base-frame TCP pose: solve IK, then MoveJ to those joints.

        Internal primitive for vision-driven motion: (x, y, z) is a target in
        the BASE frame, expected to come from the camera pipeline after the
        camera→base transform (camera-frame mm mapped through T_base_cam).
        It is never fed from hand-typed UI coordinates. GetInverseKinRef type=0
        treats the pose as absolute in the base frame. Orientation defaults to
        the robot's CURRENT orientation when rx/ry/rz are omitted — camera
        targets are positions, and reusing the current orientation avoids the
        no-IK-solution errors an arbitrary orientation can trigger.

        IK is seeded with the current joints (GetInverseKinRef) so it returns the
        solution nearest the current arm configuration — a natural, predictable
        move rather than an arbitrary elbow flip.
        """
        self._require_connection()
        current = self.robot.GetActualTCPPose()[1]
        current_joints = self.robot.GetActualJointPosDegree()[1]
        target = [
            float(x), float(y), float(z),
            current[3] if rx is None else float(rx),
            current[4] if ry is None else float(ry),
            current[5] if rz is None else float(rz),
        ]
        self.log(f"PTP → base pose {_fmt(target)}: solving IK…", "info")
        err, joints = self.robot.GetInverseKinRef(0, target, current_joints)
        if err != 0 or joints is None:
            raise RobotError(
                f"No IK solution for base pose {_fmt(target)} (error {err})."
            )
        self.log(f"IK ok → joints {_fmt(joints)}; MoveJ @ vel={vel}%", "info")
        ret = self.robot.MoveJ(joints, self.tool, self.user, desc_pos=target,
                               vel=float(vel))
        self.log(f"MoveJ returned {ret}", _ret_level(ret))
        if ret != 0:
            raise RobotError(f"MoveJ failed with error code {ret}.")
        return ret

    # --- linear motion ------------------------------------------------------
    #
    # Same rules as the PTP section: internal orchestration primitives, never
    # wired to direct-movement UI. Line endpoints come from the camera→base
    # transform; move_along_line is the weld-pass skeleton (welding I/O hooks
    # in around the traverse later).

    def move_linear(self, x, y, z, rx=None, ry=None, rz=None,
                    vel: float = DEFAULT_VEL, speed_mms: float | None = None,
                    accel: float = DEFAULT_ACCEL) -> int:
        """Straight-line MoveL to (x, y, z) in the BASE frame.

        Orientation defaults to the CURRENT one when rx/ry/rz are omitted; pass
        them to also reorient along the line (MoveL interpolates orientation as
        well as position, so the TCP travels straight while the tool smoothly
        rotates to the target RPY).

        Speed is ``vel`` percent by default. Pass ``speed_mms`` for PHYSICAL
        mode — the real travel speed in mm/s (this is how weld travel speed is
        commanded). Physical mode uses velAccParamMode=1 with ovl=mm/s and
        oacc=mm/s², and vel/acc MUST be sent as 100 (full scale) — the SDK
        defaults (vel=20, acc=0) get the move rejected with error 183.
        """
        self._require_connection()
        current = self.robot.GetActualTCPPose()[1]
        target = [
            float(x), float(y), float(z),
            current[3] if rx is None else float(rx),
            current[4] if ry is None else float(ry),
            current[5] if rz is None else float(rz),
        ]
        speed = f"{speed_mms:.0f} mm/s" if speed_mms is not None else f"{vel:.0f}%"
        self.log(f"MoveL → {_fmt(target)} @ {speed}", "info")
        ret = self._movel(target, vel, speed_mms, accel)
        self.log(f"MoveL returned {ret}", _ret_level(ret))
        if ret != 0:
            hint = (" (physical mode needs vel/acc=100 + real ovl/oacc)"
                    if ret in (182, 183) else "")
            raise RobotError(f"MoveL failed with error code {ret}.{hint}")
        return ret

    def solve_torch_down_rpy(self, x, y, z, min_down: float = 0.7) -> list[float]:
        """A torch-DOWN [rx, ry, rz] reachable at (x, y, z), nearest to now.

        Tries the candidate orientations, keeps only those whose torch axis
        points downward (base -Z component ≥ min_down), solves each with
        GetInverseKinRef seeded with the current joints, and returns the RPY of
        the solution with the least joint travel. Raises RobotError when no
        torch-down orientation is reachable (move the work closer or flip
        geometry.TORCH_AXIS).
        """
        self._require_connection()
        ref_joints = self.robot.GetActualJointPosDegree()[1]
        best: tuple[float, list[float]] | None = None
        for rx, ry, rz in geometry.orientation_candidates():
            if geometry.torch_dir_in_base(rx, ry, rz)[2] > -min_down:
                continue  # not pointing down enough
            err, joints = self.robot.GetInverseKinRef(0, [x, y, z, rx, ry, rz],
                                                      ref_joints)
            if err != 0 or joints is None:
                continue
            travel = sum(abs(a - b) for a, b in zip(joints, ref_joints, strict=True))
            if best is None or travel < best[0]:
                best = (travel, [rx, ry, rz])
        if best is None:
            raise RobotError(
                f"No torch-DOWN orientation reachable at ({x}, {y}, {z})."
            )
        return best[1]

    def move_linear_torch_down(self, x, y, z, vel: float = DEFAULT_VEL,
                               speed_mms: float | None = None,
                               accel: float = DEFAULT_ACCEL) -> int:
        """MoveL to (x, y, z) with an auto-solved torch-DOWN orientation there.

        The tool reorients along the line to the solved pose — no hand-picked
        angles needed. Raises RobotError if no torch-down orientation is
        reachable at the destination (nothing moves in that case).
        """
        rpy = self.solve_torch_down_rpy(x, y, z)
        self.log(f"Torch-down RPY at destination: {_fmt(rpy)}", "info")
        return self.move_linear(x, y, z, *rpy, vel=vel, speed_mms=speed_mms,
                                accel=accel)

    def move_along_line(self, p1, p2, vel: float = DEFAULT_VEL,
                        approach: float = 30.0, standoff: float = 10.0,
                        speed_mms: float | None = None,
                        accel: float = DEFAULT_ACCEL, align_yaw: bool = True,
                        pull: bool = True, yaw_offset: float = 0.0,
                        weld: WeldParams | None = None) -> None:
        """MoveL the TCP along the seam P1→P2: approach, descend, traverse, retract.

        ``p1``/``p2`` are full base-frame XYZ (mm) from the camera→base
        transform; each endpoint keeps its OWN Z so a tilted seam is followed in
        3D. Tilt (rx, ry) comes from the current pose — establish a tool-down
        orientation first (e.g. move_linear_torch_down to above P1).

        ``standoff`` mm is held ABOVE the surface for the whole pass so the tip
        never touches; approach/retract sit a further ``approach`` mm above
        that. Positioning legs run at ``vel`` percent; the traverse runs at
        ``speed_mms`` mm/s (physical mode) when given — that MoveL IS the weld
        travel speed.

        ``align_yaw`` rotates rz about base Z so the torch faces the line
        (rx/ry kept). ``pull=True`` — the DEFAULT and the project's welding
        technique — adds 180° so the torch DRAGS against travel (backhand);
        push is for experiments only.

        With ``weld``, the traverse becomes a weld stroke: the arc is struck at
        P1 (weld_start) and ended at P2 (weld_end). weld_end runs in a finally,
        so the arc is ALWAYS dropped — even if the traverse errors or the worker
        is torn down. If the arc does not establish, the pass aborts with no
        traverse. WeldParams defaults to a DRY weld (live=False): identical
        motion, nothing energized.

        Raises RobotError on the first failed leg (remaining legs are skipped).
        """
        self._require_connection()
        pose = self.robot.GetActualTCPPose()[1]
        rx, ry, rz = pose[3], pose[4], pose[5]
        if align_yaw:
            rz_new = geometry.yaw_to_line(rx, ry, rz, p1, p2, pull=pull,
                                          yaw_offset=yaw_offset)
            self.log(f"Yaw aligned to line ({'pull/drag' if pull else 'push'}): "
                     f"rz {rz:.1f} → {rz_new:.1f}°", "info")
            rz = rz_new

        def point(p, dz: float = 0.0) -> list[float]:
            # Surface Z + standoff (clear of the surface) + dz (approach lift).
            return [float(p[0]), float(p[1]), float(p[2]) + standoff + dz,
                    rx, ry, rz]

        tspeed = f"{speed_mms:.0f} mm/s" if speed_mms is not None else f"{vel:.0f}%"
        wtag = ("" if weld is None else
                (" WELD LIVE" if weld.live else " WELD dry-run"))
        self.log(
            f"Line pass P1({p1[0]:.1f},{p1[1]:.1f},{p1[2]:.1f}) → "
            f"P2({p2[0]:.1f},{p2[1]:.1f},{p2[2]:.1f}) standoff=+{standoff:.0f}mm "
            f"approach=+{approach:.0f}mm traverse={tspeed}{wtag}", "info")

        def leg(label: str, target, leg_speed_mms: float | None = None) -> None:
            ret = self._movel(target, vel, leg_speed_mms, accel)
            self.log(f"MoveL {label}: {ret}", _ret_level(ret))
            if ret != 0:
                hint = (" (physical mode needs vel/acc=100 + real ovl/oacc)"
                        if ret in (182, 183) else "")
                raise RobotError(f"Leg '{label}' failed with error code {ret}."
                                 f"{hint}")

        leg("approach over P1", point(p1, approach))
        leg("descend to P1", point(p1))

        # Weld stroke: strike at P1, traverse, end at P2. weld_end sits in the
        # finally so the arc NEVER stays lit, whatever happens to the traverse.
        try:
            if weld is not None:
                self.weld_start(weld)
            leg("traverse to P2" + (" (WELD stroke)" if weld is not None else ""),
                point(p2), speed_mms)
        finally:
            if weld is not None:
                self.weld_end(weld)

        leg("retract over P2", point(p2, approach))

    # --- welding I/O ----------------------------------------------------------
    #
    # Ported from farino_app test_weld_pass.py / red_line_viewer.py. The welder
    # signals live on the CO output bank (arc=CO0, wire fwd=CO1, wire rev=CO2,
    # gas=CO3), which is BLOCKED for plain SetDO ("channel configured function")
    # — these dedicated welding commands are the ONLY way to drive them.
    # A weld pass is: descend to P1 → weld_start → traverse (the travel speed
    # IS that MoveL) → weld_end → retract; see move_along_line(weld=...).

    def set_weld_current(self, amps: float, io: int = 0) -> int:
        """Override the welding current (A) on AO0 (bypasses the WebApp process)."""
        self._require_connection()
        ret = self.robot.WeldingSetCurrent(io, float(amps), 0, 0)   # AO0 = current
        self.log(f"WeldingSetCurrent {amps:.0f} A → {ret}", _ret_level(ret))
        return ret

    def set_weld_voltage(self, volts: float, io: int = 0) -> int:
        """Override the welding voltage (V) on AO1 (bypasses the WebApp process)."""
        self._require_connection()
        ret = self.robot.WeldingSetVoltage(io, float(volts), 1, 0)  # AO1 = voltage
        self.log(f"WeldingSetVoltage {volts:.1f} V → {ret}", _ret_level(ret))
        return ret

    def set_gas(self, on: bool, io: int = 0) -> int:
        """Open/close the shielding gas valve (CO3 via SetAspirated)."""
        self._require_connection()
        ret = self.robot.SetAspirated(io, 1 if on else 0)
        self.log(f"Gas {'ON' if on else 'OFF'} → {ret}", _ret_level(ret))
        return ret

    def start_wire_feed(self, reverse: bool = False, io: int = 0) -> int:
        """Run the wire feeder (cold, no arc): forward feeds, reverse retracts."""
        self._require_connection()
        if reverse:
            ret = self.robot.SetReverseWireFeed(io, 1)
        else:
            ret = self.robot.SetForwardWireFeed(io, 1)
        self.log(f"Wire feed {'REVERSE' if reverse else 'FORWARD'} → {ret}",
                 _ret_level(ret))
        return ret

    def stop_wire_feed(self, io: int = 0) -> int:
        """Stop the wire feeder (clears BOTH directions)."""
        self._require_connection()
        ret_fwd = self.robot.SetForwardWireFeed(io, 0)
        ret_rev = self.robot.SetReverseWireFeed(io, 0)
        ret = ret_fwd or ret_rev
        self.log(f"Wire feed STOP → {ret}", _ret_level(ret))
        return ret

    def weld_start(self, w: WeldParams) -> None:
        """Strike the arc for a weld stroke: current/voltage → gas → ARCStart.

        With ``w.live`` False this is a DRY WELD — the steps are logged but
        nothing is energized, so the surrounding motion is identical to a real
        pass. Raises RobotError if the arc does not establish (the caller must
        NOT traverse in that case).
        """
        self._require_connection()
        if not w.live:
            self.log("[dry weld] would set current/voltage, open gas, ARCStart "
                     "(nothing energized)", "info")
            return
        if w.current is not None:
            self.set_weld_current(w.current, w.io)
        if w.voltage is not None:
            self.set_weld_voltage(w.voltage, w.io)
        if w.gas:
            self.set_gas(True, w.io)
        self.log(f"ARCStart (process #{w.arc_num}) …", "info")
        ret = self.robot.ARCStart(w.io, w.arc_num, w.timeout_ms)
        self.log(f"ARCStart returned {ret}", _ret_level(ret))
        if ret != 0:
            raise RobotError(f"Arc did not establish (ARCStart error {ret}).")

    def weld_end(self, w: WeldParams) -> None:
        """End the arc and shut the gas. NEVER raises — safe to call twice and
        from a finally block, so the arc can always be dropped on any exit."""
        if not w.live:
            self.log("[dry weld] would ARCEnd + close gas", "info")
            return
        if self.robot is None:
            return
        ret = self.robot.ARCEnd(w.io, w.arc_num, w.timeout_ms)
        self.log(f"ARCEnd returned {ret}", _ret_level(ret))
        if w.gas:
            try:
                self.set_gas(False, w.io)
            except Exception:  # noqa: BLE001 — gas-off is best-effort cleanup
                pass

    # --- helpers ------------------------------------------------------------

    def _movel(self, target, vel_pct: float, speed_mms: float | None,
               accel: float) -> int:
        """One MoveL. Percentage mode by default; PHYSICAL mode with speed_mms."""
        if speed_mms is None:
            return self.robot.MoveL(desc_pos=target, tool=self.tool,
                                    user=self.user, vel=float(vel_pct))
        return self.robot.MoveL(desc_pos=target, tool=self.tool, user=self.user,
                                vel=100.0, acc=100.0, ovl=float(speed_mms),
                                oacc=float(accel), velAccParamMode=1)

    def _require_connection(self) -> None:
        if self.robot is None:
            raise RobotError("Not connected to the robot.")
