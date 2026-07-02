"""Isolated hardware test: the PTP and linear motion helper primitives.

Exercises the RobotController motion helpers against a LIVE robot, the way the
program uses them internally (never user-typed coordinates):

    * move_ptp_joints  — MoveJ to an explicit joint configuration (joint space).
    * move_ptp_pose    — IK a base-frame TCP pose, then MoveJ to it.
    * move_linear      — straight-line MoveL to a base-frame TCP pose.
    * pull orientation — yaw the torch to the PULL welding angle for a line,
                         KEEPING the current tool tilt (red_line_viewer approach).

Every target is a SMALL RELATIVE OFFSET from wherever the arm is standing, read
live from GetActualTCPPose / GetActualJointPosDegree. Nothing is hard-coded to a
cell, and each test moves OUT and then RETURNS to its start pose, so the arm ends
where it began. Speeds are kept low. Every stage is gated by a y/N confirm.

This does NOT weld, feed wire, or open gas — motion only.

Run (robot powered, e-stop clear, area around the arm clear):
    uv run python scripts/test_motion.py

Edit ROBOT_IP / VEL / offsets below for your cell.
"""

from __future__ import annotations

import sys

from omnicron_ui.robot import geometry
from omnicron_ui.robot.controller import RobotController, RobotError

ROBOT_IP = "192.168.58.2"
VEL = 15.0            # PTP/linear velocity (% of max) — keep low for testing

# Relative test offsets (arm returns to start after each). Small on purpose.
JOINT_NUDGE_DEG = 5.0     # J6 (wrist) nudge for the joint-space PTP test
POSE_OFFSET_MM = 30.0     # +X base-frame offset for the pose / linear tests

# Base-frame weld-line direction for the pull-orientation test (mm). The torch
# yaws to face along this line; only rz changes — the current tool tilt is kept.
LINE_DX, LINE_DY = 100.0, 0.0

TOL_DEG = 0.5             # per-joint / per-angle arrival tolerance
TOL_MM = 1.0             # per-axis XYZ arrival tolerance


def _fmt(vals) -> str:
    return "[" + ", ".join(f"{v:.1f}" for v in vals) + "]"


def _confirm(prompt: str) -> bool:
    return input(f"\n{prompt} [y/N] ").strip().lower() == "y"


def _pose_error_mm(reached, target) -> float:
    """Max per-axis XYZ error between two poses (ignores orientation)."""
    return max(abs(a - b) for a, b in zip(reached[:3], target[:3], strict=True))


def test_ptp_joints(c: RobotController) -> bool | None:
    """Nudge J6 by JOINT_NUDGE_DEG and return — pure joint-space PTP (MoveJ)."""
    start = list(c.get_state()["joints"])
    target = list(start)
    target[5] += JOINT_NUDGE_DEG
    print(f"\n[PTP joints]  start: {_fmt(start)}")
    print(f"[PTP joints]  target (J6 {JOINT_NUDGE_DEG:+.0f}°): {_fmt(target)}")
    if not _confirm("Run joint-space PTP test?"):
        print("  skipped.")
        return None

    c.move_ptp_joints(target, vel=VEL)
    reached = c.get_state()["joints"]
    err = max(abs(a - b) for a, b in zip(reached, target, strict=True))
    print(f"  reached: {_fmt(reached)}  (max joint error {err:.2f}°)")

    c.move_ptp_joints(start, vel=VEL)  # return
    ok = err <= TOL_DEG
    print(f"  {'✅ PTP joints OK' if ok else f'❌ error above {TOL_DEG}°'}")
    return ok


def test_ptp_pose(c: RobotController) -> bool | None:
    """MoveJ (via IK) to current TCP pose +POSE_OFFSET_MM in base X, then return."""
    start = list(c.get_state()["tcp"])
    target = list(start)
    target[0] += POSE_OFFSET_MM
    print(f"\n[PTP pose]  start TCP:  {_fmt(start)}")
    print(f"[PTP pose]  target (X {POSE_OFFSET_MM:+.0f}mm): {_fmt(target)}")
    if not _confirm("Run base-pose PTP (IK + MoveJ) test?"):
        print("  skipped.")
        return None

    c.move_ptp_pose(target[0], target[1], target[2])
    reached = c.get_state()["tcp"]
    err = _pose_error_mm(reached, target)
    print(f"  reached: {_fmt(reached)}  (max XYZ error {err:.2f}mm)")

    c.move_ptp_pose(start[0], start[1], start[2])  # return
    ok = err <= TOL_MM
    print(f"  {'✅ PTP pose OK' if ok else f'❌ error above {TOL_MM}mm'}")
    return ok


def test_linear(c: RobotController) -> bool | None:
    """MoveL (straight line) to current TCP pose +POSE_OFFSET_MM in base X, return."""
    start = list(c.get_state()["tcp"])
    target = list(start)
    target[0] += POSE_OFFSET_MM
    target[1] += POSE_OFFSET_MM
    print(f"\n[MoveL]  start TCP:  {_fmt(start)}")
    print(f"[MoveL]  target (X {POSE_OFFSET_MM:+.0f}mm): {_fmt(target)}")
    if not _confirm("Run straight-line MoveL test?"):
        print("  skipped.")
        return None

    c.move_linear(target[0], target[1], target[2], vel=VEL)
    reached = c.get_state()["tcp"]
    err = _pose_error_mm(reached, target)
    print(f"  reached: {_fmt(reached)}  (max XYZ error {err:.2f}mm)")

    c.move_linear(start[0], start[1], start[2], vel=VEL)  # return
    ok = err <= TOL_MM
    print(f"  {'✅ MoveL OK' if ok else f'❌ error above {TOL_MM}mm'}")
    return ok


def test_pull_orientation(c: RobotController) -> bool | None:
    """Yaw the torch into the PULL (drag/backhand) welding angle, keeping tilt.

    This is the red_line_viewer approach — NO torch-down solve. It holds the
    CURRENT tool tilt (rx, ry) and rotates ONLY rz about base Z so the torch
    faces along the weld line, +180° for pull (drag/backhand). Because just the
    wrist yaws — no reorientation to a solved tool-down pose — the motion is
    smooth, not erratic. Position (including Z) is held; only the yaw changes.

    Establish a tool-down-ish orientation by jogging first if you want the torch
    truly vertical; this test keeps whatever tilt the arm currently has.
    """
    pose = list(c.get_state()["tcp"])
    rx, ry, rz = pose[3], pose[4], pose[5]
    # A base-frame weld line through the current position (only its XY direction
    # matters for the yaw alignment).
    p1 = [pose[0], pose[1], pose[2]]
    p2 = [pose[0] + LINE_DX, pose[1] + LINE_DY, pose[2]]
    rz_pull = geometry.yaw_to_line(rx, ry, rz, p1, p2, pull=True)
    print(f"\n[pull yaw]  current orientation: rx={rx:.1f} ry={ry:.1f} rz={rz:.1f}")
    print(f"[pull yaw]  weld line dir (base): dx={LINE_DX:.0f} dy={LINE_DY:.0f}")
    print(f"[pull yaw]  pull orientation:  rz {rz:.1f}° → {rz_pull:.1f}° (tilt kept)")
    if not _confirm("Yaw torch to the pull welding orientation?"):
        print("  skipped.")
        return None

    # Hold position + tilt; rotate ONLY rz to the pull orientation.
    c.move_linear(pose[0], pose[1], pose[2], rx, ry, rz_pull, vel=VEL)
    reached = c.get_state()["tcp"]
    rz_err = abs(((reached[5] - rz_pull + 180.0) % 360.0) - 180.0)
    tilt_err = max(abs(reached[3] - rx), abs(reached[4] - ry))
    xyz_err = _pose_error_mm(reached, pose)
    print(f"  reached rz={reached[5]:.1f}° (err {rz_err:.2f}°), "
          f"tilt kept within {tilt_err:.2f}°, XYZ moved {xyz_err:.2f}mm")

    # Return to the original orientation.
    c.move_linear(pose[0], pose[1], pose[2], rx, ry, rz, vel=VEL)
    ok = rz_err <= TOL_DEG and tilt_err <= TOL_DEG and xyz_err <= TOL_MM
    print(f"  {'✅ pull orientation OK' if ok else '❌ yaw off / tilt or position drifted'}")
    return ok


def main() -> int:
    c = RobotController()  # default logger prints [level] message
    try:
        c.connect(ROBOT_IP)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED to connect: {exc}")
        return 1

    try:
        state = c.get_state()
        print(f"\nCurrent joints: {_fmt(state['joints'])}")
        print(f"Current TCP:    {_fmt(state['tcp'])}")
        print(f"Fault code:     {state['fault']}")
        print("\n⚠️  The arm will make small moves. Keep the workspace clear.")

        results: dict[str, bool | None] = {
            "PTP joints": test_ptp_joints(c),
            "PTP pose (IK)": test_ptp_pose(c),
            "MoveL linear": test_linear(c),
            "Pull orientation": test_pull_orientation(c),
        }

        print("\n=== Summary ===")
        for name, ok in results.items():
            tag = "skipped" if ok is None else ("PASS ✅" if ok else "FAIL ❌")
            print(f"  {name:<16} {tag}")

        failed = [n for n, ok in results.items() if ok is False]
        return 1 if failed else 0
    except RobotError as exc:
        print(f"\n❌ Motion test FAILED: {exc}")
        return 1
    finally:
        c.disconnect()


if __name__ == "__main__":
    sys.exit(main())
