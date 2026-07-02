"""Isolated hardware test: the PTP and linear motion helper primitives.

Exercises the RobotController motion helpers against a LIVE robot, the way the
program uses them internally (never user-typed coordinates):

    * move_ptp_joints  — MoveJ to an explicit joint configuration (joint space).
    * move_ptp_pose    — IK a base-frame TCP pose, then MoveJ to it.
    * move_linear      — straight-line MoveL to a base-frame TCP pose.

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

from omnicron_ui.robot.controller import RobotController, RobotError

ROBOT_IP = "192.168.58.2"
VEL = 15.0            # PTP/linear velocity (% of max) — keep low for testing

# Relative test offsets (arm returns to start after each). Small on purpose.
JOINT_NUDGE_DEG = 5.0     # J6 (wrist) nudge for the joint-space PTP test
POSE_OFFSET_MM = 30.0     # +X base-frame offset for the pose / linear tests

TOL_DEG = 0.5             # per-joint arrival tolerance (joint test)
TOL_MM = 1.0             # per-axis XYZ arrival tolerance (pose / linear tests)


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
