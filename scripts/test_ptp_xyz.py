"""Isolated hardware test: PTP move by base-frame XYZ (IK -> MoveJ).

Exercises RobotController.move_ptp_pose against the real robot. It connects, reads
the current TCP pose, offsets the position by a small delta in the BASE frame
(keeping the current orientation), solves IK, moves there (PTP / MoveJ), then
returns to the start.

This is the "coords in the base frame, TCP moves there" path: the target is a
base-frame TCP position and the controller solves IK seeded with the current
joints, so the arm takes the natural nearby solution.

Safe by design: small default offset, low velocity, and a y/N confirmation that
prints the exact target before any motion.

Run (robot powered + reachable):
    uv run python scripts/test_ptp_xyz.py

Edit ROBOT_IP / DELTA_* / VEL below for your cell.
"""

from __future__ import annotations

import math
import sys

from omnicron_ui.robot.controller import RobotController, RobotError

ROBOT_IP = "192.168.58.2"
DELTA_X = 30.0   # base-frame offset applied to the current TCP position (mm)
DELTA_Y = 0.0
DELTA_Z = 0.0
VEL = 15.0       # PTP velocity (% of max) — keep low for testing
RETURN_TO_START = True


def _fmt(p) -> str:
    return "[" + ", ".join(f"{v:.1f}" for v in p) + "]"


def main() -> int:
    c = RobotController()  # default logger prints [level] message
    try:
        c.connect(ROBOT_IP)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED to connect: {exc}")
        return 1

    try:
        start = list(c.get_state()["tcp"])   # [x, y, z, rx, ry, rz]
        print(f"\nCurrent TCP pose: {_fmt(start)}")

        tx, ty, tz = start[0] + DELTA_X, start[1] + DELTA_Y, start[2] + DELTA_Z
        print(f"Target position:  [{tx:.1f}, {ty:.1f}, {tz:.1f}]  "
              f"(offset {DELTA_X:+.0f},{DELTA_Y:+.0f},{DELTA_Z:+.0f} mm, orientation kept)")

        if input("\nMove to target? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return 0

        # Keep current orientation (rx/ry/rz omitted -> controller reuses current).
        c.move_ptp_pose(tx, ty, tz, vel=VEL)
        final = c.get_state()["tcp"]
        err = math.dist(final[:3], (tx, ty, tz))
        print(f"Reached: {_fmt(final)}  (position error {err:.1f} mm)")

        if RETURN_TO_START:
            print("\nReturning to start…")
            c.move_ptp_pose(start[0], start[1], start[2], vel=VEL)
            print(f"Back at: {_fmt(c.get_state()['tcp'])}")

        print("\n✅ PTP-by-XYZ (IK) test SUCCESSFUL")
        return 0
    except RobotError as exc:
        print(f"\n❌ PTP-by-XYZ (IK) test FAILED: {exc}")
        return 1
    finally:
        c.disconnect()


if __name__ == "__main__":
    sys.exit(main())
