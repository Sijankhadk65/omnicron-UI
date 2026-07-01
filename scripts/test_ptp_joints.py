"""Isolated hardware test: PTP move by JOINT values (MoveJ).

Exercises RobotController.move_ptp_joints against the real robot. It connects,
reads the current joints, nudges ONE joint by a small delta, moves there (PTP /
MoveJ), then moves back to the start — so the robot ends where it began.

Safe by design: small default delta, low velocity, and a y/N confirmation that
prints the exact target before any motion.

Run (robot powered + reachable):
    uv run python scripts/test_ptp_joints.py

Edit ROBOT_IP / JOINT_INDEX / DELTA_DEG / VEL below for your cell.
"""

from __future__ import annotations

import sys

from omnicron_ui.robot.controller import RobotController, RobotError

ROBOT_IP = "192.168.58.2"
JOINT_INDEX = 0        # which joint to nudge (0 = j1 … 5 = j6)
DELTA_DEG = 10.0       # how far to move it (degrees)
VEL = 15.0             # PTP velocity (% of max) — keep low for testing
RETURN_TO_START = True  # move back to the starting joints afterwards


def main() -> int:
    c = RobotController()  # default logger prints [level] message
    try:
        c.connect(ROBOT_IP)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED to connect: {exc}")
        return 1

    try:
        state = c.get_state()
        start = list(state["joints"])
        print(f"\nCurrent joints: {start}")

        target = list(start)
        target[JOINT_INDEX] += DELTA_DEG
        print(f"Target joints:  {target}  (j{JOINT_INDEX + 1} {DELTA_DEG:+.1f}°)")

        if input("\nMove to target? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return 0

        c.move_ptp_joints(target, vel=VEL)
        print(f"Reached: {c.get_state()['joints']}")

        if RETURN_TO_START:
            print("\nReturning to start…")
            c.move_ptp_joints(start, vel=VEL)
            print(f"Back at: {c.get_state()['joints']}")

        print("\n✅ PTP-by-joints test SUCCESSFUL")
        return 0
    except RobotError as exc:
        print(f"\n❌ PTP-by-joints test FAILED: {exc}")
        return 1
    finally:
        c.disconnect()


if __name__ == "__main__":
    sys.exit(main())
