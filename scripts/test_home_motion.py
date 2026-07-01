"""Isolated hardware test: the internal move-home primitive.

Exercises RobotController.move_home the way the program will use it — as an
internally triggered "park the arm" step between tasks, not a user-driven move.
It connects, shows the current joints next to HOME_JOINTS, and on confirmation
runs move_home() and verifies the arm arrived (max per-joint error).

There is deliberately NO way to type a target here: home is a fixed, known-good
joint configuration owned by the program (controller.HOME_JOINTS).

Run (robot powered + reachable):
    uv run python scripts/test_home_motion.py

Edit ROBOT_IP / VEL below for your cell.
"""

from __future__ import annotations

import sys

from omnicron_ui.robot.controller import HOME_JOINTS, RobotController, RobotError

ROBOT_IP = "192.168.58.2"
VEL = 15.0            # PTP velocity (% of max) — keep low for testing
TOL_DEG = 0.5         # per-joint arrival tolerance


def _fmt(vals) -> str:
    return "[" + ", ".join(f"{v:.1f}" for v in vals) + "]"


def main() -> int:
    c = RobotController()  # default logger prints [level] message
    try:
        c.connect(ROBOT_IP)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED to connect: {exc}")
        return 1

    try:
        start = list(c.get_state()["joints"])
        print(f"\nCurrent joints: {_fmt(start)}")
        print(f"Home joints:    {_fmt(HOME_JOINTS)}")

        if input("\nMove to HOME? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return 0

        c.move_home(vel=VEL)

        reached = c.get_state()["joints"]
        err = max(abs(a - b) for a, b in zip(reached, HOME_JOINTS, strict=True))
        print(f"Reached: {_fmt(reached)}  (max joint error {err:.2f}°)")
        if err > TOL_DEG:
            print(f"\n❌ Home motion FAILED: joint error above {TOL_DEG}°")
            return 1

        print("\n✅ Move-home test SUCCESSFUL")
        return 0
    except RobotError as exc:
        print(f"\n❌ Move-home test FAILED: {exc}")
        return 1
    finally:
        c.disconnect()


if __name__ == "__main__":
    sys.exit(main())
