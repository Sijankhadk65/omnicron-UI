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

from omnicron_ui.robot.sdk import Robot

# Log levels the UI understands (info/success/warn/error). Matches the color
# coding used by the log panel.
LogFn = Callable[[str, str], None]


def _default_log(msg: str, level: str = "info") -> None:
    print(f"[{level}] {msg}")


def _ret_level(ret: int) -> str:
    """Map an SDK return code to a log level (0 = success, else error)."""
    return "success" if ret == 0 else "error"


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

    # --- helpers ------------------------------------------------------------

    def _require_connection(self) -> None:
        if self.robot is None:
            raise RobotError("Not connected to the robot.")
