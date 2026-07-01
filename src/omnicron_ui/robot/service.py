"""Asynchronous robot service — keeps all blocking SDK calls off the Qt UI thread.

Why this exists
---------------
Fairino SDK calls block: ``connect`` sleeps ~2.5 s, a ``MoveL`` can take several
seconds, and the realtime reads hit a socket. Calling any of these from the Qt
main thread freezes the whole UI (unresponsive window, no repaints). So every
robot call runs on a dedicated worker thread and results come back as Qt signals.

Design
------
* ``RobotService`` is a ``QObject`` that is *moved* onto its own ``QThread``
  (the canonical Qt pattern). After the move, its slots execute on the worker
  thread while its methods are still *called* from the UI thread.
* The UI submits **jobs** — ``(name, fn)`` where ``fn`` takes the
  ``RobotController`` and returns a result. ``submit()`` only emits a queued
  signal, so it returns instantly and never blocks the UI.
* Jobs run **serialized** on the worker thread (one robot, one motion at a time).
* Everything the UI needs to react to is a signal: ``log``, ``busy``,
  ``connected``, ``disconnected``, ``state``, ``job_done``, ``job_failed``.
  Qt delivers them to the UI thread automatically (queued), so slots wired to
  widgets are safe.

Threading rules for callers
---------------------------
* Call ``submit`` / the convenience methods from the UI thread only.
* Never touch ``RobotController`` (or the RPC) from the UI thread — go through a
  job. The controller is created on the worker thread and lives there.
* A ``state`` poll shares the worker thread with commands, so state updates pause
  while a long motion is running and resume when it finishes. That is expected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal

from omnicron_ui.robot.controller import RobotController, WeldParams

# A job body: takes the controller, returns anything (delivered via job_done).
JobFn = Callable[[RobotController], Any]


@dataclass(frozen=True)
class _Job:
    name: str
    fn: JobFn


class RobotService(QObject):
    """Runs robot jobs on a worker thread and reports back through signals."""

    # --- signals emitted toward the UI (delivered on the UI thread) ---------
    log = Signal(str, str)          # message, level (info/success/warn/error)
    busy = Signal(bool)             # True while a job is running
    connected = Signal(int, int)    # active tool num, active workpiece num
    disconnected = Signal()
    state = Signal(object)          # dict from RobotController.get_state(), or None
    job_done = Signal(str, object)  # job name, result
    job_failed = Signal(str, str)   # job name, error text

    # --- internal: marshals a job onto the worker thread --------------------
    _submit = Signal(object)        # carries a _Job; queued to the worker thread

    def __init__(self, state_interval_ms: int = 300) -> None:
        super().__init__()
        self._state_interval_ms = state_interval_ms
        self._controller: RobotController | None = None
        self._state_timer: QTimer | None = None

        self._thread = QThread()
        self._thread.setObjectName("robot-worker")
        self.moveToThread(self._thread)

        # Build the controller and the poll timer once we're on the worker thread.
        self._thread.started.connect(self._on_thread_started)
        # Jobs are marshalled onto the worker thread via a queued connection.
        self._submit.connect(self._run_job, Qt.ConnectionType.QueuedConnection)
        self._thread.start()

    # --- public API (call from the UI thread) -------------------------------

    def submit(self, name: str, fn: JobFn) -> None:
        """Queue a job to run on the worker thread. Returns immediately."""
        self._submit.emit(_Job(name, fn))

    def connect_robot(self, ip: str) -> None:
        def job(ctrl: RobotController):
            tool, user = ctrl.connect(ip)
            self.connected.emit(tool, user)
            return tool, user

        self.submit("connect", job)

    def disconnect_robot(self) -> None:
        def job(ctrl: RobotController):
            ctrl.disconnect()
            self.disconnected.emit()

        self.submit("disconnect", job)

    def reset_error(self) -> None:
        self.submit("reset_error", lambda ctrl: ctrl.reset_error())

    # Motion methods are orchestration entry points for program logic (task
    # sequences, the vision pipeline) — they are never wired to direct-movement
    # UI controls.

    def move_home(self, vel: float = 20.0) -> None:
        """Park the arm at the home joint configuration (worker thread)."""
        self.submit("home", lambda ctrl: ctrl.move_home(vel))

    def move_ptp_joints(self, joints, vel: float = 20.0) -> None:
        """PTP to a known joint configuration (runs on the worker thread)."""
        self.submit("ptp_joints", lambda ctrl: ctrl.move_ptp_joints(joints, vel))

    def move_ptp_pose(self, x, y, z, rx=None, ry=None, rz=None,
                      vel: float = 20.0) -> None:
        """PTP to a camera-derived base-frame target via IK (worker thread).

        (x, y, z) is expected to come from the camera→base transform, not from
        user-typed coordinates.
        """
        self.submit(
            "ptp_pose",
            lambda ctrl: ctrl.move_ptp_pose(x, y, z, rx, ry, rz, vel),
        )

    def move_linear(self, x, y, z, rx=None, ry=None, rz=None, vel: float = 20.0,
                    speed_mms: float | None = None) -> None:
        """Straight-line MoveL to a base-frame target (worker thread).

        ``speed_mms`` switches to physical mode (real mm/s travel speed).
        """
        self.submit(
            "linear",
            lambda ctrl: ctrl.move_linear(x, y, z, rx, ry, rz, vel=vel,
                                          speed_mms=speed_mms),
        )

    def move_linear_torch_down(self, x, y, z, vel: float = 20.0) -> None:
        """MoveL to a base-frame target with an auto-solved torch-down RPY."""
        self.submit(
            "linear_torch_down",
            lambda ctrl: ctrl.move_linear_torch_down(x, y, z, vel=vel),
        )

    def move_along_line(self, p1, p2, vel: float = 20.0,
                        speed_mms: float | None = None,
                        weld: WeldParams | None = None, **kwargs) -> None:
        """Seam pass P1→P2 (approach/descend/traverse/retract; worker thread).

        Endpoints are camera-derived base-frame XYZ. Traverse runs at
        ``speed_mms`` (physical mode) when given — the weld travel speed.
        Pass ``weld`` (WeldParams) to make the traverse a weld stroke; the
        default WeldParams is a DRY weld (live=False, nothing energized).
        """
        self.submit(
            "line_pass",
            lambda ctrl: ctrl.move_along_line(p1, p2, vel=vel,
                                              speed_mms=speed_mms, weld=weld,
                                              **kwargs),
        )

    # Welding I/O primitives — for bring-up sequences run by program logic
    # (wire tension, gas check), not for direct-control UI.

    def set_gas(self, on: bool) -> None:
        """Open/close the shielding gas valve (worker thread)."""
        self.submit("gas", lambda ctrl: ctrl.set_gas(on))

    def start_wire_feed(self, reverse: bool = False) -> None:
        """Run the wire feeder cold (no arc); reverse retracts the wire."""
        self.submit("wire_feed", lambda ctrl: ctrl.start_wire_feed(reverse))

    def stop_wire_feed(self) -> None:
        """Stop the wire feeder (both directions)."""
        self.submit("wire_stop", lambda ctrl: ctrl.stop_wire_feed())

    def shutdown(self) -> None:
        """Stop the worker thread cleanly. Call from the UI thread on app exit."""
        # Queue a teardown on the worker thread (stop the timer, close the RPC),
        # then ask the event loop to quit and wait for the thread to finish.
        self._submit.emit(_Job("shutdown", self._teardown))
        self._thread.quit()
        self._thread.wait(3000)

    # --- worker-thread slots (run on the worker thread) ---------------------

    def _on_thread_started(self) -> None:
        self._controller = RobotController(log=self._emit_log)
        self._state_timer = QTimer()
        self._state_timer.setInterval(self._state_interval_ms)
        self._state_timer.timeout.connect(self._poll_state)
        self._state_timer.start()

    def _emit_log(self, msg: str, level: str = "info") -> None:
        self.log.emit(msg, level)

    def _run_job(self, job: _Job) -> None:
        self.busy.emit(True)
        try:
            result = job.fn(self._controller)
            self.job_done.emit(job.name, result)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            self.log.emit(f"{job.name} failed: {exc}", "error")
            self.job_failed.emit(job.name, str(exc))
        finally:
            self.busy.emit(False)

    def _teardown(self, ctrl: RobotController) -> None:
        if self._state_timer is not None:
            self._state_timer.stop()
        ctrl.disconnect()

    def _poll_state(self) -> None:
        ctrl = self._controller
        if ctrl is None or not ctrl.connected:
            return
        try:
            self.state.emit(ctrl.get_state())
        except Exception:  # noqa: BLE001 — a bad poll must not kill the timer
            pass
