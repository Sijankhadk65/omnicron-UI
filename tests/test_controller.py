"""Unit tests for RobotController's motion primitives — no robot, no Qt.

The controller is deliberately Qt-free and takes the SDK object as plain state
(``self.robot``), so a fake RPC can be injected directly and every orchestration
decision asserted:

* move_home delegates to joint-space PTP with HOME_JOINTS (no IK involved).
* move_ptp_joints validates input and surfaces MoveJ errors as RobotError.
* move_ptp_pose defaults orientation to the current pose, seeds IK with the
  current joints (type=0 absolute), and refuses to move when IK fails.

RobotService is NOT unit-tested here: its behaviour is cross-thread queued Qt
signals, which are unreliable under a synthetic processEvents loop — it gets
verified through the running app instead.
"""

from __future__ import annotations

import pytest

from omnicron_ui.robot import controller as controller_mod
from omnicron_ui.robot import geometry
from omnicron_ui.robot.controller import (
    ARC_TIMEOUT_MS,
    DEFAULT_VEL,
    HOME_JOINTS,
    RobotController,
    RobotError,
    WeldParams,
)

CURRENT_TCP = [400.0, -100.0, 300.0, 180.0, 0.0, 90.0]
CURRENT_JOINTS = [-90.0, -80.0, 90.0, -100.0, -90.0, 10.0]
IK_JOINTS = [-88.0, -79.0, 91.0, -101.0, -90.0, 11.0]


class FakeRPC:
    """Stands in for the SDK's Robot.RPC: records calls, returns canned values."""

    def __init__(self) -> None:
        self.movej_calls: list[dict] = []
        self.movel_calls: list[dict] = []
        self.ik_calls: list[dict] = []
        self.movej_ret = 0
        self.movel_ret: int | list[int] = 0
        self.ik_ret: tuple = (0, list(IK_JOINTS))
        self.ik_solver = None  # optional fn(pose) -> (err, joints); wins over ik_ret
        self.arc_start_ret = 0
        self.events: list[tuple] = []  # ordered motion + welding I/O calls

    def GetActualTCPPose(self):  # noqa: N802 — SDK naming
        return 0, list(CURRENT_TCP)

    def GetActualJointPosDegree(self):  # noqa: N802
        return 0, list(CURRENT_JOINTS)

    def GetRobotErrorCode(self):  # noqa: N802
        return 0, [0, 0]

    def GetInverseKinRef(self, type_, pose, ref_joints):  # noqa: N802
        self.ik_calls.append({"type": type_, "pose": pose, "ref": ref_joints})
        if self.ik_solver is not None:
            return self.ik_solver(pose)
        return self.ik_ret

    def MoveJ(self, joints, tool, user, desc_pos=None, vel=None):  # noqa: N802
        self.movej_calls.append(
            {"joints": joints, "tool": tool, "user": user, "desc_pos": desc_pos, "vel": vel}
        )
        return self.movej_ret

    def MoveL(self, desc_pos, tool, user, vel=20.0, acc=0.0, ovl=100.0,  # noqa: N802
              oacc=100.0, velAccParamMode=0):  # noqa: N803 — SDK naming
        self.movel_calls.append(
            {"desc_pos": list(desc_pos), "tool": tool, "user": user, "vel": vel,
             "acc": acc, "ovl": ovl, "oacc": oacc, "mode": velAccParamMode}
        )
        self.events.append(("MoveL", list(desc_pos)))
        if isinstance(self.movel_ret, list):  # one return code per successive call
            return self.movel_ret[len(self.movel_calls) - 1]
        return self.movel_ret

    # Welding I/O — CO bank is driven only through these dedicated commands.

    def WeldingSetCurrent(self, io, amps, ao, wait):  # noqa: N802
        self.events.append(("WeldingSetCurrent", io, amps, ao, wait))
        return 0

    def WeldingSetVoltage(self, io, volts, ao, wait):  # noqa: N802
        self.events.append(("WeldingSetVoltage", io, volts, ao, wait))
        return 0

    def SetAspirated(self, io, on):  # noqa: N802
        self.events.append(("SetAspirated", io, on))
        return 0

    def ARCStart(self, io, arc_num, timeout):  # noqa: N802
        self.events.append(("ARCStart", io, arc_num, timeout))
        return self.arc_start_ret

    def ARCEnd(self, io, arc_num, timeout):  # noqa: N802
        self.events.append(("ARCEnd", io, arc_num, timeout))
        return 0

    def SetForwardWireFeed(self, io, on):  # noqa: N802
        self.events.append(("SetForwardWireFeed", io, on))
        return 0

    def SetReverseWireFeed(self, io, on):  # noqa: N802
        self.events.append(("SetReverseWireFeed", io, on))
        return 0

    def ResetAllError(self):  # noqa: N802
        return 0

    def CloseRPC(self):  # noqa: N802
        pass


@pytest.fixture
def rpc() -> FakeRPC:
    return FakeRPC()


@pytest.fixture
def ctrl(rpc: FakeRPC) -> RobotController:
    """A connected controller with the fake RPC injected (tool/user as read)."""
    c = RobotController(log=lambda msg, level="info": None)
    c.robot = rpc
    c.tool = 1
    c.user = 0
    return c


# --- disconnected guard ------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.move_home(),
        lambda c: c.move_ptp_joints(HOME_JOINTS),
        lambda c: c.move_ptp_pose(400.0, 0.0, 300.0),
        lambda c: c.move_linear(400.0, 0.0, 300.0),
        lambda c: c.move_linear_torch_down(400.0, 0.0, 300.0),
        lambda c: c.move_along_line([0.0, 600.0, 80.0], [100.0, 600.0, 80.0]),
        lambda c: c.set_gas(True),
        lambda c: c.start_wire_feed(),
        lambda c: c.stop_wire_feed(),
        lambda c: c.weld_start(WeldParams(live=True)),
        lambda c: c.reset_error(),
    ],
)
def test_motion_requires_connection(call):
    c = RobotController(log=lambda msg, level="info": None)
    with pytest.raises(RobotError, match="Not connected"):
        call(c)


# --- move_ptp_joints ----------------------------------------------------------


def test_ptp_joints_movej_with_active_frames(ctrl, rpc):
    target = [10, -20, 30, -40, 50, -60]  # ints on purpose: must be coerced
    assert ctrl.move_ptp_joints(target, vel=25.0) == 0

    assert len(rpc.movej_calls) == 1
    call = rpc.movej_calls[0]
    assert call["joints"] == [10.0, -20.0, 30.0, -40.0, 50.0, -60.0]
    assert all(isinstance(j, float) for j in call["joints"])
    assert call["tool"] == ctrl.tool
    assert call["user"] == ctrl.user
    assert call["vel"] == 25.0
    assert call["desc_pos"] is None  # joint-space PTP: no descartes target


def test_ptp_joints_rejects_wrong_length(ctrl, rpc):
    with pytest.raises(RobotError, match="Expected 6 joint values, got 5"):
        ctrl.move_ptp_joints([0.0] * 5)
    assert rpc.movej_calls == []


def test_ptp_joints_raises_on_movej_error(ctrl, rpc):
    rpc.movej_ret = 112
    with pytest.raises(RobotError, match="error code 112"):
        ctrl.move_ptp_joints(HOME_JOINTS)


# --- move_home ----------------------------------------------------------------


def test_move_home_is_joint_space_ptp_to_home(ctrl, rpc):
    assert ctrl.move_home() == 0

    assert len(rpc.movej_calls) == 1
    call = rpc.movej_calls[0]
    assert call["joints"] == HOME_JOINTS
    assert call["vel"] == DEFAULT_VEL
    assert rpc.ik_calls == []  # home is a known configuration: never IK


def test_move_home_passes_velocity(ctrl, rpc):
    ctrl.move_home(vel=10.0)
    assert rpc.movej_calls[0]["vel"] == 10.0


# --- move_ptp_pose --------------------------------------------------------------


def test_ptp_pose_defaults_orientation_to_current(ctrl, rpc):
    """A camera target is a position; orientation must come from the robot."""
    assert ctrl.move_ptp_pose(450.0, -80.0, 250.0) == 0

    target = rpc.ik_calls[0]["pose"]
    assert target[:3] == [450.0, -80.0, 250.0]
    assert target[3:] == CURRENT_TCP[3:]


def test_ptp_pose_uses_explicit_orientation_when_given(ctrl, rpc):
    ctrl.move_ptp_pose(450.0, -80.0, 250.0, rx=170.0, ry=5.0, rz=45.0)
    assert rpc.ik_calls[0]["pose"][3:] == [170.0, 5.0, 45.0]


def test_ptp_pose_ik_absolute_and_seeded_with_current_joints(ctrl, rpc):
    ctrl.move_ptp_pose(450.0, -80.0, 250.0)

    ik = rpc.ik_calls[0]
    assert ik["type"] == 0  # absolute pose in the BASE frame
    assert ik["ref"] == CURRENT_JOINTS  # nearest-solution seed


def test_ptp_pose_movej_to_ik_joints_with_desc_pos(ctrl, rpc):
    ctrl.move_ptp_pose(450.0, -80.0, 250.0, vel=30.0)

    call = rpc.movej_calls[0]
    assert call["joints"] == IK_JOINTS
    assert call["desc_pos"] == [450.0, -80.0, 250.0, *CURRENT_TCP[3:]]
    assert call["vel"] == 30.0


def test_ptp_pose_no_motion_when_ik_fails(ctrl, rpc):
    rpc.ik_ret = (112, None)
    with pytest.raises(RobotError, match="No IK solution"):
        ctrl.move_ptp_pose(2000.0, 2000.0, 2000.0)
    assert rpc.movej_calls == []  # must not move on a failed solve


def test_ptp_pose_raises_on_movej_error(ctrl, rpc):
    rpc.movej_ret = 14
    with pytest.raises(RobotError, match="error code 14"):
        ctrl.move_ptp_pose(450.0, -80.0, 250.0)


# --- move_linear ----------------------------------------------------------------


def test_linear_defaults_orientation_to_current(ctrl, rpc):
    assert ctrl.move_linear(500.0, -50.0, 200.0) == 0

    call = rpc.movel_calls[0]
    assert call["desc_pos"] == [500.0, -50.0, 200.0, *CURRENT_TCP[3:]]
    assert call["mode"] == 0  # percentage mode by default
    assert call["vel"] == DEFAULT_VEL
    assert call["tool"] == ctrl.tool
    assert call["user"] == ctrl.user


def test_linear_uses_explicit_orientation_when_given(ctrl, rpc):
    ctrl.move_linear(500.0, -50.0, 200.0, rx=170.0, ry=5.0, rz=45.0)
    assert rpc.movel_calls[0]["desc_pos"][3:] == [170.0, 5.0, 45.0]


def test_linear_physical_mode_follows_error_183_rule(ctrl, rpc):
    """speed_mms → velAccParamMode=1, ovl/oacc physical, vel/acc pinned to 100."""
    ctrl.move_linear(500.0, -50.0, 200.0, speed_mms=8.0, accel=150.0)

    call = rpc.movel_calls[0]
    assert call["mode"] == 1
    assert call["ovl"] == 8.0     # real travel speed, mm/s
    assert call["oacc"] == 150.0  # real acceleration, mm/s^2
    assert call["vel"] == 100.0   # full scale — anything else is rejected (183)
    assert call["acc"] == 100.0


def test_linear_raises_on_movel_error(ctrl, rpc):
    rpc.movel_ret = 112
    with pytest.raises(RobotError, match="error code 112"):
        ctrl.move_linear(500.0, -50.0, 200.0)


# --- torch-down solve / move ------------------------------------------------------


def test_torch_down_only_offers_down_orientations_to_ik(ctrl, rpc):
    accepted = [0.0, 0.0, 90.0]  # a down-pointing candidate from the sweep
    rpc.ik_solver = (
        lambda pose: (0, list(IK_JOINTS)) if pose[3:] == accepted else (112, None)
    )

    ctrl.move_linear_torch_down(450.0, -80.0, 250.0)

    for call in rpc.ik_calls:
        rx, ry, rz = call["pose"][3:]
        assert geometry.torch_dir_in_base(rx, ry, rz)[2] <= -0.7
    assert rpc.movel_calls[0]["desc_pos"] == [450.0, -80.0, 250.0, *accepted]


def test_torch_down_prefers_least_joint_travel(ctrl, rpc):
    near, far = list(CURRENT_JOINTS), [j + 40.0 for j in CURRENT_JOINTS]

    def solver(pose):
        if pose[3:] == [0.0, 0.0, 0.0]:   # earlier candidate, far solution
            return (0, far)
        if pose[3:] == [0.0, 0.0, 90.0]:  # later candidate, nearest solution
            return (0, near)
        return (112, None)

    rpc.ik_solver = solver
    assert ctrl.solve_torch_down_rpy(450.0, -80.0, 250.0) == [0.0, 0.0, 90.0]


def test_torch_down_no_motion_when_unreachable(ctrl, rpc):
    rpc.ik_solver = lambda pose: (112, None)
    with pytest.raises(RobotError, match="No torch-DOWN"):
        ctrl.move_linear_torch_down(2000.0, 2000.0, 2000.0)
    assert rpc.movel_calls == []


# --- move_along_line (weld-pass skeleton) -------------------------------------------


LINE_P1 = [0.0, 600.0, 80.0]
LINE_P2 = [100.0, 600.0, 80.0]


def test_line_pass_leg_sequence_and_heights(ctrl, rpc):
    ctrl.move_along_line(LINE_P1, LINE_P2, approach=30.0, standoff=10.0,
                         align_yaw=False)

    assert [c["desc_pos"][:2] for c in rpc.movel_calls] == [
        [0.0, 600.0],    # approach over P1
        [0.0, 600.0],    # descend to P1
        [100.0, 600.0],  # traverse to P2
        [100.0, 600.0],  # retract over P2
    ]
    # standoff keeps the tip off the surface; approach lifts a further 30 mm.
    assert [c["desc_pos"][2] for c in rpc.movel_calls] == [120.0, 90.0, 90.0, 120.0]
    for c in rpc.movel_calls:  # orientation held from the current pose
        assert c["desc_pos"][3:] == CURRENT_TCP[3:]


def test_line_pass_traverse_is_the_weld_speed_leg(ctrl, rpc):
    ctrl.move_along_line(LINE_P1, LINE_P2, speed_mms=8.0, align_yaw=False)

    # Only the traverse runs in physical mode — that MoveL IS the travel speed.
    assert [c["mode"] for c in rpc.movel_calls] == [0, 0, 1, 0]
    traverse = rpc.movel_calls[2]
    assert traverse["ovl"] == 8.0
    assert traverse["vel"] == 100.0 and traverse["acc"] == 100.0


def test_line_pass_pull_drag_yaw_is_the_default(ctrl, rpc):
    """The project welds PULL: default yaw leans the torch against travel."""
    ctrl.move_along_line(LINE_P1, LINE_P2)

    # Current pose heads +Y (rz=90); the line heads +X; pull adds 180 → ±180.
    rz = rpc.movel_calls[0]["desc_pos"][5]
    assert abs(abs(rz) - 180.0) < 1e-6
    assert all(c["desc_pos"][5] == rz for c in rpc.movel_calls)
    # rx/ry (the tool-down tilt) are never touched by yaw alignment.
    assert rpc.movel_calls[0]["desc_pos"][3:5] == CURRENT_TCP[3:5]


def test_line_pass_push_differs_by_180(ctrl, rpc):
    ctrl.move_along_line(LINE_P1, LINE_P2, pull=False)
    assert abs(rpc.movel_calls[0]["desc_pos"][5]) < 1e-6


def test_line_pass_aborts_on_first_failed_leg(ctrl, rpc):
    rpc.movel_ret = [0, 112]  # approach OK, descend fails
    with pytest.raises(RobotError, match="descend to P1"):
        ctrl.move_along_line(LINE_P1, LINE_P2, align_yaw=False)
    assert len(rpc.movel_calls) == 2  # traverse/retract never sent


# --- welding I/O ----------------------------------------------------------------


def _event_names(rpc) -> list[str]:
    return [e[0] for e in rpc.events]


def test_weld_params_default_to_dry_run():
    """Safe by default: a bare WeldParams energizes nothing."""
    w = WeldParams()
    assert w.live is False
    assert w.gas is False
    assert w.current is None and w.voltage is None


def test_dry_weld_start_end_energize_nothing(ctrl, rpc):
    w = WeldParams(current=100.0, voltage=18.0, gas=True)  # still live=False
    ctrl.weld_start(w)
    ctrl.weld_end(w)
    assert rpc.events == []  # not a single welding command sent


def test_live_weld_start_sets_ao_then_gas_then_arc(ctrl, rpc):
    w = WeldParams(live=True, gas=True, current=65.0, voltage=16.5, arc_num=2)
    ctrl.weld_start(w)

    assert rpc.events == [
        ("WeldingSetCurrent", 0, 65.0, 0, 0),  # AO0 = current
        ("WeldingSetVoltage", 0, 16.5, 1, 0),  # AO1 = voltage
        ("SetAspirated", 0, 1),                # gas before the arc
        ("ARCStart", 0, 2, ARC_TIMEOUT_MS),
    ]


def test_live_weld_uses_webapp_process_when_no_overrides(ctrl, rpc):
    """No current/voltage given → they come from the arc_num process, not AO writes."""
    ctrl.weld_start(WeldParams(live=True))
    assert _event_names(rpc) == ["ARCStart"]


def test_weld_start_raises_when_arc_fails(ctrl, rpc):
    rpc.arc_start_ret = 1
    with pytest.raises(RobotError, match="Arc did not establish"):
        ctrl.weld_start(WeldParams(live=True))


def test_weld_end_drops_arc_then_gas(ctrl, rpc):
    ctrl.weld_end(WeldParams(live=True, gas=True))
    assert rpc.events == [
        ("ARCEnd", 0, 1, ARC_TIMEOUT_MS),
        ("SetAspirated", 0, 0),
    ]


def test_weld_end_never_raises_even_disconnected():
    # weld_end must be safe from a finally block on ANY exit path.
    c = RobotController(log=lambda msg, level="info": None)
    c.weld_end(WeldParams(live=True))  # no connection: no-op, no exception


def test_wire_feed_forward_and_stop(ctrl, rpc):
    ctrl.start_wire_feed()
    ctrl.stop_wire_feed()
    assert rpc.events == [
        ("SetForwardWireFeed", 0, 1),
        ("SetForwardWireFeed", 0, 0),  # stop clears BOTH directions
        ("SetReverseWireFeed", 0, 0),
    ]


def test_wire_feed_reverse(ctrl, rpc):
    ctrl.start_wire_feed(reverse=True)
    assert rpc.events == [("SetReverseWireFeed", 0, 1)]


# --- weld pass (move_along_line + weld) ---------------------------------------------


def test_weld_pass_arc_wraps_the_traverse_only(ctrl, rpc):
    """Arc strikes after the descend, ends before the retract."""
    ctrl.move_along_line(LINE_P1, LINE_P2, align_yaw=False,
                         weld=WeldParams(live=True, gas=True))
    assert _event_names(rpc) == [
        "MoveL",         # approach over P1
        "MoveL",         # descend to P1
        "SetAspirated",  # gas on
        "ARCStart",
        "MoveL",         # traverse = weld stroke
        "ARCEnd",
        "SetAspirated",  # gas off
        "MoveL",         # retract over P2
    ]


def test_weld_pass_dry_run_is_pure_motion(ctrl, rpc):
    ctrl.move_along_line(LINE_P1, LINE_P2, align_yaw=False, weld=WeldParams())
    assert _event_names(rpc) == ["MoveL"] * 4  # identical path, nothing energized


def test_weld_pass_no_traverse_when_arc_fails(ctrl, rpc):
    rpc.arc_start_ret = 1
    with pytest.raises(RobotError, match="Arc did not establish"):
        ctrl.move_along_line(LINE_P1, LINE_P2, align_yaw=False,
                             weld=WeldParams(live=True))
    # Approach + descend happened; after the failed strike the finally still
    # sends ARCEnd, and the traverse/retract never run.
    assert _event_names(rpc) == ["MoveL", "MoveL", "ARCStart", "ARCEnd"]


def test_weld_pass_arc_always_ended_when_traverse_fails(ctrl, rpc):
    rpc.movel_ret = [0, 0, 112]  # approach OK, descend OK, traverse fails
    with pytest.raises(RobotError, match="traverse"):
        ctrl.move_along_line(LINE_P1, LINE_P2, align_yaw=False,
                             weld=WeldParams(live=True))
    # The arc is dropped even though the weld stroke errored mid-pass.
    assert _event_names(rpc) == ["MoveL", "MoveL", "ARCStart", "MoveL", "ARCEnd"]


# --- connect ------------------------------------------------------------------


def test_connect_runs_error_154_sequence(monkeypatch):
    """Mode(0) → RobotEnable(1) → ResetAllError must run before any motion."""
    calls: list[tuple] = []

    class FakeConnectRPC(FakeRPC):
        def __init__(self, ip: str) -> None:
            super().__init__()
            calls.append(("RPC", ip))

        def Mode(self, mode):  # noqa: N802
            calls.append(("Mode", mode))
            return 0

        def RobotEnable(self, enable):  # noqa: N802
            calls.append(("RobotEnable", enable))
            return 0

        def ResetAllError(self):  # noqa: N802
            calls.append(("ResetAllError",))
            return 0

        def GetActualTCPNum(self):  # noqa: N802
            return 0, 3

        def GetActualWObjNum(self):  # noqa: N802
            return 0, 2

    monkeypatch.setattr(controller_mod.Robot, "RPC", FakeConnectRPC)
    monkeypatch.setattr(controller_mod.time, "sleep", lambda s: None)

    c = RobotController(log=lambda msg, level="info": None)
    tool, user = c.connect("192.168.58.2")

    assert calls == [
        ("RPC", "192.168.58.2"),
        ("Mode", 0),
        ("RobotEnable", 1),
        ("ResetAllError",),
    ]
    # Active frames read on connect are what later MoveJ calls must use
    # (mismatch causes error 14).
    assert (tool, user) == (3, 2)
    assert (c.tool, c.user) == (3, 2)
    assert c.connected
