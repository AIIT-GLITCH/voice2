"""Unit tests — StateController."""
import pytest
from ..shared_state import SharedState
from ..state_controller import StateController
from ..enums import EngineState, FloorOwner, TransitionReason


def make_ctrl():
    return StateController(SharedState())


def test_valid_transition():
    ctrl = make_ctrl()
    assert ctrl.transition(EngineState.LISTENING, TransitionReason.SPEECH_DETECTED)
    assert ctrl.get_state() == EngineState.LISTENING


def test_invalid_transition_rejected():
    ctrl = make_ctrl()
    # Can't go IDLE -> SPEAKING directly
    result = ctrl.transition(EngineState.SPEAKING, TransitionReason.PLAYBACK_START)
    assert not result
    assert ctrl.get_state() == EngineState.IDLE


def test_stopped_blocks_all():
    ctrl = make_ctrl()
    ctrl.transition(EngineState.LISTENING, TransitionReason.SPEECH_DETECTED)
    ctrl.transition(EngineState.STOPPED, TransitionReason.SHUTDOWN)
    result = ctrl.transition(EngineState.IDLE, TransitionReason.STARTUP)
    assert not result


def test_floor_blocks_agent_speaking():
    shared = SharedState()
    ctrl = StateController(shared)
    # Set floor to USER then try to go SPEAKING
    ctrl.set_floor(FloorOwner.USER, "test")
    ctrl.transition(EngineState.LISTENING, TransitionReason.SPEECH_DETECTED)
    result = ctrl.transition(EngineState.THINKING, TransitionReason.ASR_COMPLETE)
    # THINKING is allowed regardless of floor
    assert result
    result = ctrl.transition(EngineState.SPEAKING, TransitionReason.PLAYBACK_START)
    assert not result  # USER has floor — blocked


def test_turn_counter():
    ctrl = make_ctrl()
    t1 = ctrl.start_new_turn()
    t2 = ctrl.start_new_turn()
    assert t2 == t1 + 1


def test_snapshot():
    ctrl = make_ctrl()
    snap = ctrl.snapshot()
    assert snap["state"] == "IDLE"
    assert snap["floor_owner"] == "NONE"
