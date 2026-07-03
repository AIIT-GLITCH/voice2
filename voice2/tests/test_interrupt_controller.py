"""Unit tests — InterruptController debounce and trigger."""
import time
from ..shared_state import SharedState
from ..state_controller import StateController
from ..floor_manager import FloorManager
from ..interrupt_controller import InterruptController
from ..enums import InterruptSource, EngineState, TransitionReason


def make_stack():
    shared = SharedState()
    ctrl = StateController(shared)
    floor = FloorManager(shared, ctrl)
    interrupt = InterruptController(shared, ctrl, floor, debounce_ms=200)
    return shared, ctrl, floor, interrupt


def test_trigger_sets_event():
    shared, ctrl, floor, interrupt = make_stack()
    ctrl.transition(EngineState.LISTENING, TransitionReason.SPEECH_DETECTED)
    ctrl.transition(EngineState.THINKING, TransitionReason.ASR_COMPLETE)
    # Need SPEAKING state for transition to INTERRUPTING
    # go through full path
    result = interrupt.trigger(InterruptSource.KEYBOARD, "test")
    assert shared.interrupted.is_set()


def test_debounce_blocks_second():
    shared, ctrl, floor, interrupt = make_stack()
    ctrl.transition(EngineState.LISTENING, TransitionReason.SPEECH_DETECTED)
    interrupt.trigger(InterruptSource.KEYBOARD, "first")
    result = interrupt.trigger(InterruptSource.KEYBOARD, "second")
    assert not result  # debounced


def test_debounce_allows_after_window():
    shared, ctrl, floor, interrupt = make_stack()
    interrupt._debounce_sec = 0.05
    ctrl.transition(EngineState.LISTENING, TransitionReason.SPEECH_DETECTED)
    interrupt.trigger(InterruptSource.KEYBOARD, "first")
    time.sleep(0.1)
    shared.interrupted.clear()  # simulate resolution
    # Now trigger again — should work
    interrupt._last_trigger_ts = 0.0
    result = interrupt.trigger(InterruptSource.KEYBOARD, "second")
    assert result


def test_clear():
    shared, ctrl, floor, interrupt = make_stack()
    shared.interrupted.set()
    interrupt.clear()
    assert not shared.interrupted.is_set()
