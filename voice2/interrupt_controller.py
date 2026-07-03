"""InterruptController — central interrupt trigger with debounce."""
import threading
import time
from .enums import InterruptSource, EngineState, TransitionReason
from .shared_state import SharedState
from .state_controller import StateController
from .floor_manager import FloorManager
from .logging_util import LatencyTrace
from . import logging_util as log


class InterruptController:
    def __init__(
        self,
        shared: SharedState,
        ctrl: StateController,
        floor: FloorManager,
        debounce_ms: int = 200,
    ) -> None:
        self._shared = shared
        self._ctrl = ctrl
        self._floor = floor
        self._debounce_sec = debounce_ms / 1000.0
        self._lock = threading.Lock()
        self._last_trigger_ts: float = 0.0
        self._last_source: InterruptSource | None = None
        self._trace: LatencyTrace | None = None

    def set_trace(self, trace: LatencyTrace) -> None:
        self._trace = trace

    def trigger(self, source: InterruptSource, reason: str = "", **meta) -> bool:
        with self._lock:
            now = time.monotonic()
            # Debounce — first trigger wins within the window
            since = now - self._last_trigger_ts
            if since < self._debounce_sec:
                log.event("interrupt", "debounced",
                          state=self._ctrl.get_state().name,
                          floor_owner=self._ctrl.get_floor_owner().name,
                          turn_id=self._shared.current_turn_id,
                          source=source.name, since_ms=int(since * 1000))
                return False

            self._last_trigger_ts = now
            self._last_source = source

        # Mark timing
        if self._trace:
            self._trace.mark("interrupt_detected")

        log.event("interrupt", "triggered",
                  state=self._ctrl.get_state().name,
                  floor_owner=self._ctrl.get_floor_owner().name,
                  turn_id=self._shared.current_turn_id,
                  source=source.name, reason=reason, **meta)

        # 1. Set interrupted event — playback loop polls this
        self._shared.interrupted.set()

        # 2. Floor immediately back to user
        self._floor.handle_interrupt(source=source, reason=reason)

        # 3. State → INTERRUPTING
        self._ctrl.transition(
            EngineState.INTERRUPTING,
            TransitionReason.INTERRUPT,
            source=source.name,
            reason_text=reason,
        )

        return True

    def clear(self) -> None:
        """Called after interrupt is fully resolved and we're back to listening."""
        self._shared.interrupted.clear()
        log.event("interrupt", "cleared",
                  state=self._ctrl.get_state().name,
                  floor_owner=self._ctrl.get_floor_owner().name,
                  turn_id=self._shared.current_turn_id)

    def is_interrupted(self) -> bool:
        return self._shared.interrupted.is_set()
