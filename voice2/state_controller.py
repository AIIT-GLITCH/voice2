"""StateController — owns all state transitions. Single lock. Atomic only."""
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .enums import EngineState, FloorOwner, TransitionReason
from .shared_state import SharedState
from . import logging_util as log


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable atomic snapshot. Callers compare against this, never hold lock."""
    state: EngineState
    floor_owner: FloorOwner
    turn_id: int
    ts: float

# Transition validation table: (from_state, to_state) -> allowed
_VALID = {
    (EngineState.IDLE, EngineState.LISTENING),
    (EngineState.IDLE, EngineState.THINKING),   # submit_text() path: no LISTENING step
    (EngineState.IDLE, EngineState.FALLBACK_TEXT),
    (EngineState.IDLE, EngineState.ERROR),
    (EngineState.IDLE, EngineState.STOPPED),
    (EngineState.LISTENING, EngineState.THINKING),
    (EngineState.LISTENING, EngineState.IDLE),
    (EngineState.LISTENING, EngineState.INTERRUPTING),
    (EngineState.LISTENING, EngineState.ERROR),
    (EngineState.LISTENING, EngineState.STOPPED),
    (EngineState.THINKING, EngineState.SPEAKING),
    (EngineState.THINKING, EngineState.IDLE),
    (EngineState.THINKING, EngineState.INTERRUPTING),
    (EngineState.THINKING, EngineState.ERROR),
    (EngineState.THINKING, EngineState.STOPPED),
    (EngineState.SPEAKING, EngineState.IDLE),
    (EngineState.SPEAKING, EngineState.LISTENING),
    (EngineState.SPEAKING, EngineState.INTERRUPTING),
    (EngineState.SPEAKING, EngineState.ERROR),
    (EngineState.SPEAKING, EngineState.STOPPED),
    (EngineState.INTERRUPTING, EngineState.LISTENING),
    (EngineState.INTERRUPTING, EngineState.IDLE),
    (EngineState.INTERRUPTING, EngineState.ERROR),
    (EngineState.INTERRUPTING, EngineState.STOPPED),
    (EngineState.ERROR, EngineState.IDLE),
    (EngineState.ERROR, EngineState.STOPPED),
    (EngineState.FALLBACK_TEXT, EngineState.THINKING),
    (EngineState.FALLBACK_TEXT, EngineState.STOPPED),
}

# Floor ownership rules: cannot enter SPEAKING if floor is USER
_FLOOR_BLOCKS: dict[EngineState, set[FloorOwner]] = {
    EngineState.SPEAKING: {FloorOwner.USER},
}


@dataclass
class TransitionRecord:
    ts: float
    from_state: EngineState
    to_state: EngineState
    reason: TransitionReason
    floor_owner: FloorOwner
    turn_id: int
    meta: dict = field(default_factory=dict)


class StateController:
    def __init__(self, shared: SharedState) -> None:
        self._shared = shared
        self._lock = threading.RLock()
        self._state = EngineState.IDLE
        self._floor = FloorOwner.NONE
        self._history: list[TransitionRecord] = []

    # ── Reads ──

    def get_state(self) -> EngineState:
        with self._lock:
            return self._state

    def get_floor_owner(self) -> FloorOwner:
        with self._lock:
            return self._floor

    def get_snapshot(self) -> StateSnapshot:
        """Atomic frozen snapshot — callers may read freely without holding lock."""
        with self._lock:
            return StateSnapshot(
                state=self._state,
                floor_owner=self._floor,
                turn_id=self._shared.current_turn_id,
                ts=time.monotonic(),
            )

    def snapshot(self) -> dict[str, Any]:
        s = self.get_snapshot()
        return {
            "state": s.state.name,
            "floor_owner": s.floor_owner.name,
            "turn_id": s.turn_id,
            "history_len": len(self._history),
        }

    # ── Turn counter ──

    def start_new_turn(self) -> int:
        with self._lock:
            self._shared.current_turn_id += 1
            log.event("state", "new_turn",
                      state=self._state.name, floor_owner=self._floor.name,
                      turn_id=self._shared.current_turn_id)
            return self._shared.current_turn_id

    def current_turn(self) -> int:
        return self._shared.current_turn_id

    # ── State transition ──

    def transition(
        self,
        new_state: EngineState,
        reason: TransitionReason,
        **meta: Any,
    ) -> bool:
        with self._lock:
            if self._state == EngineState.STOPPED and new_state != EngineState.STOPPED:
                log.event("state", "transition_rejected",
                          from_state=self._state.name, to_state=new_state.name,
                          floor_owner=self._floor.name,
                          turn_id=self._shared.current_turn_id,
                          reason="STOPPED_no_exit")
                return False

            pair = (self._state, new_state)
            if pair not in _VALID:
                log.event("state", "transition_rejected",
                          from_state=self._state.name, to_state=new_state.name,
                          floor_owner=self._floor.name,
                          turn_id=self._shared.current_turn_id,
                          reason="invalid_transition")
                return False

            blocked = _FLOOR_BLOCKS.get(new_state, set())
            if self._floor in blocked:
                log.event("state", "transition_rejected",
                          from_state=self._state.name, to_state=new_state.name,
                          floor_owner=self._floor.name,
                          turn_id=self._shared.current_turn_id,
                          reason="floor_blocked")
                return False

            rec = TransitionRecord(
                ts=time.monotonic(),
                from_state=self._state,
                to_state=new_state,
                reason=reason,
                floor_owner=self._floor,
                turn_id=self._shared.current_turn_id,
                meta=meta,
            )
            self._history.append(rec)
            self._state = new_state

            _RESERVED = {"turn_id", "state", "floor_owner", "reason",
                         "from_state", "to_state", "old_state", "new_state"}
            safe_meta = {k: v for k, v in meta.items() if k not in _RESERVED}
            log.event("state", "transition",
                      from_state=self._history[-1].from_state.name,
                      state=new_state.name, floor_owner=self._floor.name,
                      turn_id=self._shared.current_turn_id,
                      reason=reason.name, **safe_meta)
            return True

    # ── Floor control ──

    def set_floor(self, owner: FloorOwner, reason: str = "", **meta: Any) -> None:
        with self._lock:
            _RESERVED = {"turn_id", "state", "floor_owner", "reason"}
            safe_meta = {k: v for k, v in meta.items() if k not in _RESERVED}
            self._floor = owner
            log.event("floor", "floor_set",
                      state=self._state.name, floor_owner=owner.name,
                      turn_id=self._shared.current_turn_id,
                      reason=reason, **safe_meta)
