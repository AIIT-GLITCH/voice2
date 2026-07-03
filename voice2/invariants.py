"""
System invariants — called periodically or on every transition.
Violations are logged critical and trigger forced repair.

INVARIANT 1: SPEAKING => FloorOwner == AGENT
INVARIANT 2: interrupted.is_set() => no new TTS begins
INVARIANT 3: Only ListenWorker advances committed turn_id
INVARIANT 4: Response plays only if turn_id matches + floor + no interrupt + not shutdown
INVARIANT 5: No direct state mutation outside StateController
INVARIANT 6: No subscriber failure may stall mic capture
"""

import threading
from .enums import EngineState, FloorOwner
from .shared_state import SharedState
from .state_controller import StateController
from . import logging_util as log


class InvariantChecker:
    def __init__(self, shared: SharedState, ctrl: StateController) -> None:
        self._shared = shared
        self._ctrl = ctrl
        self._lock = threading.Lock()

    def check_all(self) -> list[str]:
        """Run all invariant checks. Returns list of violation descriptions."""
        violations = []
        violations += self._check_speaking_floor()
        violations += self._check_interrupt_blocks_playback()
        for v in violations:
            log.event("invariant", "VIOLATION",
                      state=self._ctrl.get_state().name,
                      floor_owner=self._ctrl.get_floor_owner().name,
                      turn_id=self._shared.current_turn_id,
                      violation=v)
        return violations

    def _check_speaking_floor(self) -> list[str]:
        state = self._ctrl.get_state()
        floor = self._ctrl.get_floor_owner()
        if state == EngineState.SPEAKING and floor != FloorOwner.AGENT:
            msg = f"INV1: SPEAKING but floor={floor.name}, expected AGENT"
            # Force repair
            self._ctrl.set_floor(FloorOwner.AGENT, reason="invariant_repair")
            return [msg]
        return []

    def _check_interrupt_blocks_playback(self) -> list[str]:
        # Invariant 2 is enforced structurally in PlaybackWorker._speak()
        # Here we just audit and log if something slipped through
        if (self._shared.interrupted.is_set()
                and self._ctrl.get_state() == EngineState.SPEAKING):
            return ["INV2: speaking while interrupted flag is set"]
        return []

    def playback_is_allowed(self, turn_id: int) -> tuple[bool, str]:
        """
        Invariant 4: A response may only play if:
          - turn_id == current_turn_id
          - floor manager grants agent floor
          - interrupt flag is clear
          - not shutting down
        Returns (allowed, reason).
        """
        if self._shared.shutdown.is_set():
            return False, "shutdown"
        if self._shared.interrupted.is_set():
            return False, "interrupted"
        if turn_id != self._shared.current_turn_id:
            return False, f"stale_turn:{turn_id}!={self._shared.current_turn_id}"
        if self._ctrl.get_floor_owner() == FloorOwner.USER:
            return False, "user_owns_floor"
        return True, "ok"

    def run_loop(self, interval_sec: float = 2.0) -> None:
        """Background invariant monitor. Run in daemon thread."""
        import time
        while not self._shared.shutdown.is_set():
            time.sleep(interval_sec)
            self.check_all()
