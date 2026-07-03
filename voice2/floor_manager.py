"""FloorManager — policy object for turn-taking arbitration."""
import threading
from .enums import FloorOwner, EngineState, TransitionReason, InterruptSource
from .shared_state import SharedState
from .state_controller import StateController
from . import logging_util as log


class FloorManager:
    def __init__(self, shared: SharedState, ctrl: StateController) -> None:
        self._shared = shared
        self._ctrl = ctrl
        self._lock = threading.RLock()  # reentrant — request_agent_floor calls can_agent_speak

    def can_agent_speak(self, turn_id: int = 0) -> bool:
        """Agent may speak only if floor is NONE or AGENT and state allows.
        If turn_id provided, also verifies the turn is still current."""
        with self._lock:
            state = self._ctrl.get_state()
            floor = self._ctrl.get_floor_owner()
            if state in (EngineState.STOPPED, EngineState.ERROR):
                return False
            if floor == FloorOwner.USER:
                return False
            if turn_id and turn_id != self._shared.current_turn_id:
                return False
            return True

    def request_user_floor(self, reason: str = "", turn_id: int = 0) -> bool:
        with self._lock:
            self._ctrl.set_floor(FloorOwner.USER, reason=reason, turn_id=turn_id)
            return True

    def request_agent_floor(self, reason: str = "", turn_id: int = 0) -> bool:
        with self._lock:
            if not self.can_agent_speak():
                log.event("floor", "agent_floor_denied",
                          state=self._ctrl.get_state().name,
                          floor_owner=self._ctrl.get_floor_owner().name,
                          turn_id=turn_id, reason=reason)
                return False
            self._ctrl.set_floor(FloorOwner.AGENT, reason=reason, turn_id=turn_id)
            return True

    def release_floor(self, reason: str = "", turn_id: int = 0) -> None:
        with self._lock:
            self._ctrl.set_floor(FloorOwner.NONE, reason=reason, turn_id=turn_id)

    def should_commit_user_audio(self) -> bool:
        """Is ASR result from user currently valid to act on?"""
        floor = self._ctrl.get_floor_owner()
        state = self._ctrl.get_state()
        if state in (EngineState.STOPPED, EngineState.ERROR):
            return False
        # User speech is always commitable if user has or can take floor
        return floor in (FloorOwner.USER, FloorOwner.NONE)

    def handle_interrupt(self, source: InterruptSource, reason: str = "") -> None:
        """User is interrupting. Floor goes back to USER immediately."""
        with self._lock:
            self._shared.speaking.clear()
            self._ctrl.set_floor(FloorOwner.USER,
                                  reason=f"interrupt:{source.name}:{reason}")
            log.event("floor", "interrupt_floor_taken",
                      state=self._ctrl.get_state().name,
                      floor_owner=FloorOwner.USER.name,
                      turn_id=self._shared.current_turn_id,
                      source=source.name)
