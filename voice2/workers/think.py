"""ThinkWorker — consumes ASR transcripts, calls LLM, submits to playback."""
import queue
import threading

from ..enums import EngineState, TransitionReason
from ..shared_state import SharedState
from ..state_controller import StateController
from ..floor_manager import FloorManager
from ..interrupt_controller import InterruptController
from ..logging_util import LatencyTrace
from .. import logging_util as log


class ThinkWorker(threading.Thread):
    def __init__(
        self,
        transcript_queue: queue.Queue,
        shared: SharedState,
        ctrl: StateController,
        floor: FloorManager,
        interrupt: InterruptController,
        llm_backend,
        playback_worker,
        tts_backend,
        cues=None,
    ) -> None:
        super().__init__(name="voice-think", daemon=True)
        self._cues = cues
        self._in_q = transcript_queue
        self._shared = shared
        self._ctrl = ctrl
        self._floor = floor
        self._interrupt = interrupt
        self._llm = llm_backend
        self._playback = playback_worker
        self._tts = tts_backend

    def run(self) -> None:
        while not self._shared.shutdown.is_set():
            try:
                text, turn_id, trace = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Stale generation guard — newer turn already started
            if turn_id != self._ctrl.current_turn():
                log.event("think", "stale_discarded", turn_id=turn_id,
                          current=self._ctrl.current_turn())
                continue

            self._ctrl.transition(EngineState.THINKING, TransitionReason.THINK_START,
                                   turn_id=turn_id)
            self._shared.thinking.set()
            if self._cues:
                self._cues.thinking()
            trace.mark("think_start")
            log.event("think", "start",
                      state=EngineState.THINKING.name,
                      floor_owner=self._ctrl.get_floor_owner().name,
                      turn_id=turn_id, input_preview=text[:80])

            try:
                reply = self._llm.reply(text)
                trace.mark("first_token")
                trace.mark("think_end")
                self._shared.thinking.clear()

                log.event("think", "complete", turn_id=turn_id,
                          think_ms=int((trace.duration("think_start", "think_end") or 0) * 1000),
                          reply_preview=reply[:80])

                # Stale check again — could have been interrupted while thinking
                if turn_id != self._ctrl.current_turn():
                    log.event("think", "post_think_stale_suppressed", turn_id=turn_id)
                    continue

                if not self._floor.can_agent_speak(turn_id):
                    log.event("think", "floor_blocked_post_think", turn_id=turn_id)
                    self._ctrl.transition(EngineState.IDLE, TransitionReason.THINK_COMPLETE,
                                          turn_id=turn_id)
                    continue

                if reply:
                    # Hand off to PlaybackWorker — it owns the SPEAKING transition
                    self._interrupt.set_trace(trace)
                    self._playback.submit(reply, turn_id, self._tts, trace)
                    log.event("think", "submitted_to_playback", turn_id=turn_id)
                else:
                    self._ctrl.transition(EngineState.IDLE, TransitionReason.THINK_COMPLETE,
                                          turn_id=turn_id)

            except Exception as e:
                self._shared.thinking.clear()
                log.event("think", "error", turn_id=turn_id, error=str(e))
                self._ctrl.transition(EngineState.IDLE, TransitionReason.ERROR_RECOVERY,
                                      turn_id=turn_id)
