"""InterruptDetectorWorker — fast VAD/energy barge-in during SPEAKING only."""
import queue
import time
import threading
import numpy as np

from ..enums import EngineState, InterruptSource
from ..shared_state import SharedState
from ..state_controller import StateController
from ..interrupt_controller import InterruptController
from ..config import InterruptVADConfig
from .. import logging_util as log


class InterruptDetectorWorker(threading.Thread):
    def __init__(
        self,
        audio_queue: queue.Queue,
        shared: SharedState,
        ctrl: StateController,
        interrupt: InterruptController,
        cfg: InterruptVADConfig,
    ) -> None:
        super().__init__(name="voice-interrupt_det", daemon=True)
        self._q = audio_queue
        self._shared = shared
        self._ctrl = ctrl
        self._interrupt = interrupt
        self._cfg = cfg

    def run(self) -> None:
        """Runs in daemon thread. Only hot during SPEAKING."""
        baseline_rms = 0.001
        consecutive = 0
        last_refractory = 0.0

        while not self._shared.shutdown.is_set():
            # Only active during SPEAKING — idle otherwise
            if self._ctrl.get_state() != EngineState.SPEAKING:
                # Drain queue so it doesn't fill up while inactive
                try:
                    self._q.get(timeout=0.05)
                except queue.Empty:
                    pass
                consecutive = 0
                baseline_rms = 0.001
                continue

            try:
                frame = self._q.get(timeout=0.05)
            except queue.Empty:
                continue

            # Compute RMS energy
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

            # Slowly update baseline from speaker bleed (slower = more stable floor)
            baseline_rms = 0.995 * baseline_rms + 0.005 * rms

            # Threshold: multiplier × baseline, but NEVER below 0.06 absolute floor
            # 0.06 blocks ambient noise and most speaker bleed; a normal speaking voice clears it
            threshold = max(baseline_rms * self._cfg.energy_multiplier, 0.06)

            if rms > threshold:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= self._cfg.consecutive_frames_required:
                now = time.monotonic()
                refractory_sec = self._cfg.refractory_ms / 1000.0
                if (now - last_refractory) > refractory_sec:
                    last_refractory = now
                    consecutive = 0
                    log.event("interrupt_detector", "vad_triggered",
                              state=self._ctrl.get_state().name,
                              floor_owner=self._ctrl.get_floor_owner().name,
                              turn_id=self._shared.current_turn_id,
                              rms=round(rms, 4),
                              baseline=round(baseline_rms, 4))
                    self._interrupt.trigger(
                        InterruptSource.VAD,
                        reason="energy_threshold_exceeded",
                        rms=round(rms, 4),
                    )
