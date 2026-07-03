"""PlaybackWorker — chunk-by-chunk TTS playback with hard interrupt guarantee."""
import queue
import socket
import threading
import time
import numpy as np
import sounddevice as sd

# Fan out played TTS samples to localhost UDP so companion visualizers
# (e.g. edge_glow) can derive a sample-accurate envelope without guessing
# through PulseAudio monitor routing. Fire-and-forget; listener optional.
_TAP_HOST = "127.0.0.1"
_TAP_PORT = 47121

from ..enums import EngineState, TransitionReason, FloorOwner
from ..shared_state import SharedState
from ..state_controller import StateController
from ..floor_manager import FloorManager
from ..interrupt_controller import InterruptController
from ..logging_util import LatencyTrace
from .. import logging_util as log


class PlaybackWorker(threading.Thread):
    def __init__(
        self,
        shared: SharedState,
        ctrl: StateController,
        floor: FloorManager,
        interrupt: InterruptController,
        sample_rate: int = 22050,
        chunk_ms: int = 20,
        output_device: int | None = None,
        cues=None,
    ) -> None:
        super().__init__(name="voice-playback", daemon=True)
        self._cues = cues
        self._shared = shared
        self._ctrl = ctrl
        self._floor = floor
        self._interrupt = interrupt
        self._sr = sample_rate
        self._chunk_samples = int(sample_rate * chunk_ms / 1000)
        self._device = output_device
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()
        try:
            self._tap_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._tap_sock.setblocking(False)
        except Exception:
            self._tap_sock = None

    def submit(self, text: str, turn_id: int, tts_backend, trace: LatencyTrace) -> None:
        """Non-blocking. Puts work on internal queue."""
        self._queue.put_nowait((text, turn_id, tts_backend, trace))

    def run(self) -> None:
        """Worker loop. Run in its own daemon thread."""
        log.event("playback", "worker_started")
        try:
            while not self._shared.shutdown.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                text, turn_id, tts_backend, trace = item
                self._speak(text, turn_id, tts_backend, trace)
        except Exception as e:
            log.event("playback", "worker_crashed", error=str(e))
            raise

    def _speak(self, text: str, turn_id: int, tts_backend, trace: LatencyTrace) -> None:
        log.event("playback", "speak_entry", turn_id=turn_id,
                  current_turn=self._ctrl.current_turn(),
                  floor=self._ctrl.get_floor_owner().name,
                  state=self._ctrl.get_state().name)
        # Stale generation guard — if turn moved on, suppress
        if turn_id != self._ctrl.current_turn():
            log.event("playback", "stale_suppressed", turn_id=turn_id,
                      current_turn=self._ctrl.current_turn())
            return

        if not self._floor.request_agent_floor(reason="playback_start", turn_id=turn_id):
            log.event("playback", "floor_denied", turn_id=turn_id)
            return

        ok = self._ctrl.transition(EngineState.SPEAKING, TransitionReason.PLAYBACK_START,
                                   turn_id=turn_id)
        if not ok:
            self._floor.release_floor(reason="transition_failed", turn_id=turn_id)
            return

        self._shared.speaking.set()
        if self._cues:
            self._cues.speaking()
        trace.mark("tts_start")
        trace.mark("playback_start")
        log.event("playback", "start",
                  state=EngineState.SPEAKING.name,
                  floor_owner=FloorOwner.AGENT.name,
                  turn_id=turn_id)

        try:
            stream = sd.OutputStream(
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                device=self._device,
                blocksize=self._chunk_samples,
            )
            stream.start()
            first = True

            for chunk in tts_backend.synthesize(text):
                # Split into target chunk sizes for tight interrupt polling
                for subchunk in self._split(chunk):
                    if self._shared.shutdown.is_set():
                        stream.abort()
                        return
                    if self._interrupt.is_interrupted():
                        if self._cues:
                            self._cues.interrupted()
                        trace.mark("interrupt_executed")
                        log.event("playback", "interrupt_executed",
                                  turn_id=turn_id,
                                  queued_chunks_discarded=self._queue.qsize())
                        stream.abort()
                        # Drain pending TTS for this turn
                        self._drain(turn_id)
                        return
                    if first:
                        trace.mark("tts_first_sample")
                        first = False
                    stream.write(subchunk)
                    if self._tap_sock is not None:
                        try:
                            self._tap_sock.sendto(
                                subchunk.tobytes(), (_TAP_HOST, _TAP_PORT)
                            )
                        except Exception:
                            pass

            stream.stop()
            stream.close()
            trace.mark("playback_end")
            log.event("playback", "complete", turn_id=turn_id,
                      duration_ms=int((trace.duration("playback_start", "playback_end") or 0) * 1000))

        except Exception as e:
            log.event("playback", "error", turn_id=turn_id, error=str(e))
        finally:
            self._shared.speaking.clear()
            self._floor.release_floor(reason="playback_finished", turn_id=turn_id)
            state = self._ctrl.get_state()
            if state == EngineState.SPEAKING:
                self._ctrl.transition(EngineState.LISTENING,
                                      TransitionReason.PLAYBACK_COMPLETE,
                                      turn_id=turn_id)

    def _split(self, chunk: np.ndarray):
        """Yield chunk_samples-sized sub-arrays for tight interrupt polling."""
        for i in range(0, len(chunk), self._chunk_samples):
            yield chunk[i:i + self._chunk_samples].astype(np.float32)

    def _drain(self, turn_id: int) -> None:
        """Discard queued TTS items for this turn only; preserve items for other turns."""
        keep = []
        drained = 0
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if len(item) >= 2 and item[1] == turn_id:
                drained += 1
            else:
                keep.append(item)
        for item in keep:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass  # queue filled while we were draining — item lost
        if drained:
            log.event("playback", "drained", turn_id=turn_id, count=drained)
