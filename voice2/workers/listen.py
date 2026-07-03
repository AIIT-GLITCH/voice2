"""ListenWorker — patient VAD utterance capture with pre-roll."""
import queue
import re
import threading
import time
import numpy as np

from ..enums import EngineState, FloorOwner, TransitionReason
from ..shared_state import SharedState
from ..state_controller import StateController
from ..floor_manager import FloorManager
from ..interrupt_controller import InterruptController
from ..ring_buffer import RingAudioBuffer
from ..logging_util import LatencyTrace
from ..config import VADConfig
from .. import logging_util as log


class ListenWorker(threading.Thread):
    def __init__(
        self,
        audio_queue: queue.Queue,
        shared: SharedState,
        ctrl: StateController,
        floor: FloorManager,
        interrupt: InterruptController,
        ring: RingAudioBuffer,
        vad_cfg: VADConfig,
        sample_rate: int,
        transcript_queue: queue.Queue,
        asr_backend,
        cues=None,
    ) -> None:
        super().__init__(name="voice-listen", daemon=True)
        self._cues = cues
        self._q = audio_queue
        self._shared = shared
        self._ctrl = ctrl
        self._floor = floor
        self._interrupt = interrupt
        self._ring = ring
        self._cfg = vad_cfg
        self._sr = sample_rate
        self._out_q = transcript_queue
        self._asr = asr_backend
        self._silence_threshold = 0.02   # onset: need intentional voice, not ambient noise
        self._collect_silence_threshold = 0.015  # during collection: slightly more lenient
        self._vad_model = None
        self._torch = None

    def run(self) -> None:
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            (get_speech_timestamps, _, read_audio, *_) = utils
            self._vad_model = model
            self._torch = torch
        except Exception as e:
            log.event("listen", "vad_load_error", error=str(e))
            self._vad_model = None
            self._torch = None

        while not self._shared.shutdown.is_set():
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Always feed ring buffer
            self._ring.append(frame)

            # Only capture if engine is in a state that allows it
            state = self._ctrl.get_state()
            if state not in (EngineState.IDLE, EngineState.LISTENING,
                             EngineState.INTERRUPTING):
                continue

            # Simple energy gate to detect speech onset
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            if rms < self._silence_threshold:
                continue

            # Speech detected — start an utterance
            turn_id = self._ctrl.start_new_turn()
            trace = LatencyTrace(turn_id)
            trace.mark("turn_start")
            trace.mark("speech_detect_start")

            self._floor.request_user_floor(reason="speech_onset", turn_id=turn_id)
            # Only transition if not already LISTENING
            if self._ctrl.get_state() != EngineState.LISTENING:
                self._ctrl.transition(EngineState.LISTENING, TransitionReason.SPEECH_DETECTED,
                                      turn_id=turn_id)
            self._interrupt.clear()

            log.event("listen", "utterance_start",
                      state=EngineState.LISTENING.name,
                      floor_owner=FloorOwner.USER.name,
                      turn_id=turn_id)
            if self._cues:
                self._cues.listening()

            # Collect utterance until silence
            utterance = self._collect_utterance(turn_id, trace)

            if utterance is None or len(utterance) == 0:
                self._ctrl.transition(EngineState.IDLE, TransitionReason.SPEECH_ENDED,
                                      turn_id=turn_id)
                continue

            quality = self._speech_quality(utterance)
            if not quality["ok"]:
                log.event("listen", "utterance_discarded",
                          turn_id=turn_id,
                          reason=quality["reason"],
                          duration_ms=quality["duration_ms"],
                          voiced_ms=quality["voiced_ms"],
                          max_prob=round(quality["max_prob"], 3),
                          voiced_ratio=round(quality["voiced_ratio"], 3))
                self._floor.release_floor(reason="audio_quality_gate", turn_id=turn_id)
                self._ctrl.transition(EngineState.IDLE, TransitionReason.SPEECH_ENDED,
                                      turn_id=turn_id)
                continue

            trace.mark("speech_detect_end")
            trace.mark("asr_start")
            log.event("listen", "asr_start", turn_id=turn_id,
                      duration_sec=round(len(utterance) / self._sr, 2))

            try:
                text = self._clean_transcript(self._asr.transcribe(utterance))
                trace.mark("asr_end")
                log.event("listen", "asr_complete", turn_id=turn_id,
                          text_preview=text[:80],
                          asr_ms=int((trace.duration("asr_start", "asr_end") or 0) * 1000))

                if self._is_transcript_noise(text, quality):
                    log.event("listen", "asr_discarded", turn_id=turn_id,
                              reason="noise_transcript",
                              text_preview=text[:80])
                    self._floor.release_floor(reason="noise_transcript", turn_id=turn_id)
                    self._ctrl.transition(EngineState.IDLE, TransitionReason.SPEECH_ENDED)
                elif text and self._floor.should_commit_user_audio():
                    # Release floor so agent can respond
                    self._floor.release_floor(reason="asr_committed", turn_id=turn_id)
                    self._out_q.put_nowait((text, turn_id, trace))
                else:
                    log.event("listen", "asr_discarded", turn_id=turn_id,
                              reason="empty_or_floor_denied")
                    self._floor.release_floor(reason="asr_discarded", turn_id=turn_id)
                    self._ctrl.transition(EngineState.IDLE, TransitionReason.SPEECH_ENDED)

            except Exception as e:
                log.event("listen", "asr_error", turn_id=turn_id, error=str(e))
                self._floor.release_floor(reason="asr_error", turn_id=turn_id)
                self._ctrl.transition(EngineState.IDLE, TransitionReason.SPEECH_ENDED)

    def _collect_utterance(self, turn_id: int, trace: LatencyTrace) -> np.ndarray | None:
        """Collect frames until end_silence_ms of quiet. Returns full audio."""
        frames = []
        # Pre-roll from ring buffer
        pre_roll_sec = self._cfg.pre_roll_ms / 1000.0
        pre = self._ring.get_last(pre_roll_sec)
        if len(pre) > 0:
            frames.append(pre)

        silence_samples = int(self._sr * self._cfg.end_silence_ms / 1000.0)
        max_samples = int(self._sr * self._cfg.max_utterance_sec)
        silent_count = 0
        total_samples = 0

        deadline = time.monotonic() + self._cfg.max_utterance_sec + 2

        while not self._shared.shutdown.is_set():
            if time.monotonic() > deadline:
                break
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            self._ring.append(frame)
            frames.append(frame)
            total_samples += len(frame)

            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            if rms < self._collect_silence_threshold:
                silent_count += len(frame)
                if silent_count >= silence_samples:
                    break
            else:
                silent_count = 0

            if total_samples >= max_samples:
                break

        if not frames:
            return None

        audio = np.concatenate(frames)
        min_samples = int(self._sr * self._cfg.min_utterance_ms / 1000.0)
        if len(audio) < min_samples:
            return None
        return audio

    def _speech_quality(self, audio: np.ndarray) -> dict:
        duration_ms = int(len(audio) / self._sr * 1000)
        result = {
            "ok": True,
            "reason": "ok",
            "duration_ms": duration_ms,
            "voiced_ms": 0,
            "max_prob": 0.0,
            "voiced_ratio": 0.0,
        }
        if duration_ms < self._cfg.min_utterance_ms:
            result.update(ok=False, reason="too_short")
            return result

        if self._vad_model is None or self._torch is None:
            # Better to allow text through the post-ASR noise gate than dead-stop
            # voice if Silero failed to load.
            return result

        frame_len = 512
        probs = []
        audio_f32 = audio.astype(np.float32)
        for start in range(0, max(len(audio_f32) - frame_len + 1, 0), frame_len):
            frame = audio_f32[start:start + frame_len]
            if frame.shape[0] != frame_len:
                continue
            if float(np.sqrt(np.mean(frame ** 2))) < 0.003:
                probs.append(0.0)
                continue
            try:
                with self._torch.no_grad():
                    prob = float(self._vad_model(self._torch.from_numpy(frame), self._sr).item())
            except Exception as e:
                log.event("listen", "quality_vad_error", error=str(e))
                return result
            probs.append(prob)

        if not probs:
            result.update(ok=False, reason="no_frames")
            return result

        voiced = [p for p in probs if p >= self._cfg.min_voice_prob]
        voiced_ms = int(len(voiced) * frame_len / self._sr * 1000)
        voiced_ratio = len(voiced) / max(len(probs), 1)
        max_prob = max(probs)
        result.update(
            voiced_ms=voiced_ms,
            max_prob=max_prob,
            voiced_ratio=voiced_ratio,
        )
        if max_prob < self._cfg.min_voice_prob:
            result.update(ok=False, reason="no_speech_probability")
        elif voiced_ms < self._cfg.min_voiced_ms:
            result.update(ok=False, reason="too_little_voiced_audio")
        elif voiced_ratio < self._cfg.min_voiced_ratio:
            result.update(ok=False, reason="voiced_ratio_low")
        return result

    @staticmethod
    def _clean_transcript(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    @staticmethod
    def _is_transcript_noise(text: str, quality: dict) -> bool:
        norm = re.sub(r"[^a-z0-9' ]+", "", (text or "").lower()).strip()
        noise = {
            "",
            ".",
            "...",
            "uh",
            "um",
            "umm",
            "hmm",
            "hm",
            "mm",
            "ah",
            "oh",
            "you",
            "thank you",
            "thanks",
            "thanks for watching",
            "bye",
            "bye bye",
            "subtitles by the amaraorg community",
        }
        if norm in noise:
            return True
        if quality.get("duration_ms", 0) < 1100:
            words = [w for w in norm.split() if w]
            if len(words) <= 1 and len(norm) <= 5:
                return True
        return len(norm) <= 1
