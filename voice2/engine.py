"""VoiceEngine — orchestration only. Wires all workers, manages lifecycle."""
import os
import queue
import subprocess
import sys
import threading
from typing import Callable

import sounddevice as sd
import numpy as np

# Optional companion visualizer: point VOICE2_EDGE_GLOW at a script that tails
# the engine's JSONL log (e.g. a screen-edge glow while speaking). Off by default.
EDGE_GLOW_PATH = os.path.expanduser(os.environ.get('VOICE2_EDGE_GLOW', ''))

from .config import VoiceConfig
from .enums import EngineState, TransitionReason
from .shared_state import SharedState
from .state_controller import StateController
from .floor_manager import FloorManager
from .interrupt_controller import InterruptController
from .audio_broadcaster import AudioBroadcaster
from .ring_buffer import RingAudioBuffer
from .workers.listen import ListenWorker
from .workers.interrupt_detector import InterruptDetectorWorker
from .workers.think import ThinkWorker
from .workers.playback import PlaybackWorker
from .workers.keyboard import KeyboardWorker
from .backends.asr import FasterWhisperASR
from .backends.tts import PiperTTS
from .backends.llm import CallableLLM
from .invariants import InvariantChecker
from .tones import UICues
from . import logging_util as log


class VoiceEngine:
    def __init__(self, config: VoiceConfig, ask_fn: Callable[[str], str]) -> None:
        self.cfg = config
        self._ask_fn = ask_fn
        self._started = False

        # Control plane
        self.shared = SharedState()
        self.ctrl = StateController(self.shared)
        self.floor = FloorManager(self.shared, self.ctrl)
        self.interrupt = InterruptController(
            self.shared, self.ctrl, self.floor,
            debounce_ms=config.interrupt.debounce_ms,
        )

        # Data plane
        self.broadcaster = AudioBroadcaster(self.shared.shutdown)
        self.ring = RingAudioBuffer(
            config.audio.sample_rate, config.audio.channels,
            config.ring.seconds,
        )
        self._transcript_q: queue.Queue = queue.Queue(maxsize=50)
        self._threads: list[threading.Thread] = []
        self._stream: sd.InputStream | None = None

        # UI cues
        self.cues = UICues(
            sample_rate=config.tts.sample_rate,
            device=config.audio.output_device,
        )

        # Backends (lazy loaded)
        self._asr = None
        self._tts = None
        self._llm = None

        # Workers (built after load_models)
        self._listen_worker = None
        self._interrupt_det = None
        self._think_worker = None
        self._playback_worker = None
        self._keyboard_worker = None

    # ── Lifecycle ──

    def load_models(self) -> None:
        log.event("engine", "load_models_start")
        try:
            self._asr = FasterWhisperASR(
                self.cfg.asr.model_size,
                self.cfg.asr.device,
                self.cfg.asr.compute_type,
            )
        except Exception as e:
            log.event("engine", "asr_load_failed", error=str(e))
            self.shared.asr_available = False

        try:
            self._tts = PiperTTS(self.cfg.tts.model_path, self.cfg.tts.sample_rate)
        except Exception as e:
            log.event("engine", "tts_load_failed", error=str(e))
            self.shared.voice_out_available = False

        self._llm = CallableLLM(self._ask_fn)
        log.event("engine", "load_models_done")

    def start(self) -> None:
        if self._started:
            return

        log.init(self.cfg.log_file)
        log.event("engine", "starting")

        if self._asr is None:
            self.load_models()

        # Test audio I/O
        self._test_audio_in()
        self._test_audio_out()

        # Build workers
        qcfg = self.cfg.queues
        listen_q = self.broadcaster.subscribe("listen", maxsize=qcfg.broadcast_listen_maxsize)
        interrupt_q = self.broadcaster.subscribe("interrupt_det", maxsize=qcfg.broadcast_interrupt_maxsize)

        self._playback_worker = PlaybackWorker(
            self.shared, self.ctrl, self.floor, self.interrupt,
            sample_rate=self.cfg.tts.sample_rate,
            chunk_ms=self.cfg.playback.chunk_ms,
            output_device=self.cfg.audio.output_device,
            cues=self.cues,
        )
        self._think_worker = ThinkWorker(
            self._transcript_q, self.shared, self.ctrl, self.floor,
            self.interrupt, self._llm, self._playback_worker, self._tts,
            cues=self.cues,
        )
        self._listen_worker = ListenWorker(
            listen_q, self.shared, self.ctrl, self.floor, self.interrupt,
            self.ring, self.cfg.vad, self.cfg.audio.sample_rate,
            self._transcript_q, self._asr, cues=self.cues,
        )
        self._interrupt_det = InterruptDetectorWorker(
            interrupt_q, self.shared, self.ctrl, self.interrupt,
            self.cfg.interrupt_vad,
        )
        self._keyboard_worker = KeyboardWorker(
            self.shared, self.interrupt, self.cfg.interrupt.keyboard_key,
        )

        # Launch workers (Thread subclasses — call .start() directly)
        for worker in (self._listen_worker, self._interrupt_det,
                       self._think_worker, self._playback_worker):
            worker.start()
            self._threads.append(worker)

        # Keyboard in isolated try — failure here doesn't kill engine
        try:
            self._keyboard_worker.start()
            self._threads.append(self._keyboard_worker)
        except Exception as e:
            log.event("engine", "keyboard_start_failed", error=str(e))
            self.shared.keyboard_interrupt_available = False

        # Start mic stream
        self._start_mic()

        # Invariant monitor
        self._invariants = InvariantChecker(self.shared, self.ctrl)
        self._spawn("invariants", lambda: self._invariants.run_loop(interval_sec=2.0))

        # Engine starts in IDLE — no transition needed, just record it
        self._started = True
        log.event("engine", "online", state=self.ctrl.get_state().name)

        # Launch optional edge-glow companion (VOICE2_EDGE_GLOW).
        # Runs under system python so it can use GUI bindings not in this venv.
        self._edge_glow = None
        if (self.cfg.log_file and EDGE_GLOW_PATH
                and os.path.exists(EDGE_GLOW_PATH) and os.environ.get('DISPLAY')):
            try:
                self._edge_glow = subprocess.Popen(
                    ['/usr/bin/python3', EDGE_GLOW_PATH,
                     '--log', self.cfg.log_file,
                     '--rate', str(self.cfg.tts.sample_rate)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                log.event("engine", "edge_glow_started", pid=self._edge_glow.pid)
            except Exception as e:
                log.event("engine", "edge_glow_start_failed", error=str(e))

    def stop(self) -> None:
        if not self._started:
            return
        log.event("engine", "stopping")
        self.shared.shutdown.set()
        self.shared.interrupted.set()  # unblock any active playback loop immediately
        # Shut down cue stream first (no concurrent sd.play calls after this)
        if self.cues:
            try:
                self.cues.close()
            except Exception:
                pass
        if self._stream:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        if getattr(self, "_edge_glow", None):
            try:
                self._edge_glow.terminate()
                self._edge_glow.wait(timeout=2.0)
            except Exception:
                pass
        self._started = False
        log.event("engine", "stopped")

    def submit_text(self, text: str) -> None:
        """Fallback: inject text directly as if ASR produced it."""
        from .logging_util import LatencyTrace
        turn_id = self.ctrl.start_new_turn()
        trace = LatencyTrace(turn_id)
        trace.mark("turn_start")
        self._transcript_q.put_nowait((text, turn_id, trace))

    def status(self) -> dict:
        return {
            **self.ctrl.snapshot(),
            "started": self._started,
            "interrupted": self.shared.interrupted.is_set(),
            "speaking": self.shared.speaking.is_set(),
            "thinking": self.shared.thinking.is_set(),
            "listening": self.shared.listening.is_set(),
            "capabilities": {
                "voice_in_available": self.shared.voice_in_available,
                "voice_out_available": self.shared.voice_out_available,
                "asr_available": self.shared.asr_available,
                "llm_available": self.shared.llm_available,
                "keyboard_interrupt_available": self.shared.keyboard_interrupt_available,
                "vad_interrupt_available": self.shared.vad_interrupt_available,
            },
            "worker_health": {
                t.name: t.is_alive()
                for t in self._threads
            },
        }

    def join(self) -> None:
        for t in self._threads:
            t.join()

    # ── Internal ──

    def _spawn(self, name: str, target) -> None:
        t = threading.Thread(target=target, name=f"voice-{name}", daemon=True)
        t.start()
        self._threads.append(t)

    def _start_mic(self) -> None:
        cfg = self.cfg.audio

        def _callback(indata, frames, time_info, status):
            frame = indata[:, 0].copy()
            self.ring.append(frame)
            self.broadcaster.publish(frame)

        try:
            self._stream = sd.InputStream(
                samplerate=cfg.sample_rate,
                channels=cfg.channels,
                dtype=cfg.dtype,
                blocksize=cfg.blocksize,
                device=cfg.input_device,
                callback=_callback,
            )
            self._stream.start()
            log.event("engine", "mic_started")
        except Exception as e:
            log.event("engine", "mic_failed", error=str(e))
            self.shared.voice_in_available = False

    def _test_audio_in(self) -> None:
        try:
            sd.check_input_settings(
                device=self.cfg.audio.input_device,
                samplerate=self.cfg.audio.sample_rate,
            )
        except Exception as e:
            log.event("engine", "audio_in_unavailable", error=str(e))
            self.shared.voice_in_available = False

    def _test_audio_out(self) -> None:
        try:
            sd.check_output_settings(
                device=self.cfg.audio.output_device,
                samplerate=self.cfg.tts.sample_rate,
            )
        except Exception as e:
            log.event("engine", "audio_out_unavailable", error=str(e))
            self.shared.voice_out_available = False
