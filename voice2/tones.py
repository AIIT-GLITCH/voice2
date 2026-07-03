"""Soft UI tones — dedicated persistent OutputStream, no sd.play() calls."""
import queue
import threading
import numpy as np
import sounddevice as sd


def _tone(
    freq: float,
    duration_ms: int,
    volume: float,
    sample_rate: int,
    fade_ms: int = 30,
) -> np.ndarray:
    """Sine wave with fade-in and fade-out to avoid clicks."""
    n = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32) * volume

    fade = int(sample_rate * fade_ms / 1000)
    fade = min(fade, n // 2)
    ramp = np.linspace(0, 1, fade)
    wave[:fade] *= ramp
    wave[-fade:] *= ramp[::-1]
    return wave


def _chord(freqs, duration_ms, volume, sample_rate, fade_ms=30) -> np.ndarray:
    wave = sum(_tone(f, duration_ms, volume / len(freqs), sample_rate, fade_ms) for f in freqs)
    return wave.astype(np.float32)


def _sequence(tones, gap_ms, sample_rate) -> np.ndarray:
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    parts = []
    for i, t in enumerate(tones):
        parts.append(t)
        if i < len(tones) - 1:
            parts.append(gap)
    return np.concatenate(parts)


class UICues:
    """
    Play soft tones for voice engine state transitions.

    Uses a dedicated persistent sd.OutputStream so tone playback never
    conflicts with the TTS OutputStream in PlaybackWorker.  All public
    methods are non-blocking: they enqueue the pre-rendered numpy array
    and return immediately.
    """

    def __init__(self, sample_rate: int = 22050, device=None, volume: float = 0.18):
        self._sr = sample_rate
        self._dev = device
        self._vol = volume
        self._q: queue.Queue = queue.Queue(maxsize=16)
        self._stream: sd.OutputStream | None = None
        self._thread: threading.Thread | None = None
        self._open()

    # ── Lifecycle ──

    def _open(self) -> None:
        """Open dedicated cue stream and start drain thread."""
        try:
            self._stream = sd.OutputStream(
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                device=self._dev,
                blocksize=512,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            return  # cues silently disabled

        self._thread = threading.Thread(target=self._drain, daemon=True, name="voice-cues")
        self._thread.start()

    def _drain(self) -> None:
        """Pull tone arrays from queue and write to the open stream."""
        while True:
            try:
                samples = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stream is None:
                    break
                continue
            if samples is None:  # sentinel
                break
            if self._stream is None:
                continue
            try:
                self._stream.write(samples)
            except Exception:
                pass

    def close(self) -> None:
        """Graceful shutdown — call from engine.stop()."""
        self._q.put(None)  # sentinel to drain thread (blocking — guaranteed delivery)
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _play(self, samples: np.ndarray) -> None:
        """Non-blocking enqueue. Drops silently if queue full."""
        if self._stream is None:
            return
        try:
            self._q.put_nowait(samples)
        except queue.Full:
            pass

    # ── Cues ──

    def listening(self) -> None:
        """Soft rising two-note: ready to hear you."""
        tones = [
            _tone(660, 80, self._vol, self._sr),
            _tone(880, 100, self._vol, self._sr),
        ]
        self._play(_sequence(tones, 20, self._sr))

    def thinking(self) -> None:
        """Single soft mid-note ping: processing."""
        self._play(_tone(528, 90, self._vol * 0.7, self._sr))

    def speaking(self) -> None:
        """Gentle chord: about to speak."""
        self._play(_chord([528, 660], 100, self._vol * 0.8, self._sr))

    def interrupted(self) -> None:
        """Soft descending note: cut off."""
        tones = [
            _tone(660, 70, self._vol * 0.7, self._sr),
            _tone(440, 90, self._vol * 0.6, self._sr),
        ]
        self._play(_sequence(tones, 15, self._sr))

    def error(self) -> None:
        """Low soft thud: something wrong."""
        self._play(_tone(220, 150, self._vol * 0.5, self._sr))
