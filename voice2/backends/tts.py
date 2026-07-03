"""TTS backend — yields PCM chunks. Swap without touching PlaybackWorker."""
from typing import Iterable, Protocol
import io
import os
import subprocess
import numpy as np


class TTSBackend(Protocol):
    def synthesize(self, text: str) -> Iterable[np.ndarray]: ...


class PiperTTS:
    """Yields one PCM chunk per sentence. Starts playback before full synthesis."""

    def __init__(self, model_path: str, sample_rate: int = 22050) -> None:
        self._model = model_path
        self._sr = sample_rate
        self._length_scale = os.environ.get("VOICE2_TTS_LENGTH_SCALE")
        self._noise_scale = os.environ.get("VOICE2_TTS_NOISE_SCALE")
        self._noise_w = os.environ.get("VOICE2_TTS_NOISE_W")
        self._volume = os.environ.get("VOICE2_TTS_VOLUME")

    def synthesize(self, text: str) -> Iterable[np.ndarray]:
        # Piper reads text from stdin, outputs raw PCM on stdout
        cmd = ["piper", "--model", self._model, "--output-raw"]
        if self._length_scale:
            cmd += ["--length-scale", self._length_scale]
        if self._noise_scale:
            cmd += ["--noise-scale", self._noise_scale]
        if self._noise_w:
            cmd += ["--noise-w-scale", self._noise_w]
        if self._volume:
            cmd += ["--volume", self._volume]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write(text.encode())
        proc.stdin.close()

        chunk_bytes = int(self._sr * 0.020) * 2  # 20ms of int16
        while True:
            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                break
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            yield pcm

        proc.wait()
