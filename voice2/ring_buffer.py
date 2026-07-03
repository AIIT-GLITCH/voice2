"""RingAudioBuffer — thread-safe rolling mic history for pre-roll."""
import threading
from collections import deque
import numpy as np


class RingAudioBuffer:
    def __init__(self, sample_rate: int, channels: int, seconds: float,
                 dtype: str = "float32") -> None:
        self._sr = sample_rate
        self._ch = channels
        self._dtype = dtype
        self._capacity_samples = int(sample_rate * seconds)
        self._buf: deque[np.ndarray] = deque()
        self._size = 0  # samples stored
        self._lock = threading.Lock()

    def append(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append(frame)
            self._size += len(frame)
            # Trim oldest frames to stay within capacity
            while self._size > self._capacity_samples and self._buf:
                oldest = self._buf.popleft()
                self._size -= len(oldest)

    def get_last(self, seconds: float) -> np.ndarray:
        """Return up to `seconds` of most recent audio."""
        n = int(self._sr * seconds)
        with self._lock:
            if not self._buf:
                return np.zeros(0, dtype=self._dtype)
            frames = list(self._buf)
        combined = np.concatenate(frames)
        return combined[-n:] if len(combined) > n else combined

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._size = 0

    def size_seconds(self) -> float:
        with self._lock:
            return self._size / self._sr
