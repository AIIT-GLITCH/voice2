"""Unit tests — RingAudioBuffer."""
import numpy as np
from ..ring_buffer import RingAudioBuffer


def test_append_and_get():
    buf = RingAudioBuffer(sample_rate=16000, channels=1, seconds=2.0)
    frame = np.ones(1600, dtype="float32")
    buf.append(frame)
    out = buf.get_last(0.1)
    assert len(out) == 1600


def test_capacity_trimming():
    buf = RingAudioBuffer(sample_rate=16000, channels=1, seconds=1.0)
    for _ in range(20):
        buf.append(np.ones(1600, dtype="float32"))
    # Should not exceed capacity
    assert buf.size_seconds() <= 1.1


def test_preroll_extraction():
    buf = RingAudioBuffer(sample_rate=16000, channels=1, seconds=2.0)
    buf.append(np.zeros(8000, dtype="float32"))
    buf.append(np.ones(8000, dtype="float32"))
    out = buf.get_last(0.5)
    assert len(out) == 8000
    # Last 0.5s should be the ones
    assert np.all(out == 1.0)


def test_clear():
    buf = RingAudioBuffer(sample_rate=16000, channels=1, seconds=2.0)
    buf.append(np.ones(1600, dtype="float32"))
    buf.clear()
    assert buf.size_seconds() == 0.0
