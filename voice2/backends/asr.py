"""ASR backend wrapper. Swap internals without touching workers."""
from typing import Protocol
import numpy as np


class ASRBackend(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class FasterWhisperASR:
    def __init__(self, model_size: str = "small.en", device: str = "cpu",
                 compute_type: str = "int8") -> None:
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device=device,
                                   compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        audio_f32 = audio.astype("float32")
        segments, _ = self._model.transcribe(audio_f32, language="en",
                                              beam_size=1, vad_filter=False,
                                              condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()
