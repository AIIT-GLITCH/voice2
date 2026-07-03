"""Voice engine configuration. Dataclasses only. No logic here."""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    block_ms: int = 32
    input_device: int | None = None
    output_device: int | None = None

    @property
    def blocksize(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)


@dataclass
class VADConfig:
    """Normal utterance capture — patient."""
    threshold: float = 0.5
    pre_roll_ms: int = 400
    min_utterance_ms: int = 300
    min_voiced_ms: int = 320
    min_voice_prob: float = 0.45
    min_voiced_ratio: float = 0.08
    end_silence_ms: int = 5000   # patient: give the speaker room to think mid-sentence
    max_utterance_sec: float = 90.0


@dataclass
class InterruptVADConfig:
    """Aggressive barge-in detection during SPEAKING. Fast and cheap."""
    threshold: float = 0.4
    consecutive_frames_required: int = 4   # ~128ms at 32ms blocks
    energy_multiplier: float = 2.0         # must exceed N× playback bleed baseline
    refractory_ms: int = 500               # don't re-trigger within this window
    check_interval_ms: int = 32            # match block_ms


@dataclass
class ASRConfig:
    model_size: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1
    vad_filter: bool = False
    condition_on_previous_text: bool = False


@dataclass
class TTSConfig:
    model_path: str = os.environ.get(
        "VOICE2_PIPER_MODEL",
        os.path.expanduser("~/.local/share/piper-voices/en_US-ryan-high.onnx"),
    )
    sample_rate: int = 22050
    chunk_ms: int = 20          # playback chunk size — 20ms = fast interrupt response


@dataclass
class PlaybackConfig:
    chunk_ms: int = 20          # must match TTSConfig.chunk_ms
    discard_on_interrupt: bool = True


@dataclass
class InterruptConfig:
    debounce_ms: int = 200
    keyboard_key: str = " "     # spacebar


@dataclass
class RingBufferConfig:
    seconds: float = 2.0


@dataclass
class QueueConfig:
    """Sizes for all internal queues. Tune for latency vs. drop behavior."""
    broadcast_listen_maxsize: int = 500
    broadcast_interrupt_maxsize: int = 200
    transcript_maxsize: int = 50
    playback_maxsize: int = 200


@dataclass
class VoiceConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    interrupt_vad: InterruptVADConfig = field(default_factory=InterruptVADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    interrupt: InterruptConfig = field(default_factory=InterruptConfig)
    ring: RingBufferConfig = field(default_factory=RingBufferConfig)
    queues: QueueConfig = field(default_factory=QueueConfig)
    log_file: str | None = "voice_engine.jsonl"
