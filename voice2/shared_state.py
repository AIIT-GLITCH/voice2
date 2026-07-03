"""SharedState — synchronized primitives only. No business logic."""
import threading
from dataclasses import dataclass, field


@dataclass
class SharedState:
    # Lifecycle
    shutdown: threading.Event = field(default_factory=threading.Event)

    # Floor signals
    interrupted: threading.Event = field(default_factory=threading.Event)
    speaking: threading.Event = field(default_factory=threading.Event)
    thinking: threading.Event = field(default_factory=threading.Event)
    listening: threading.Event = field(default_factory=threading.Event)
    mic_gated: threading.Event = field(default_factory=threading.Event)

    # Capability flags — set once at startup, read-only after that
    voice_in_available: bool = True
    voice_out_available: bool = True
    asr_available: bool = True
    llm_available: bool = True
    keyboard_interrupt_available: bool = True
    vad_interrupt_available: bool = True

    # Turn tracking — only StateController writes this
    current_turn_id: int = 0
