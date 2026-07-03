"""All enums for the voice engine. Single source of truth."""
from enum import Enum, auto


class EngineState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    INTERRUPTING = auto()
    FALLBACK_TEXT = auto()
    ERROR = auto()
    STOPPED = auto()


class FloorOwner(Enum):
    NONE = auto()
    USER = auto()
    AGENT = auto()


class TransitionReason(Enum):
    STARTUP = auto()
    SPEECH_DETECTED = auto()
    SPEECH_ENDED = auto()
    ASR_COMPLETE = auto()
    THINK_START = auto()
    THINK_COMPLETE = auto()
    PLAYBACK_START = auto()
    PLAYBACK_COMPLETE = auto()
    INTERRUPT = auto()
    INTERRUPT_RESOLVED = auto()
    ERROR_RECOVERY = auto()
    CAPABILITY_FAILURE = auto()
    SHUTDOWN = auto()
    FALLBACK = auto()


class InterruptSource(Enum):
    KEYBOARD = auto()
    VAD = auto()
    PROGRAMMATIC = auto()
    PARTIAL_ASR = auto()  # hook for future use
