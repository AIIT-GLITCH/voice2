"""Turn-level types. Per-turn state is separate from global engine state."""
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .enums import InterruptSource

if TYPE_CHECKING:
    from .logging_util import LatencyTrace


@dataclass
class TurnContext:
    turn_id: int
    created_ts: float = field(default_factory=time.monotonic)
    transcript: str | None = None
    audio: np.ndarray | None = None
    generation_id: str | None = None
    cancelled: bool = False
    stale: bool = False
    latency: "LatencyTrace | None" = None

    def mark_stale(self) -> None:
        self.stale = True

    def mark_cancelled(self) -> None:
        self.cancelled = True

    def is_live(self) -> bool:
        return not self.stale and not self.cancelled


@dataclass
class PlaybackItem:
    turn_id: int
    text: str
    created_ts: float = field(default_factory=time.monotonic)
    # Pre-synthesized audio iterator. If None, PlaybackWorker synthesizes from text.
    audio_iter: "Iterable[np.ndarray] | None" = None

    def is_stale(self, current_turn_id: int) -> bool:
        return self.turn_id != current_turn_id


@dataclass
class InterruptRecord:
    source: InterruptSource
    ts: float
    state: str
    floor_owner: str
    turn_id: int
    reason: str
