"""Structured JSON-line event logger for the voice engine."""
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any


_log_lock = threading.Lock()
_log_file: str | None = None


def init(path: str | None) -> None:
    global _log_file
    _log_file = path


def event(subsystem: str, ev: str, **meta: Any) -> None:
    """All metadata is keyword-only. No positional args beyond subsystem + ev."""
    turn_id = meta.get("turn_id", 0)
    state = meta.get("state", "")
    record = {
        "ts": time.monotonic(),
        "wall_ts": datetime.now(timezone.utc).isoformat(),
        "subsystem": subsystem,
        "event": ev,
        **meta,
    }
    line = json.dumps(record, default=str)
    with _log_lock:
        print(f"[{subsystem}] {ev}" + (f" | turn={turn_id}" if turn_id else "")
              + (f" | state={state}" if state else ""))
        if _log_file:
            try:
                with open(_log_file, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass


class LatencyTrace:
    """Per-turn monotonic timing. Thread-safe reads, single-writer assumed."""

    FIELDS = [
        "turn_start", "speech_detect_start", "speech_detect_end",
        "asr_start", "asr_end", "think_start", "first_token", "think_end",
        "tts_start", "tts_first_sample", "playback_start",
        "interrupt_detected", "interrupt_executed",
        "playback_end", "turn_end",
    ]

    def __init__(self, turn_id: int) -> None:
        self.turn_id = turn_id
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._marks[name] = time.monotonic()

    def duration(self, a: str, b: str) -> float | None:
        if a in self._marks and b in self._marks:
            return round(self._marks[b] - self._marks[a], 4)
        return None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"turn_id": self.turn_id, "marks": {}}
        for k, v in self._marks.items():
            out["marks"][k] = round(v, 4)
        # Key durations
        out["durations"] = {
            "asr_ms": int((self.duration("asr_start", "asr_end") or 0) * 1000),
            "think_ms": int((self.duration("think_start", "think_end") or 0) * 1000),
            "interrupt_stop_ms": int((self.duration("interrupt_detected", "interrupt_executed") or 0) * 1000),
            "total_turn_ms": int((self.duration("turn_start", "turn_end") or 0) * 1000),
        }
        return out
