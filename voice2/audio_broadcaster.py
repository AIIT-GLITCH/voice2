"""AudioBroadcaster — fan-out mic frames to N consumers. No frame theft."""
import queue
import threading
from collections import defaultdict
from typing import Dict
import numpy as np
from . import logging_util as log


class AudioBroadcaster:
    def __init__(self, shutdown: threading.Event) -> None:
        self._shutdown = shutdown
        self._lock = threading.Lock()
        self._consumers: Dict[str, queue.Queue] = {}
        self._dropped: Dict[str, int] = defaultdict(int)

    def subscribe(self, name: str, maxsize: int = 500) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._consumers[name] = q
        log.event("broadcaster", "subscribed", meta={"name": name, "maxsize": maxsize})
        return q

    def unsubscribe(self, name: str) -> None:
        with self._lock:
            self._consumers.pop(name, None)

    def publish(self, frame: np.ndarray) -> None:
        with self._lock:
            consumers = list(self._consumers.items())
        for name, q in consumers:
            try:
                q.put_nowait(frame)
            except queue.Full:
                # Drop oldest, push newest — never block the publisher
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                    self._dropped[name] += 1
                except Exception:
                    pass

    def dropped_count(self, name: str) -> int:
        return self._dropped.get(name, 0)
