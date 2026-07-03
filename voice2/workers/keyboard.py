"""KeyboardWorker — spacebar interrupt. Isolated so failure doesn't kill engine."""
import threading
import sys
import termios
import tty

from ..enums import InterruptSource
from ..shared_state import SharedState
from ..interrupt_controller import InterruptController
from .. import logging_util as log


def _read_char() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class KeyboardWorker(threading.Thread):
    def __init__(
        self,
        shared: SharedState,
        interrupt: InterruptController,
        key: str = " ",
    ) -> None:
        super().__init__(name="voice-keyboard", daemon=True)
        self._shared = shared
        self._interrupt = interrupt
        self._key = key

    def run(self) -> None:
        log.event("keyboard", "listener_started", meta={"key": repr(self._key)})
        while not self._shared.shutdown.is_set():
            try:
                ch = _read_char()
                if ch == self._key:
                    log.event("keyboard", "spacebar_pressed")
                    self._interrupt.trigger(
                        InterruptSource.KEYBOARD,
                        reason="spacebar_pressed",
                    )
                elif ch in ("\x03", "\x04"):  # Ctrl+C or Ctrl+D
                    self._shared.shutdown.set()
                    break
            except Exception as e:
                log.event("keyboard", "listener_error", meta={"error": str(e)})
                # Don't kill the engine over keyboard issues
                break
        log.event("keyboard", "listener_stopped")
