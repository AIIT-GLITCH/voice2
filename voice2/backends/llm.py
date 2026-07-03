"""LLM backend wrapper. Wire any callable(text) -> str here."""
from typing import Callable, Protocol


class LLMBackend(Protocol):
    def reply(self, text: str) -> str: ...


class CallableLLM:
    """Wraps any callable(text) -> str as an LLM backend."""
    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn

    def reply(self, text: str) -> str:
        return self._fn(text)
