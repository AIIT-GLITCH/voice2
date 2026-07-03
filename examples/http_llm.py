"""
Wire the voice engine to any local LLM behind an HTTP API.

Works with anything that takes text and returns text — llama.cpp server,
Ollama, vLLM, or your own FastAPI wrapper. Edit `ask()` to match your API.

Usage:
    LLM_URL=http://localhost:8000/chat python examples/http_llm.py
"""
import os
import signal

import httpx

from voice2 import VoiceEngine, VoiceConfig

LLM_URL = os.environ.get("LLM_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("LLM_MODEL", "llama3.2")


def ask(text: str) -> str:
    """Ollama-style example. Adapt the payload/response to your server."""
    try:
        r = httpx.post(
            LLM_URL,
            json={"model": MODEL, "prompt": text, "stream": False},
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip() or "..."
    except Exception as e:
        return f"Backend error: {e}"


def main() -> None:
    engine = VoiceEngine(VoiceConfig(), ask)
    engine.load_models()
    engine.start()
    print("Online. Talk naturally. Space = interrupt, Ctrl+C = quit.")
    try:
        signal.pause()
    except KeyboardInterrupt:
        pass
    engine.stop()


if __name__ == "__main__":
    main()
