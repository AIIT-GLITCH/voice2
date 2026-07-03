"""
Minimal bootstrap — run the engine with an echo backend, no LLM required.

Usage:
    python -m voice2.main

Wire your own model by replacing `backend` with any callable(text) -> str.
See examples/ for an HTTP LLM backend.
"""
import os
import signal

from voice2 import VoiceEngine, VoiceConfig
from voice2.config import VADConfig, InterruptVADConfig, InterruptConfig


def backend(text: str) -> str:
    """Echo backend — proves the full loop (mic -> ASR -> reply -> TTS)."""
    return f"You said: {text}"


def main() -> None:
    cfg = VoiceConfig(
        vad=VADConfig(end_silence_ms=5000, max_utterance_sec=90.0),
        interrupt_vad=InterruptVADConfig(
            consecutive_frames_required=4,
            energy_multiplier=2.0,
        ),
        interrupt=InterruptConfig(debounce_ms=200, keyboard_key=" "),
        log_file=os.path.expanduser("~/voice_engine.jsonl"),
    )

    engine = VoiceEngine(cfg, backend)

    print("=" * 60)
    print("  VOICE ENGINE v2 — Full-Duplex Interruptible")
    print("  Space = interrupt | Ctrl+C = quit")
    print("=" * 60)

    print("[boot] Loading models...")
    engine.load_models()

    print("[boot] Starting engine...")
    engine.start()

    print(f"[boot] Online. Status: {engine.status()}")
    print("[boot] Listening. Talk naturally.\n")

    # Text fallback if voice_in failed
    if not engine.shared.voice_in_available:
        print("[fallback] Mic unavailable — TEXT MODE. Type messages, Enter to send.")
        while not engine.shared.shutdown.is_set():
            try:
                text = input("You: ").strip()
                if text:
                    engine.submit_text(text)
            except (EOFError, KeyboardInterrupt):
                break
    else:
        try:
            signal.pause()
        except KeyboardInterrupt:
            pass

    print("\n[shutdown] Stopping engine...")
    engine.stop()
    print("[shutdown] Done.")


if __name__ == "__main__":
    main()
