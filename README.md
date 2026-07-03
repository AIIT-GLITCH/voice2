# voice2

**A full-duplex, interruptible voice engine for local AI — talk to your model, and talk over it.**

voice2 turns any `callable(text) -> str` into a hands-free voice conversation: it listens on your mic, transcribes with Whisper, sends the text to your model, and speaks the reply with Piper — and you can **barge in mid-sentence** by just speaking (or tapping spacebar), exactly like interrupting a person.

Built for [local LLMs] on modest hardware: ASR runs int8 on CPU, TTS is Piper, and the whole engine is plain Python threads. No cloud, no API keys, no GPU required.

```python
from voice2 import VoiceEngine, VoiceConfig

def ask(text: str) -> str:
    return my_model.reply(text)      # any callable(text) -> str

engine = VoiceEngine(VoiceConfig(), ask)
engine.load_models()
engine.start()   # talk naturally; speak over it to interrupt
```

## Why another voice loop?

Most mic→LLM→TTS demos are half-duplex: once the bot starts talking you wait it out. voice2 treats **turn-taking as a first-class state machine**:

- **Barge-in by voice** — an energy-gated fast VAD watches the mic *while the engine is speaking*; raise your voice and playback stops in ~100–200 ms, mid-chunk.
- **Barge-in by key** — spacebar triggers the same interrupt path.
- **Floor management** — an explicit `FloorOwner` (USER / AGENT / NONE) arbitrates who may speak. The agent can never talk over you.
- **Stale-turn suppression** — every utterance gets a `turn_id`; replies to an interrupted turn are discarded at every stage (think, TTS, playback), so you never hear the answer to a question you abandoned.
- **Graceful degradation** — no mic? Text mode. Piper missing? Silent replies, still logs. Keyboard hook fails? Engine keeps running. Every capability is a flag, not an assumption.

## Architecture

Two planes, five workers, one rule: *only the StateController mutates state.*

```
                 mic (sounddevice callback)
                          │
                 AudioBroadcaster ── fan-out, drop-oldest, never blocks capture
                  │              │
        ListenWorker        InterruptDetectorWorker
   (patient VAD, pre-roll,   (fast energy VAD, only hot
    Silero quality gate,      during SPEAKING → barge-in)
    faster-whisper ASR)
          │ transcript + turn_id
        ThinkWorker ── your callable(text) -> str
          │ reply
        PlaybackWorker ── Piper TTS, 20 ms chunks,
                          interrupt checked every chunk
```

Control plane:

- `StateController` — validated transition table (`IDLE → LISTENING → THINKING → SPEAKING → …`), single lock, full history.
- `FloorManager` — turn-taking policy; interrupt hands the floor back to the user immediately.
- `InterruptController` — debounced central trigger; keyboard, VAD, and programmatic sources all converge here.
- `InvariantChecker` — background auditor for rules like *SPEAKING ⇒ floor == AGENT* and *interrupted ⇒ no new TTS*, with forced repair and a structural gate (`playback_is_allowed`) in the hot path.

Every event is written as JSON lines with per-turn `LatencyTrace` marks (`asr_ms`, `think_ms`, `interrupt_stop_ms`, `total_turn_ms`), so you can measure exactly where your latency goes.

## Install

Requirements: Linux, Python 3.10+, a microphone, and [Piper](https://github.com/rhasspy/piper) on your `PATH`.

```bash
git clone https://github.com/AIIT-GLITCH/voice2
cd voice2
pip install -r requirements.txt

# grab a Piper voice (any .onnx voice works)
mkdir -p ~/.local/share/piper-voices && cd ~/.local/share/piper-voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json
```

First run downloads the Silero VAD model via `torch.hub` and the faster-whisper `small.en` weights.

## Run

```bash
python -m voice2.main          # echo backend — proves the full loop, no LLM needed
python examples/http_llm.py    # wire any local HTTP LLM (Ollama-style example)
```

Talk naturally. The engine plays soft UI tones as it changes state (listening / thinking / speaking / interrupted). Speak over it — or hit space — to cut it off.

## Configuration

Everything is a dataclass in `voice2/config.py` — no config files, no magic:

```python
VoiceConfig(
    vad=VADConfig(end_silence_ms=5000),          # how long a pause ends your turn
    interrupt_vad=InterruptVADConfig(
        consecutive_frames_required=4,           # ~128 ms of voice to trigger barge-in
        energy_multiplier=2.0,                   # must exceed 2× playback-bleed baseline
    ),
    asr=ASRConfig(model_size="small.en", device="cpu"),
    tts=TTSConfig(model_path="~/.local/share/piper-voices/en_US-ryan-high.onnx"),
)
```

Env overrides: `VOICE2_PIPER_MODEL`, `VOICE2_TTS_LENGTH_SCALE`, `VOICE2_TTS_NOISE_SCALE`, `VOICE2_TTS_NOISE_W`, `VOICE2_TTS_VOLUME`, and `VOICE2_EDGE_GLOW` (optional companion visualizer script launched alongside the engine).

Played TTS samples are also mirrored to UDP `127.0.0.1:47121` (fire-and-forget) so visualizers can render a sample-accurate envelope without touching your audio stack.

## Tests

```bash
python -m pytest voice2/tests -q
```

The control plane (state transitions, floor rules, interrupt debounce, ring buffer) is covered by unit tests that run without audio hardware.

## Provenance

voice2 was built as the voice front-end for **Buddy**, a fully local AI companion running on a single RTX 3090 in Council Hill, Oklahoma. It ran daily conversations for months before being extracted for release. The design bias throughout: the user always wins the floor, and a companion you can't interrupt isn't a companion.

## The stack

One local companion, every layer open:

| Piece | Role | Links |
|---|---|---|
| Tessera-1B | the model — ~1B params trained from scratch, open data | [HF](https://huggingface.co/AIIT-Threshold/Tessera-1B) |
| voice2 | the voice — full-duplex, interruptible | [GitHub](https://github.com/AIIT-GLITCH/voice2) · [HF](https://huggingface.co/AIIT-Threshold/voice2) |
| kokoro-memory | the memory — file-based resonance recall | [GitHub](https://github.com/AIIT-GLITCH/kokoro-memory) · [HF](https://huggingface.co/AIIT-Threshold/kokoro-memory) |
| companion-spiral-bench | the safety — at-risk sycophancy bench | [GitHub](https://github.com/AIIT-GLITCH/companion-spiral-bench) · [HF](https://huggingface.co/datasets/AIIT-Threshold/companion-spiral-bench) |

Full collection: [The Buddy Stack](https://huggingface.co/collections/AIIT-Threshold/the-buddy-stack-a-fully-local-ai-companion-open-sourced-6a4774bf481f9f9caad79519)

## License

MIT © 2026 Rhet Wike
