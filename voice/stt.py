"""Speech-to-Text abstraction.

Backends (set STT_BACKEND):
  whisper : offline local STT via faster-whisper + sounddevice mic capture
            (push-to-talk: press Enter, speak, it records a few seconds).
  text    : typed input (default; always works, used in CI/demos).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402


class TextSTT:
    backend = "text"

    def listen(self, prompt: str = "BORROWER> ") -> str:
        try:
            return input(prompt).strip()
        except EOFError:
            return ""


class WhisperSTT:
    backend = "whisper"

    def __init__(self, seconds: float = 6.0, samplerate: int = 16000):
        from faster_whisper import WhisperModel  # lazy
        self.model = WhisperModel(settings.WHISPER_MODEL, device="cpu", compute_type="int8")
        self.seconds = seconds
        self.samplerate = samplerate

    def listen(self, prompt: str = "BORROWER> ") -> str:
        import numpy as np
        import sounddevice as sd
        input(f"{prompt}(press Enter, then speak for ~{int(self.seconds)}s) ")
        audio = sd.rec(int(self.seconds * self.samplerate), samplerate=self.samplerate, channels=1, dtype="float32")
        sd.wait()
        segments, _ = self.model.transcribe(audio.flatten(), language="en")
        text = " ".join(s.text for s in segments).strip()
        print(f"(heard) {text}")
        return text


def get_stt():
    if settings.STT_BACKEND == "whisper":
        try:
            return WhisperSTT()
        except Exception as e:
            print(f"(whisper unavailable: {e}; falling back to text input)")
    return TextSTT()
