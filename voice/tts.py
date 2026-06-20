"""Text-to-Speech abstraction.

Backends (set TTS_BACKEND):
  pyttsx3 : offline local speech (Windows SAPI5 / macOS / espeak)
  print   : prints the agent's line (default; always works, used in CI/demos)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402


class PrintTTS:
    backend = "print"

    def speak(self, text: str) -> None:
        print(f"\nAGENT  > {text}\n")


class Pyttsx3TTS:
    backend = "pyttsx3"

    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 175)

    def speak(self, text: str) -> None:
        print(f"\nAGENT  > {text}\n")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception:
            pass


def get_tts():
    if settings.TTS_BACKEND == "pyttsx3":
        try:
            return Pyttsx3TTS()
        except Exception as e:
            print(f"(pyttsx3 unavailable: {e}; falling back to text)")
    return PrintTTS()
