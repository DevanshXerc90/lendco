"""Diagnosis Agent — owns intent + gap analysis + next-question selection."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.diagnosis import DiagnosisLayer, DiagnosisResult  # noqa: E402


class DiagnosisAgent:
    def __init__(self):
        self.layer = DiagnosisLayer()

    def diagnose(self, utterance: str, context: dict, intent: str | None = None,
                 already_asked: set[str] | None = None) -> DiagnosisResult:
        return self.layer.diagnose(utterance, context, intent=intent, already_asked=already_asked)
