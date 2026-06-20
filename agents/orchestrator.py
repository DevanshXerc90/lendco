"""Orchestrator — coordinates the multi-agent system across a call.

Per session it:
  1. ContextAgent  builds the unified context + memory recall.
  2. Greets with a MEMORY-AWARE opener (references open commitments) — this is
     what makes a returning borrower's call start ahead of where the last ended.
  3. Per turn: DiagnosisAgent finds intent + gaps; if an essential ask-only slot
     is missing it asks ONE focused follow-up (dynamic questioning); otherwise
     the VoiceAgent composes a grounded, tool-backed answer (LLM or rule-based).
  4. MemoryAgent captures Q&A / FAQs / commitments and, at session end, records
     the resolution path + outcome so future calls improve (continuous learning).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.context_agent import ContextAgent  # noqa: E402
from agents.diagnosis_agent import DiagnosisAgent  # noqa: E402
from agents.knowledge_agent import KnowledgeAgent  # noqa: E402
from agents.memory_agent import MemoryAgent  # noqa: E402
from agents.voice_agent import VoiceAgent, rupee  # noqa: E402
from core.tools import ToolContext  # noqa: E402
from core.llm import get_llm  # noqa: E402
from core.memory import get_memory  # noqa: E402

_SENTIMENT = {
    "angry": ["angry", "furious", "ridiculous", "worst", "useless", "fed up", "cheated", "harass"],
    "frustrated": ["frustrat", "annoyed", "again", "still not", "why is this", "not working", "tired of"],
    "anxious": ["worried", "scared", "afraid", "anxious", "stress", "please help", "urgent"],
    "positive": ["thank", "great", "appreciate", "helpful", "perfect"],
}


def analyze_sentiment(text: str) -> str:
    t = text.lower()
    for label, kws in _SENTIMENT.items():
        if any(k in t for k in kws):
            return label
    return "neutral"


ESSENTIAL_PRIORITY = 3  # ask-only slots at/above this priority are asked before answering


class Orchestrator:
    def __init__(self, memory=None, llm=None):
        self.memory = memory or get_memory()
        self.llm = llm or get_llm()
        self.context_agent = ContextAgent(self.memory)
        self.diagnosis_agent = DiagnosisAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.memory_agent = MemoryAgent(self.memory)
        self.voice_agent = VoiceAgent(self.llm, self.memory)
        self.session = None

    # ----------------------------------------------------------------- session
    def start(self, borrower_id: str | None = None, phone: str | None = None, session_id: str = "S-LIVE") -> dict:
        ctx = self.context_agent.build(borrower_id=borrower_id, phone=phone)
        if not ctx.found:
            return {"found": False, "text": "I'm sorry, I couldn't locate an account with those details. "
                                             "Could you share your registered phone number or loan ID?"}
        bid = ctx.borrower_id
        recall = self.memory_agent.recall(bid)
        # persist baseline preferences for future calls
        self.memory_agent.set_preference(bid, "language", ctx.profile.get("preferred_language", "English"))

        self.session = {
            "session_id": session_id,
            "borrower_id": bid,
            "context": ctx.to_dict(),
            "recall": recall,
            "history": [],
            "collected": {},
            "asked_slots": set(recall.get("previously_asked_slots", [])),
            "pending_slot": None,
            "intent": None,
            "sentiment": "neutral",
            "tool_trace": [],
            "actions": [],
            "turns": 0,
            "escalated": False,
            "started_at": "2026-06-09T10:00:00",
        }

        name = ctx.profile.get("name", "there")
        opener = f"Hello {name}, thank you for calling LendCo support, this is Aarav. "
        open_commitments = recall.get("open_commitments", [])
        if open_commitments:
            c = open_commitments[0]
            reason = c["reason"].strip()
            for lead in ("because of ", "because ", "due to ", "as ", "since "):
                if reason.lower().startswith(lead):
                    reason = reason[len(lead):]
                    break
            opener += (f"During our previous conversation, you mentioned that due to {reason} you planned to "
                       f"pay {rupee(c['amount'])} by {c['date']}. Were you able to complete that payment?")
            self.session["intent"] = "promise_to_pay"
            self.session["expecting_ptp_followup"] = True
        else:
            opener += "How can I help you with your loan today?"

        self.session["history"].append({"role": "agent", "text": opener})
        return {"found": True, "borrower_id": bid, "text": opener,
                "memory_aware": bool(open_commitments), "recall": recall}

    def _tool_ctx(self) -> ToolContext:
        s = self.session
        return ToolContext(borrower_id=s["borrower_id"], context=s["context"],
                           memory=self.memory, session_id=s["session_id"], trace=[])

    def handle(self, utterance: str) -> dict:
        s = self.session
        if s is None:
            raise RuntimeError("call start() before handle()")
        s["history"].append({"role": "borrower", "text": utterance})
        s["turns"] += 1
        s["sentiment"] = analyze_sentiment(utterance)

        # follow-up answer to a previously asked slot?
        if s["pending_slot"]:
            slot = s["pending_slot"]
            s["collected"][slot] = utterance.strip().rstrip(".,;")
            self.memory_agent.capture_qa(s["session_id"], s["borrower_id"], slot, slot, utterance)
            s["asked_slots"].add(slot)
            s["pending_slot"] = None
        elif s.get("expecting_ptp_followup"):
            # response to the memory-aware "did you pay?" opener
            s.pop("expecting_ptp_followup", None)
            kept = any(w in utterance.lower() for w in ["yes", "paid", "done", "did", "completed", "cleared"])
            self.memory_agent.resolve_open_commitments(s["borrower_id"], kept)
            if kept:
                txt = "That's wonderful, thank you for honouring your commitment. Is there anything else I can help you with today?"
            else:
                txt = ("No problem, these things happen. When do you now expect to be able to make the payment?")
                s["intent"] = "promise_to_pay"
                s["pending_slot"] = "ptp_date"
            s["history"].append({"role": "agent", "text": txt})
            return self._turn_result(txt, s.get("intent", "promise_to_pay"), None)

        # diagnosis (sticky intent across a multi-turn slot fill)
        intent = s["intent"]
        diagnosis = self.diagnosis_agent.diagnose(
            utterance, s["context"], intent=intent,
            already_asked=s["asked_slots"] | set(s["collected"].keys()),
        )
        if s["intent"] is None:
            s["intent"] = diagnosis.intent
            self.memory_agent.capture_faq(s["borrower_id"], diagnosis.intent, utterance)
        s["last_intent"] = diagnosis.intent

        # dynamic follow-up: ask one essential ask-only question we still lack
        if not self.llm.available:
            essential = [q for q in diagnosis.questions
                         if q["ask_only"] and q["priority"] <= ESSENTIAL_PRIORITY
                         and q["slot"] not in s["collected"]]
            if essential:
                q = essential[0]
                s["pending_slot"] = q["slot"]
                s["asked_slots"].add(q["slot"])
                s["history"].append({"role": "agent", "text": q["question"]})
                return self._turn_result(q["question"], diagnosis.intent, None, asking=q["slot"])

        # compose the grounded answer
        tc = self._tool_ctx()
        if self.llm.available:
            out = self.voice_agent.respond_llm(s["context"], s["recall"], diagnosis, s["history"], tc)
        else:
            out = self.voice_agent.respond_rule_based(s["context"], s["recall"], diagnosis, s["collected"], tc)

        s["tool_trace"].extend(out["tool_trace"])
        if out.get("action"):
            s["actions"].append(out["action"])
            if out["action"] in ("ticket_created",) and diagnosis.intent in ("settlement_request",):
                s["escalated"] = True
        s["history"].append({"role": "agent", "text": out["text"]})
        result = self._turn_result(out["text"], diagnosis.intent, out.get("action"), diagnosis=diagnosis)
        # an answer completed this intent — reset so the next utterance is re-classified
        # afresh (intent only stays sticky while a slot-fill follow-up is pending).
        s["intent"] = None
        s["collected"] = {}
        return result

    def _turn_result(self, text, intent, action, asking=None, diagnosis=None):
        return {
            "text": text, "intent": intent, "action": action, "asking": asking,
            "sentiment": self.session["sentiment"],
            "tool_trace": self.session["tool_trace"][-6:],
            "diagnosis": diagnosis.to_dict() if diagnosis else None,
        }

    def end(self) -> dict:
        s = self.session
        if s is None:
            return {}
        success = not s["escalated"]
        last_intent = s.get("last_intent") or s.get("intent") or "general_faq"
        path = self.memory_agent.learn(last_intent, s["tool_trace"], success, s["turns"])
        outstanding = [] if success else [{"intent": last_intent, "note": "escalated for human review"}]
        self.memory_agent.save_session({
            "session_id": s["session_id"], "borrower_id": s["borrower_id"], "intent": last_intent,
            "started_at": s["started_at"], "resolved": success, "sentiment": s["sentiment"],
            "turns": s["turns"], "transcript": s["history"], "outstanding": outstanding,
        })
        summary = {"borrower_id": s["borrower_id"], "intent": last_intent, "turns": s["turns"],
                   "resolved": success, "resolution_path": path, "actions": s["actions"],
                   "tools_used": [t["tool"] for t in s["tool_trace"]]}
        self.session = None
        return summary
