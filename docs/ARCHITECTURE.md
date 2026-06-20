# Architecture

## 1. System context

```mermaid
flowchart LR
    B([Borrower]) -- voice --> STT[STT<br/>faster-whisper / text]
    STT --> ORC
    ORC --> TTS[TTS<br/>pyttsx3 / print]
    TTS -- voice --> B

    subgraph AGENTS[Multi-Agent Orchestrator]
        ORC{{Orchestrator}}
        CA[Context Agent]
        DA[Diagnosis Agent]
        KA[Knowledge Agent / RAG]
        MA[Memory Agent]
        VA[Voice Agent + LLM tool-loop]
        ORC --> CA & DA & KA & MA & VA
    end

    CA --> CE[Context Engine]
    MA --> MEM[(Memory DB<br/>SQLite)]
    KA --> VEC[(Vector index<br/>TF-IDF / embeddings)]

    subgraph PLAT[5 Independent Platforms]
        CRM[CRM<br/>:8101]
        PAY[Payments<br/>:8102]
        SUP[Support<br/>:8103]
        KB[Knowledge<br/>:8104]
        WF[Workflow<br/>:8105]
    end

    CE --> CRM & PAY & SUP
    VA -- tools --> CRM & PAY & SUP & WF
    KA --> KB
```

## 2. Request lifecycle (one call)

```mermaid
sequenceDiagram
    participant B as Borrower
    participant O as Orchestrator
    participant Ctx as Context Engine
    participant Mem as Memory
    participant Dx as Diagnosis
    participant V as Voice Agent
    participant T as Tools/Platforms
    participant R as RAG/KB

    B->>O: inbound call (id/phone)
    O->>Ctx: build unified context
    Ctx->>T: CRM + Payments + Support reads
    O->>Mem: recall(borrower)
    Mem-->>O: prefs, open commitments, past Q&A, best path
    O-->>B: memory-aware greeting (e.g. "did you pay as promised?")
    loop each turn
        B->>O: utterance
        O->>Dx: diagnose (intent, known, gaps, next question)
        alt essential info missing
            O-->>B: one focused follow-up question
        else enough to act
            O->>V: compose (LLM tool-loop OR deterministic planner)
            V->>T: get_gateway_logs / create_ticket / payment_link / ...
            V->>R: search_knowledge_base (ground + cite)
            V-->>O: grounded answer + action
            O-->>B: spoken response
        end
        O->>Mem: capture Q&A / commitment / FAQ
    end
    O->>Mem: record resolution path + outcome (learning)
```

## 3. Components

| Layer | Module | Responsibility |
|---|---|---|
| **Platforms** | `platforms/{crm,payments,support,knowledge,workflow}` | Five independent FastAPI services, each owning a domain slice + its own writes. Emulate Zoho/HubSpot, Razorpay/Stripe, Freshdesk, Notion, n8n. |
| **Clients** | `core/clients.py` | Typed HTTP wrappers with graceful degradation (read fallback to local data if a service is down). |
| **Context Engine** | `core/context_engine.py` | Aggregates borrower + interaction + operational context; derives loan analytics (EMIs paid/remaining, interest/principal split, outstanding, next due, overdue, penalties). |
| **Diagnosis Layer** | `core/diagnosis.py` | Intent classification + slot-based gap analysis; emits prioritized, de-duplicated, dynamically-phrased follow-ups; suggests tools + KB queries. |
| **RAG** | `core/rag.py` | Chunk + index KB; pluggable TF-IDF (default) or neural embeddings; citation-ready retrieval. |
| **Memory** | `core/memory.py` | Borrower / conversation / agent memory in SQLite; `recall()` + `best_resolution_path()` drive continuous learning. |
| **Tools** | `core/tools.py` | 11 real actions (read systems, create ticket, payment link, reversal, callback, escalate, record commitment). Shared by LLM loop and deterministic planner. |
| **LLM** | `core/llm.py` | Pluggable reasoning: Anthropic tool-use loop (with prompt caching) or null-LLM fallback. |
| **Agents** | `agents/*` | Context / Diagnosis / Knowledge / Memory / Voice agents + the Orchestrator that coordinates them. |
| **Voice** | `voice/{stt,tts}.py` | Local STT/TTS with text fallbacks. |
| **Eval + Dashboard** | `eval/`, `dashboard/` | Evaluation harness + self-contained HTML analytics dashboard. |

## 4. Key design decisions
1. **Diagnosis drives questioning, not a script.** Slots are *derivable* (resolved from
   systems — never asked) or *ask-only* (only the borrower knows). Only missing,
   essential ask-only slots become questions → no redundant questioning.
2. **Tools are shared between LLM and fallback.** The same tool registry powers the
   Anthropic tool-use loop and the offline deterministic planner, so behaviour and
   side-effects are identical with or without a key.
3. **Grounding is enforced.** Policy intents must cite a KB doc-id; the evaluator
   measures grounding rate.
4. **Memory is the differentiator.** `recall()` runs before every call; the opener is
   memory-aware; agent memory records which resolution paths actually work.
5. **Degrade gracefully.** Missing services, missing LLM key, missing voice deps — each
   has a fallback. This is a production-readiness stance, not a demo shortcut.
```
