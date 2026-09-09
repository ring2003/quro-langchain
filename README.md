# Quro-Thinking MVP

Minimal verification that a **Reasoning Session** can exist independently from a
Conversation Session.  Built with LangGraph without `MessagesState`.

---

## Quick Start

### Prerequisites

- Python >= 3.11
- An OpenAI-compatible API (OpenAI, Ollama, vLLM, LM Studio, OpenRouter…)

### Setup

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install package and dev dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
```

Edit `.env` — set at least `QURO_API_KEY`:

```ini
QURO_API_KEY=sk-your-key-here
QURO_BASE_URL=https://api.openai.com/v1      # or Ollama http://localhost:11434/v1
QURO_MODEL=gpt-4o-mini                        # or qwen2.5:7b, deepseek-r1, etc.
QURO_TEMPERATURE=0
QURO_MAX_STEPS=5
```

### Run Tests (offline, no API key needed)

```bash
pytest
```

All P0 tests use `FakeBackend` — no network calls, runs in < 1 s.

### Run MVP (requires a working LLM endpoint)

```bash
python scripts/run_mvp.py --problem-file examples/mvp/pr_fail.txt
```

Or pass the problem inline:

```bash
python scripts/run_mvp.py --problem "Why does my PR fail CI?"
```

The console will print:

- Step-by-step Planner / Worker call traces (first 300 chars of each response)
- A summary with final status, step count, conclusions, and final answer

---

## Switching Models

Change **only** environment variables — no code changes needed:

| Provider | `QURO_BASE_URL` | Example `QURO_MODEL` |
|----------|-----------------|----------------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4o` |
| Ollama (local) | `http://localhost:11434/v1` | `qwen2.5:7b`, `deepseek-r1:8b` |
| vLLM (self-hosted) | `http://your-host:8000/v1` | `meta-llama/Llama-3.1-8B` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-3.5-sonnet` |
| LM Studio | `http://localhost:1234/v1` | model loaded in UI |

Set `QURO_API_KEY` to the corresponding key (Ollama and LM Studio may accept
any dummy value).

---

## Architecture

```
Problem
  → Planner (maintains Reasoning plain-text state)
  → Worker (temporary conversation, returns only Conclusion)
  → Planner (updates Reasoning, decides continue / finish)
  → … / finish → Final Answer
```

### Layers

| Layer | Package | Responsibility |
|-------|---------|---------------|
| **Domain** | `quro/core/` | PlainText protocol, Reasoning session, Worker session |
| **Runtime** | `quro/runtime/` | `IRuntimeBackend` — pluggable LLM backend |
| **Graph** | `quro/graph/` | LangGraph `StateGraph` with custom `ReasoningState` |
| **Config** | `quro/config/` | Environment → `Settings` |
| **App** | `quro/app/` | Wiring: config + backend + graph → `run_mvp()` |
| **CLI** | `scripts/run_mvp.py` | Argument parsing and entry point |

### Design Principles

- **Reasoning ≠ Conversation** — global state is PlainText, not `Messages`
- **Worker is disposable** — only the Conclusion survives; the conversation is
  destroyed
- **Backend is pluggable** — switch model provider via `QURO_BASE_URL` only
- **`core/` avoids LangGraph** — can be reused with a different orchestrator

---

## Project Status

MVP phases 0–5 complete. See [docs/mvp/implementation.md](docs/mvp/implementation.md)
for the full spec.
