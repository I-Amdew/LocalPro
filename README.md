# LocalPro Multi-Agent Research App

Local "ChatGPT Pro"-style web app that orchestrates multiple LM Studio-hosted agents and Tavily live web research with full SQLite persistence. Upgraded to a micromanager, step-based orchestrator that can backtrack/rerun steps, adjust reasoning depth, switch model endpoints, persist memory, and support speech-to-text input.

## Features
- FastAPI backend with configurable LM Studio endpoints (per role) and Tavily search/extract wrapper.
- Micromanager Orchestrator (GPT-OSS-20B) builds a step plan (task graph), executes step-by-step with backtracking/reruns/added substeps; budgets and step counts respect reasoning depth (LOW/MED/HIGH/ULTRA).
- Field agents (Qwen3/Qwen4 profiles: primary/recency/adversarial research, math, critic, summarizer, JSON repair, verifier).
- Memory system: durable memory items, auto-memory toggle, retrieval prior to planning, memory panel with search/pin/edit/delete.
- Live activity feed over SSE with operational summaries only (no chain-of-thought); events include memory retrieval/save and control actions.
- Speech-to-text input (Web Speech API) with language selection.
- SQLite "information dump": runs, messages, tasks, searches, extracts, sources, claims, drafts, verifier reports, events, configs, step plans, step runs, artifacts, control actions, memory tables.
- Simple SPA UI (vanilla HTML/JS/CSS) with chat bubbles, collapsible activity/evidence/memory panels, settings modal, hotkeys, and reasoning depth slider.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # edit values or use settings modal
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Configuration
- Per-role endpoints + model IDs (set via Settings UI):
  - Orchestrator, Worker A/B/C, Router, Summarizer, Verifier (each base URL + model id)
  - Discovery mode: provide base URLs; app calls `/v1/models` and shows results.
- Legacy defaults: `http://localhost:1234/v1`, orchestrator `openai/gpt-oss-20b`, worker `qwen/qwen3-v1-8b`, router/summarizer `qwen/qwen-4b`.
- Tavily API key: set in `.env` or via UI.
- Settings are saved to `config.json` and snapshotted in SQLite (`configs` table).

## Reasoning depth (AUTO or manual)
- UI slider + AUTO toggle. When AUTO: Router chooses depth; otherwise user selection wins.
- Depth mapping (deterministic):
  - LOW: ~6 steps, 1 research round, search budget ~4, extracts 3–6
  - MED: ~10 steps, 2 research rounds, moderate budget, extracts 6–10
  - HIGH: ~14 steps, 3 rounds, advanced preferred, extracts 10–16
  - ULTRA: ~18–24 steps, 3 rounds, advanced required, extracts 16–24, strict verify looping
- Depth affects: plan size, tool budgets, verification strictness, retry/backtrack thresholds, search depth defaults.

## Memory
- Tables: `memory_items`, `run_memory_links`.
- Retrieval before planning (keyword/tag/recency) with `memory_retrieved` event; injected into planning context.
- Auto-memory toggle: saves summarized answer into memory, links to run, emits `memory_saved`.
- Memory panel: search, refresh, pin/unpin, delete, edit (quick update via pin toggle + delete).

## UI
- Chat bubbles, Enter to send (Shift+Enter newline), Ctrl+K opens settings.
- Speech-to-text button (browser Web Speech API) with language input.
- Collapsible Activity feed, Evidence dump, Memory panel.
- Settings modal for endpoints, models, discovery, and Tavily key.

## Running a query
1. Open the web UI, enter a question (or dictate via mic).
2. Choose AUTO/manual reasoning depth, search depth, max results, strict mode, auto-memory, evidence dump.
3. Submit. Activity feed streams router decision, plan creation, per-step start/finish, Tavily queries/extracts, control actions (rerun/backtrack), memory events, verifier verdict, loops, archive.
4. Final answer shows confidence and sources. Evidence dump toggle shows claims -> URLs/excerpts.

## Acceptance behaviors / manual checks
- Changing reasoning depth in UI changes plan size/tool budgets (observe activity steps and search/extract counts).
- Switching model base URLs/IDs via settings works without restart; discovery lists models per base URL.
- Speech-to-text inserts transcript into the message box (or shows unsupported warning).
- Memory panel shows saved items; retrieval events appear and influence planning context.
- Startup validation calls `/v1/models` per configured endpoint; UI shows missing models.

## Troubleshooting
- LM Studio not running / wrong base URL: update in settings modal or `.env`, restart server.
- Model ID mismatch: load the expected model in LM Studio; warnings appear in header.
- Tavily key missing/invalid: searches/extracts will return `missing_api_key` in DB; update key in settings.
- Concurrency queueing: LM Studio may serialize requests; watch activity feed for queued steps or replays.
- Reset DB: stop server and delete `app_data.db` (or change `DATABASE_PATH`).

## Project layout
- `app/main.py` FastAPI entry, routes, SSE, discovery, memory endpoints.
- `app/db.py` SQLite schema + helpers (runs, steps, artifacts, memory, etc.).
- `app/config.py` settings load/save with per-role endpoints.
- `app/llm.py` LM Studio client wrapper (override base_url per call).
- `app/tavily.py` Tavily API wrapper.
- `app/orchestrator.py` micromanager orchestration loop + event bus + depth budgets.
- `app/agents.py` prompt profiles.
- `app/schemas.py` pydantic models.
- `app/web/static/` front-end assets (HTML/CSS/JS).
