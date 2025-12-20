# LocalPro Multi-Agent Research App

Local “ChatGPT Pro”-style web app that orchestrates multiple LM Studio-hosted agents and Tavily live web research with full SQLite persistence.

## Features
- FastAPI backend with LM Studio (OpenAI-compatible) client and Tavily search/extract wrapper.
- Router → Orchestrator → 3 concurrent research lanes (A/B/C) → merge → draft → verifier loop.
- Live activity feed over Server-Sent Events (no chain-of-thought; operational summaries only).
- SQLite “information dump”: runs, messages, tasks, searches, extracts, sources, claims, drafts, verifier reports, events, configs.
- Simple SPA UI (vanilla HTML/JS/CSS) with chat history, toggles, evidence dump, and settings modal.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # edit values or use settings modal
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Configuration
- Default LM Studio base URL: `http://localhost:1234/v1`
- Models (adjust to what you loaded in LM Studio):
  - Orchestrator: `openai/gpt-oss-20b`
  - Worker/Verifier: `qwen/qwen3-v1-8b`
  - Router/Summarizer: any Qwen 4B (set in settings)
- Tavily API key: set in `.env` or via UI.
- Config saved to `config.json` and latest snapshot recorded in SQLite (`configs` table).

## Running a query
1. Open the web UI, enter a question.
2. Choose auto/manual reasoning level, search depth, max results, strict mode, evidence dump.
3. Submit. Activity feed streams router decision, lane searches/extracts, merge, verifier verdict, loops, archive.
4. Final answer shows confidence and sources. Toggle Evidence dump to see claims → URLs and excerpts.

## Acceptance behaviors
- Startup calls `/v1/models`; missing models are shown in the header warning.
- Router decides web need + reasoning level; web questions trigger 3 research lanes with Tavily search+extract stored.
- Simple math prompts route to non-web path and still run draft+verify.
- Evidence dump uses stored claims and URLs from SQLite.

## Troubleshooting
- LM Studio not running / wrong base URL: update in settings modal or `.env`, restart server.
- Model ID mismatch: load the expected model in LM Studio; warning appears in header.
- Tavily key missing/invalid: searches/extracts will return `missing_api_key` in DB; update key in settings.
- Concurrency queueing: LM Studio may serialize requests; lanes still run with asyncio gather—watch activity feed for queueing.
- Reset DB: stop server and delete `app_data.db` (or change `DATABASE_PATH`).

## Project layout
- `app/main.py` FastAPI entry, routes, SSE.
- `app/db.py` SQLite schema + helpers.
- `app/config.py` settings load/save.
- `app/llm.py` LM Studio client wrapper.
- `app/tavily.py` Tavily API wrapper.
- `app/orchestrator.py` orchestration loop + event bus.
- `app/agents.py` prompt profiles.
- `app/schemas.py` pydantic models.
- `app/web/static/` front-end assets.
