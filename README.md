# LocalPro Multi-Agent Research App

Local "ChatGPT Pro"-style web app with LM Studio-hosted agents, Tavily search, and SQLite persistence.

## Quick start (from repo root)
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill Tavily key + LM Studio URLs/models or use Settings UI
python -m app.main  # binds 0.0.0.0 so phones on Wi-Fi can reach it
# or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# (if you previously ran uvicorn without --host, stop that process first)
```

### PowerShell variant
```powershell
py -3 -m venv .venv  # or python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m app.main  # binds 0.0.0.0 so phones on Wi-Fi can reach it
# or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# (if you previously ran uvicorn without --host, stop that process first)
```
Notes: if activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that shell. If `python` opens the Store, use `py -3`.

## Open it
- Desktop: http://127.0.0.1:8000
- Same Wi-Fi: http://<your-lan-ip>:8000 (get IP via `ipconfig` or `ifconfig`)
- Allow inbound on port 8000 in Windows Firewall for Private networks if you want LAN/mobile access.

## Minimal config
- `.env` / Settings modal: base URLs + model IDs per role, Tavily API key. Defaults target LM Studio at `http://localhost:1234/v1`.
- Server bind is controlled by `HOST`/`PORT` (`HOST=0.0.0.0` by default so LAN/mobile can connect).
- DB lives in `app_data.db`; delete to reset.

## Features (short list)
- Step-based micromanager orchestrator with reruns/backtracking and depth control.
- Per-role LM Studio endpoints + Tavily search/extract; live activity feed via SSE.
- Model tiers: LocalFast (8B linear + tools), LocalDeep (auto chooses OSS vs. Mini Pro lane or lock it manually), LocalAuto (automatically routes each question to the tier that matches its detail level), LocalPro (full OSS + four-level reasoning slider).
- Memory panel with retrieval/save, speech-to-text input, simple SPA UI.
- Uploads: drop images/PDFs on desktop or mobile; 8B vision labels them and 4B secretary summarizes into the plan.
