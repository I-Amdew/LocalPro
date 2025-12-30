# LocalPro

Windows quick start:
```powershell
.\start.ps1
```

Open: use the URL printed by the script (defaults to http://127.0.0.1:8000; if busy it picks the next free port).
Stop: Ctrl+C in the same window.

Local Wi-Fi/LAN access:
- Ensure `HOST=0.0.0.0` in `.env` (default) or `config.json` so the server binds to your LAN interface.
- Current LAN IP on this machine: `192.168.5.112` (use `http://192.168.5.112:8000` or the port printed by the script).
- If the IP changes, run `ipconfig` or `Get-NetIPAddress -AddressFamily IPv4` and use the IPv4 address for your active adapter.

Needs: Windows, Python 3.10+, LM Studio running with models loaded. Tavily is optional.
If `python` is not found, install from python.org and check "Add Python to PATH".
If scripts are blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Settings precedence:
- `config.json` (UI-saved settings) wins over `.env` by default.
- Set `LOCALPRO_ENV_OVERRIDES_CONFIG=1` to allow environment overrides (for deployments).

Optional auto-reload:
```powershell
.\start.ps1 -Reload
```

If you need to reclaim port 8000:
```powershell
.\start.ps1 -ForcePort
```

Live trace helper (CLI):
```powershell
# Start the app first, then run a prompt and stream the live events (human view by default).
python .\watch_run.py --strict --reasoning-mode manual --manual-level HIGH "Compare SQLite vs Postgres for run logs."

# Full debug event stream:
python .\watch_run.py --view debug --strict --reasoning-mode manual --manual-level HIGH "Compare SQLite vs Postgres for run logs."

# If the app is running on a non-default port:
python .\watch_run.py --base-url http://127.0.0.1:8001 "Your prompt here"

# Attach to a run you started in the UI:
python .\watch_run.py --run-id <RUN_ID>
```

Model profiling:
```powershell
python .\localpro_cli.py models profile
python .\localpro_cli.py models profile --force --wait
```

Accuracy spot-check:
```powershell
python .\accuracy_benchmark.py --port 8000 --model-tier pro
```

Live plan edits (API):
```powershell
# Add constraints without stopping the run.
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/run/<RUN_ID>/control `
  -ContentType "application/json" `
  -Body '{"new_constraints":{"focus":"Prioritize primary sources and flag assumptions"}}'

# Append a new step.
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/run/<RUN_ID>/control `
  -ContentType "application/json" `
  -Body '{"control":"ADD_STEPS","steps":[{"name":"Scan for recent reports","type":"research","agent_profile":"ResearchRecency","depends_on":[1]}]}'
```

Planning overhaul (v2):
- See `docs/plan_overhaul.md` for PlanStore views, executor scheduling, and prereq resolution.
- Extensive modes (plan granularity + expansion):
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/run `
  -ContentType "application/json" `
  -Body '{"question":"List every district and summarize key stats.","model_tier":"pro","plan_reasoning_mode":"extensive","planning_mode":"extensive","reasoning_level":4}'
```

Dynamic model management:
- See `docs/model_management.md` for discovery, profiling, autoscaling, and LM Studio parallel instance details.
