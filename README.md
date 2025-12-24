# LocalPro

Windows quick start:
```powershell
.\start.ps1
```

Open: http://127.0.0.1:8000 (the script prints the port if it changes).
Stop: Ctrl+C in the same window.

Needs: Windows, Python 3.10+, LM Studio running with models loaded. Tavily is optional.
If `python` is not found, install from python.org and check "Add Python to PATH".
If scripts are blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Optional auto-reload:
```powershell
.\start.ps1 -Reload
```
