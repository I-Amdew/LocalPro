# Dynamic Backend Management

LocalPro discovers, profiles, and routes across compute backends dynamically instead of binding fixed worker slots.

## What changes
- Backends are discovered from enabled runtimes (LM Studio by default).
- Capabilities are profiled in the background (tool use, JSON schema, latency).
- Routing picks instances based on required capabilities and objectives.
- Autoscaling loads/unloads instances as backlog and resources change.
- Resource profiling loads one instance at a time, measures RAM/VRAM peaks, and runs a 5-test suite.

## LM Studio parallel instances
LocalPro relies on LM Studio identifiers to run multiple instances of the same runtime model:
- `lms load <model_key> --identifier <unique_id> --ttl <seconds>`
- API calls use the identifier as the `model` value.

This enables parallel calls across N instances of the same backend key.

## Settings highlights
`config.json` includes:
- `backends.lmstudio`: host, port, CLI, TTL settings.
- `model_candidates`: `auto` / `allowlist` / `denylist`, plus `prefer`.
- `autoscaling`: global/per-backend caps and minimums.
- `profiling`: auto-profile toggle, repeats, and output token cap.
- `ram_headroom_pct` / `vram_headroom_pct`: enforce 10% headroom by default.

The UI shows discovered backends with profile metrics and allows prefer/deny toggles.

## Bootstrap behavior
On first run, LocalPro:
- Discovers available candidates.
- Starts a tool-capable instance automatically.
- Continues profiling other instances in the background.

## Profiling
LocalPro profiles each backend by loading a single instance, running a 5-test suite, and capturing RAM/VRAM peaks.
Profiles are cached per backend/config signature and used for capacity planning and autoscaling.

CLI helpers:
```powershell
python .\localpro_cli.py models profile
python .\localpro_cli.py models profile --force --wait
```