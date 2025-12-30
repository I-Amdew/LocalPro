# Dynamic Model Management

LocalPro now discovers, profiles, and routes across models dynamically instead of binding fixed worker slots.

## What changes
- Models are discovered from enabled backends (LM Studio by default).
- Capabilities are profiled in the background (tool use, JSON schema, latency).
- Routing picks instances based on required capabilities and objectives.
- Autoscaling loads/unloads instances as backlog and resources change.
- Resource profiling loads one instance at a time, measures RAM/VRAM peaks, and runs a 5-test suite.

## LM Studio parallel instances
LocalPro relies on LM Studio identifiers to run multiple instances of the same model:
- `lms load <model_key> --identifier <unique_id> --ttl <seconds>`
- API calls use the identifier as the `model` value.

This enables parallel calls across N instances of the same model key.

## Settings highlights
`config.json` now includes:
- `backends.lmstudio`: host, port, CLI, TTL settings.
- `model_candidates`: `auto` / `allowlist` / `denylist`, plus `prefer`.
- `autoscaling`: global/per-backend caps and minimums.
- `profiling`: auto-profile toggle, repeats, and output token cap.
- `ram_headroom_pct` / `vram_headroom_pct`: enforce 10% headroom by default.

The UI shows discovered models with their profile metrics and allows prefer/deny toggles.

## Bootstrap behavior
On first run, LocalPro:
- Discovers available candidates.
- Starts a tool-capable model instance automatically.
- Continues profiling other models in the background.

## Model profiling
LocalPro profiles each model by loading a single instance, running a 5-test suite, and capturing RAM/VRAM peaks.
Profiles are cached per model/config signature and used for capacity planning and autoscaling.

CLI helpers:
```powershell
python .\localpro_cli.py models profile
python .\localpro_cli.py models profile --force --wait
```
