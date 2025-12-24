import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional, Set


def _bytes_to_gb(val: float) -> float:
    try:
        return round(float(val) / (1024**3), 2)
    except Exception:
        return 0.0


def _mb_to_gb(val: float) -> float:
    try:
        return round(float(val) / 1024, 2)
    except Exception:
        return 0.0


def get_resource_snapshot() -> Dict[str, Any]:
    """Return a lightweight snapshot of system RAM and GPU VRAM (if available)."""
    ram: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        ram = {
            "total_gb": _bytes_to_gb(mem.total),
            "available_gb": _bytes_to_gb(mem.available),
            "used_gb": _bytes_to_gb(mem.total - mem.available),
            "percent": round(float(mem.percent), 2),
        }
    except Exception:
        ram = {}

    gpus = []
    smi_path = shutil.which("nvidia-smi")
    if smi_path:
        try:
            cmd = [
                smi_path,
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
            raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=2)
            for line in raw.strip().splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 3:
                    total_mb = float(parts[1])
                    used_mb = float(parts[2])
                    total_gb = _mb_to_gb(total_mb)
                    used_gb = _mb_to_gb(used_mb)
                    gpus.append(
                        {
                            "name": parts[0],
                            "total_gb": total_gb,
                            "used_gb": used_gb,
                            "free_gb": max(total_gb - used_gb, 0.0),
                        }
                    )
        except Exception as exc:
            gpus.append({"error": str(exc)})

    return {"ram": ram, "gpus": gpus, "captured_at": datetime.utcnow().isoformat() + "Z"}


def _worker_variants(model_map: Dict[str, Dict[str, str]], availability: Optional[Dict[str, Any]]) -> int:
    """Count distinct worker-like model IDs that look usable."""
    roles = ("worker", "worker_b", "worker_c")
    configured: Set[str] = set()
    for role in roles:
        cfg = model_map.get(role) or {}
        mid = cfg.get("model")
        if mid:
            configured.add(mid)
    if availability:
        prefixes = {m.split(":")[0] for m in configured if m}
        discovered: Set[str] = set()
        for role in roles:
            info = availability.get(role) if isinstance(availability, dict) else None
            if not isinstance(info, dict):
                continue
            for mid in info.get("available", []) or []:
                if not mid:
                    continue
                prefix = mid.split(":")[0]
                if not prefixes or prefix in prefixes:
                    discovered.add(mid)
        if discovered:
            configured |= discovered
    return max(1, len(configured)) if configured else 1


def compute_worker_slots(
    model_map: Dict[str, Dict[str, str]],
    model_tier: str,
    availability: Optional[Dict[str, Any]] = None,
    resources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Estimate how many concurrent worker agents we should schedule, factoring in:
    - Configured worker endpoints (worker/worker_b/worker_c)
    - Detected variants from LM Studio availability data
    - RAM/VRAM headroom by tier
    """
    variants = _worker_variants(model_map, availability)
    worker_roles = ("worker", "worker_b", "worker_c")
    configured_slots = sum(
        1
        for role in worker_roles
        if (model_map.get(role) or {}).get("model") and (model_map.get(role) or {}).get("base_url")
    )
    if configured_slots <= 0:
        configured_slots = 1
    base_slots = max(variants, configured_slots)

    ram_slots: Optional[int] = None
    vram_slots: Optional[int] = None
    ram_pressure = False
    vram_pressure = False
    ram_headroom = 2.0 if model_tier == "fast" else 2.5 if model_tier == "deep" else 3.5
    vram_headroom = 2.0 if model_tier == "fast" else 3.0 if model_tier == "deep" else 4.5

    if resources:
        ram_info = resources.get("ram") or {}
        available_gb = ram_info.get("available_gb")
        if available_gb is not None:
            available_gb = float(available_gb)
            ram_slots = max(1, int(available_gb // ram_headroom))
            percent = ram_info.get("percent")
            if percent is not None:
                ram_pressure = float(percent) >= 90.0
            if available_gb < ram_headroom:
                ram_pressure = True
        gpu_list = [
            g for g in resources.get("gpus") or [] if isinstance(g, dict) and g.get("free_gb") is not None
        ]
        if gpu_list:
            free_gpu = max((g.get("free_gb") or 0.0 for g in gpu_list), default=0.0)
            vram_slots = max(1, int(float(free_gpu) // vram_headroom))
            if float(free_gpu) < vram_headroom:
                vram_pressure = True

    if ram_slots is None:
        ram_slots = base_slots
    if vram_slots is None:
        # If GPU info is missing, don't cap concurrency below RAM-derived capacity.
        vram_slots = ram_slots

    safe_cap = max(1, min(ram_slots, vram_slots))
    # Prefer configured worker slots unless memory pressure is high.
    if ram_pressure or vram_pressure:
        max_parallel = safe_cap
    else:
        max_parallel = max(base_slots, safe_cap)
    return {
        "max_parallel": max_parallel,
        "configured": configured_slots,
        "variants": variants,
        "ram_slots": ram_slots,
        "vram_slots": vram_slots,
        "ram_headroom_gb": ram_headroom,
        "vram_headroom_gb": vram_headroom,
        "ram_pressure": ram_pressure,
        "vram_pressure": vram_pressure,
    }
