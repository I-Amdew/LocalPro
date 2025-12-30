import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from .resource_telemetry import ResourceTelemetry


# No fixed minimum; parallelism adapts to live resource headroom.
MIN_PARALLEL_SLOTS = 1
_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b", re.IGNORECASE)


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


def _model_size_hint(model_id: str) -> Optional[float]:
    if not model_id:
        return None
    match = _MODEL_SIZE_RE.search(model_id)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def get_resource_snapshot() -> Dict[str, Any]:
    """Return a lightweight snapshot of system RAM and GPU VRAM (if available)."""
    telemetry = ResourceTelemetry()
    snapshot = telemetry.snapshot()
    ram: Dict[str, Any] = {}
    ram_raw = snapshot.get("ram") or {}
    if ram_raw:
        total_bytes = ram_raw.get("total_bytes") or 0
        available_bytes = ram_raw.get("available_bytes") or 0
        used_bytes = ram_raw.get("used_bytes") or 0
        ram = {
            "total_gb": _bytes_to_gb(total_bytes),
            "available_gb": _bytes_to_gb(available_bytes),
            "used_gb": _bytes_to_gb(used_bytes),
            "percent": round(float(ram_raw.get("used_pct") or 0.0), 2),
        }

    process_snapshot: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        processes = []
        total_rss = 0
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                info = proc.info
                mem_info = info.get("memory_info")
                rss = getattr(mem_info, "rss", 0) if mem_info else 0
            except Exception:
                continue
            total_rss += rss or 0
            processes.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "rss_gb": _bytes_to_gb(rss or 0),
                }
            )
        processes.sort(key=lambda item: item.get("rss_gb") or 0, reverse=True)
        process_snapshot = {
            "count": len(processes),
            "rss_total_gb": _bytes_to_gb(total_rss),
            "top": processes[:5],
        }
    except Exception:
        process_snapshot = {}

    gpus = []
    for gpu in snapshot.get("gpus") or []:
        total_mb = gpu.get("vram_total_mb") or 0.0
        used_mb = gpu.get("vram_used_mb") or 0.0
        total_gb = _mb_to_gb(total_mb)
        used_gb = _mb_to_gb(used_mb)
        gpus.append(
            {
                "name": gpu.get("name"),
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": max(total_gb - used_gb, 0.0),
                "gpu_id": gpu.get("gpu_id"),
                "vram_total_mb": total_mb,
                "vram_used_mb": used_mb,
            }
        )

    return {
        "ram": ram,
        "gpus": gpus,
        "processes": process_snapshot,
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


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
    Estimate how many concurrent worker agents we should schedule, primarily from:
    - RAM/VRAM headroom by tier
    - Configured worker endpoints (worker/worker_b/worker_c) as a fallback when resources are unknown
    - Detected variants from LM Studio availability data
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
    size_hints = [
        hint
        for role in worker_roles
        for hint in (_model_size_hint((model_map.get(role) or {}).get("model") or ""),)
        if hint
    ]
    if size_hints:
        max_size = max(size_hints)
        if max_size <= 4:
            ram_headroom = min(ram_headroom, 2.0)
            vram_headroom = min(vram_headroom, 2.5)
        elif max_size <= 8:
            ram_headroom = min(ram_headroom, 2.8)
            vram_headroom = min(vram_headroom, 3.25)

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

    if ram_slots is None and vram_slots is None:
        ram_slots = base_slots
        vram_slots = base_slots
        max_parallel = base_slots
    else:
        if ram_slots is None:
            ram_slots = base_slots
        if vram_slots is None:
            # If GPU info is missing, don't cap concurrency below RAM-derived capacity.
            vram_slots = ram_slots
        safe_cap = max(1, min(ram_slots, vram_slots))
        max_parallel = safe_cap
    resource_capped = bool(resources and (resources.get("ram") or resources.get("gpus")))
    if MIN_PARALLEL_SLOTS > 1 and max_parallel < MIN_PARALLEL_SLOTS and not resource_capped:
        max_parallel = MIN_PARALLEL_SLOTS
        ram_slots = max(ram_slots or 0, MIN_PARALLEL_SLOTS)
        vram_slots = max(vram_slots or 0, MIN_PARALLEL_SLOTS)
        ram_pressure = True
        vram_pressure = True
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
