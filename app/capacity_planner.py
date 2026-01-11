import math
from typing import Any, Dict, Optional

from .resource_profiles import ModelResourceProfileStore
from .resource_telemetry import ResourceTelemetry


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _baseline_vram_used(profile: Dict[str, Any]) -> Dict[int, float]:
    baseline = profile.get("baseline_snapshot") or {}
    gpus = baseline.get("gpus") or []
    used: Dict[int, float] = {}
    for gpu in gpus:
        try:
            idx = int(gpu.get("gpu_id") or 0)
        except Exception:
            idx = 0
        used[idx] = _safe_float(gpu.get("vram_used_mb"))
    return used


def _baseline_ram_used(profile: Dict[str, Any]) -> float:
    baseline = profile.get("baseline_snapshot") or {}
    return _safe_float(baseline.get("ram_used_bytes"))


def _total_ram_bytes(snapshot: Dict[str, Any], profile: Dict[str, Any]) -> float:
    ram = snapshot.get("ram") or {}
    total = _safe_float(ram.get("total_bytes"))
    if total:
        return total
    baseline = profile.get("baseline_snapshot") or {}
    return _safe_float(baseline.get("ram_total_bytes"))


def _gpu_totals(snapshot: Dict[str, Any]) -> Dict[int, float]:
    totals: Dict[int, float] = {}
    for gpu in snapshot.get("gpus") or []:
        try:
            idx = int(gpu.get("gpu_id") or 0)
        except Exception:
            idx = 0
        totals[idx] = _safe_float(gpu.get("vram_total_mb"))
    return totals


def _sanitize_vram_peaks(profile: Dict[str, Any], totals: Dict[int, float]) -> Dict[str, float]:
    vram_peak_by_gpu = profile.get("vram_instance_peak_mb_by_gpu") or {}
    load_delta = profile.get("load_delta_vram_mb_by_gpu") or {}
    sanitized: Dict[str, float] = {}
    for gid_str, peak_val in vram_peak_by_gpu.items():
        try:
            gid = int(gid_str)
        except Exception:
            gid = 0
        peak = _safe_float(peak_val, 0.0)
        total = totals.get(gid, 0.0)
        fallback = _safe_float(load_delta.get(str(gid)) or load_delta.get(gid), 0.0)
        if fallback > 0.0 and peak > (fallback * 1.5):
            peak = fallback
        if total and peak > (total * 2.0):
            peak = fallback if fallback > 0.0 else 0.0
        if peak > 0.0:
            sanitized[str(gid)] = peak
    if not sanitized and profile.get("vram_estimate_only_mb") is not None:
        sanitized = {"0": _safe_float(profile.get("vram_estimate_only_mb"), 0.0)}
    return sanitized


class CapacityPlanner:
    def __init__(
        self,
        *,
        telemetry: ResourceTelemetry,
        store: ModelResourceProfileStore,
        ram_headroom_pct: float = 10.0,
        vram_headroom_pct: float = 10.0,
    ) -> None:
        self.telemetry = telemetry
        self.store = store
        self.ram_headroom_pct = ram_headroom_pct
        self.vram_headroom_pct = vram_headroom_pct

    async def capacity_for(
        self,
        backend_id: str,
        model_key: str,
        config_signature: str,
    ) -> Optional[Dict[str, Any]]:
        profile = await self.store.get(backend_id, model_key, config_signature)
        if not profile:
            return None
        return self.compute_capacity(profile, self.telemetry.snapshot())

    def compute_capacity(self, profile: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
        ram_peak = _safe_float(profile.get("ram_instance_peak_bytes"))
        load_delta_ram = _safe_float(profile.get("load_delta_ram_bytes"))
        totals = _gpu_totals(snapshot)
        vram_peak_by_gpu = _sanitize_vram_peaks(profile, totals)
        baseline_ram_used = _baseline_ram_used(profile)
        baseline_vram_used = _baseline_vram_used(profile)
        total_ram = _total_ram_bytes(snapshot, profile)
        allowed_ram = total_ram * (1.0 - (self.ram_headroom_pct / 100.0)) if total_ram else 0.0
        usable_ram = allowed_ram - baseline_ram_used
        if load_delta_ram > 0.0 and (ram_peak <= 0.0 or ram_peak > (load_delta_ram * 4.0)):
            ram_peak = load_delta_ram
        if usable_ram > 0.0 and ram_peak > usable_ram:
            if load_delta_ram > 0.0 and load_delta_ram <= usable_ram:
                ram_peak = load_delta_ram
            else:
                ram_peak = usable_ram
        max_by_ram: Optional[int] = None
        if ram_peak > 0:
            if usable_ram > 0:
                max_by_ram = max(int(math.floor(usable_ram / ram_peak)), 0)
            else:
                max_by_ram = 0

        max_instances_by_gpu: Dict[str, int] = {}
        if totals and vram_peak_by_gpu:
            for gid_str, peak in vram_peak_by_gpu.items():
                try:
                    gid = int(gid_str)
                except Exception:
                    gid = 0
                total = totals.get(gid, 0.0)
                allowed_vram = total * (1.0 - (self.vram_headroom_pct / 100.0)) if total else 0.0
                usable_vram = allowed_vram - baseline_vram_used.get(gid, 0.0)
                max_by_vram = 0
                if peak and usable_vram > 0:
                    max_by_vram = max(int(math.floor(usable_vram / peak)), 0)
                if max_by_ram is None:
                    max_instances_by_gpu[str(gid)] = max_by_vram
                else:
                    max_instances_by_gpu[str(gid)] = max(0, min(max_by_vram, max_by_ram))
        elif max_by_ram is not None:
            max_instances_by_gpu["ram_only"] = max_by_ram

        max_instances = max(max_instances_by_gpu.values()) if max_instances_by_gpu else 0
        return {
            "max_instances": max_instances,
            "max_instances_by_gpu": max_instances_by_gpu,
            "max_by_ram": max_by_ram if max_by_ram is not None else 0,
            "ram_instance_peak_bytes": ram_peak,
            "vram_instance_peak_mb_by_gpu": vram_peak_by_gpu,
        }
