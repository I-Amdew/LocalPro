import threading
from typing import Any, Dict, Optional

from .resource_telemetry import ResourceTelemetry


def _gb_to_mb(value: float) -> float:
    return float(value or 0.0) * 1024.0


def _mb_to_bytes(value: float) -> float:
    return float(value or 0.0) * 1024.0 * 1024.0


def _bytes_to_mb(value: float) -> float:
    return float(value or 0.0) / (1024.0 * 1024.0)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class ResourceBackend:
    def __init__(self, telemetry: Optional[ResourceTelemetry] = None) -> None:
        self.telemetry = telemetry or ResourceTelemetry()

    def snapshot(self) -> Dict[str, Any]:
        return self.telemetry.snapshot()


class ResourceManager:
    """Track resource reservations and enforce headroom limits."""

    def __init__(
        self,
        backend: Optional[ResourceBackend] = None,
        ram_headroom_pct: float = 10.0,
        vram_headroom_pct: float = 10.0,
        max_concurrent_runs: Optional[int] = None,
        per_model_class_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        self.backend = backend or ResourceBackend()
        self.ram_headroom_pct = ram_headroom_pct
        self.vram_headroom_pct = vram_headroom_pct
        self.max_concurrent_runs = max_concurrent_runs
        self.per_model_class_limits = per_model_class_limits or {}
        self._lock = threading.Lock()
        self._reservations: Dict[str, Dict[str, Any]] = {}

    def snapshot(self) -> Dict[str, Any]:
        snap = self.backend.snapshot()
        with self._lock:
            snap["reservations"] = dict(self._reservations)
        return snap

    def model_profile(self, model_class_or_id: str) -> Dict[str, Any]:
        # Heuristic estimates; can be overridden by config externally.
        base = str(model_class_or_id or "").lower()
        if "cheap" in base:
            return {"vram_est": 1500, "cpu_est": 10, "tps_est": 30}
        if "strong" in base:
            return {"vram_est": 4000, "cpu_est": 25, "tps_est": 12}
        if "summarizer" in base:
            return {"vram_est": 2000, "cpu_est": 15, "tps_est": 25}
        if "synth" in base:
            return {"vram_est": 4500, "cpu_est": 30, "tps_est": 10}
        return {"vram_est": 2500, "cpu_est": 15, "tps_est": 20}

    def reserve(self, run_id: str, budgets: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self.max_concurrent_runs is not None and len(self._reservations) >= self.max_concurrent_runs:
                return {"granted": False, "reason": "max_concurrent_runs"}
            model_class = budgets.get("model_class")
            if model_class:
                limit = self.per_model_class_limits.get(str(model_class))
                if limit is not None:
                    active = sum(
                        1 for res in self._reservations.values() if res.get("model_class") == model_class
                    )
                    if active >= limit:
                        return {"granted": False, "reason": "model_class_limit"}
            snapshot = self.backend.snapshot()
            gpu_id = budgets.get("gpu_id")
            vram_mb = float(budgets.get("vram_mb") or 0.0)
            ram_bytes = float(budgets.get("ram_bytes") or 0.0)
            if ram_bytes <= 0 and budgets.get("ram_mb") is not None:
                ram_bytes = _mb_to_bytes(_to_float(budgets.get("ram_mb")))
            if vram_mb > 0 and snapshot.get("gpus"):
                gpus = snapshot.get("gpus") or []
                target_gpu = None
                if gpu_id is not None:
                    for gpu in gpus:
                        try:
                            if int(gpu.get("gpu_id") or 0) == int(gpu_id):
                                target_gpu = gpu
                                break
                        except Exception:
                            continue
                if target_gpu is None:
                    target_gpu = gpus[0]
                total_mb = _to_float(target_gpu.get("vram_total_mb"))
                if total_mb <= 0:
                    total_mb = _gb_to_mb(_to_float(target_gpu.get("total_gb")))
                used_mb = _to_float(target_gpu.get("vram_used_mb"))
                if used_mb <= 0 and target_gpu.get("used_gb") is not None:
                    used_mb = _gb_to_mb(_to_float(target_gpu.get("used_gb")))
                reserved_mb = sum(
                    _to_float(res.get("vram_mb") or 0.0)
                    for res in self._reservations.values()
                    if res.get("gpu_id") == gpu_id
                )
                baseline_used = max(used_mb, reserved_mb)
                allowed_used = total_mb * (1.0 - (self.vram_headroom_pct / 100.0))
                if total_mb > 0 and (baseline_used + vram_mb) > allowed_used:
                    return {"granted": False, "reason": "vram_headroom"}
            if ram_bytes > 0 and snapshot.get("ram"):
                ram_info = snapshot.get("ram") or {}
                total_bytes = _to_float(ram_info.get("total_bytes"))
                used_bytes = _to_float(ram_info.get("used_bytes"))
                if total_bytes <= 0 and ram_info.get("total_gb") is not None:
                    total_bytes = _mb_to_bytes(_gb_to_mb(_to_float(ram_info.get("total_gb"))))
                if used_bytes <= 0 and ram_info.get("used_gb") is not None:
                    used_bytes = _mb_to_bytes(_gb_to_mb(_to_float(ram_info.get("used_gb"))))
                reserved_ram = sum(
                    _mb_to_bytes(_to_float(res.get("ram_mb")))
                    if res.get("ram_mb") is not None
                    else _to_float(res.get("ram_bytes") or 0.0)
                    for res in self._reservations.values()
                )
                baseline_used = max(used_bytes, reserved_ram)
                allowed_used = total_bytes * (1.0 - (self.ram_headroom_pct / 100.0))
                if total_bytes > 0 and (baseline_used + ram_bytes) > allowed_used:
                    return {"granted": False, "reason": "ram_headroom"}
            self._reservations[run_id] = dict(budgets)
            return {"granted": True}

    def release(self, run_id: str) -> None:
        with self._lock:
            self._reservations.pop(run_id, None)
