import asyncio
import uuid
from typing import Any, Dict, List, Optional

from .resource_telemetry import ResourceTelemetry


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _ram_bytes_from_budget(budgets: Dict[str, Any]) -> float:
    if budgets.get("ram_bytes") is not None:
        return _safe_float(budgets.get("ram_bytes"))
    if budgets.get("ram_mb") is not None:
        return _safe_float(budgets.get("ram_mb")) * 1024.0 * 1024.0
    return 0.0


def _vram_mb_from_budget(budgets: Dict[str, Any]) -> float:
    if budgets.get("vram_mb") is not None:
        return _safe_float(budgets.get("vram_mb"))
    return 0.0


class SpawnGovernor:
    def __init__(
        self,
        *,
        telemetry: ResourceTelemetry,
        ram_headroom_pct: float = 10.0,
        vram_headroom_pct: float = 10.0,
        max_concurrent_loads: int = 1,
    ) -> None:
        self.telemetry = telemetry
        self.ram_headroom_pct = ram_headroom_pct
        self.vram_headroom_pct = vram_headroom_pct
        self._load_sem = asyncio.Semaphore(max(1, max_concurrent_loads))

    def _headroom_allows(
        self,
        snapshot: Dict[str, Any],
        *,
        add_ram_bytes: float = 0.0,
        add_vram_mb: float = 0.0,
        gpu_id: Optional[int] = None,
    ) -> bool:
        ram = snapshot.get("ram") or {}
        total_ram = _safe_float(ram.get("total_bytes"))
        used_ram = _safe_float(ram.get("used_bytes"))
        if total_ram:
            allowed = total_ram * (1.0 - (self.ram_headroom_pct / 100.0))
            if used_ram + add_ram_bytes > allowed:
                return False
        gpus = snapshot.get("gpus") or []
        if not gpus:
            return True
        for gpu in gpus:
            try:
                idx = int(gpu.get("gpu_id") or 0)
            except Exception:
                idx = 0
            if gpu_id is not None and idx != gpu_id:
                continue
            total = _safe_float(gpu.get("vram_total_mb"))
            used = _safe_float(gpu.get("vram_used_mb"))
            if total:
                allowed = total * (1.0 - (self.vram_headroom_pct / 100.0))
                if used + add_vram_mb > allowed:
                    return False
        return True

    async def spawn_instance(
        self,
        *,
        backend: Any,
        model_key: str,
        opts: Dict[str, Any],
        budgets: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        budgets = budgets or {}
        snapshot = self.telemetry.snapshot()
        add_ram = _ram_bytes_from_budget(budgets)
        add_vram = _vram_mb_from_budget(budgets)
        gpu_id = None
        if budgets.get("gpu_id") is not None:
            try:
                gpu_id = int(budgets.get("gpu_id"))
            except Exception:
                gpu_id = None
        if not self._headroom_allows(snapshot, add_ram_bytes=add_ram, add_vram_mb=add_vram, gpu_id=gpu_id):
            return None
        async with self._load_sem:
            try:
                if not opts.get("identifier"):
                    opts["identifier"] = f"{model_key}-{uuid.uuid4().hex[:8]}"
                instance = await backend.load_instance(model_key, opts)
            except Exception:
                return None
        post_load = self.telemetry.snapshot()
        if not self._headroom_allows(post_load, add_ram_bytes=0.0, add_vram_mb=0.0, gpu_id=gpu_id):
            try:
                await backend.unload_instance(getattr(instance, "instance_id", instance))
            except Exception:
                pass
            return None
        return instance

    async def scale_up(
        self,
        *,
        backend: Any,
        model_key: str,
        opts: Dict[str, Any],
        count: int,
        budgets: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        instances = []
        for _ in range(max(0, count)):
            inst = await self.spawn_instance(
                backend=backend,
                model_key=model_key,
                opts=dict(opts),
                budgets=budgets,
            )
            if not inst:
                break
            instances.append(inst)
        return instances
