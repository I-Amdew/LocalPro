import json
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _parse_mb(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        text = str(val)
        num = ""
        for ch in text:
            if ch.isdigit() or ch == ".":
                num += ch
            elif num:
                break
        return float(num) if num else None


def _gpu_snapshot_nvml() -> Optional[List[Dict[str, Any]]]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    gpus: List[Dict[str, Any]] = []
    try:
        count = int(pynvml.nvmlDeviceGetCount())
        for idx in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                util = None
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except Exception:
                    util = None
                total_mb = _to_float(mem.total) / (1024.0 * 1024.0)
                used_mb = _to_float(mem.used) / (1024.0 * 1024.0)
                gpus.append(
                    {
                        "gpu_id": idx,
                        "name": str(name) if name else None,
                        "vram_total_mb": total_mb,
                        "vram_used_mb": used_mb,
                        "vram_free_mb": max(total_mb - used_mb, 0.0),
                        "util": _to_float(util) if util is not None else None,
                    }
                )
            except Exception:
                continue
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpus


def _gpu_snapshot_nvidia_smi() -> Optional[List[Dict[str, Any]]]:
    smi_path = shutil.which("nvidia-smi")
    if not smi_path:
        return None
    queries = [
        "index,name,memory.total,memory.used,utilization.gpu",
        "index,name,memory.total,memory.used",
        "name,memory.total,memory.used",
    ]
    raw = None
    for query in queries:
        try:
            cmd = [smi_path, f"--query-gpu={query}", "--format=csv,noheader,nounits"]
            raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=2)
            if raw:
                break
        except Exception:
            raw = None
    if not raw:
        return None
    gpus: List[Dict[str, Any]] = []
    for idx, line in enumerate(raw.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        if len(parts) >= 4 and parts[0].isdigit():
            gpu_id = int(parts[0])
            name = parts[1]
            total = _parse_mb(parts[2]) or 0.0
            used = _parse_mb(parts[3]) or 0.0
            util = _parse_mb(parts[4]) if len(parts) >= 5 else None
        else:
            gpu_id = idx
            name = parts[0]
            total = _parse_mb(parts[1]) or 0.0
            used = _parse_mb(parts[2]) or 0.0
            util = _parse_mb(parts[3]) if len(parts) >= 4 else None
        gpus.append(
            {
                "gpu_id": gpu_id,
                "name": name or None,
                "vram_total_mb": _to_float(total),
                "vram_used_mb": _to_float(used),
                "vram_free_mb": max(_to_float(total) - _to_float(used), 0.0),
                "util": _to_float(util) if util is not None else None,
            }
        )
    return gpus


def _gpu_snapshot_rocm_smi() -> Optional[List[Dict[str, Any]]]:
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    try:
        raw = subprocess.check_output(
            [exe, "--showmeminfo", "vram", "--showproductname", "--json"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=2,
        )
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    gpus: List[Dict[str, Any]] = []
    for key, info in payload.items():
        if not isinstance(info, dict):
            continue
        name = (
            info.get("Card series")
            or info.get("Card Series")
            or info.get("Card model")
            or info.get("Card Model")
            or info.get("Card type")
            or info.get("Card Type")
            or info.get("Product Name")
        )
        total = None
        used = None
        for k, v in info.items():
            label = str(k).lower()
            if "vram total" in label and total is None:
                total = _parse_mb(v)
            if "vram used" in label and used is None:
                used = _parse_mb(v)
        if total is None and used is None:
            continue
        gpu_id = None
        digits = "".join(ch for ch in str(key) if ch.isdigit())
        if digits:
            try:
                gpu_id = int(digits)
            except Exception:
                gpu_id = None
        gpus.append(
            {
                "gpu_id": gpu_id if gpu_id is not None else len(gpus),
                "name": str(name) if name else None,
                "vram_total_mb": _to_float(total),
                "vram_used_mb": _to_float(used),
                "vram_free_mb": max(_to_float(total) - _to_float(used), 0.0),
                "util": None,
            }
        )
    return gpus


class ResourceTelemetry:
    """Collect live RAM/VRAM stats with optional sampling for peak tracking."""

    def __init__(
        self,
        gpu_providers: Optional[List[Callable[[], Optional[List[Dict[str, Any]]]]]] = None,
    ) -> None:
        self._gpu_providers = gpu_providers or [
            _gpu_snapshot_nvml,
            _gpu_snapshot_nvidia_smi,
            _gpu_snapshot_rocm_smi,
        ]
        self._monitors: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def snapshot(self) -> Dict[str, Any]:
        ram: Dict[str, Any] = {}
        try:
            import psutil  # type: ignore

            mem = psutil.virtual_memory()
            total = int(mem.total)
            available = int(getattr(mem, "available", 0))
            used = int(getattr(mem, "used", total - available))
            free = int(getattr(mem, "free", 0))
            ram = {
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "available_bytes": available,
                "used_pct": round(_to_float(mem.percent), 2),
            }
        except Exception:
            ram = {}
        gpus: List[Dict[str, Any]] = []
        for provider in self._gpu_providers:
            result = provider()
            if result is None:
                continue
            gpus = result
            break
        return {"ram": ram, "gpus": gpus, "captured_at": _utc_iso()}

    def monitor_start(self, sample_interval_ms: int = 250) -> str:
        monitor_id = uuid.uuid4().hex
        stop_event = threading.Event()
        monitor: Dict[str, Any] = {
            "stop_event": stop_event,
            "samples": [],
            "interval_ms": max(50, int(sample_interval_ms)),
            "started_at": time.time(),
        }

        def _run() -> None:
            interval_s = monitor["interval_ms"] / 1000.0
            while not stop_event.is_set():
                monitor["samples"].append(self.snapshot())
                time.sleep(interval_s)

        thread = threading.Thread(target=_run, name=f"telemetry-monitor-{monitor_id}", daemon=True)
        monitor["thread"] = thread
        with self._lock:
            self._monitors[monitor_id] = monitor
        thread.start()
        return monitor_id

    def monitor_stop(self, monitor_id: str) -> Dict[str, Any]:
        with self._lock:
            monitor = self._monitors.pop(monitor_id, None)
        if not monitor:
            snap = self.snapshot()
            return {
                "peak_snapshot": snap,
                "samples_summary": {"count": 1, "duration_ms": 0},
            }
        monitor["stop_event"].set()
        thread = monitor.get("thread")
        if thread:
            thread.join(timeout=1.0)
        samples = monitor.get("samples") or []
        peak = _peak_snapshot(samples) if samples else self.snapshot()
        duration_ms = max(0, int((time.time() - monitor.get("started_at", time.time())) * 1000))
        return {
            "peak_snapshot": peak,
            "samples_summary": {"count": len(samples), "duration_ms": duration_ms},
        }


def _peak_snapshot(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}
    peak_ram = max(samples, key=lambda s: (s.get("ram") or {}).get("used_bytes", 0))
    peak_snapshot = {
        "ram": dict(peak_ram.get("ram") or {}),
        "gpus": [],
        "captured_at": peak_ram.get("captured_at"),
    }
    gpu_peaks: Dict[int, Dict[str, Any]] = {}
    for sample in samples:
        for gpu in sample.get("gpus") or []:
            try:
                gpu_id = int(gpu.get("gpu_id") or 0)
            except Exception:
                gpu_id = 0
            used = _to_float(gpu.get("vram_used_mb"), 0.0)
            existing = gpu_peaks.get(gpu_id)
            if not existing or used > _to_float(existing.get("vram_used_mb"), 0.0):
                gpu_peaks[gpu_id] = dict(gpu)
    if gpu_peaks:
        peak_snapshot["gpus"] = [gpu_peaks[idx] for idx in sorted(gpu_peaks.keys())]
    return peak_snapshot
