import asyncio
import json
import logging
import time
import calendar
import uuid
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import aiosqlite

from .capacity_planner import CapacityPlanner
from .load_profiler import LoadProfiler, build_config_signature
from .resource_profiles import PROFILE_VERSION, ModelResourceProfileStore
from .resource_telemetry import ResourceTelemetry
from .spawn_governor import SpawnGovernor


PROFILE_DEFAULT_TTL_S = 6 * 60 * 60
PROFILE_MAX_CONCURRENT = 1
INSTANCE_BUSY_TTL_S = 60 * 5
INSTANCE_LOAD_TIMEOUT_S = 25
INSTANCE_ACQUIRE_WAIT_S = 6
INSTANCE_ACQUIRE_POLL_S = 0.25
RESOURCE_PRESSURE_THRESHOLD = 0.9
DEFAULT_CONTEXT_LENGTH = 8192
RAM_EXCLUSIVE_VRAM_MB = 512.0
RAM_EXCLUSIVE_MIN_RAM_MB = 1024.0
RAM_EXCLUSIVE_RATIO = 3.0
logger = logging.getLogger("uvicorn.error")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_utc_timestamp(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return float(calendar.timegm(parsed))
    except Exception:
        return None


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _budget_ram_mb(budgets: Dict[str, Any]) -> float:
    if budgets.get("ram_bytes") is not None:
        return _safe_float(budgets.get("ram_bytes")) / (1024.0 * 1024.0)
    if budgets.get("ram_mb") is not None:
        return _safe_float(budgets.get("ram_mb"))
    return 0.0


def _budget_vram_mb(budgets: Dict[str, Any]) -> float:
    if budgets.get("vram_mb") is None:
        return 0.0
    return _safe_float(budgets.get("vram_mb"))


def _budgets_require_exclusive_ram(budgets: Optional[Dict[str, Any]]) -> bool:
    if not budgets:
        return False
    flagged = budgets.get("exclusive_ram")
    if flagged is not None:
        return bool(flagged)
    ram_mb = _budget_ram_mb(budgets)
    vram_mb = _budget_vram_mb(budgets)
    if ram_mb <= 0.0:
        return False
    if vram_mb <= 0.0:
        return True
    if vram_mb <= RAM_EXCLUSIVE_VRAM_MB and ram_mb >= max(RAM_EXCLUSIVE_MIN_RAM_MB, vram_mb * RAM_EXCLUSIVE_RATIO):
        return True
    return False


def _snapshot_ram_used(snapshot: Dict[str, Any]) -> float:
    ram = snapshot.get("ram") or {}
    return _safe_float(ram.get("used_bytes"), 0.0)


def _gpu_used_map(snapshot: Dict[str, Any]) -> Dict[int, float]:
    used: Dict[int, float] = {}
    for gpu in snapshot.get("gpus") or []:
        try:
            idx = int(gpu.get("gpu_id") or 0)
        except Exception:
            idx = 0
        used[idx] = _safe_float(gpu.get("vram_used_mb"), 0.0)
    return used


def _gpu_total_map(snapshot: Dict[str, Any]) -> Dict[int, float]:
    totals: Dict[int, float] = {}
    for gpu in snapshot.get("gpus") or []:
        try:
            idx = int(gpu.get("gpu_id") or 0)
        except Exception:
            idx = 0
        totals[idx] = _safe_float(gpu.get("vram_total_mb"), 0.0)
    return totals


def _gpu_delta_map(after: Dict[int, float], before: Dict[int, float]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for idx, used in after.items():
        base = before.get(idx, 0.0)
        delta = used - base
        if delta > 0.0:
            deltas[str(idx)] = delta
    return deltas


def _merge_peak_map(existing: Dict[str, Any], observed: Dict[str, float]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for key, val in (existing or {}).items():
        merged[str(key)] = _safe_float(val, 0.0)
    for key, val in (observed or {}).items():
        merged[str(key)] = max(merged.get(str(key), 0.0), _safe_float(val, 0.0))
    return merged


def _normalize_vram_deltas(
    vram_deltas: Dict[str, float],
    *,
    base_snapshot: Dict[str, Any],
    peak_snapshot: Dict[str, Any],
) -> Dict[str, float]:
    if not vram_deltas:
        return {}
    totals = _gpu_total_map(base_snapshot)
    peak_totals = _gpu_total_map(peak_snapshot)
    for gid, total in peak_totals.items():
        totals[gid] = max(totals.get(gid, 0.0), total)
    normalized: Dict[str, float] = {}
    for gid_str, value in vram_deltas.items():
        try:
            gid = int(gid_str)
        except Exception:
            gid = 0
        cleaned = _safe_float(value, 0.0)
        total = totals.get(gid, 0.0)
        if total > 0.0 and cleaned > (total * 8.0):
            cleaned = cleaned / (1024.0 * 1024.0)
        if total > 0.0 and cleaned > (total * 8.0):
            cleaned = total
        if cleaned > 0.0:
            normalized[str(gid)] = cleaned
    return normalized


def _model_size_hint(model_key: str) -> Optional[float]:
    if not model_key:
        return None
    lower = model_key.lower()
    for token in lower.replace("-", " ").replace("_", " ").split():
        if token.endswith("b"):
            try:
                return float(token[:-1])
            except Exception:
                return None
    return None


def _is_coder_key(model_key: str) -> bool:
    return "coder" in (model_key or "").lower()


def _is_profile_key(model_key: str) -> bool:
    return str(model_key or "").startswith("profile-")


def _candidate_class(candidate: "ModelCandidate") -> str:
    if not candidate:
        return "default"
    family = str(candidate.metadata.get("family") or "").strip().lower()
    if family:
        return family
    model_type = str(candidate.metadata.get("type") or "").strip().lower()
    if model_type:
        return model_type
    if candidate.capabilities.get("coding"):
        return "coder"
    if candidate.capabilities.get("vision"):
        return "vision"
    if candidate.capabilities.get("embeddings"):
        return "embeddings"
    return "default"


def _model_tokens(model_key: str) -> List[str]:
    if not model_key:
        return []
    return [token for token in re.split(r"[\\/_\-.]+", model_key.lower()) if token]


def _metadata_flag(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "y", "on"}:
            return True
        if cleaned in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _infer_candidate_capabilities(candidate: "ModelCandidate") -> None:
    tokens = _model_tokens(candidate.model_key)
    token_set = set(tokens)
    model_key_lower = (candidate.model_key or "").lower()
    display_lower = (candidate.display_name or "").lower()
    metadata = candidate.metadata or {}
    architecture = str(metadata.get("architecture") or "").lower()
    publisher = str(metadata.get("publisher") or "").lower()
    trained_tools = _metadata_flag(metadata.get("trainedForToolUse") or metadata.get("trained_for_tool_use"))
    vision_flag = _metadata_flag(metadata.get("vision") or metadata.get("isVision") or metadata.get("vision_supported"))
    max_context = metadata.get("maxContextLength") or metadata.get("max_context_length") or metadata.get("context_length")
    if max_context is not None and "context_length" not in candidate.metadata:
        try:
            candidate.metadata["context_length"] = int(max_context)
        except Exception:
            pass
    if trained_tools is True:
        candidate.capabilities["tool_use"] = True
    if vision_flag is True:
        candidate.capabilities["vision"] = True
    if "coder" in token_set or "code" in token_set:
        candidate.capabilities["coding"] = True
        candidate.metadata.setdefault("family", "coder")
    if "coder" in display_lower or "coder" in architecture:
        candidate.capabilities["coding"] = True
        candidate.metadata.setdefault("family", "coder")
    if "oss" in token_set or "gpt-oss" in model_key_lower or "oss" in architecture or "oss" in publisher:
        candidate.metadata.setdefault("family", "oss")
    if (
        "embedding" in token_set
        or "embeddings" in token_set
        or "embed" in token_set
        or "embedding" in model_key_lower
        or "embeddings" in model_key_lower
    ):
        candidate.metadata.setdefault("type", "embedding")
        candidate.capabilities["embeddings"] = True
    if "vl" in token_set or "vision" in token_set or "multimodal" in token_set:
        candidate.capabilities["vision"] = True
        candidate.metadata.setdefault("family", "vision")


def _is_coder_candidate(candidate: "ModelCandidate") -> bool:
    if candidate.capabilities.get("coding"):
        return True
    if _is_coder_key(candidate.model_key):
        return True
    display = (candidate.display_name or "").lower()
    if "coder" in display:
        return True
    architecture = str(candidate.metadata.get("architecture") or "").lower()
    return "coder" in architecture


@dataclass
class ModelCandidate:
    backend_id: str
    model_key: str
    display_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    profile: Dict[str, Any] = field(default_factory=dict)
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "model_key": self.model_key,
            "display_name": self.display_name,
            "metadata": dict(self.metadata),
            "capabilities": dict(self.capabilities),
            "profile": dict(self.profile),
            "updated_at": self.updated_at,
        }


@dataclass
class ModelInstanceInfo:
    backend_id: str
    instance_id: str
    model_key: str
    api_identifier: str
    endpoint: str
    status: str = "ready"
    last_used_at: float = 0.0
    ttl_seconds: Optional[int] = None
    context_length: Optional[int] = None
    resource_reservation: Dict[str, Any] = field(default_factory=dict)
    measured_perf: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "instance_id": self.instance_id,
            "model_key": self.model_key,
            "api_identifier": self.api_identifier,
            "endpoint": self.endpoint,
            "status": self.status,
            "last_used_at": self.last_used_at,
            "ttl_seconds": self.ttl_seconds,
            "context_length": self.context_length,
            "resource_reservation": dict(self.resource_reservation),
            "measured_perf": dict(self.measured_perf),
        }


@dataclass
class ResourceEstimate:
    vram_mb: Optional[float] = None
    ram_mb: Optional[float] = None
    cpu_pct: Optional[float] = None
    gpu_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vram_mb": self.vram_mb,
            "ram_mb": self.ram_mb,
            "cpu_pct": self.cpu_pct,
            "gpu_id": self.gpu_id,
        }


class ModelBackend(Protocol):
    id: str

    async def discover(self) -> List[ModelCandidate]:
        ...

    async def list_loaded(self) -> List[ModelInstanceInfo]:
        ...

    async def ensure_server_running(self) -> None:
        ...

    async def load_instance(self, model_key: str, opts: Dict[str, Any]) -> ModelInstanceInfo:
        ...

    async def unload_instance(self, instance_id_or_identifier: str) -> None:
        ...

    async def estimate_resources(self, model_key: str, opts: Dict[str, Any]) -> Optional[ResourceEstimate]:
        ...

    async def call_chat_completion(self, instance: ModelInstanceInfo, request: Dict[str, Any]) -> Dict[str, Any]:
        ...

    async def call_responses(self, instance: ModelInstanceInfo, request: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def supports_tools(self) -> bool:
        ...


class ModelProfileStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def upsert(self, candidate: ModelCandidate, ttl_seconds: int) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO model_profiles(backend_id, model_key, display_name, metadata_json, "
                "capabilities_json, profile_json, updated_at, ttl_seconds) VALUES (?,?,?,?,?,?,?,?)",
                (
                    candidate.backend_id,
                    candidate.model_key,
                    candidate.display_name,
                    json.dumps(candidate.metadata, ensure_ascii=True),
                    json.dumps(candidate.capabilities, ensure_ascii=True),
                    json.dumps(candidate.profile, ensure_ascii=True),
                    candidate.updated_at or utc_now(),
                    ttl_seconds,
                ),
            )
            await db.commit()

    async def get(self, backend_id: str, model_key: str) -> Optional[ModelCandidate]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT display_name, metadata_json, capabilities_json, profile_json, updated_at, ttl_seconds "
                "FROM model_profiles WHERE backend_id=? AND model_key=?",
                (backend_id, model_key),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if not row:
            return None
        return ModelCandidate(
            backend_id=backend_id,
            model_key=model_key,
            display_name=row["display_name"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            capabilities=json.loads(row["capabilities_json"] or "{}"),
            profile=json.loads(row["profile_json"] or "{}"),
            updated_at=row["updated_at"],
        )

    async def list_all(self) -> List[ModelCandidate]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT backend_id, model_key, display_name, metadata_json, capabilities_json, profile_json, updated_at "
                "FROM model_profiles"
            )
            rows = await cursor.fetchall()
            await cursor.close()
        results: List[ModelCandidate] = []
        for row in rows:
            results.append(
                ModelCandidate(
                    backend_id=row["backend_id"],
                    model_key=row["model_key"],
                    display_name=row["display_name"],
                    metadata=json.loads(row["metadata_json"] or "{}"),
                    capabilities=json.loads(row["capabilities_json"] or "{}"),
                    profile=json.loads(row["profile_json"] or "{}"),
                    updated_at=row["updated_at"],
                )
            )
        return results


class WorkerPool:
    def __init__(self) -> None:
        self._instances: Dict[str, ModelInstanceInfo] = {}
        self._busy: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def add(self, instance: ModelInstanceInfo) -> None:
        async with self._lock:
            self._instances[instance.instance_id] = instance

    async def remove(self, instance_id: str) -> Optional[ModelInstanceInfo]:
        async with self._lock:
            self._busy.pop(instance_id, None)
            return self._instances.pop(instance_id, None)

    async def mark_busy(self, instance_id: str) -> None:
        async with self._lock:
            self._busy[instance_id] = time.monotonic()
            inst = self._instances.get(instance_id)
            if inst:
                inst.status = "busy"
                inst.last_used_at = time.monotonic()

    async def release(self, instance_id: str) -> None:
        async with self._lock:
            self._busy.pop(instance_id, None)
            inst = self._instances.get(instance_id)
            if inst:
                inst.status = "ready"

    async def list_instances(self) -> List[ModelInstanceInfo]:
        async with self._lock:
            return list(self._instances.values())

    async def list_ready(self) -> List[ModelInstanceInfo]:
        async with self._lock:
            now = time.monotonic()
            ready: List[ModelInstanceInfo] = []
            for instance_id, inst in self._instances.items():
                last_busy = self._busy.get(instance_id)
                if last_busy and (now - last_busy) > INSTANCE_BUSY_TTL_S:
                    self._busy.pop(instance_id, None)
                    inst.status = "ready"
                if inst.status != "busy":
                    ready.append(inst)
            return ready

    async def busy_count(self) -> int:
        async with self._lock:
            return len(self._busy)


class ModelSelector:
    def __init__(self, candidates: List[ModelCandidate], prefer: Optional[Iterable[str]] = None):
        self.candidates = candidates
        self.prefer = {p for p in (prefer or []) if p}

    def _prefer_bonus(self, objective: str) -> float:
        if objective == "best_latency":
            return 1.0
        if objective == "best_quality":
            return 0.2
        if objective == "balanced":
            return 0.4
        return 0.3

    def _vision_penalty(self, candidate: ModelCandidate, required: List[str], objective: str) -> float:
        if "vision" in required:
            return 0.0
        if candidate.capabilities.get("vision") or candidate.metadata.get("family") == "vision":
            if objective == "best_quality":
                return 1.5
            if objective == "balanced":
                return 0.75
            if objective == "best_latency":
                return 0.25
        return 0.0

    def _capability_score(self, candidate: ModelCandidate, required: List[str]) -> Optional[float]:
        if candidate.metadata.get("type") in ("embedding", "embeddings") and "embeddings" not in required:
            return None
        score = 0.0
        for cap in required:
            val = candidate.capabilities.get(cap)
            if val is False:
                return None
            if val is True:
                score += 2.0
            else:
                score -= 0.5
        return score

    def _quality_score(self, candidate: ModelCandidate) -> float:
        profile = candidate.profile or {}
        quality = (
            _safe_float(profile.get("tool_call_success_rate"), 0.0)
            + _safe_float(profile.get("json_schema_success_rate"), 0.0)
            + (1.0 - _safe_float(profile.get("error_rate"), 0.0))
        )
        size_hint = _model_size_hint(candidate.model_key)
        has_quality = any(key in profile for key in ("tool_call_success_rate", "json_schema_success_rate", "error_rate"))
        has_perf = any(key in profile for key in ("tps", "latency_ms", "tps_samples", "latency_samples"))
        if not has_quality and not has_perf:
            if size_hint is not None:
                return size_hint / 10.0
            return quality
        if size_hint is not None:
            quality += size_hint / 10.0
        return quality

    def _latency_score(self, candidate: ModelCandidate) -> float:
        profile = candidate.profile or {}
        tps = _safe_float(profile.get("tps"), 0.0)
        latency = _safe_float(profile.get("latency_ms"), 0.0)
        if tps == 0.0 and latency == 0.0:
            size_hint = _model_size_hint(candidate.model_key)
            if size_hint is not None:
                return -size_hint
        return tps - (latency / 100.0)

    def _resource_penalty(self, candidate: ModelCandidate) -> float:
        profile = candidate.profile or {}
        ram_bytes = _safe_float(profile.get("ram_instance_peak_bytes"), 0.0)
        vram_map = profile.get("vram_instance_peak_mb_by_gpu") or {}
        vram_peak = 0.0
        for val in vram_map.values():
            vram_peak = max(vram_peak, _safe_float(val, 0.0))
        if vram_peak <= 0.0 and profile.get("vram_estimate_only_mb") is not None:
            vram_peak = _safe_float(profile.get("vram_estimate_only_mb"), 0.0)
        ram_gb = ram_bytes / (1024.0**3) if ram_bytes else 0.0
        vram_gb = vram_peak / 1024.0 if vram_peak else 0.0
        if ram_gb <= 0.0 and vram_gb <= 0.0:
            return 0.5
        return ram_gb + (vram_gb * 1.5)

    def choose_candidate(
        self,
        required_capabilities: Optional[List[str]] = None,
        objective: str = "balanced",
        prefer_small: bool = False,
    ) -> Optional[ModelCandidate]:
        required_capabilities = required_capabilities or []
        best: Optional[Tuple[float, ModelCandidate]] = None
        for candidate in self.candidates:
            cap_score = self._capability_score(candidate, required_capabilities)
            if cap_score is None:
                continue
            if objective == "best_latency":
                score = cap_score + self._latency_score(candidate)
            elif objective == "best_quality":
                score = cap_score + self._quality_score(candidate)
            else:
                score = cap_score + (self._quality_score(candidate) + self._latency_score(candidate)) / 2.0
            if candidate.model_key in self.prefer:
                score += self._prefer_bonus(objective)
            score -= self._vision_penalty(candidate, required_capabilities, objective)
            if prefer_small:
                score -= self._resource_penalty(candidate)
            if best is None or score > best[0]:
                best = (score, candidate)
        return best[1] if best else None

    def choose_instance(
        self,
        instances: List[ModelInstanceInfo],
        candidates_by_key: Dict[str, ModelCandidate],
        required_capabilities: Optional[List[str]] = None,
        objective: str = "balanced",
        prefer_small: bool = False,
    ) -> Optional[ModelInstanceInfo]:
        required_capabilities = required_capabilities or []
        best: Optional[Tuple[float, ModelInstanceInfo]] = None
        for inst in instances:
            candidate = candidates_by_key.get(inst.model_key)
            if not candidate:
                continue
            if candidate.metadata.get("type") in ("embedding", "embeddings") and "embeddings" not in required_capabilities:
                continue
            cap_score = self._capability_score(candidate, required_capabilities)
            if cap_score is None:
                continue
            if objective == "best_latency":
                score = cap_score + self._latency_score(candidate)
            elif objective == "best_quality":
                score = cap_score + self._quality_score(candidate)
            else:
                score = cap_score + (self._quality_score(candidate) + self._latency_score(candidate)) / 2.0
            if candidate.model_key in self.prefer:
                score += self._prefer_bonus(objective)
            score -= self._vision_penalty(candidate, required_capabilities, objective)
            if inst.status == "busy":
                score -= 1.0
            if prefer_small:
                score -= self._resource_penalty(candidate)
            if best is None or score > best[0]:
                best = (score, inst)
        return best[1] if best else None


class Autoscaler:
    def __init__(
        self,
        *,
        enabled: bool = True,
        global_max_instances: Optional[int] = None,
        per_backend_max_instances: Optional[Dict[str, Optional[int]]] = None,
        min_instances: Optional[Dict[str, int]] = None,
    ) -> None:
        self.enabled = enabled
        self.global_max_instances = global_max_instances
        self.per_backend_max_instances = per_backend_max_instances or {}
        self.min_instances = min_instances or {}

    def _cap_max(self, backend_id: str, current: int, desired: int) -> int:
        max_backend = self.per_backend_max_instances.get(backend_id)
        if max_backend is not None:
            desired = min(desired, max_backend)
        if self.global_max_instances is not None:
            desired = min(desired, self.global_max_instances)
        return max(current, desired)

    def desired_instances(self, backend_id: str, backlog: int, ready: int) -> int:
        min_req = int(self.min_instances.get("executor") or 0)
        desired = max(min_req, ready)
        if backlog > ready:
            desired = max(desired, backlog)
        return self._cap_max(backend_id, ready, desired)


class ModelManager:
    def __init__(
        self,
        *,
        db_path: str,
        backends: Dict[str, ModelBackend],
        resource_manager: Optional[Any] = None,
        telemetry: Optional[ResourceTelemetry] = None,
        ram_headroom_pct: float = 10.0,
        vram_headroom_pct: float = 10.0,
        model_candidates_mode: str = "auto",
        allow: Optional[List[str]] = None,
        deny: Optional[List[str]] = None,
        prefer: Optional[List[str]] = None,
        autoscaler: Optional[Autoscaler] = None,
        routing_objective: str = "balanced",
        tool_required_by_default: bool = True,
        profile_ttl_s: int = PROFILE_DEFAULT_TTL_S,
        profiling: Optional[Dict[str, Any]] = None,
        capacity_planner: Optional[CapacityPlanner] = None,
        spawn_governor: Optional[SpawnGovernor] = None,
        max_concurrent_loads: int = 1,
    ) -> None:
        self.db_path = db_path
        self.backends = backends
        self.resource_manager = resource_manager
        self.telemetry = telemetry or ResourceTelemetry()
        self.ram_headroom_pct = ram_headroom_pct
        self.vram_headroom_pct = vram_headroom_pct
        self.profile_store = ModelProfileStore(db_path)
        self.resource_profile_store = ModelResourceProfileStore(db_path)
        self.model_candidates_mode = model_candidates_mode or "auto"
        self.allow = allow or []
        self.deny = deny or []
        self.prefer = prefer or []
        self.autoscaler = autoscaler or Autoscaler()
        self.routing_objective = routing_objective
        self.tool_required_by_default = tool_required_by_default
        self.profile_ttl_s = profile_ttl_s
        self.profiling_config = profiling or {}
        self.worker_pool = WorkerPool()
        self._candidates: List[ModelCandidate] = []
        self._candidate_lock = asyncio.Lock()
        self._acquire_lock = asyncio.Lock()
        self._profile_tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        max_profiles = int(self.profiling_config.get("max_concurrent_profiles") or PROFILE_MAX_CONCURRENT)
        self._probe_sem = asyncio.Semaphore(max(1, max_profiles))
        self._profiling_active = 0
        self._profiling_lock = asyncio.Lock()
        self._manual_profile_task: Optional[asyncio.Task] = None
        self._load_sem = asyncio.Semaphore(max(1, max_concurrent_loads))
        self._profile_status: Dict[str, Any] = {
            "running": False,
            "current": None,
            "completed": 0,
            "total": 0,
            "errors": [],
        }
        self.load_profiler = LoadProfiler(
            telemetry=self.telemetry,
            store=self.resource_profile_store,
            ram_headroom_pct=self.ram_headroom_pct,
            vram_headroom_pct=self.vram_headroom_pct,
            enforce_headroom=bool(self.profiling_config.get("enforce_headroom", False)),
            sample_interval_ms=int(self.profiling_config.get("sample_interval_ms") or 250),
            repeats=int(self.profiling_config.get("repeats") or 1),
            test_timeout_s=int(self.profiling_config.get("test_timeout_s") or 120),
            settle_timeout_s=int(self.profiling_config.get("settle_timeout_s") or 12),
            max_output_tokens=self.profiling_config.get("max_output_tokens"),
            load_sem=self._load_sem,
        )
        self.capacity_planner = capacity_planner or CapacityPlanner(
            telemetry=self.telemetry,
            store=self.resource_profile_store,
            ram_headroom_pct=self.ram_headroom_pct,
            vram_headroom_pct=self.vram_headroom_pct,
        )
        self.spawn_governor = spawn_governor or SpawnGovernor(
            telemetry=self.telemetry,
            ram_headroom_pct=self.ram_headroom_pct,
            vram_headroom_pct=self.vram_headroom_pct,
            max_concurrent_loads=max_concurrent_loads,
            load_sem=self._load_sem,
        )

    async def refresh(self) -> List[ModelCandidate]:
        for backend in self.backends.values():
            try:
                await backend.ensure_server_running()
            except Exception:
                continue
        candidates = await self.discover_candidates()
        await self._sync_loaded_instances()
        await self._schedule_profiles(candidates)
        return candidates

    async def discover_candidates(self) -> List[ModelCandidate]:
        discovered: List[ModelCandidate] = []
        for backend_id, backend in self.backends.items():
            try:
                found = await backend.discover()
            except Exception:
                found = []
            for candidate in found:
                candidate.backend_id = backend_id
                _infer_candidate_capabilities(candidate)
                discovered.append(candidate)
        filtered = self._filter_candidates(discovered)
        async with self._candidate_lock:
            self._candidates = filtered
        return filtered

    async def _sync_loaded_instances(self) -> None:
        desired_context = self.profiling_config.get("context_length")
        if desired_context is None:
            desired_context = DEFAULT_CONTEXT_LENGTH
        for backend_id, backend in self.backends.items():
            reachable = True
            check_reachable = getattr(backend, "is_server_reachable", None)
            if callable(check_reachable):
                try:
                    reachable = await check_reachable()
                except Exception:
                    reachable = False
            if not reachable:
                continue
            loaded: Optional[List[ModelInstanceInfo]] = None
            try:
                loaded = await backend.list_loaded()
            except Exception:
                loaded = None
            if loaded is None:
                continue
            if desired_context:
                filtered: List[ModelInstanceInfo] = []
                for instance in loaded:
                    if (
                        instance.context_length
                        and int(instance.context_length) < int(desired_context)
                    ):
                        continue
                    filtered.append(instance)
                loaded = filtered
            loaded_ids = {inst.instance_id for inst in loaded if inst.instance_id}
            existing = await self.worker_pool.list_instances()
            for inst in existing:
                if inst.backend_id != backend_id:
                    continue
                if inst.instance_id not in loaded_ids:
                    await self.worker_pool.remove(inst.instance_id)
            for instance in loaded:
                if _is_profile_key(instance.instance_id) or _is_profile_key(instance.model_key):
                    try:
                        await backend.unload_instance(instance.instance_id)
                    except Exception:
                        pass
                    continue
                if not instance.instance_id:
                    instance.instance_id = f"{backend_id}:{instance.api_identifier}"
                await self.worker_pool.add(instance)

    def _filter_candidates(self, candidates: List[ModelCandidate]) -> List[ModelCandidate]:
        mode = (self.model_candidates_mode or "auto").lower()
        allow = {m for m in self.allow if m}
        deny = {m for m in self.deny if m}
        filtered: List[ModelCandidate] = []
        for candidate in candidates:
            model_key = candidate.model_key
            if _is_profile_key(model_key):
                continue
            if mode == "allowlist" and allow and model_key not in allow:
                continue
            if mode == "denylist" and model_key in deny:
                continue
            filtered.append(candidate)
        return filtered

    async def _schedule_profiles(self, candidates: List[ModelCandidate]) -> None:
        now = time.time()
        profile_ttl = int(self.profiling_config.get("profile_ttl_s") or self.profile_ttl_s)
        profiling_enabled = bool(self.profiling_config.get("enabled", True))
        auto_profile = bool(self.profiling_config.get("auto_profile", True))
        for candidate in candidates:
            key = (candidate.backend_id, candidate.model_key)
            if key in self._profile_tasks and not self._profile_tasks[key].done():
                continue
            cached = await self.profile_store.get(candidate.backend_id, candidate.model_key)
            if cached and cached.updated_at:
                candidate.profile = cached.profile
                if cached.capabilities:
                    candidate.capabilities.update(cached.capabilities)
                profile_status = str(candidate.profile.get("profile_status") or "").lower()
                if profile_status != "ok":
                    _infer_candidate_capabilities(candidate)
                updated_ts = _parse_utc_timestamp(cached.updated_at)
                if updated_ts and (now - updated_ts) < profile_ttl:
                    pass
            opts = self._profile_opts(candidate)
            config_sig = build_config_signature(candidate, opts)
            resource_profile = await self.resource_profile_store.get(
                candidate.backend_id, candidate.model_key, config_sig
            )
            if resource_profile:
                self._apply_resource_profile(candidate, resource_profile)
                profile_status = str(resource_profile.get("status") or "").lower()
                if profile_status != "ok":
                    _infer_candidate_capabilities(candidate)
                profiled_at = _parse_utc_timestamp(resource_profile.get("profiled_at"))
                if profiled_at and (now - profiled_at) < profile_ttl and profile_status == "ok":
                    continue
            if not profiling_enabled or not auto_profile:
                continue
            self._profile_tasks[key] = asyncio.create_task(self._profile_candidate(candidate, opts))

    async def _profile_candidate(self, candidate: ModelCandidate, opts: Optional[Dict[str, Any]] = None) -> None:
        async with self._probe_sem:
            async with self._profiling_lock:
                self._profiling_active += 1
            try:
                await self._run_probes(candidate, opts)
            except Exception:
                return
            finally:
                async with self._profiling_lock:
                    self._profiling_active = max(0, self._profiling_active - 1)
            candidate.updated_at = utc_now()
            ttl = int(self.profiling_config.get("profile_ttl_s") or self.profile_ttl_s)
            await self.profile_store.upsert(candidate, ttl)

    async def _run_probes(self, candidate: ModelCandidate, opts: Optional[Dict[str, Any]] = None) -> None:
        backend = self.backends.get(candidate.backend_id)
        if not backend:
            return
        ttl = int(self.profiling_config.get("profile_ttl_s") or self.profile_ttl_s)
        profile = await self.load_profiler.profile_candidate(
            backend=backend,
            candidate=candidate,
            opts=opts or {},
            ttl_seconds=ttl,
        )
        self._apply_resource_profile(candidate, profile)

    def _apply_resource_profile(self, candidate: ModelCandidate, profile: Dict[str, Any]) -> None:
        if not isinstance(profile, dict):
            return
        candidate.profile.update(
            {
                "ram_instance_peak_bytes": profile.get("ram_instance_peak_bytes"),
                "vram_instance_peak_mb_by_gpu": profile.get("vram_instance_peak_mb_by_gpu"),
                "vram_estimate_only_mb": profile.get("vram_estimate_only_mb"),
                "resource_profiled_at": profile.get("profiled_at"),
                "resource_config_signature": profile.get("config_signature"),
                "tps": profile.get("tps"),
                "latency_ms": profile.get("latency_ms"),
                "error_rate": profile.get("error_rate"),
                "tool_call_success_rate": profile.get("tool_call_success_rate"),
                "json_schema_success_rate": profile.get("json_schema_success_rate"),
                "profile_status": profile.get("status"),
                "profile_error": profile.get("error"),
            }
        )
        if profile.get("tool_call_success_rate") is not None:
            candidate.capabilities["tool_use"] = bool(profile.get("tool_call_success_rate") >= 0.5)
        if profile.get("json_schema_success_rate") is not None:
            candidate.capabilities["structured_output"] = bool(profile.get("json_schema_success_rate") >= 0.5)

    def _profile_opts(self, candidate: ModelCandidate) -> Dict[str, Any]:
        opts: Dict[str, Any] = {}
        cfg_context = self.profiling_config.get("context_length")
        if cfg_context is None:
            cfg_context = DEFAULT_CONTEXT_LENGTH
        if cfg_context:
            opts["context_length"] = int(cfg_context)
        return opts

    def _observe_on_use_enabled(self) -> bool:
        if self.profiling_config.get("enabled", True) is False:
            return False
        observe = self.profiling_config.get("observe_on_use")
        return observe is not False

    async def _record_usage_profile(
        self,
        *,
        backend_id: str,
        model_key: str,
        base_snapshot: Dict[str, Any],
        peak_snapshot: Dict[str, Any],
        gpu_id: Optional[int] = None,
        samples_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not model_key or _is_profile_key(model_key):
            return
        try:
            candidates = await self.get_candidates()
            candidate = next(
                (c for c in candidates if c.backend_id == backend_id and c.model_key == model_key),
                None,
            )
            if not candidate:
                return
            base_ram_used = _snapshot_ram_used(base_snapshot)
            peak_ram_used = _snapshot_ram_used(peak_snapshot)
            ram_delta = max(peak_ram_used - base_ram_used, 0.0)
            base_vram = _gpu_used_map(base_snapshot)
            peak_vram = _gpu_used_map(peak_snapshot)
            vram_deltas = _gpu_delta_map(peak_vram, base_vram)
            vram_deltas = _normalize_vram_deltas(
                vram_deltas,
                base_snapshot=base_snapshot,
                peak_snapshot=peak_snapshot,
            )
            if gpu_id is not None:
                gpu_key = str(gpu_id)
                if gpu_key in vram_deltas:
                    vram_deltas = {gpu_key: vram_deltas[gpu_key]}
                else:
                    vram_deltas = {}
            if ram_delta <= 0.0 and not vram_deltas:
                return
            opts = self._profile_opts(candidate)
            config_sig = build_config_signature(candidate, opts)
            existing = await self.resource_profile_store.get(backend_id, model_key, config_sig) or {}
            updated = dict(existing)
            existing_status = str(
                updated.get("status")
                or updated.get("profile_status")
                or ""
            ).lower()
            observed_at = utc_now()
            updated.setdefault("backend_id", backend_id)
            updated.setdefault("model_key", model_key)
            updated["config_signature"] = config_sig
            updated.setdefault("profile_version", PROFILE_VERSION)
            updated["profiled_at"] = observed_at
            if not updated.get("status"):
                updated["status"] = "observed"
            updated["observed_at"] = observed_at
            updated["observed_samples"] = _safe_int(updated.get("observed_samples"), 0) + 1
            if samples_summary:
                duration_ms = samples_summary.get("duration_ms")
                if isinstance(duration_ms, (int, float)):
                    updated["observed_duration_ms"] = int(duration_ms)
            if existing_status == "ok":
                existing_obs_ram = _safe_float(updated.get("observed_ram_peak_bytes"), 0.0)
                updated["observed_ram_peak_bytes"] = max(existing_obs_ram, ram_delta)
                merged_obs_vram = _merge_peak_map(updated.get("observed_vram_peak_mb_by_gpu") or {}, vram_deltas)
                if merged_obs_vram:
                    updated["observed_vram_peak_mb_by_gpu"] = merged_obs_vram
            else:
                existing_ram = _safe_float(updated.get("ram_instance_peak_bytes"), 0.0)
                updated["ram_instance_peak_bytes"] = max(existing_ram, ram_delta)
                merged_vram = _merge_peak_map(updated.get("vram_instance_peak_mb_by_gpu") or {}, vram_deltas)
                if merged_vram:
                    updated["vram_instance_peak_mb_by_gpu"] = merged_vram
            ttl = int(self.profiling_config.get("profile_ttl_s") or self.profile_ttl_s)
            await self.resource_profile_store.upsert(updated, ttl)
            async with self._candidate_lock:
                for cand in self._candidates:
                    if cand.backend_id == backend_id and cand.model_key == model_key:
                        self._apply_resource_profile(cand, updated)
                        cand.updated_at = observed_at
                        break
        except Exception:
            return

    def profiling_active(self) -> bool:
        if self._profiling_active > 0:
            return True
        if self._manual_profile_task and not self._manual_profile_task.done():
            return True
        for task in self._profile_tasks.values():
            if task and not task.done():
                return True
        return False

    async def auto_profile_status(self) -> Dict[str, Any]:
        candidates = await self.get_candidates()
        total = len(candidates)
        profiling_enabled = bool(self.profiling_config.get("enabled", True))
        auto_profile = bool(self.profiling_config.get("auto_profile", True))
        enabled = profiling_enabled and auto_profile
        profile_ttl = int(self.profiling_config.get("profile_ttl_s") or self.profile_ttl_s)
        now = time.time()
        completed = 0
        pending = 0
        current = None
        candidate_keys = {(c.backend_id, c.model_key) for c in candidates}
        running = any(
            task
            and not task.done()
            and key in candidate_keys
            for key, task in self._profile_tasks.items()
        )

        for candidate in candidates:
            opts = self._profile_opts(candidate)
            config_sig = build_config_signature(candidate, opts)
            profiled_at = _parse_utc_timestamp(candidate.profile.get("resource_profiled_at"))
            profile_sig = candidate.profile.get("resource_config_signature")
            profile_status = str(candidate.profile.get("profile_status") or "").lower()
            fresh = bool(
                profiled_at
                and (now - profiled_at) < profile_ttl
                and profile_sig == config_sig
                and profile_status == "ok"
            )
            if fresh:
                completed += 1
                continue
            pending += 1
            key = (candidate.backend_id, candidate.model_key)
            task = self._profile_tasks.get(key)
            if not current and task and not task.done():
                current = candidate.model_key

        return {
            "enabled": enabled,
            "running": running if enabled else False,
            "current": current if enabled else None,
            "completed": completed,
            "total": total,
            "pending": pending,
        }

    async def wait_for_profiles(
        self,
        timeout_s: Optional[float] = None,
        *,
        cancel_on_timeout: bool = True,
    ) -> None:
        tasks = [task for task in self._profile_tasks.values() if task and not task.done()]
        if self._manual_profile_task and not self._manual_profile_task.done():
            tasks.append(self._manual_profile_task)
        if not tasks:
            return
        if timeout_s and timeout_s > 0:
            if cancel_on_timeout:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_s)
            else:
                await asyncio.wait(tasks, timeout=timeout_s)
        else:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _resource_pressure(self) -> float:
        snapshot = self.telemetry.snapshot()
        ratios: List[float] = []
        ram = snapshot.get("ram") or {}
        total_ram = _safe_float(ram.get("total_bytes"))
        used_ram = _safe_float(ram.get("used_bytes"))
        if total_ram:
            allowed = total_ram * (1.0 - (self.ram_headroom_pct / 100.0))
            if allowed > 0:
                ratios.append(used_ram / allowed)
        for gpu in snapshot.get("gpus") or []:
            total = _safe_float(gpu.get("vram_total_mb"))
            used = _safe_float(gpu.get("vram_used_mb"))
            if total:
                allowed = total * (1.0 - (self.vram_headroom_pct / 100.0))
                if allowed > 0:
                    ratios.append(used / allowed)
        return max(ratios) if ratios else 0.0

    def _is_instance_unavailable_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        if "model unloaded" in text or "invalid model identifier" in text or "model_not_found" in text:
            return True
        if any(
            token in text
            for token in (
                "context length",
                "context window",
                "context overflow",
                "context the overflows",
                "prediction-error",
            )
        ):
            return True
        response = getattr(exc, "response", None)
        if response is None:
            return False
        message = ""
        code = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                err = payload.get("error") or {}
                message = str(err.get("message") or "")
                code = str(err.get("code") or "")
        except Exception:
            payload = None
        if "model unloaded" in message.lower() or "not llm" in message.lower() or code.lower() == "model_not_found":
            return True
        try:
            body = (response.text or "").lower()
        except Exception:
            body = ""
        if "model unloaded" in body or "model not found" in body or "not llm" in body:
            return True
        return any(
            token in body
            for token in (
                "context length",
                "context window",
                "context overflow",
                "context the overflows",
                "prediction-error",
            )
        )

    async def _drop_instance(self, instance: ModelInstanceInfo) -> None:
        await self.worker_pool.remove(instance.instance_id)
        reservation_id = instance.resource_reservation.get("reservation_id")
        if reservation_id and self.resource_manager:
            self.resource_manager.release(reservation_id)

    def _reservation_size_score(self, reservation: Dict[str, Any]) -> float:
        ram_bytes = _safe_float(reservation.get("ram_bytes"), 0.0)
        if ram_bytes <= 0.0 and reservation.get("ram_mb") is not None:
            ram_bytes = _safe_float(reservation.get("ram_mb")) * 1024.0 * 1024.0
        vram_mb = _safe_float(reservation.get("vram_mb"), 0.0)
        ram_gb = ram_bytes / (1024.0**3) if ram_bytes else 0.0
        vram_gb = vram_mb / 1024.0 if vram_mb else 0.0
        return ram_gb + (vram_gb * 1.5)

    async def _evict_idle_instance(self, exclude_model_key: Optional[str] = None) -> bool:
        instances = await self.worker_pool.list_instances()
        now = time.monotonic()
        candidates = []
        for inst in instances:
            if inst.status == "busy":
                continue
            if exclude_model_key and inst.model_key == exclude_model_key:
                continue
            score = self._reservation_size_score(inst.resource_reservation or {})
            idle_age = now - inst.last_used_at if inst.last_used_at else now
            candidates.append((score, idle_age, inst))
        if not candidates:
            return False
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        target = candidates[0][2]
        backend = self.backends.get(target.backend_id)
        if not backend:
            return False
        try:
            await backend.unload_instance(target.instance_id)
        except Exception:
            return False
        await self.worker_pool.remove(target.instance_id)
        reservation_id = target.resource_reservation.get("reservation_id")
        if reservation_id and self.resource_manager:
            self.resource_manager.release(reservation_id)
        return True

    async def trim_idle_instances(self, target_ready: int) -> int:
        target_ready = max(0, int(target_ready))
        removed = 0
        while True:
            instances = await self.worker_pool.list_instances()
            ready = [inst for inst in instances if inst.status != "busy"]
            if len(ready) <= target_ready:
                break
            if not await self._evict_idle_instance():
                break
            removed += 1
        return removed

    async def _probe_tool_call(self, backend: ModelBackend, instance: ModelInstanceInfo) -> bool:
        if not backend.supports_tools():
            return False
        request = {
            "model": instance.api_identifier,
            "messages": [
                {"role": "system", "content": "Return a tool call only."},
                {"role": "user", "content": "Call the ping tool with {\"value\": 7}."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "description": "Ping tool.",
                        "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "ping"}},
            "temperature": 0.0,
            "max_tokens": 100,
        }
        try:
            response = await backend.call_chat_completion(instance, request)
        except Exception:
            return False
        tool_calls = None
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            args = tool_calls[0].get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return isinstance(args, dict) and args.get("value") == 7
        return False

    async def _probe_json_schema(self, backend: ModelBackend, instance: ModelInstanceInfo) -> bool:
        schema = {
            "name": "probe",
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        }
        request = {
            "model": instance.api_identifier,
            "messages": [
                {"role": "system", "content": "Return JSON that matches the schema."},
                {"role": "user", "content": "Return {\"ok\": true}."},
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0.0,
            "max_tokens": 60,
        }
        try:
            response = await backend.call_chat_completion(instance, request)
        except Exception:
            return False
        content = ""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content") or ""
        if not content:
            return False
        try:
            parsed = json.loads(content)
        except Exception:
            return False
        return isinstance(parsed, dict) and parsed.get("ok") is True

    async def _probe_latency(self, backend: ModelBackend, instance: ModelInstanceInfo) -> Dict[str, Any]:
        request = {
            "model": instance.api_identifier,
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0.0,
            "max_tokens": 30,
        }
        start = time.monotonic()
        try:
            response = await backend.call_chat_completion(instance, request)
        except Exception:
            return {}
        elapsed = max(time.monotonic() - start, 0.001)
        content = ""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content") or ""
        token_est = max(len(str(content).split()), 1)
        return {"latency_ms": round(elapsed * 1000, 2), "tps": round(token_est / elapsed, 2)}

    async def get_candidates(self) -> List[ModelCandidate]:
        async with self._candidate_lock:
            return list(self._candidates)

    async def bootstrap(self) -> Optional[ModelInstanceInfo]:
        candidates = await self.get_candidates()
        if not candidates:
            candidates = await self.refresh()
        ready_instances = await self.worker_pool.list_ready()
        if ready_instances:
            selector = ModelSelector(candidates, prefer=self.prefer)
            prefer_small = self._resource_pressure() >= RESOURCE_PRESSURE_THRESHOLD
            candidates_by_key = {c.model_key: c for c in candidates}
            preferred = selector.choose_instance(
                ready_instances,
                candidates_by_key,
                required_capabilities=["tool_use"] if self.tool_required_by_default else [],
                objective=self.routing_objective,
                prefer_small=prefer_small,
            )
            if preferred:
                return preferred
            for instance in ready_instances:
                candidate = candidates_by_key.get(instance.model_key)
                if candidate and candidate.metadata.get("type") not in ("embedding", "embeddings"):
                    return instance
        selector = ModelSelector(candidates, prefer=self.prefer)
        prefer_small = self._resource_pressure() >= RESOURCE_PRESSURE_THRESHOLD
        tool_capable = [c for c in candidates if c.capabilities.get("tool_use")]
        ordered = tool_capable if tool_capable else candidates
        ordered.sort(key=lambda c: _model_size_hint(c.model_key) or 999.0)
        for candidate in ordered:
            instance = await self._load_instance(candidate)
            if not instance:
                continue
            if not tool_capable:
                # If not known, run tool probe and mark capability.
                backend = self.backends.get(candidate.backend_id)
                if backend and await self._probe_tool_call(backend, instance):
                    candidate.capabilities["tool_use"] = True
            return instance
        return None

    async def _next_instance_identifier(self, model_key: str) -> str:
        instances = await self.worker_pool.list_instances()
        used_slots: set[int] = set()
        prefix = f"{model_key}:"
        for inst in instances:
            if inst.model_key != model_key:
                continue
            identifier = inst.api_identifier or inst.instance_id or ""
            if not identifier:
                continue
            if inst.backend_id and identifier.startswith(f"{inst.backend_id}:"):
                identifier = identifier.split(":", 1)[1]
            if identifier == model_key:
                used_slots.add(1)
                continue
            if identifier.startswith(prefix):
                suffix = identifier[len(prefix):]
                if suffix.isdigit():
                    used_slots.add(int(suffix))
        slot = 1
        while slot in used_slots:
            slot += 1
        return f"{model_key}:{slot}"

    async def _load_instance(self, candidate: ModelCandidate) -> Optional[ModelInstanceInfo]:
        backend = self.backends.get(candidate.backend_id)
        if not backend:
            return None
        opts = {"ttl_seconds": candidate.metadata.get("ttl_seconds")}
        opts.update(self._profile_opts(candidate))
        if not opts.get("identifier"):
            opts["identifier"] = await self._next_instance_identifier(candidate.model_key)
        budgets = await self._instance_budgets(candidate, opts)
        reservation: Dict[str, Any] = {}
        reservation_id = None
        logger.info("ModelManager load_instance: model=%s identifier=%s", candidate.model_key, opts.get("identifier"))
        if budgets and self.resource_manager:
            reservation = dict(budgets)
            reservation_id = f"instance:{candidate.model_key}:{uuid.uuid4()}"
            reserve = self.resource_manager.reserve(reservation_id, reservation)
            if not reserve.get("granted"):
                return None
        try:
            instance = await asyncio.wait_for(
                self.spawn_governor.spawn_instance(
                    backend=backend,
                    model_key=candidate.model_key,
                    opts=opts,
                    budgets=reservation or {},
                ),
                timeout=INSTANCE_LOAD_TIMEOUT_S,
            )
            if not instance:
                raise RuntimeError("spawn_governor_denied")
            if reservation:
                instance.resource_reservation = reservation
                if reservation_id:
                    instance.resource_reservation["reservation_id"] = reservation_id
            await self.worker_pool.add(instance)
            return instance
        except asyncio.TimeoutError:
            if reservation and self.resource_manager and reservation_id:
                self.resource_manager.release(reservation_id)
            return None
        except Exception:
            if reservation and self.resource_manager and reservation_id:
                self.resource_manager.release(reservation_id)
            return None

    def _sanitize_vram_budget(
        self,
        vram_mb: Optional[float],
        *,
        gpu_id: Optional[int],
        load_delta: Dict[str, Any],
    ) -> Optional[float]:
        if vram_mb is None:
            return None
        try:
            snapshot = self.telemetry.snapshot()
        except Exception:
            return vram_mb
        totals: Dict[int, float] = {}
        for gpu in snapshot.get("gpus") or []:
            try:
                idx = int(gpu.get("gpu_id") or 0)
            except Exception:
                idx = 0
            totals[idx] = _safe_float(gpu.get("vram_total_mb"), 0.0)
        if not totals:
            return vram_mb
        if gpu_id is not None:
            total = totals.get(gpu_id, 0.0)
        else:
            total = max(totals.values(), default=0.0)
        if total and vram_mb > (total * 2.0):
            fallback = _safe_float(load_delta.get(str(gpu_id)) or load_delta.get(gpu_id), 0.0)
            return fallback if fallback > 0.0 else None
        return vram_mb

    async def _instance_budgets(self, candidate: ModelCandidate, opts: Dict[str, Any]) -> Dict[str, Any]:
        budgets: Dict[str, Any] = {"model_class": _candidate_class(candidate)}
        config_sig = build_config_signature(candidate, opts)
        profile = await self.resource_profile_store.get(candidate.backend_id, candidate.model_key, config_sig)
        if profile:
            profile_status = str(profile.get("status") or "").lower()
            if profile_status != "ok":
                profile = None
        if profile:
            ram_peak = _safe_float(profile.get("ram_instance_peak_bytes"), 0.0)
            load_delta_ram = _safe_float(profile.get("load_delta_ram_bytes"), 0.0)
            if load_delta_ram > 0.0 and ram_peak > (load_delta_ram * 4.0):
                ram_peak = load_delta_ram
            if ram_peak > 0.0:
                try:
                    snapshot = self.telemetry.snapshot()
                    ram = snapshot.get("ram") or {}
                    total_ram = _safe_float(ram.get("total_bytes"), 0.0)
                    used_ram = _safe_float(ram.get("used_bytes"), 0.0)
                    if total_ram > 0.0:
                        allowed = total_ram * (1.0 - (self.ram_headroom_pct / 100.0))
                        usable = allowed - used_ram
                        if usable > 0.0 and ram_peak > usable:
                            ram_peak = usable
                except Exception:
                    pass
                budgets["ram_bytes"] = ram_peak
            vram_map = profile.get("vram_instance_peak_mb_by_gpu") or {}
            load_delta = profile.get("load_delta_vram_mb_by_gpu") or {}
            gpu_id = opts.get("gpu") if opts.get("gpu") is not None else opts.get("gpu_id")
            vram_mb = None
            if gpu_id is not None:
                vram_mb = vram_map.get(str(gpu_id))
            if vram_mb is None and vram_map:
                vals = []
                for val in vram_map.values():
                    try:
                        vals.append(float(val))
                    except Exception:
                        continue
                if vals:
                    vram_mb = max(vals)
            if vram_mb is None and load_delta:
                vals = []
                for val in load_delta.values():
                    try:
                        vals.append(float(val))
                    except Exception:
                        continue
                if vals:
                    vram_mb = max(vals)
            if load_delta:
                if gpu_id is not None:
                    delta_val = _safe_float(load_delta.get(str(gpu_id)) or load_delta.get(gpu_id), 0.0)
                else:
                    delta_val = max((_safe_float(val, 0.0) for val in load_delta.values()), default=0.0)
                if delta_val > 0.0 and vram_mb is not None and vram_mb > (delta_val * 1.5):
                    vram_mb = delta_val
            vram_mb = self._sanitize_vram_budget(vram_mb, gpu_id=gpu_id, load_delta=load_delta)
            if vram_mb is None and profile.get("vram_estimate_only_mb") is not None:
                vram_mb = profile.get("vram_estimate_only_mb")
            if vram_mb is not None:
                budgets["vram_mb"] = vram_mb
            if gpu_id is not None:
                budgets["gpu_id"] = gpu_id
            if _budgets_require_exclusive_ram(budgets):
                budgets["exclusive_ram"] = True
            return budgets
        try:
            estimate = await self.backends[candidate.backend_id].estimate_resources(candidate.model_key, opts)
        except Exception:
            estimate = None
        if estimate:
            budgets = estimate.to_dict()
            if budgets.get("ram_mb") is not None and budgets.get("ram_bytes") is None:
                try:
                    budgets["ram_bytes"] = float(budgets["ram_mb"]) * 1024.0 * 1024.0
                except Exception:
                    pass
            budgets.setdefault("model_class", _candidate_class(candidate))
        if (
            budgets.get("vram_mb") is None
            and budgets.get("ram_mb") is None
            and budgets.get("ram_bytes") is None
        ):
            size_hint = _model_size_hint(candidate.model_key)
            if size_hint is not None:
                vram_mb = float(size_hint) * 900.0
                ram_mb = float(size_hint) * 700.0
                budgets["vram_mb"] = vram_mb
                budgets["ram_mb"] = ram_mb
                budgets["ram_bytes"] = ram_mb * 1024.0 * 1024.0
        budgets.setdefault("model_class", _candidate_class(candidate))
        if _budgets_require_exclusive_ram(budgets):
            budgets["exclusive_ram"] = True
        return budgets

    async def acquire_instance(
        self,
        *,
        required_capabilities: Optional[List[str]] = None,
        objective: Optional[str] = None,
        backlog: int = 1,
        avoid_coder: bool = False,
        prefer_families: Optional[List[str]] = None,
        prefer_models: Optional[List[str]] = None,
    ) -> Optional[ModelInstanceInfo]:
        if self.profiling_config.get("pause_execution") and self.profiling_active():
            return None
        async def _ensure_budget(instance: ModelInstanceInfo, candidates_by_key: Dict[str, ModelCandidate]) -> None:
            if instance.resource_reservation:
                return
            candidate = candidates_by_key.get(instance.model_key)
            if not candidate:
                return
            opts = self._profile_opts(candidate)
            budgets = await self._instance_budgets(candidate, opts)
            if budgets:
                instance.resource_reservation = dict(budgets)
        def _candidate_supports(candidate: ModelCandidate, required: List[str]) -> bool:
            if candidate.metadata.get("type") in ("embedding", "embeddings") and "embeddings" not in required:
                return False
            for cap in required:
                if candidate.capabilities.get(cap) is False:
                    return False
            return True
        async with self._acquire_lock:
            candidates = await self.get_candidates()
            if not candidates:
                candidates = await self.refresh()
            if not candidates:
                return None
            exclusive_model_key = None
            for inst in await self.worker_pool.list_instances():
                if _budgets_require_exclusive_ram(inst.resource_reservation):
                    exclusive_model_key = inst.model_key
                    break
            if exclusive_model_key:
                candidates = [c for c in candidates if c.model_key == exclusive_model_key]
                if not candidates:
                    return None
            prefer_set = {m for m in (prefer_models or []) if m}
            prefer_family_set = {str(f).strip().lower() for f in (prefer_families or []) if f}
            if prefer_family_set:
                for cand in candidates:
                    family = str(cand.metadata.get("family") or "").strip().lower()
                    if family and family in prefer_family_set:
                        prefer_set.add(cand.model_key)
            if not prefer_set:
                prefer_set = {m for m in self.prefer if m}
            required_caps = list(required_capabilities or [])
            if required_caps:
                relaxed: List[str] = []
                for cap in required_caps:
                    supported = False
                    for candidate in candidates:
                        if candidate.metadata.get("type") in ("embedding", "embeddings") and cap != "embeddings":
                            continue
                        if candidate.capabilities.get(cap) is not False:
                            supported = True
                            break
                    if supported:
                        relaxed.append(cap)
                required_caps = relaxed
            prefer_small = self._resource_pressure() >= RESOURCE_PRESSURE_THRESHOLD
            candidate_sets: List[List[ModelCandidate]] = [candidates]
            if avoid_coder:
                non_coder = [c for c in candidates if not _is_coder_candidate(c)]
                if non_coder:
                    candidate_sets = [non_coder, candidates]

            for attempt_candidates in candidate_sets:
                selector = ModelSelector(attempt_candidates, prefer=prefer_set)
                candidates_by_key = {c.model_key: c for c in attempt_candidates}
                eligible_keys = {c.model_key for c in attempt_candidates if _candidate_supports(c, required_caps)}
                if not eligible_keys:
                    eligible_keys = set(candidates_by_key)
                ready = await self.worker_pool.list_ready()
                if not ready:
                    await self.refresh()
                    ready = await self.worker_pool.list_ready()
                ready = [inst for inst in ready if inst.model_key in candidates_by_key]
                all_ready = list(ready)
                preferred_keys: set[str] = set()
                prefer_only = False
                if prefer_family_set:
                    preferred_keys = {
                        c.model_key
                        for c in attempt_candidates
                        if str(c.metadata.get("family") or "").strip().lower() in prefer_family_set
                    }
                    if preferred_keys:
                        preferred_ready = [inst for inst in ready if inst.model_key in preferred_keys]
                        if preferred_ready:
                            ready = preferred_ready
                        else:
                            ready = []
                            prefer_only = True
                instance = selector.choose_instance(
                    ready,
                    candidates_by_key,
                    required_capabilities=required_caps,
                    objective=objective or self.routing_objective,
                    prefer_small=prefer_small,
                )
                if instance:
                    await _ensure_budget(instance, candidates_by_key)
                    await self.worker_pool.mark_busy(instance.instance_id)
                    return instance
                if not self.autoscaler.enabled:
                    continue
                if self._profiling_active > 0 and ready:
                    continue
                backend_id = attempt_candidates[0].backend_id if attempt_candidates else None
                if backend_id:
                    active_keys = set(eligible_keys)
                    if prefer_only and preferred_keys:
                        active_keys = active_keys.intersection(preferred_keys)
                    instances = await self.worker_pool.list_instances()
                    active_instances = [
                        inst
                        for inst in instances
                        if inst.backend_id == backend_id and inst.model_key in active_keys
                    ]
                    active_total = len(active_instances)
                    min_instances = int(self.autoscaler.min_instances.get("executor") or 0)
                    total_cap = self.autoscaler.per_backend_max_instances.get(backend_id)
                    if self.autoscaler.global_max_instances is not None:
                        if total_cap is None:
                            total_cap = self.autoscaler.global_max_instances
                        else:
                            total_cap = min(total_cap, self.autoscaler.global_max_instances)
                    desired_total = max(min_instances, active_total, backlog)
                    if total_cap is not None:
                        desired_total = min(desired_total, total_cap)
                    if desired_total > active_total:
                        logger.info(
                            "ModelManager autoscale: backend=%s backlog=%s active=%s desired=%s eligible=%s",
                            backend_id,
                            backlog,
                            active_total,
                            desired_total,
                            len(eligible_keys),
                        )
                    pending_candidates = list(attempt_candidates)
                    if prefer_only and preferred_keys:
                        pending_candidates = [c for c in pending_candidates if c.model_key in preferred_keys]
                    target_candidate = None
                    if pending_candidates:
                        selector = ModelSelector(pending_candidates, prefer=prefer_set)
                        target_candidate = selector.choose_candidate(
                            required_capabilities=required_caps,
                            objective=objective or self.routing_objective,
                            prefer_small=prefer_small,
                        )
                    target_key = target_candidate.model_key if target_candidate else None
                    target_active = (
                        sum(1 for inst in active_instances if inst.model_key == target_key) if target_key else 0
                    )
                    desired_target = max(min_instances, target_active, backlog)
                    if total_cap is not None:
                        available_slots = max(total_cap - (active_total - target_active), 0)
                        desired_target = min(desired_target, available_slots)
                    if desired_target > target_active:
                        logger.info(
                            "ModelManager autoscale_target: backend=%s model=%s backlog=%s active=%s desired=%s total_active=%s total_cap=%s",
                            backend_id,
                            target_key,
                            backlog,
                            target_active,
                            desired_target,
                            active_total,
                            total_cap,
                        )
                    prefer_small_load = prefer_small
                    if target_candidate:
                        while target_active < desired_target:
                            if not await self._capacity_allows(target_candidate):
                                logger.info(
                                    "ModelManager capacity_denied: model=%s",
                                    target_candidate.model_key,
                                )
                                break
                            loaded = await self._load_instance(target_candidate)
                            if loaded:
                                target_active += 1
                                active_total += 1
                                logger.info(
                                    "ModelManager loaded_instance: model=%s identifier=%s active=%s/%s",
                                    target_candidate.model_key,
                                    loaded.api_identifier,
                                    target_active,
                                    desired_target,
                                )
                            else:
                                prefer_small_load = True
                                break
                            if await self._evict_idle_instance(exclude_model_key=target_candidate.model_key):
                                continue
                    if active_total < desired_total:
                        pending_candidates = [c for c in pending_candidates if c.model_key != target_key]
                        while active_total < desired_total and pending_candidates:
                            selector = ModelSelector(pending_candidates, prefer=prefer_set)
                            candidate = selector.choose_candidate(
                                required_capabilities=required_caps,
                                objective=objective or self.routing_objective,
                                prefer_small=prefer_small_load,
                            )
                            if not candidate:
                                break
                            if not await self._capacity_allows(candidate):
                                logger.info(
                                    "ModelManager capacity_denied: model=%s",
                                    candidate.model_key,
                                )
                                pending_candidates = [
                                    c for c in pending_candidates if c.model_key != candidate.model_key
                                ]
                                continue
                            loaded = await self._load_instance(candidate)
                            if loaded:
                                active_total += 1
                                logger.info(
                                    "ModelManager loaded_instance: model=%s identifier=%s active=%s/%s",
                                    candidate.model_key,
                                    loaded.api_identifier,
                                    active_total,
                                    desired_total,
                                )
                            else:
                                pending_candidates = [
                                    c for c in pending_candidates if c.model_key != candidate.model_key
                                ]
                                prefer_small_load = True
                            if await self._evict_idle_instance(exclude_model_key=candidate.model_key):
                                continue
                    ready = await self.worker_pool.list_ready()
                    ready = [inst for inst in ready if inst.model_key in candidates_by_key]
                    if prefer_only and preferred_keys:
                        preferred_ready = [inst for inst in ready if inst.model_key in preferred_keys]
                        if preferred_ready:
                            ready = preferred_ready
                ready = await self.worker_pool.list_ready()
                ready = [inst for inst in ready if inst.model_key in candidates_by_key]
                if prefer_only and preferred_keys:
                    preferred_ready = [inst for inst in ready if inst.model_key in preferred_keys]
                    if preferred_ready:
                        ready = preferred_ready
                instance = selector.choose_instance(
                    ready,
                    candidates_by_key,
                    required_capabilities=required_caps,
                    objective=objective or self.routing_objective,
                    prefer_small=prefer_small,
                )
                if instance:
                    await _ensure_budget(instance, candidates_by_key)
                    await self.worker_pool.mark_busy(instance.instance_id)
                    return instance
                if prefer_only and all_ready:
                    instance = selector.choose_instance(
                        all_ready,
                        candidates_by_key,
                        required_capabilities=required_caps,
                        objective=objective or self.routing_objective,
                        prefer_small=prefer_small,
                    )
                    if instance:
                        await _ensure_budget(instance, candidates_by_key)
                        await self.worker_pool.mark_busy(instance.instance_id)
                        return instance
            return None

    async def _capacity_allows(self, candidate: ModelCandidate) -> bool:
        opts = self._profile_opts(candidate)
        config_sig = build_config_signature(candidate, opts)
        profile = await self.resource_profile_store.get(candidate.backend_id, candidate.model_key, config_sig)
        if profile:
            profile_status = str(profile.get("status") or "").lower()
            if profile_status != "ok":
                profile = None
        budgets = await self._instance_budgets(candidate, opts)
        instances = await self.worker_pool.list_instances()
        if _budgets_require_exclusive_ram(budgets):
            if instances:
                logger.info(
                    "ModelManager exclusive_ram_block: model=%s active_instances=%s",
                    candidate.model_key,
                    len(instances),
                )
                return False
        else:
            if any(_budgets_require_exclusive_ram(inst.resource_reservation) for inst in instances):
                logger.info(
                    "ModelManager exclusive_ram_active: model=%s",
                    candidate.model_key,
                )
                return False
        snapshot = self.telemetry.snapshot()
        if not profile:
            max_instances = self._estimate_capacity_from_budgets(budgets, snapshot)
            if max_instances is None:
                return True
            if max_instances <= 0:
                return False
            active = sum(
                1
                for inst in instances
                if inst.backend_id == candidate.backend_id and inst.model_key == candidate.model_key
            )
            return active < max_instances
        capacity = self.capacity_planner.compute_capacity(profile, snapshot)
        max_instances = int(capacity.get("max_instances") or 0)
        if max_instances <= 0:
            return False
        instances = await self.worker_pool.list_instances()
        active = sum(
            1
            for inst in instances
            if inst.backend_id == candidate.backend_id and inst.model_key == candidate.model_key
        )
        return active < max_instances

    def _estimate_capacity_from_budgets(
        self,
        budgets: Optional[Dict[str, Any]],
        snapshot: Dict[str, Any],
    ) -> Optional[int]:
        if not budgets:
            return None
        ram_bytes = budgets.get("ram_bytes")
        if ram_bytes is None and budgets.get("ram_mb") is not None:
            ram_bytes = _safe_float(budgets.get("ram_mb")) * 1024.0 * 1024.0
        vram_mb = budgets.get("vram_mb")
        gpu_id = budgets.get("gpu_id")
        slots: List[int] = []
        if ram_bytes:
            ram = snapshot.get("ram") or {}
            total_ram = _safe_float(ram.get("total_bytes"), 0.0)
            used_ram = _safe_float(ram.get("used_bytes"), 0.0)
            if total_ram > 0:
                allowed = total_ram * (1.0 - (self.ram_headroom_pct / 100.0)) - used_ram
                slots.append(max(int(allowed // max(ram_bytes, 1.0)), 0))
        if vram_mb:
            gpus = snapshot.get("gpus") or []
            target = None
            if gpu_id is not None:
                for gpu in gpus:
                    try:
                        if int(gpu.get("gpu_id") or 0) == int(gpu_id):
                            target = gpu
                            break
                    except Exception:
                        continue
            if target is None and gpus:
                target = max(
                    gpus,
                    key=lambda g: _safe_float(g.get("vram_free_mb"), 0.0),
                )
            if target is not None:
                total_mb = _safe_float(target.get("vram_total_mb"), 0.0)
                used_mb = _safe_float(target.get("vram_used_mb"), 0.0)
                if total_mb > 0:
                    allowed = total_mb * (1.0 - (self.vram_headroom_pct / 100.0)) - used_mb
                    slots.append(max(int(allowed // max(float(vram_mb), 1.0)), 0))
        if not slots:
            return None
        return max(min(slots), 0)

    async def release_instance(self, instance_id: str) -> None:
        await self.worker_pool.release(instance_id)

    async def unload_idle(self, idle_seconds: int = 180) -> int:
        removed = 0
        now = time.monotonic()
        for instance in await self.worker_pool.list_instances():
            if instance.status == "busy":
                continue
            if instance.last_used_at and (now - instance.last_used_at) < idle_seconds:
                continue
            backend = self.backends.get(instance.backend_id)
            if not backend:
                continue
            try:
                await backend.unload_instance(instance.instance_id)
            except Exception:
                continue
            await self.worker_pool.remove(instance.instance_id)
            reservation_id = instance.resource_reservation.get("reservation_id")
            if reservation_id and self.resource_manager:
                self.resource_manager.release(reservation_id)
            removed += 1
        return removed

    async def call(
        self,
        *,
        required_capabilities: Optional[List[str]] = None,
        objective: Optional[str] = None,
        request: Dict[str, Any],
        avoid_coder: bool = False,
        prefer_families: Optional[List[str]] = None,
        prefer_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        response, _ = await self.call_with_instance(
            required_capabilities=required_capabilities,
            objective=objective,
            request=request,
            avoid_coder=avoid_coder,
            prefer_families=prefer_families,
            prefer_models=prefer_models,
        )
        return response

    async def call_with_instance(
        self,
        *,
        required_capabilities: Optional[List[str]] = None,
        objective: Optional[str] = None,
        request: Dict[str, Any],
        avoid_coder: bool = False,
        prefer_families: Optional[List[str]] = None,
        prefer_models: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], ModelInstanceInfo]:
        async def _acquire() -> Optional[ModelInstanceInfo]:
            instance = await self.acquire_instance(
                required_capabilities=required_capabilities,
                objective=objective,
                backlog=1,
                avoid_coder=avoid_coder,
                prefer_families=prefer_families,
                prefer_models=prefer_models,
            )
            if not instance and INSTANCE_ACQUIRE_WAIT_S > 0:
                deadline = time.monotonic() + INSTANCE_ACQUIRE_WAIT_S
                while time.monotonic() < deadline:
                    await asyncio.sleep(INSTANCE_ACQUIRE_POLL_S)
                    instance = await self.acquire_instance(
                        required_capabilities=required_capabilities,
                        objective=objective,
                        backlog=1,
                        avoid_coder=avoid_coder,
                        prefer_families=prefer_families,
                        prefer_models=prefer_models,
                    )
                    if instance:
                        break
            return instance

        last_error: Optional[Exception] = None
        for _ in range(2):
            instance = await _acquire()
            if not instance:
                break
            backend = self.backends.get(instance.backend_id)
            if not backend:
                await self.release_instance(instance.instance_id)
                raise RuntimeError("Backend unavailable.")
            monitor_id = None
            base_snapshot = None
            if self._observe_on_use_enabled():
                try:
                    base_snapshot = self.telemetry.snapshot()
                    sample_interval_ms = int(self.profiling_config.get("sample_interval_ms") or 250)
                    monitor_id = self.telemetry.monitor_start(sample_interval_ms)
                except Exception:
                    monitor_id = None
                    base_snapshot = None
            try:
                response = await backend.call_chat_completion(instance, request)
                return response, instance
            except Exception as exc:
                last_error = exc
                if self._is_instance_unavailable_error(exc):
                    await self._drop_instance(instance)
                    continue
                raise
            finally:
                peak_snapshot = None
                samples_summary = None
                if monitor_id:
                    try:
                        result = self.telemetry.monitor_stop(monitor_id)
                        peak_snapshot = result.get("peak_snapshot") or {}
                        samples_summary = result.get("samples_summary") or {}
                    except Exception:
                        peak_snapshot = None
                        samples_summary = None
                if base_snapshot and peak_snapshot:
                    gpu_id = None
                    if instance.resource_reservation and instance.resource_reservation.get("gpu_id") is not None:
                        try:
                            gpu_id = int(instance.resource_reservation.get("gpu_id"))
                        except Exception:
                            gpu_id = None
                    asyncio.create_task(
                        self._record_usage_profile(
                            backend_id=instance.backend_id,
                            model_key=instance.model_key,
                            base_snapshot=base_snapshot,
                            peak_snapshot=peak_snapshot,
                            gpu_id=gpu_id,
                            samples_summary=samples_summary,
                        )
                    )
                await self.release_instance(instance.instance_id)
        if last_error:
            raise last_error
        raise RuntimeError("No suitable model instance available.")

    async def catalog(self) -> Dict[str, Any]:
        candidates = await self.get_candidates()
        snapshot = self.telemetry.snapshot()
        for candidate in candidates:
            config_sig = build_config_signature(candidate, self._profile_opts(candidate))
            profile = await self.resource_profile_store.get(candidate.backend_id, candidate.model_key, config_sig)
            if profile:
                capacity = self.capacity_planner.compute_capacity(profile, snapshot)
                candidate.profile["capacity"] = capacity
        return {
            "candidates": [c.to_dict() for c in candidates],
            "instances": [i.to_dict() for i in await self.worker_pool.list_instances()],
        }

    async def start_profiling(
        self,
        *,
        model_keys: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        if self._manual_profile_task and not self._manual_profile_task.done():
            return dict(self._profile_status)
        candidates = await self.get_candidates()
        if not candidates:
            candidates = await self.refresh()
        if model_keys:
            allowed = {str(key) for key in model_keys if key}
            candidates = [c for c in candidates if c.model_key in allowed]
        self._profile_status = {
            "running": True,
            "current": None,
            "completed": 0,
            "total": len(candidates),
            "errors": [],
        }

        async def _runner() -> None:
            ttl = int(self.profiling_config.get("profile_ttl_s") or self.profile_ttl_s)
            for candidate in candidates:
                self._profile_status["current"] = candidate.model_key
                opts = self._profile_opts(candidate)
                if not force:
                    config_sig = build_config_signature(candidate, opts)
                    cached = await self.resource_profile_store.get(
                        candidate.backend_id, candidate.model_key, config_sig
                    )
                    profiled_at = _parse_utc_timestamp((cached or {}).get("profiled_at"))
                    if profiled_at and (time.time() - profiled_at) < ttl:
                        self._profile_status["completed"] += 1
                        continue
                try:
                    await self._profile_candidate(candidate, opts)
                except Exception as exc:
                    self._profile_status["errors"].append(
                        {"model_key": candidate.model_key, "error": str(exc)}
                    )
                finally:
                    self._profile_status["completed"] += 1
            self._profile_status["running"] = False
            self._profile_status["current"] = None

        self._manual_profile_task = asyncio.create_task(_runner())
        return dict(self._profile_status)

    def profile_status(self) -> Dict[str, Any]:
        return dict(self._profile_status)
