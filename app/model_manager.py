import asyncio
import json
import time
import calendar
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import aiosqlite

from .capacity_planner import CapacityPlanner
from .load_profiler import LoadProfiler, build_config_signature
from .resource_profiles import ModelResourceProfileStore
from .resource_telemetry import ResourceTelemetry
from .spawn_governor import SpawnGovernor


PROFILE_DEFAULT_TTL_S = 6 * 60 * 60
PROFILE_MAX_CONCURRENT = 1
INSTANCE_BUSY_TTL_S = 60 * 5
INSTANCE_LOAD_TIMEOUT_S = 25


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
        return (
            _safe_float(profile.get("tool_call_success_rate"), 0.0)
            + _safe_float(profile.get("json_schema_success_rate"), 0.0)
            + (1.0 - _safe_float(profile.get("error_rate"), 0.0))
        )

    def _latency_score(self, candidate: ModelCandidate) -> float:
        profile = candidate.profile or {}
        tps = _safe_float(profile.get("tps"), 0.0)
        latency = _safe_float(profile.get("latency_ms"), 0.0)
        if tps == 0.0 and latency == 0.0:
            size_hint = _model_size_hint(candidate.model_key)
            if size_hint is not None:
                return -size_hint
        return tps - (latency / 100.0)

    def choose_candidate(
        self,
        required_capabilities: Optional[List[str]] = None,
        objective: str = "balanced",
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
                score += 1.5
            if best is None or score > best[0]:
                best = (score, candidate)
        return best[1] if best else None

    def choose_instance(
        self,
        instances: List[ModelInstanceInfo],
        candidates_by_key: Dict[str, ModelCandidate],
        required_capabilities: Optional[List[str]] = None,
        objective: str = "balanced",
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
                score += 1.5
            if inst.status == "busy":
                score -= 1.0
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
                discovered.append(candidate)
        filtered = self._filter_candidates(discovered)
        async with self._candidate_lock:
            self._candidates = filtered
        return filtered

    async def _sync_loaded_instances(self) -> None:
        for backend_id, backend in self.backends.items():
            try:
                loaded = await backend.list_loaded()
            except Exception:
                loaded = []
            for instance in loaded:
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
                candidate.capabilities = cached.capabilities
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
                profiled_at = _parse_utc_timestamp(resource_profile.get("profiled_at"))
                if profiled_at and (now - profiled_at) < profile_ttl:
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
        if cfg_context:
            opts["context_length"] = int(cfg_context)
        meta_context = candidate.metadata.get("context_length")
        if meta_context and "context_length" not in opts:
            opts["context_length"] = int(meta_context)
        return opts

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
            candidates_by_key = {c.model_key: c for c in candidates}
            preferred = selector.choose_instance(
                ready_instances,
                candidates_by_key,
                required_capabilities=["tool_use"] if self.tool_required_by_default else [],
                objective=self.routing_objective,
            )
            if preferred:
                return preferred
            for instance in ready_instances:
                candidate = candidates_by_key.get(instance.model_key)
                if candidate and candidate.metadata.get("type") not in ("embedding", "embeddings"):
                    return instance
        selector = ModelSelector(candidates, prefer=self.prefer)
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

    async def _load_instance(self, candidate: ModelCandidate) -> Optional[ModelInstanceInfo]:
        backend = self.backends.get(candidate.backend_id)
        if not backend:
            return None
        opts = {"ttl_seconds": candidate.metadata.get("ttl_seconds")}
        opts.update(self._profile_opts(candidate))
        budgets = await self._instance_budgets(candidate, opts)
        reservation: Dict[str, Any] = {}
        reservation_id = None
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

    async def _instance_budgets(self, candidate: ModelCandidate, opts: Dict[str, Any]) -> Dict[str, Any]:
        budgets: Dict[str, Any] = {}
        config_sig = build_config_signature(candidate, opts)
        profile = await self.resource_profile_store.get(candidate.backend_id, candidate.model_key, config_sig)
        if profile:
            ram_peak = profile.get("ram_instance_peak_bytes")
            if ram_peak:
                budgets["ram_bytes"] = ram_peak
            vram_map = profile.get("vram_instance_peak_mb_by_gpu") or {}
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
            if vram_mb is None and profile.get("vram_estimate_only_mb") is not None:
                vram_mb = profile.get("vram_estimate_only_mb")
            if vram_mb is not None:
                budgets["vram_mb"] = vram_mb
            if gpu_id is not None:
                budgets["gpu_id"] = gpu_id
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
        return budgets

    async def acquire_instance(
        self,
        *,
        required_capabilities: Optional[List[str]] = None,
        objective: Optional[str] = None,
        backlog: int = 1,
        avoid_coder: bool = False,
    ) -> Optional[ModelInstanceInfo]:
        async with self._acquire_lock:
            candidates = await self.get_candidates()
            if not candidates:
                candidates = await self.refresh()
            if not candidates:
                return None
            candidate_sets: List[List[ModelCandidate]] = [candidates]
            if avoid_coder:
                non_coder = [c for c in candidates if not _is_coder_key(c.model_key)]
                if non_coder:
                    candidate_sets = [non_coder, candidates]

            for attempt_candidates in candidate_sets:
                selector = ModelSelector(attempt_candidates, prefer=self.prefer)
                candidates_by_key = {c.model_key: c for c in attempt_candidates}
                ready = await self.worker_pool.list_ready()
                ready = [inst for inst in ready if inst.model_key in candidates_by_key]
                instance = selector.choose_instance(
                    ready,
                    candidates_by_key,
                    required_capabilities=required_capabilities,
                    objective=objective or self.routing_objective,
                )
                if instance:
                    await self.worker_pool.mark_busy(instance.instance_id)
                    return instance
                if not self.autoscaler.enabled:
                    continue
                if self._profiling_active > 0:
                    continue
                backend_id = attempt_candidates[0].backend_id if attempt_candidates else None
                if backend_id:
                    desired = self.autoscaler.desired_instances(backend_id, backlog, len(ready))
                    pending_candidates = list(attempt_candidates)
                    while len(ready) < desired and pending_candidates:
                        selector = ModelSelector(pending_candidates, prefer=self.prefer)
                        candidate = selector.choose_candidate(
                            required_capabilities=required_capabilities, objective=objective or self.routing_objective
                        )
                        if not candidate:
                            break
                        if not await self._capacity_allows(candidate):
                            pending_candidates = [c for c in pending_candidates if c.model_key != candidate.model_key]
                            continue
                        loaded = await self._load_instance(candidate)
                        if not loaded:
                            break
                        ready = await self.worker_pool.list_ready()
                        ready = [inst for inst in ready if inst.model_key in candidates_by_key]
                ready = await self.worker_pool.list_ready()
                ready = [inst for inst in ready if inst.model_key in candidates_by_key]
                instance = selector.choose_instance(
                    ready,
                    candidates_by_key,
                    required_capabilities=required_capabilities,
                    objective=objective or self.routing_objective,
                )
                if instance:
                    await self.worker_pool.mark_busy(instance.instance_id)
                    return instance
            return None

    async def _capacity_allows(self, candidate: ModelCandidate) -> bool:
        opts = self._profile_opts(candidate)
        config_sig = build_config_signature(candidate, opts)
        profile = await self.resource_profile_store.get(candidate.backend_id, candidate.model_key, config_sig)
        if not profile:
            return True
        capacity = self.capacity_planner.compute_capacity(profile, self.telemetry.snapshot())
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
    ) -> Dict[str, Any]:
        response, _ = await self.call_with_instance(
            required_capabilities=required_capabilities,
            objective=objective,
            request=request,
            avoid_coder=avoid_coder,
        )
        return response

    async def call_with_instance(
        self,
        *,
        required_capabilities: Optional[List[str]] = None,
        objective: Optional[str] = None,
        request: Dict[str, Any],
        avoid_coder: bool = False,
    ) -> Tuple[Dict[str, Any], ModelInstanceInfo]:
        instance = await self.acquire_instance(
            required_capabilities=required_capabilities,
            objective=objective,
            backlog=1,
            avoid_coder=avoid_coder,
        )
        if not instance:
            raise RuntimeError("No suitable model instance available.")
        backend = self.backends.get(instance.backend_id)
        if not backend:
            await self.release_instance(instance.instance_id)
            raise RuntimeError("Backend unavailable.")
        try:
            response = await backend.call_chat_completion(instance, request)
            return response, instance
        finally:
            await self.release_instance(instance.instance_id)

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
