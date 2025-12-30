import asyncio
import json
import time
import uuid
from statistics import median
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .resource_profiles import PROFILE_VERSION, ModelResourceProfileStore, utc_now
from .resource_telemetry import ResourceTelemetry

if TYPE_CHECKING:
    from .model_manager import ModelBackend, ModelCandidate, ModelInstanceInfo


DEFAULT_SAMPLE_INTERVAL_MS = 250
DEFAULT_TEST_TIMEOUT_S = 120
DEFAULT_SETTLE_TIMEOUT_S = 12
DEFAULT_SETTLE_RAM_TOLERANCE_BYTES = 256 * 1024 * 1024
DEFAULT_SETTLE_VRAM_TOLERANCE_MB = 256.0


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return default


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    try:
        return float(median(values))
    except Exception:
        return float(values[-1])


def _tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(len(text.split()), 1)


def _extract_text(response: Dict[str, Any]) -> str:
    if not isinstance(response, dict):
        return ""
    if response.get("choices"):
        choice = response.get("choices")[0] if response.get("choices") else {}
        message = choice.get("message") or {}
        content = message.get("content") or ""
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=True)
    return ""


def _extract_usage_tokens(response: Dict[str, Any]) -> Optional[int]:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage") or {}
    for key in ("completion_tokens", "output_tokens", "total_tokens"):
        if key in usage:
            try:
                return int(usage[key])
            except Exception:
                continue
    return None


def _parse_tool_call(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    call = tool_calls[0]
    func = call.get("function") or {}
    args = func.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"raw": args}
    return {"name": func.get("name"), "arguments": args}


def _parse_structured_text(content: str) -> Optional[dict]:
    if not content:
        return None
    try:
        return json.loads(content)
    except Exception:
        return None


def _pad_text(base: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        return base
    base_tokens = len(base.split())
    if base_tokens >= target_tokens:
        return base
    filler = " filler"
    needed = max(target_tokens - base_tokens, 0)
    return base + (filler * needed)


def _gpu_used_map(snapshot: Dict[str, Any]) -> Dict[int, float]:
    used: Dict[int, float] = {}
    for gpu in snapshot.get("gpus") or []:
        try:
            idx = int(gpu.get("gpu_id") or 0)
        except Exception:
            idx = 0
        used[idx] = _safe_float(gpu.get("vram_used_mb"))
    return used


def _gpu_total_map(snapshot: Dict[str, Any]) -> Dict[int, float]:
    total: Dict[int, float] = {}
    for gpu in snapshot.get("gpus") or []:
        try:
            idx = int(gpu.get("gpu_id") or 0)
        except Exception:
            idx = 0
        total[idx] = _safe_float(gpu.get("vram_total_mb"))
    return total


def _gpu_delta_map(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for idx, used in after.items():
        base = before.get(idx, 0.0)
        deltas[str(idx)] = max(used - base, 0.0)
    return deltas


def _compute_return_pct(baseline: float, peak: float, current: float) -> float:
    peak_delta = max(peak - baseline, 0.0)
    if peak_delta <= 0:
        return 100.0
    remaining = max(current - baseline, 0.0)
    return max(0.0, min(100.0, (1.0 - (remaining / peak_delta)) * 100.0))


def build_config_signature(candidate: "ModelCandidate", opts: Dict[str, Any]) -> str:
    base = {
        "context_length": opts.get("context_length") or candidate.metadata.get("context_length"),
        "quantization": candidate.metadata.get("quantization") or candidate.metadata.get("quant"),
        "gpu_id": opts.get("gpu"),
    }
    return json.dumps(base, sort_keys=True, ensure_ascii=True)


def _build_test_suite() -> List[Dict[str, Any]]:
    schema = {
        "name": "profile_structured",
        "schema": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "count": {"type": "integer"},
                "label": {"type": "string"},
            },
            "required": ["ok", "count", "label"],
            "additionalProperties": False,
        },
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Ping tool.",
                "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
            },
        }
    ]
    return [
        {
            "name": "sanity_short",
            "messages": [
                {"role": "system", "content": "Reply with a single word only."},
                {"role": "user", "content": "Reply with: pong"},
            ],
            "max_tokens": 24,
        },
        {
            "name": "medium_completion",
            "messages": [
                {"role": "system", "content": "Summarize in three short bullet points."},
                {
                    "role": "user",
                    "content": "Explain how a solar eclipse happens and why it is temporary.",
                },
            ],
            "max_tokens": 160,
            "padding_tokens": 200,
        },
        {
            "name": "long_completion",
            "messages": [
                {"role": "system", "content": "Write a detailed, multi-paragraph explanation."},
                {
                    "role": "user",
                    "content": "Explain photosynthesis step by step, with examples.",
                },
            ],
            "max_tokens": 420,
            "padding_tokens": 800,
        },
        {
            "name": "structured_json",
            "messages": [
                {"role": "system", "content": "Return JSON that matches the schema."},
                {"role": "user", "content": "Return {\"ok\": true, \"count\": 3, \"label\": \"alpha\"}."},
            ],
            "max_tokens": 120,
            "response_format": {"type": "json_schema", "json_schema": schema},
            "requires_structured": True,
            "fallback_messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Return {\"ok\": true, \"count\": 3, \"label\": \"alpha\"}."},
            ],
        },
        {
            "name": "tool_call",
            "messages": [
                {"role": "system", "content": "Return a tool call only."},
                {"role": "user", "content": "Call the ping tool with {\"value\": 7}."},
            ],
            "max_tokens": 120,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "ping"}},
            "requires_tools": True,
            "fallback_messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Return {\"tool\": \"ping\", \"arguments\": {\"value\": 7}}."},
            ],
        },
    ]


class LoadProfiler:
    def __init__(
        self,
        *,
        telemetry: ResourceTelemetry,
        store: ModelResourceProfileStore,
        ram_headroom_pct: float = 10.0,
        vram_headroom_pct: float = 10.0,
        enforce_headroom: bool = False,
        sample_interval_ms: int = DEFAULT_SAMPLE_INTERVAL_MS,
        repeats: int = 1,
        test_timeout_s: int = DEFAULT_TEST_TIMEOUT_S,
        settle_timeout_s: int = DEFAULT_SETTLE_TIMEOUT_S,
        max_output_tokens: Optional[int] = None,
        settle_ram_tolerance_bytes: int = DEFAULT_SETTLE_RAM_TOLERANCE_BYTES,
        settle_vram_tolerance_mb: float = DEFAULT_SETTLE_VRAM_TOLERANCE_MB,
    ) -> None:
        self.telemetry = telemetry
        self.store = store
        self.ram_headroom_pct = ram_headroom_pct
        self.vram_headroom_pct = vram_headroom_pct
        self.enforce_headroom = enforce_headroom
        self.sample_interval_ms = sample_interval_ms
        self.repeats = max(1, repeats)
        self.test_timeout_s = test_timeout_s
        self.settle_timeout_s = settle_timeout_s
        self.max_output_tokens = max_output_tokens
        self.settle_ram_tolerance_bytes = settle_ram_tolerance_bytes
        self.settle_vram_tolerance_mb = settle_vram_tolerance_mb
        self.tests = _build_test_suite()

    def _headroom_allows(
        self,
        snapshot: Dict[str, Any],
        vram_need_mb: Optional[float],
        *,
        ram_need_bytes: float = 0.0,
        gpu_id: Optional[int] = None,
    ) -> bool:
        if not self.enforce_headroom:
            return True
        ram = snapshot.get("ram") or {}
        total_ram = _safe_float(ram.get("total_bytes"))
        used_ram = _safe_float(ram.get("used_bytes"))
        if total_ram:
            allowed = total_ram * (1.0 - (self.ram_headroom_pct / 100.0))
            if used_ram + ram_need_bytes > allowed:
                return False
        gpus = snapshot.get("gpus") or []
        if not gpus or vram_need_mb is None:
            return True
        for gpu in gpus:
            total = _safe_float(gpu.get("vram_total_mb"))
            used = _safe_float(gpu.get("vram_used_mb"))
            if gpu_id is not None:
                try:
                    if int(gpu.get("gpu_id") or 0) != gpu_id:
                        continue
                except Exception:
                    continue
            if total:
                allowed = total * (1.0 - (self.vram_headroom_pct / 100.0))
                if used + vram_need_mb > allowed:
                    return False
        return True

    async def profile_candidate(
        self,
        *,
        backend: "ModelBackend",
        candidate: "ModelCandidate",
        opts: Optional[Dict[str, Any]] = None,
        ttl_seconds: int = 0,
    ) -> Dict[str, Any]:
        opts = dict(opts or {})
        config_sig = build_config_signature(candidate, opts)
        await backend.ensure_server_running()
        estimate = None
        try:
            estimate = await backend.estimate_resources(candidate.model_key, opts)
        except Exception:
            estimate = None
        estimate_vram = _safe_float(getattr(estimate, "vram_mb", None)) if estimate else None
        estimate_ram = _safe_float(getattr(estimate, "ram_mb", None)) if estimate else None
        estimate_ram_bytes = estimate_ram * 1024.0 * 1024.0 if estimate_ram else None
        base_snapshot = self.telemetry.snapshot()
        gpu_id = None
        if opts.get("gpu") is not None:
            try:
                gpu_id = int(opts.get("gpu"))
            except Exception:
                gpu_id = None
        if gpu_id is None and opts.get("gpu_id") is not None:
            try:
                gpu_id = int(opts.get("gpu_id"))
            except Exception:
                gpu_id = None
        if not self._headroom_allows(
            base_snapshot,
            estimate_vram,
            ram_need_bytes=estimate_ram_bytes or 0.0,
            gpu_id=gpu_id,
        ):
            return {
                "backend_id": candidate.backend_id,
                "model_key": candidate.model_key,
                "config_signature": config_sig,
                "profile_version": PROFILE_VERSION,
                "profiled_at": utc_now(),
                "status": "skipped_headroom",
                "baseline_snapshot": base_snapshot,
            }

        instance = None
        unloaded = False
        instance_id = f"profile-{candidate.model_key}-{uuid.uuid4().hex[:8]}"
        opts.setdefault("identifier", instance_id)
        opts.setdefault("ttl_seconds", 240)
        try:
            instance = await backend.load_instance(candidate.model_key, opts)
        except Exception as exc:
            return {
                "backend_id": candidate.backend_id,
                "model_key": candidate.model_key,
                "config_signature": config_sig,
                "profile_version": PROFILE_VERSION,
                "profiled_at": utc_now(),
                "status": "load_failed",
                "error": str(exc),
                "baseline_snapshot": base_snapshot,
            }

        try:
            post_load = self.telemetry.snapshot()
            if not self._headroom_allows(post_load, 0.0, ram_need_bytes=0.0, gpu_id=gpu_id):
                try:
                    await backend.unload_instance(instance.instance_id)
                except Exception:
                    pass
                unloaded = True
                return {
                    "backend_id": candidate.backend_id,
                    "model_key": candidate.model_key,
                    "config_signature": config_sig,
                    "profile_version": PROFILE_VERSION,
                    "profiled_at": utc_now(),
                    "status": "headroom_exceeded",
                    "baseline_snapshot": base_snapshot,
                    "post_load_snapshot": post_load,
                }
            load_delta_ram = max(
                _safe_float((post_load.get("ram") or {}).get("used_bytes"))
                - _safe_float((base_snapshot.get("ram") or {}).get("used_bytes")),
                0.0,
            )
            baseline_vram_used = _gpu_used_map(base_snapshot)
            load_vram_used = _gpu_used_map(post_load)
            load_delta_vram = _gpu_delta_map(load_vram_used, baseline_vram_used)
            vram_estimate_only = False
            if not load_delta_vram and estimate_vram is not None:
                load_delta_vram = {"0": estimate_vram}
                vram_estimate_only = True
            per_test_runs: Dict[str, List[Dict[str, Any]]] = {t["name"]: [] for t in self.tests}
            tool_supported = backend.supports_tools()
            structured_supported: Optional[bool] = candidate.capabilities.get("structured_output")
            for repeat_idx in range(self.repeats):
                for test in self.tests:
                    result = await self._run_test(
                        backend=backend,
                        instance=instance,
                        test=test,
                        base_snapshot=base_snapshot,
                        tool_supported=tool_supported,
                        structured_supported=structured_supported,
                    )
                    if test.get("requires_structured") and structured_supported is None:
                        structured_supported = result.get("capability_supported")
                    per_test_runs[test["name"]].append({"repeat": repeat_idx + 1, **result})
            aggregated_tests = []
            tool_success = []
            structured_success = []
            latency_samples = []
            tps_samples = []
            error_count = 0
            total_runs = 0
            peak_ram_values = [load_delta_ram]
            peak_vram_by_gpu: Dict[str, float] = dict(load_delta_vram)
            for test in self.tests:
                runs = per_test_runs.get(test["name"], [])
                total_runs += len(runs)
                errors = sum(1 for r in runs if not r.get("success"))
                error_count += errors
                latencies = [r.get("latency_ms", 0.0) for r in runs if r.get("latency_ms")]
                tps_values = [r.get("tokens_per_sec", 0.0) for r in runs if r.get("tokens_per_sec")]
                tokens_values = [r.get("tokens_out", 0) for r in runs if r.get("tokens_out")]
                peak_ram = max((r.get("peak_delta_ram_bytes", 0.0) for r in runs), default=0.0)
                peak_ram_values.append(peak_ram)
                peak_vram = {}
                for r in runs:
                    for gid, val in (r.get("peak_delta_vram_mb_by_gpu") or {}).items():
                        peak_vram[gid] = max(peak_vram.get(gid, 0.0), _safe_float(val))
                        peak_vram_by_gpu[gid] = max(peak_vram_by_gpu.get(gid, 0.0), _safe_float(val))
                success_rate = 1.0 if runs and errors == 0 else (1.0 - (errors / max(len(runs), 1)))
                capability_values = [
                    1.0 if r.get("capability_supported") else 0.0
                    for r in runs
                    if r.get("capability_supported") is not None
                ]
                capability_rate = _median(capability_values) if capability_values else None
                capability_supported = None
                if capability_rate is not None:
                    capability_supported = capability_rate >= 0.5
                aggregated_tests.append(
                    {
                        "name": test["name"],
                        "runs": len(runs),
                        "success_rate": round(success_rate, 3),
                        "latency_ms": round(_median(latencies), 2) if latencies else 0.0,
                        "tokens_out": int(_median(tokens_values)) if tokens_values else 0,
                        "tokens_per_sec": round(_median(tps_values), 2) if tps_values else 0.0,
                        "peak_delta_ram_bytes": peak_ram,
                        "peak_delta_vram_mb_by_gpu": peak_vram,
                        "capability_supported": capability_supported,
                    }
                )
                if test.get("requires_tools"):
                    if capability_rate is not None:
                        tool_success.append(capability_rate)
                if test.get("requires_structured"):
                    if capability_rate is not None:
                        structured_success.append(capability_rate)
                if latencies:
                    latency_samples.append(_median(latencies))
                if tps_values:
                    tps_samples.append(_median(tps_values))
            peak_ram = max(peak_ram_values) if peak_ram_values else 0.0
            peak_vram = peak_vram_by_gpu
            post_unload_snapshot, memory_returned_pct = await self._unload_and_settle(
                backend=backend,
                instance=instance,
                baseline=base_snapshot,
                peak_ram=peak_ram,
                peak_vram_by_gpu=peak_vram,
            )
            error_rate = error_count / max(total_runs, 1)
            tps_summary = round(_median(tps_samples), 2) if tps_samples else 0.0
            latency_summary = round(_median(latency_samples), 2) if latency_samples else 0.0
            tool_rate = _median(tool_success) if tool_success else 0.0
            structured_rate = _median(structured_success) if structured_success else 0.0
            profile = {
                "backend_id": candidate.backend_id,
                "model_key": candidate.model_key,
                "config_signature": config_sig,
                "profile_version": PROFILE_VERSION,
                "profiled_at": utc_now(),
                "status": "ok",
                "baseline_snapshot": _summarize_snapshot(base_snapshot),
                "post_unload_snapshot": _summarize_snapshot(post_unload_snapshot),
                "load_delta_ram_bytes": load_delta_ram,
                "load_delta_vram_mb_by_gpu": load_delta_vram,
                "per_test": aggregated_tests,
                "ram_instance_peak_bytes": peak_ram,
                "vram_instance_peak_mb_by_gpu": peak_vram,
                "memory_returned_pct": memory_returned_pct,
                "tool_call_success_rate": round(tool_rate, 3),
                "json_schema_success_rate": round(structured_rate, 3),
                "tps": tps_summary,
                "latency_ms": latency_summary,
                "error_rate": round(error_rate, 3),
                "confidence": self._confidence_score(),
            }
            if vram_estimate_only and estimate_vram is not None:
                profile["vram_estimate_only_mb"] = estimate_vram
            await self.store.upsert(profile, ttl_seconds=ttl_seconds)
            unloaded = True
            return profile
        finally:
            # Ensure the instance is unloaded if not handled elsewhere.
            if instance and not unloaded:
                try:
                    await backend.unload_instance(instance.instance_id)
                except Exception:
                    pass

    async def _run_test(
        self,
        *,
        backend: "ModelBackend",
        instance: "ModelInstanceInfo",
        test: Dict[str, Any],
        base_snapshot: Dict[str, Any],
        tool_supported: bool,
        structured_supported: Optional[bool],
    ) -> Dict[str, Any]:
        messages = test.get("messages") or []
        if test.get("padding_tokens"):
            padded = _pad_text(messages[-1]["content"], int(test.get("padding_tokens")))
            messages = [*messages[:-1], {**messages[-1], "content": padded}]
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self._max_tokens_for(test),
            "use_responses": True,
        }
        capability_supported = None
        if test.get("requires_tools") and not tool_supported:
            payload["messages"] = test.get("fallback_messages") or messages
            capability_supported = False
        else:
            if test.get("tools"):
                payload["tools"] = test.get("tools")
            if test.get("tool_choice"):
                payload["tool_choice"] = test.get("tool_choice")
        if test.get("requires_structured"):
            if structured_supported is False:
                payload["messages"] = test.get("fallback_messages") or messages
                capability_supported = False
            else:
                payload["response_format"] = test.get("response_format")
        result = await self._run_payload(
            backend=backend,
            instance=instance,
            payload=payload,
            base_snapshot=base_snapshot,
        )
        if result.get("error") and test.get("requires_structured") and structured_supported is None:
            capability_supported = False
        if test.get("requires_tools") and tool_supported:
            parsed = _parse_tool_call(result.get("response") or {})
            if parsed and parsed.get("arguments", {}).get("value") == 7:
                capability_supported = True
            else:
                fallback = test.get("fallback_messages")
                if fallback:
                    fallback_payload = dict(payload)
                    fallback_payload.pop("tools", None)
                    fallback_payload.pop("tool_choice", None)
                    fallback_payload["messages"] = fallback
                    result = await self._run_payload(
                        backend=backend,
                        instance=instance,
                        payload=fallback_payload,
                        base_snapshot=base_snapshot,
                    )
                capability_supported = False
        if test.get("requires_structured"):
            parsed = _parse_structured_text(_extract_text(result.get("response") or {}))
            if parsed and parsed.get("ok") is True and parsed.get("label") == "alpha":
                if structured_supported is None:
                    capability_supported = True
            else:
                fallback = test.get("fallback_messages")
                if fallback and structured_supported is None:
                    fallback_payload = dict(payload)
                    fallback_payload.pop("response_format", None)
                    fallback_payload["messages"] = fallback
                    result = await self._run_payload(
                        backend=backend,
                        instance=instance,
                        payload=fallback_payload,
                        base_snapshot=base_snapshot,
                    )
                if structured_supported is None:
                    capability_supported = False
        return {
            "success": bool(result.get("success")),
            "error": result.get("error"),
            "latency_ms": result.get("latency_ms", 0.0),
            "tokens_out": result.get("tokens_out", 0),
            "tokens_per_sec": result.get("tokens_per_sec", 0.0),
            "peak_delta_ram_bytes": result.get("peak_delta_ram_bytes", 0.0),
            "peak_delta_vram_mb_by_gpu": result.get("peak_delta_vram_mb_by_gpu", {}),
            "capability_supported": capability_supported,
        }

    def _max_tokens_for(self, test: Dict[str, Any]) -> int:
        base = int(test.get("max_tokens") or 120)
        if self.max_output_tokens is None:
            return base
        return int(min(base, self.max_output_tokens))

    async def _run_payload(
        self,
        *,
        backend: "ModelBackend",
        instance: "ModelInstanceInfo",
        payload: Dict[str, Any],
        base_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        monitor_id = self.telemetry.monitor_start(self.sample_interval_ms)
        start = time.monotonic()
        error = None
        response: Dict[str, Any] = {}
        try:
            response = await asyncio.wait_for(
                backend.call_chat_completion(instance, payload),
                timeout=self.test_timeout_s,
            )
        except Exception as exc:
            error = str(exc)
        elapsed = max(time.monotonic() - start, 0.001)
        monitor = self.telemetry.monitor_stop(monitor_id)
        peak = monitor.get("peak_snapshot") or {}
        peak_ram_delta = max(
            _safe_float((peak.get("ram") or {}).get("used_bytes"))
            - _safe_float((base_snapshot.get("ram") or {}).get("used_bytes")),
            0.0,
        )
        peak_vram_delta = _gpu_delta_map(_gpu_used_map(peak), _gpu_used_map(base_snapshot))
        tokens_out = _extract_usage_tokens(response)
        if tokens_out is None:
            tokens_out = _tokens_from_text(_extract_text(response))
        tps = round(tokens_out / elapsed, 2) if tokens_out else 0.0
        return {
            "success": error is None,
            "error": error,
            "latency_ms": round(elapsed * 1000, 2),
            "tokens_out": int(tokens_out or 0),
            "tokens_per_sec": tps,
            "peak_delta_ram_bytes": peak_ram_delta,
            "peak_delta_vram_mb_by_gpu": peak_vram_delta,
            "response": response,
        }

    async def _unload_and_settle(
        self,
        *,
        backend: "ModelBackend",
        instance: "ModelInstanceInfo",
        baseline: Dict[str, Any],
        peak_ram: float,
        peak_vram_by_gpu: Dict[str, float],
    ) -> Tuple[Dict[str, Any], float]:
        try:
            await backend.unload_instance(instance.instance_id)
        except Exception:
            pass
        start = time.monotonic()
        last_snapshot = self.telemetry.snapshot()
        while (time.monotonic() - start) < self.settle_timeout_s:
            snapshot = self.telemetry.snapshot()
            ram_used = _safe_float((snapshot.get("ram") or {}).get("used_bytes"))
            base_used = _safe_float((baseline.get("ram") or {}).get("used_bytes"))
            ram_ok = (ram_used - base_used) <= self.settle_ram_tolerance_bytes
            vram_ok = True
            baseline_vram = _gpu_used_map(baseline)
            current_vram = _gpu_used_map(snapshot)
            for gid, base_val in baseline_vram.items():
                current = current_vram.get(gid, base_val)
                if (current - base_val) > self.settle_vram_tolerance_mb:
                    vram_ok = False
                    break
            last_snapshot = snapshot
            if ram_ok and vram_ok:
                break
            await asyncio.sleep(0.25)
        base_ram = _safe_float((baseline.get("ram") or {}).get("used_bytes"))
        current_ram = _safe_float((last_snapshot.get("ram") or {}).get("used_bytes"))
        ram_return_pct = _compute_return_pct(base_ram, base_ram + peak_ram, current_ram)
        return_pct = ram_return_pct
        baseline_vram = _gpu_used_map(baseline)
        current_vram = _gpu_used_map(last_snapshot)
        for gid, peak_val in peak_vram_by_gpu.items():
            base = baseline_vram.get(int(gid), 0.0)
            current = current_vram.get(int(gid), base)
            return_pct = min(return_pct, _compute_return_pct(base, base + peak_val, current))
        return last_snapshot, round(return_pct, 2)

    def _confidence_score(self) -> float:
        if self.repeats <= 1:
            return 0.6
        if self.repeats == 2:
            return 0.8
        return 1.0


def _summarize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    ram = snapshot.get("ram") or {}
    gpus = snapshot.get("gpus") or []
    return {
        "ram_used_bytes": _safe_float(ram.get("used_bytes")),
        "ram_total_bytes": _safe_float(ram.get("total_bytes")),
        "gpus": [
            {
                "gpu_id": gpu.get("gpu_id"),
                "vram_used_mb": _safe_float(gpu.get("vram_used_mb")),
                "vram_total_mb": _safe_float(gpu.get("vram_total_mb")),
            }
            for gpu in gpus
        ],
    }
