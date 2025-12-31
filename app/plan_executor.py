import asyncio
import re
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .artifact_store import ArtifactStore
from .executor_state_store import ExecutorStateStore
from .plan_store import PlanStore
from .model_manager import ModelManager, ModelInstanceInfo
from .request_store import RequestStore
from .resource_manager import ResourceManager
from .tavily import TavilyClient


class PlanExecutor:
    """Resource-aware executor that syncs plan diffs and schedules runnable steps."""

    def __init__(
        self,
        plan_store: PlanStore,
        artifact_store: ArtifactStore,
        request_store: RequestStore,
        resource_manager: ResourceManager,
        state_store: ExecutorStateStore,
        model_manager: Optional[ModelManager] = None,
        tavily_client: Optional[TavilyClient] = None,
        bus: Optional[Any] = None,
        run_id: Optional[str] = None,
        *,
        max_parallel: int = 4,
        page_size: int = 200,
        draft_token_budget: int = 1200,
    ) -> None:
        self.plan_store = plan_store
        self.artifact_store = artifact_store
        self.request_store = request_store
        self.resource_manager = resource_manager
        self.state_store = state_store
        self.model_manager = model_manager
        self.tavily = tavily_client
        self._bus = bus
        self._bus_run_id = run_id
        self.max_parallel = max(1, max_parallel)
        self.page_size = max(50, page_size)
        self.draft_token_budget = draft_token_budget
        self._paused_steps: set[str] = set()
        self._paused_by_finding: Dict[str, List[str]] = {}
        self._resource_pressure_threshold = 0.9
        self._plan_meta: Dict[str, Any] = {}
        self._question = ""

    async def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self._bus or not self._bus_run_id:
            return
        try:
            await self._bus.emit(self._bus_run_id, event_type, payload)
        except Exception:
            return

    async def run(self, plan_id: str, stop_event: Optional[asyncio.Event] = None) -> Dict[str, Any]:
        plan = await self.plan_store.get(plan_id)
        if plan:
            metadata = plan.get("metadata") or {}
            if isinstance(metadata, dict):
                self._plan_meta = metadata
                self._question = str(metadata.get("query") or metadata.get("question") or "")
        state = await self.state_store.get(plan_id)
        last_revision = int(state.get("last_revision") or 0)
        handled_findings = {str(fid) for fid in (state.get("handled_findings") or [])}
        scheduled_replans = {str(fid) for fid in (state.get("scheduled_replans") or [])}
        applied_patches = {str(pid) for pid in (state.get("applied_patches") or [])}
        auto_verified = {str(sid) for sid in (state.get("auto_verified") or [])}
        self._paused_steps = {str(sid) for sid in (state.get("paused_steps") or [])}
        paused_by_finding = state.get("paused_by_finding") or {}
        self._paused_by_finding = {
            str(fid): [str(sid) for sid in (step_ids or [])]
            for fid, step_ids in paused_by_finding.items()
            if fid
        }
        running_tasks: Dict[str, asyncio.Task] = {}
        running_step_ids: Dict[str, str] = {}
        final_ref: Optional[Dict[str, Any]] = None

        async def _handle_step(step: Dict[str, Any], run_id: str, instance: Optional[ModelInstanceInfo]) -> Dict[str, Any]:
            step_id = step["step_id"]
            attempt = int(step.get("attempt") or 0) + 1
            step_payload = {
                "step_id": step_id,
                "name": step.get("title") or step.get("name") or "",
                "type": step.get("step_type") or step.get("type") or "",
                "agent_profile": "executor",
            }
            base_meta = dict(step.get("run_metadata") or {})
            if instance:
                base_meta.update(
                    {
                        "run_id": run_id,
                        "model_instance": instance.api_identifier,
                        "model_key": instance.model_key,
                        "backend_id": instance.backend_id,
                    }
                )
            else:
                base_meta.update({"run_id": run_id})
            await self.plan_store.mark_running(plan_id, step_id, run_id, run_metadata=base_meta)
            await self._emit("step_started", step_payload)
            try:
                output_refs = await self._execute_step(plan_id, step, instance=instance)
                await self.plan_store.mark_done(plan_id, step_id, output_refs)
                await self._emit("step_completed", step_payload)
                return {"ok": True, "step_id": step_id, "output_refs": output_refs}
            except Exception as exc:
                await self.plan_store.mark_failed(plan_id, step_id, str(exc), retryable=True)
                max_retries = int(step.get("max_retries") or 0)
                if attempt < max_retries:
                    await self.plan_store.update_step(plan_id, step_id, {"status": "READY"})
                await self._emit("step_error", {**step_payload, "message": str(exc)})
                return {"ok": False, "step_id": step_id, "error": str(exc)}
            finally:
                if instance and self.model_manager:
                    await self.model_manager.release_instance(instance.instance_id)
                self.resource_manager.release(run_id)

        while True:
            if stop_event and stop_event.is_set():
                break
            overview = await self.plan_store.get_overview(plan_id)
            diff = await self.plan_store.get_diff(plan_id, last_revision, limit=self.page_size)
            last_revision = diff.get("revision", last_revision)
            await self._auto_create_verifiers(plan_id, diff.get("changes") or [], auto_verified)
            await self._triage_findings(plan_id, diff.get("finding_changes") or [], handled_findings, scheduled_replans, overview)
            await self._apply_validated_patches(plan_id, diff.get("patch_changes") or [], applied_patches, scheduled_replans)
            await self.state_store.set(
                plan_id,
                {
                    "last_revision": last_revision,
                    "handled_findings": sorted(handled_findings),
                    "scheduled_replans": sorted(scheduled_replans),
                    "applied_patches": sorted(applied_patches),
                    "auto_verified": sorted(auto_verified),
                    "paused_steps": sorted(self._paused_steps),
                    "paused_by_finding": self._paused_by_finding,
                },
            )
            await self._fulfill_requests(plan_id)

            # Clean up completed tasks
            if running_tasks:
                done, _ = await asyncio.wait(
                    list(running_tasks.values()),
                    timeout=0.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    result = task.result()
                    step_id = result.get("step_id")
                    if step_id:
                        running_tasks.pop(step_id, None)
                        running_step_ids.pop(step_id, None)

            capacity = self.max_parallel - len(running_tasks)
            reserve_denied = False
            if capacity > 0:
                candidates = await self._fetch_candidates(plan_id, capacity)
                for step in candidates:
                    step_id = step["step_id"]
                    claim = await self.plan_store.claim_step(plan_id, step_id, "executor")
                    if not claim.get("ok"):
                        continue
                    run_id = f"{plan_id}:{step_id}:{uuid.uuid4()}"
                    instance = await self._acquire_instance(step)
                    if instance is None and self.model_manager is not None and not self._allow_fallback(step):
                        await self.plan_store.update_step(plan_id, step_id, {"status": "READY", "claimed_by": None})
                        reserve_denied = True
                        continue
                    budgets = self._reservation_budgets(step, instance)
                    reserve = self.resource_manager.reserve(run_id, budgets)
                    if not reserve.get("granted"):
                        if instance and self.model_manager:
                            await self.model_manager.release_instance(instance.instance_id)
                        await self.plan_store.update_step(plan_id, step_id, {"status": "READY", "claimed_by": None})
                        reserve_denied = True
                        continue
                    running_step_ids[step_id] = step_id
                    running_tasks[step_id] = asyncio.create_task(_handle_step(step, run_id, instance))
            if running_tasks and capacity <= 0:
                await asyncio.wait(list(running_tasks.values()), return_when=asyncio.FIRST_COMPLETED)

            counts = overview.get("counts_by_status") or {}
            pending_left = sum(
                counts.get(status, 0) for status in ("PENDING", "READY", "CLAIMED", "RUNNING", "STALE")
            )
            if pending_left == 0 and not running_tasks:
                break
            if not running_tasks:
                await asyncio.sleep(0.2 if reserve_denied else 0.05)

        final_ref = await self._find_final_output(plan_id)
        return {"status": "done", "final_ref": final_ref}

    async def _fetch_candidates(self, plan_id: str, capacity: int) -> List[Dict[str, Any]]:
        pressure = self._resource_pressure()
        prefer_light = pressure >= self._resource_pressure_threshold
        fetch_limit = min(self.page_size, max(capacity * (3 if prefer_light else 1), capacity))
        ready = await self.plan_store.list_steps(
            plan_id,
            status="READY",
            limit=fetch_limit,
            fields=[
                "step_id",
                "title",
                "description",
                "step_type",
                "status",
                "prereq_step_ids",
                "priority",
                "cost_hint",
                "max_retries",
                "attempt",
                "tags",
                "run_metadata",
                "notes",
            ],
        )
        steps: List[Dict[str, Any]] = []
        needs_resolve = False
        for step in (ready.get("steps") or []):
            if step.get("step_id") in self._paused_steps:
                continue
            if self._has_unresolved_notes(step):
                needs_resolve = True
                await self.plan_store.update_step(plan_id, step["step_id"], {"status": "PENDING"})
                continue
            steps.append(step)
        if len(steps) >= capacity and not prefer_light:
            return steps
        pending = await self.plan_store.list_steps(
            plan_id,
            status="PENDING",
            limit=fetch_limit,
            fields=[
                "step_id",
                "title",
                "description",
                "step_type",
                "status",
                "prereq_step_ids",
                "priority",
                "cost_hint",
                "max_retries",
                "attempt",
                "tags",
                "run_metadata",
                "notes",
            ],
        )
        pending_steps = [s for s in (pending.get("steps") or []) if s.get("step_id") not in self._paused_steps]
        remaining = max(0, capacity - len(steps) - len(pending_steps))
        stale = {"steps": []}
        if remaining:
            stale = await self.plan_store.list_steps(
                plan_id,
                status="STALE",
                limit=fetch_limit,
                fields=[
                    "step_id",
                    "title",
                    "description",
                    "step_type",
                    "status",
                    "prereq_step_ids",
                    "priority",
                    "cost_hint",
                    "max_retries",
                    "attempt",
                    "tags",
                    "run_metadata",
                    "notes",
                ],
            )
        for step in pending_steps:
            if self._has_unresolved_notes(step):
                needs_resolve = True
                continue
            if not step.get("prereq_step_ids"):
                await self.plan_store.update_step(plan_id, step["step_id"], {"status": "READY"})
                steps.append({**step, "status": "READY"})
            else:
                steps.append(step)
        for step in [s for s in (stale.get("steps") or []) if s.get("step_id") not in self._paused_steps]:
            if self._has_unresolved_notes(step):
                needs_resolve = True
                await self.plan_store.update_step(plan_id, step["step_id"], {"status": "PENDING"})
                continue
            if not step.get("prereq_step_ids"):
                await self.plan_store.update_step(plan_id, step["step_id"], {"status": "READY"})
                steps.append({**step, "status": "READY"})
            else:
                await self.plan_store.update_step(plan_id, step["step_id"], {"status": "PENDING"})
                steps.append({**step, "status": "PENDING"})
        dependents: Dict[str, int] = {}
        for step in steps:
            for prereq in step.get("prereq_step_ids") or []:
                dependents[prereq] = dependents.get(prereq, 0) + 1
        if prefer_light:
            def _score(item: Dict[str, Any]) -> int:
                priority = int(item.get("priority") or 0)
                deps = dependents.get(item.get("step_id"), 0)
                cost = self._step_cost_score(item)
                resolve_bonus = 5000 if needs_resolve and "phase:resolve" in (item.get("tags") or []) else 0
                return (priority * 1000) + resolve_bonus + (deps * 10) - cost

            steps.sort(key=_score, reverse=True)
        else:
            def _priority(item: Dict[str, Any]) -> int:
                base = int(item.get("priority") or 0)
                if needs_resolve and "phase:resolve" in (item.get("tags") or []):
                    return base + 100
                return base

            steps.sort(
                key=lambda item: (
                    _priority(item),
                    dependents.get(item.get("step_id"), 0),
                ),
                reverse=True,
            )
        return steps[:capacity]

    async def _acquire_instance(self, step: Dict[str, Any]) -> Optional[ModelInstanceInfo]:
        if not self.model_manager:
            return None
        step_kind = self._resolve_step_type(step)
        tags = set(step.get("tags") or [])
        if step_kind in {"VERIFIER", "PATCH_VERIFY"}:
            return None
        if tags.intersection({"phase:expand", "phase:resolve", "phase:draft", "phase:finalize"}):
            return None
        cost_hint = step.get("cost_hint") or {}
        required = cost_hint.get("required_capabilities") or []
        required = await self._resolve_required_capabilities(required)
        objective = cost_hint.get("preferred_objective") or self.model_manager.routing_objective
        if isinstance(objective, str):
            cleaned = objective.strip().lower()
            if cleaned in ("latency", "fast", "speed"):
                objective = "best_latency"
            elif cleaned in ("quality", "best_quality", "accuracy"):
                objective = "best_quality"
            elif cleaned in ("balanced", "balance", ""):
                objective = "balanced"
        return await self.model_manager.acquire_instance(
            required_capabilities=list(required),
            objective=objective,
            backlog=1,
        )

    def _reservation_budgets(self, step: Dict[str, Any], instance: Optional[ModelInstanceInfo]) -> Dict[str, Any]:
        budgets = dict(step.get("cost_hint") or {})
        if instance is None:
            step_type = self._resolve_step_type(step)
            tags = set(step.get("tags") or [])
            if step_type in {"VERIFIER", "PATCH_VERIFY"} or tags.intersection(
                {"phase:expand", "phase:resolve", "phase:draft", "phase:finalize"}
            ):
                return {}
        if instance and instance.resource_reservation:
            for key in ("vram_mb", "ram_mb", "ram_bytes", "cpu_pct", "gpu_id"):
                if key in instance.resource_reservation and instance.resource_reservation[key] is not None:
                    budgets.setdefault(key, instance.resource_reservation[key])
        if instance and not instance.resource_reservation:
            return budgets
        if not budgets.get("vram_mb"):
            profile = self.resource_manager.model_profile(budgets.get("model_class") or "dynamic")
            budgets.setdefault("vram_mb", profile.get("vram_est"))
            budgets.setdefault("cpu_pct", profile.get("cpu_est"))
        return budgets

    def _resource_pressure(self) -> float:
        snapshot = self.resource_manager.snapshot()
        ratios: List[float] = []
        ram = snapshot.get("ram") or {}
        total_ram = float(ram.get("total_bytes") or 0.0)
        used_ram = float(ram.get("used_bytes") or 0.0)
        if total_ram:
            allowed = total_ram * (1.0 - (self.resource_manager.ram_headroom_pct / 100.0))
            if allowed > 0:
                ratios.append(used_ram / allowed)
        for gpu in snapshot.get("gpus") or []:
            total = float(gpu.get("vram_total_mb") or 0.0)
            used = float(gpu.get("vram_used_mb") or 0.0)
            if total:
                allowed = total * (1.0 - (self.resource_manager.vram_headroom_pct / 100.0))
                if allowed > 0:
                    ratios.append(used / allowed)
        return max(ratios) if ratios else 0.0

    def _step_cost_score(self, step: Dict[str, Any]) -> int:
        cost_hint = step.get("cost_hint") or {}
        tokens = cost_hint.get("estimated_tokens")
        try:
            tokens_val = int(tokens)
        except Exception:
            tokens_val = 0
        if tokens_val <= 0:
            tokens_val = 1000
        return tokens_val

    def _resolve_step_type(self, step: Dict[str, Any]) -> str:
        explicit = step.get("step_type")
        if explicit:
            return str(explicit).upper()
        tags = set(step.get("tags") or [])
        if "phase:verify" in tags:
            return "VERIFIER"
        if "phase:replan" in tags:
            return "REPLAN_PATCH"
        if "phase:patch_verify" in tags:
            return "PATCH_VERIFY"
        return ""

    async def _resolve_required_capabilities(self, required: List[str]) -> List[str]:
        required = [str(cap) for cap in (required or []) if cap]
        if not self.model_manager or "structured_output" not in required:
            return required
        try:
            candidates = await self.model_manager.get_candidates()
        except Exception:
            return required
        has_structured = any(c.capabilities.get("structured_output") for c in candidates)
        if not has_structured:
            return [cap for cap in required if cap != "structured_output"]
        return required

    async def _call_with_fallback(
        self,
        request: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.model_manager:
            raise RuntimeError("Model manager unavailable.")
        required = await self._resolve_required_capabilities(required_capabilities or [])
        cleaned_request = dict(request)
        if "structured_output" not in required and "response_format" in cleaned_request:
            cleaned_request.pop("response_format", None)
        try:
            return await self.model_manager.call(
                required_capabilities=required,
                objective=objective,
                request=cleaned_request,
            )
        except RuntimeError as exc:
            if "No suitable model instance available." not in str(exc) or "structured_output" not in required:
                raise
            fallback_required = [cap for cap in required if cap != "structured_output"]
            fallback_request = dict(request)
            fallback_request.pop("response_format", None)
            return await self.model_manager.call(
                required_capabilities=fallback_required,
                objective=objective,
                request=fallback_request,
            )

    def _allow_fallback(self, step: Dict[str, Any]) -> bool:
        meta = step.get("run_metadata") or {}
        if meta.get("allow_fallback"):
            return True
        step_type = self._resolve_step_type(step)
        if step_type in {"VERIFIER", "REPLAN_PATCH", "PATCH_VERIFY"}:
            return True
        tags = set(step.get("tags") or [])
        if tags.intersection({"phase:expand", "phase:resolve", "phase:draft", "phase:finalize"}):
            return True
        return False

    def _coerce_str_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        if value is None:
            return []
        return [str(value)]

    def _normalize_queries(self, value: Any) -> List[str]:
        queries = self._coerce_str_list(value)
        return [q.strip() for q in queries if q and str(q).strip()]

    def _utc_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _sanitize_text(self, value: str) -> str:
        if not value:
            return value
        replacements = {
            "\u2013": "-",
            "\u2014": "-",
            "\u2019": "'",
            "\u2018": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2026": "...",
            "\u00a0": " ",
        }
        cleaned = value
        for src, repl in replacements.items():
            cleaned = cleaned.replace(src, repl)
        return cleaned.encode("ascii", "ignore").decode("ascii")

    def _reduce_source_text(self, text: str, limit: int = 2000) -> str:
        if not text:
            return text
        if len(text) <= limit:
            return text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        hits: List[str] = []
        for line in lines:
            lowered = line.lower()
            if any(token in lowered for token in ("golf", "country club", "club")):
                if len(line) <= 200:
                    hits.append(line)
            if len(hits) >= 40:
                break
        if hits:
            snippet = "\n".join(hits)
            return snippet[:limit]
        return text[:limit]

    def _has_unresolved_notes(self, step: Dict[str, Any]) -> bool:
        notes = step.get("notes")
        if notes is None:
            return False
        if isinstance(notes, str):
            return bool(notes.strip())
        return True

    async def _auto_create_verifiers(
        self,
        plan_id: str,
        changes: List[Dict[str, Any]],
        auto_verified: set[str],
    ) -> None:
        inserted = [
            change.get("step_id")
            for change in changes
            if "insert" in (change.get("changed_fields") or [])
        ]
        inserted = [sid for sid in inserted if sid and sid not in auto_verified]
        if not inserted:
            return
        step_rows = await self.plan_store.get_steps(
            plan_id,
            inserted,
            fields=[
                "step_id",
                "title",
                "tags",
                "step_type",
                "priority",
                "partition_key",
                "prereq_step_ids",
                "status",
            ],
        )
        new_verifiers: List[Dict[str, Any]] = []
        candidate_steps: List[Dict[str, Any]] = []
        verify_ids: List[str] = []
        for row in step_rows.get("steps") or []:
            step_id = row.get("step_id")
            if not step_id or step_id in auto_verified:
                continue
            if str(row.get("status") or "").upper() in ("DONE", "FAILED", "CANCELED"):
                auto_verified.add(step_id)
                continue
            step_type = str(row.get("step_type") or "").upper()
            tags = set(row.get("tags") or [])
            if step_type in ("VERIFIER", "REPLAN_PATCH", "PATCH_VERIFY"):
                auto_verified.add(step_id)
                continue
            if any(tag.startswith("phase:") and tag != "phase:execute" for tag in tags):
                auto_verified.add(step_id)
                continue
            if "no-verify" in tags:
                auto_verified.add(step_id)
                continue
            if "phase:execute" not in tags:
                auto_verified.add(step_id)
                continue
            verify_step_id = f"verify-{step_id}"
            candidate_steps.append(row)
            verify_ids.append(verify_step_id)
        existing_verifiers: set[str] = set()
        if verify_ids:
            existing = await self.plan_store.get_steps(plan_id, verify_ids, fields=["step_id"])
            existing_verifiers = {row["step_id"] for row in existing.get("steps") or []}
        for row in candidate_steps:
            step_id = row.get("step_id")
            verify_step_id = f"verify-{step_id}"
            if verify_step_id in existing_verifiers:
                auto_verified.add(step_id)
                continue
            priority = max(0, int(row.get("priority") or 0) - 1)
            new_verifiers.append(
                {
                    "step_id": verify_step_id,
                    "title": f"Verify {row.get('title') or step_id}",
                    "description": f"Verify outputs for {row.get('title') or step_id}.",
                    "step_type": "VERIFIER",
                    "status": "PENDING",
                    "tags": ["phase:verify", f"verifies:{step_id}"],
                    "partition_key": row.get("partition_key"),
                    "priority": priority,
                    "prereq_step_ids": [step_id],
                    "run_metadata": {"verify_step_id": step_id, "allow_fallback": True},
                    "cost_hint": {
                        "required_capabilities": ["structured_output"],
                        "preferred_objective": "best_quality",
                        "estimated_tokens": 200,
                    },
                    "created_by": {"type": "executor", "id": "auto_verifier"},
                }
            )
            auto_verified.add(step_id)
        if new_verifiers:
            await self.plan_store.add_steps(plan_id, new_verifiers)

    async def _triage_findings(
        self,
        plan_id: str,
        finding_changes: List[Dict[str, Any]],
        handled_findings: set[str],
        scheduled_replans: set[str],
        overview: Dict[str, Any],
    ) -> None:
        for change in finding_changes:
            payload = change.get("payload") or {}
            action = payload.get("action")
            if action not in ("created", "updated"):
                continue
            finding_id = payload.get("finding_id") or change.get("entity_id")
            if not finding_id or finding_id in handled_findings:
                continue
            finding = await self.plan_store.get_finding(plan_id, finding_id)
            if not finding:
                handled_findings.add(finding_id)
                continue
            if finding.get("status") != "OPEN":
                handled_findings.add(finding_id)
                continue
            severity = str(finding.get("severity") or "INFO").upper()
            if severity in ("INFO", "WARN"):
                handled_findings.add(finding_id)
                continue
            seeds = []
            if finding.get("source_step_id"):
                seeds.append(finding.get("source_step_id"))
            seeds.extend(finding.get("impacted_step_ids") or [])
            seeds = [sid for sid in self._coerce_str_list(seeds) if sid]
            impact = await self.plan_store.compute_impact(plan_id, seeds, mode="downstream")
            impacted_ids = impact.get("impacted_step_ids") or []
            if impacted_ids:
                self._paused_steps.update(impacted_ids)
                self._paused_by_finding[str(finding_id)] = impacted_ids
            recommended_invalidations = impact.get("recommended_invalidations") or []
            if recommended_invalidations:
                await self.plan_store.invalidate_steps(
                    plan_id,
                    recommended_invalidations,
                    reason=f"finding:{finding_id}",
                    behavior="SET_STATUS_STALE",
                )
            if finding_id not in scheduled_replans:
                await self._schedule_replan_patch(plan_id, finding, impact, overview)
                scheduled_replans.add(finding_id)
            handled_findings.add(finding_id)

    async def _schedule_replan_patch(
        self,
        plan_id: str,
        finding: Dict[str, Any],
        impact: Dict[str, Any],
        overview: Dict[str, Any],
    ) -> None:
        replan_step_id = str(uuid.uuid4())
        patch_verify_id = str(uuid.uuid4())
        base_revision = int(overview.get("revision") or 0)
        finding_id = str(finding.get("finding_id") or "")
        steps = [
            {
                "step_id": replan_step_id,
                "title": "Replan patch",
                "description": f"Propose plan patch for finding {finding_id}.",
                "step_type": "REPLAN_PATCH",
                "status": "READY",
                "tags": ["phase:replan"],
                "priority": 9,
                "run_metadata": {
                    "finding_ids": [finding_id] if finding_id else [],
                    "impact": impact,
                    "base_revision": base_revision,
                    "allow_fallback": True,
                },
                "cost_hint": {
                    "required_capabilities": ["structured_output", "tool_use"],
                    "preferred_objective": "best_quality",
                    "estimated_tokens": 900,
                },
                "created_by": {"type": "executor", "id": "replan_scheduler"},
            },
            {
                "step_id": patch_verify_id,
                "title": "Verify patch",
                "description": "Validate patch proposal before apply.",
                "step_type": "PATCH_VERIFY",
                "status": "PENDING",
                "tags": ["phase:patch_verify"],
                "priority": 8,
                "prereq_step_ids": [replan_step_id],
                "run_metadata": {"replan_step_id": replan_step_id, "allow_fallback": True},
                "cost_hint": {
                    "required_capabilities": ["structured_output"],
                    "preferred_objective": "best_quality",
                    "estimated_tokens": 400,
                },
                "created_by": {"type": "executor", "id": "replan_scheduler"},
            },
        ]
        await self.plan_store.add_steps(plan_id, steps)

    async def _apply_validated_patches(
        self,
        plan_id: str,
        patch_changes: List[Dict[str, Any]],
        applied_patches: set[str],
        scheduled_replans: set[str],
    ) -> None:
        for change in patch_changes:
            payload = change.get("payload") or {}
            patch_id = payload.get("patch_id") or change.get("entity_id")
            status = str(payload.get("status") or "").upper()
            if not patch_id or patch_id in applied_patches:
                continue
            if status != "VALIDATED":
                continue
            patch = await self.plan_store.get_patch(plan_id, patch_id)
            if not patch or patch.get("status") != "VALIDATED":
                continue
            result = await self.plan_store.apply_patch(plan_id, patch_id, approver_id="executor")
            if result.get("conflict"):
                conflict_id = await self.plan_store.raise_finding(
                    plan_id,
                    None,
                    {
                        "severity": "ERROR",
                        "category": "CONTRADICTION",
                        "summary": f"Patch conflict for {patch_id}",
                        "details": json.dumps(result, ensure_ascii=True),
                        "linked_patch_id": patch_id,
                    },
                )
                await self._schedule_replan_patch(
                    plan_id,
                    {"finding_id": conflict_id},
                    {"impacted_step_ids": []},
                    await self.plan_store.get_overview(plan_id),
                )
                scheduled_replans.add(conflict_id)
                continue
            applied_patches.add(patch_id)
            for fid in patch.get("linked_finding_ids") or []:
                await self.plan_store.resolve_finding(
                    plan_id,
                    fid,
                    resolution_note=f"Resolved via patch {patch_id}.",
                    linked_patch_id=patch_id,
                )
                scheduled_replans.discard(str(fid))
                paused = self._paused_by_finding.pop(str(fid), [])
                for sid in paused:
                    self._paused_steps.discard(str(sid))
            await self.plan_store.resolve_prereqs(plan_id)

    async def _execute_step(
        self,
        plan_id: str,
        step: Dict[str, Any],
        instance: Optional[ModelInstanceInfo] = None,
    ) -> List[Dict[str, Any]]:
        step_kind = self._resolve_step_type(step)
        step_title = str(step.get("title") or "")
        step_type = step_title.strip().lower()
        tags = set(step.get("tags") or [])
        if step_kind == "VERIFIER":
            return await self._verify_step(plan_id, step)
        if step_kind == "REPLAN_PATCH":
            return await self._replan_patch(plan_id, step, instance=instance)
        if step_kind == "PATCH_VERIFY":
            return await self._patch_verify(plan_id, step)
        if "phase:expand" in tags or step_type.startswith("expand"):
            return await self._expand_partitions(plan_id, step)
        if "phase:resolve" in tags:
            await self.plan_store.resolve_prereqs(plan_id)
            ref = await self.artifact_store.put(
                None, {"status": "resolved"}, metadata={"step_id": step["step_id"]}, kind="json"
            )
            return [ref]
        if "phase:draft" in tags:
            return await self._draftbook(plan_id, step)
        if "phase:finalize" in tags:
            return await self._finalize(plan_id, step)
        if "phase:execute" in tags:
            return await self._execute_partition(plan_id, step, instance=instance)
        mock_output = (step.get("run_metadata") or {}).get("mock_output")
        if mock_output is not None:
            ref = await self.artifact_store.put(
                None, mock_output, metadata={"step_id": step["step_id"]}, kind="text"
            )
            await self._maybe_raise_worker_finding(plan_id, step, [ref])
            return [ref]
        if not self.model_manager or not instance:
            output = f"{step.get('title')}: complete."
            ref = await self.artifact_store.put(
                None, output, metadata={"step_id": step["step_id"]}, kind="text"
            )
            await self._maybe_raise_worker_finding(plan_id, step, [ref])
            return [ref]
        prompt = f"Task: {step.get('title')}\nDescription: {step.get('description') or ''}\n"
        input_refs = step.get("input_refs") or []
        if input_refs:
            packed = await self.artifact_store.pack_context(step["step_id"], input_refs, 800)
            context_text = await self.artifact_store.get(packed["ref_id"])
            prompt += f"Context: {context_text}"
        required_caps = (step.get("cost_hint") or {}).get("required_capabilities") or []
        response = await self.model_manager.backends[instance.backend_id].call_chat_completion(
            instance,
            {
                "messages": [{"role": "system", "content": "Complete the task."}, {"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": int((step.get("cost_hint") or {}).get("estimated_tokens") or 500),
                "use_responses": True,
            },
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
        ref = await self.artifact_store.put(
            None, content, metadata={"step_id": step["step_id"], "required_capabilities": required_caps}, kind="text"
        )
        await self._maybe_raise_worker_finding(plan_id, step, [ref])
        return [ref]

    async def _maybe_raise_worker_finding(
        self,
        plan_id: str,
        step: Dict[str, Any],
        output_refs: List[Dict[str, Any]],
    ) -> None:
        meta = step.get("run_metadata") or {}
        finding_payload = meta.get("raise_finding")
        if not finding_payload:
            return
        if isinstance(finding_payload, dict):
            payload = dict(finding_payload)
        else:
            payload = {"summary": str(finding_payload)}
        payload.setdefault("severity", meta.get("finding_severity") or "ERROR")
        payload.setdefault("category", meta.get("finding_category") or "OTHER")
        payload.setdefault("evidence_refs", output_refs)
        payload.setdefault("impacted_step_ids", [step.get("step_id")])
        await self.plan_store.raise_finding(plan_id, step.get("step_id"), payload)

    async def _collect_output_text(self, refs: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for ref in refs:
            content = await self.artifact_store.get(ref.get("ref_id"))
            if isinstance(content, dict):
                chunks.append(json.dumps(content, ensure_ascii=True))
            else:
                chunks.append(str(content))
        return "\n".join(chunks)

    def _extract_verifier_target(self, step: Dict[str, Any]) -> Optional[str]:
        meta = step.get("run_metadata") or {}
        target = meta.get("verify_step_id")
        if target:
            return str(target)
        for tag in step.get("tags") or []:
            if isinstance(tag, str) and tag.startswith("verifies:"):
                return tag.split("verifies:", 1)[-1]
        return None

    async def _verify_step(self, plan_id: str, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        target_step_id = self._extract_verifier_target(step)
        issues: List[Dict[str, Any]] = []
        verdict = "PASS"
        output_refs: List[Dict[str, Any]] = []
        if not target_step_id:
            verdict = "NEEDS_REVISION"
            issues.append({"reason": "missing_verify_target"})
        else:
            target_rows = await self.plan_store.get_steps(
                plan_id,
                [target_step_id],
                fields=["step_id", "title", "status", "output_refs"],
            )
            target = (target_rows.get("steps") or [None])[0]
            if not target:
                verdict = "NEEDS_REVISION"
                issues.append({"reason": "target_not_found", "step_id": target_step_id})
            elif target.get("status") != "DONE":
                verdict = "NEEDS_REVISION"
                issues.append({"reason": "target_not_done", "status": target.get("status")})
            else:
                output_refs = target.get("output_refs") or []
                text = await self._collect_output_text(output_refs)
                meta = step.get("run_metadata") or {}
                checks = meta.get("checks") or {}
                must_contain = self._coerce_str_list(checks.get("must_contain") or meta.get("must_contain"))
                must_not_contain = self._coerce_str_list(checks.get("must_not_contain") or meta.get("must_not_contain"))
                invalidate_if_contains = self._coerce_str_list(
                    checks.get("invalidate_if_contains") or meta.get("invalidate_if_contains")
                )
                require_non_empty = checks.get("require_non_empty", True)
                if require_non_empty and not text.strip():
                    verdict = "NEEDS_REVISION"
                    issues.append({"reason": "empty_output"})
                for token in must_contain:
                    if token and token not in text:
                        verdict = "NEEDS_REVISION"
                        issues.append({"reason": "missing_required_text", "token": token})
                for token in must_not_contain:
                    if token and token in text:
                        verdict = "NEEDS_REVISION"
                        issues.append({"reason": "disallowed_text_present", "token": token})
                for token in invalidate_if_contains:
                    if token and token in text:
                        verdict = "NEEDS_REVISION"
                        issues.append({"reason": "assumption_invalidated", "token": token})

        if verdict != "PASS":
            meta = step.get("run_metadata") or {}
            payload = {
                "severity": meta.get("severity") or "ERROR",
                "category": meta.get("category") or "QUALITY_FAILURE",
                "summary": f"Verification failed for {target_step_id or step.get('step_id')}",
                "details": json.dumps(issues, ensure_ascii=True),
                "evidence_refs": output_refs,
                "impacted_step_ids": [target_step_id] if target_step_id else [],
            }
            await self.plan_store.raise_finding(plan_id, step.get("step_id"), payload)

        report = {"verdict": verdict, "issues": issues, "target_step_id": target_step_id}
        ref = await self.artifact_store.put(
            None,
            report,
            metadata={"step_id": step["step_id"], "verifier": True},
            kind="json",
        )
        return [ref]

    async def _build_fallback_patch(
        self,
        plan_id: str,
        findings: List[Dict[str, Any]],
        impact: Dict[str, Any],
    ) -> Dict[str, Any]:
        operations: List[Dict[str, Any]] = []
        impacted_ids = impact.get("impacted_step_ids") or []
        step_rows = await self.plan_store.get_steps(
            plan_id,
            impacted_ids,
            fields=["step_id", "prereq_step_ids", "tags"],
        )
        impacted_map = {row["step_id"]: row for row in step_rows.get("steps") or []}
        for finding in findings or [{}]:
            source_step_id = finding.get("source_step_id")
            new_step_id = str(uuid.uuid4())
            title = "Address finding"
            source_prereqs: List[str] = []
            if source_step_id:
                title = f"Rework {source_step_id}"
                source_row = impacted_map.get(str(source_step_id))
                if source_row:
                    source_prereqs = list(source_row.get("prereq_step_ids") or [])
                operations.append(
                    {
                        "op": "CANCEL_STEP",
                        "step_id": str(source_step_id),
                        "reason": "Superseded by replan patch.",
                    }
                )
            operations.append(
                {
                    "op": "ADD_STEP",
                    "step": {
                        "step_id": new_step_id,
                        "title": title,
                        "description": finding.get("summary") or "Address verification finding.",
                        "status": "PENDING",
                        "tags": ["phase:execute", "replan:fix"],
                        "priority": 6,
                        "prereq_step_ids": source_prereqs,
                        "cost_hint": {
                            "required_capabilities": ["tool_use"],
                            "preferred_objective": "best_quality",
                            "estimated_tokens": 600,
                        },
                        "created_by": {"type": "planner", "id": "fallback_replan"},
                    },
                }
            )
            for step_id, row in impacted_map.items():
                if step_id in (source_step_id, new_step_id):
                    continue
                tags = set(row.get("tags") or [])
                if "phase:replan" in tags or "phase:patch_verify" in tags:
                    continue
                prereqs = list(row.get("prereq_step_ids") or [])
                if source_step_id and source_step_id in prereqs:
                    prereqs = [pid for pid in prereqs if pid != source_step_id]
                if new_step_id not in prereqs:
                    prereqs.append(new_step_id)
                operations.append(
                    {"op": "SET_PREREQS", "step_id": step_id, "prereq_step_ids": prereqs}
                )
        return {"operations": operations, "rationale": "Fallback replan based on findings."}

    async def _replan_patch(
        self,
        plan_id: str,
        step: Dict[str, Any],
        instance: Optional[ModelInstanceInfo] = None,
    ) -> List[Dict[str, Any]]:
        meta = step.get("run_metadata") or {}
        finding_ids = self._coerce_str_list(meta.get("finding_ids"))
        base_revision = int(meta.get("base_revision") or 0)
        if base_revision <= 0:
            overview = await self.plan_store.get_overview(plan_id)
            base_revision = int(overview.get("revision") or 0)
        findings: List[Dict[str, Any]] = []
        for fid in finding_ids:
            finding = await self.plan_store.get_finding(plan_id, fid)
            if finding:
                findings.append(finding)
        rationale = meta.get("rationale") or "Replan based on findings."
        operations: List[Dict[str, Any]] = []
        if instance and self.model_manager:
            prompt = {
                "findings": findings,
                "impact": meta.get("impact") or {},
                "base_revision": base_revision,
            }
            schema = {
                "name": "plan_patch",
                "schema": {
                    "type": "object",
                    "properties": {
                        "rationale": {"type": "string"},
                        "operations": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["operations"],
                    "additionalProperties": True,
                },
            }
            try:
                response = await self.model_manager.backends[instance.backend_id].call_chat_completion(
                    instance,
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "Propose a plan patch as JSON with rationale and operations.",
                            },
                            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 800,
                        "response_format": {"type": "json_schema", "json_schema": schema},
                        "use_responses": True,
                    },
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
                parsed = json.loads(content) if content else {}
                if isinstance(parsed, dict):
                    rationale = parsed.get("rationale") or rationale
                    ops = parsed.get("operations") or []
                    if isinstance(ops, list):
                        operations = ops
            except Exception:
                operations = []
        if not operations:
            fallback = await self._build_fallback_patch(plan_id, findings, meta.get("impact") or {})
            operations = fallback.get("operations") or []
            rationale = fallback.get("rationale") or rationale
        patch_id = await self.plan_store.propose_patch(
            plan_id,
            base_revision,
            rationale,
            finding_ids,
            operations,
            created_by={"type": "executor", "id": step.get("step_id")},
        )
        report = {
            "patch_id": patch_id,
            "rationale": rationale,
            "operations": operations,
            "finding_ids": finding_ids,
        }
        ref = await self.artifact_store.put(
            None,
            report,
            metadata={"step_id": step["step_id"], "patch_id": patch_id},
            kind="json",
        )
        return [ref]

    async def _patch_verify(self, plan_id: str, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        meta = step.get("run_metadata") or {}
        patch_id = meta.get("patch_id")
        if not patch_id:
            replan_step_id = meta.get("replan_step_id")
            if replan_step_id:
                rows = await self.plan_store.get_steps(plan_id, [replan_step_id], fields=["output_refs"])
                refs = (rows.get("steps") or [{}])[0].get("output_refs") or []
                if refs:
                    payload = await self.artifact_store.get(refs[0]["ref_id"])
                    if isinstance(payload, dict):
                        patch_id = payload.get("patch_id")
        report: Dict[str, Any]
        if not patch_id:
            report = {"ok": False, "errors": ["patch_id_missing"], "warnings": []}
        else:
            report = await self.plan_store.validate_patch(plan_id, patch_id)
            if not report.get("ok"):
                await self.plan_store.raise_finding(
                    plan_id,
                    step.get("step_id"),
                    {
                        "severity": "ERROR",
                        "category": "QUALITY_FAILURE",
                        "summary": f"Patch validation failed for {patch_id}",
                        "details": json.dumps(report, ensure_ascii=True),
                        "linked_patch_id": patch_id,
                    },
                )
        ref = await self.artifact_store.put(
            None,
            report,
            metadata={"step_id": step["step_id"], "patch_id": patch_id},
            kind="json",
        )
        return [ref]

    async def _expand_partitions(self, plan_id: str, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        expansion_count = int(step.get("run_metadata", {}).get("expansion_count") or 50)
        question = self._question or str(self._plan_meta.get("query") or "")
        needs_web = bool(self._plan_meta.get("needs_web", True))
        tool_budget = self._plan_meta.get("tool_budget") or {}
        search_budget = int(tool_budget.get("tavily_search", 0) or 0)
        planning_mode = str(self._plan_meta.get("planning_mode") or "").lower()
        reasoning_mode = str(self._plan_meta.get("reasoning_mode") or "").lower()
        extensive = planning_mode == "extensive" or reasoning_mode == "extensive"
        real_estate_info = self._real_estate_query_info(question)
        if needs_web:
            base_budget = search_budget if search_budget > 0 else 8
            min_target = 4
            max_target = 25
            if extensive:
                min_target = 8
                max_target = 40
            if real_estate_info:
                min_target = max(min_target, 12)
                max_target = max(max_target, 40)
            budget_cap = max(min_target, min(max_target, base_budget if base_budget > 0 else max_target))
            target_count = min(expansion_count, budget_cap)
        else:
            target_count = min(expansion_count, max(6, min(50, expansion_count)))
        if target_count <= 0:
            target_count = 0
        max_results = int(self._plan_meta.get("max_results") or 6)
        if max_results < 1:
            max_results = 1
        search_depth = str(self._plan_meta.get("search_depth") or self._plan_meta.get("search_depth_mode") or "basic")
        if search_depth not in ("basic", "advanced"):
            search_depth = "basic"
        extract_depth = str(self._plan_meta.get("extract_depth") or "basic")
        if extract_depth not in ("basic", "advanced"):
            extract_depth = "basic"
        discovery_queries: List[str] = []
        discovered_developments: List[Dict[str, Any]] = []
        partition_specs: List[Dict[str, Any]] = []
        if real_estate_info and needs_web and target_count > 0:
            discovery_queries = self._real_estate_discovery_queries(real_estate_info)
            if discovery_queries:
                discovered_developments = await self._discover_real_estate_developments(
                    question,
                    discovery_queries,
                    search_depth,
                    max_results,
                    target_count,
                )
            if discovered_developments:
                partition_specs = self._real_estate_development_specs(
                    discovered_developments,
                    real_estate_info,
                    extensive,
                )
        if not partition_specs:
            partition_specs = self._real_estate_partition_specs(question)
        if not partition_specs and question and self.model_manager and target_count > 0:
            partition_specs = await self._generate_partition_specs(question, target_count, needs_web)
        if not partition_specs and target_count > 0:
            partition_specs = self._fallback_partition_specs(question, target_count, needs_web)
        if target_count > 0 and len(partition_specs) < target_count:
            missing = target_count - len(partition_specs)
            partition_specs.extend(self._fallback_partition_specs(question, missing, needs_web))
        if len(partition_specs) > target_count:
            partition_specs = partition_specs[:target_count]
        max_queries = 1
        if needs_web and search_budget and target_count and search_budget >= (target_count * 2):
            max_queries = 2
        if real_estate_info and extensive:
            max_queries = max(max_queries, 2)
        fallback_queries = self._build_fallback_queries(question, max(6, target_count))
        new_steps: List[Dict[str, Any]] = []
        discovery_step_id = None
        if real_estate_info and needs_web and discovery_queries:
            discovery_step_id = str(uuid.uuid4())
            discovery_label = "Golf communities discovery"
            discovery_run_metadata = {
                "partition_label": discovery_label,
                "queries": discovery_queries,
                "focus": "Discover golf communities and developments.",
                "needs_web": needs_web,
                "max_queries": max(max_queries, min(3, len(discovery_queries))),
                "max_results": max_results,
                "search_depth": search_depth,
                "extract_depth": extract_depth,
                "topic": self._plan_meta.get("topic") or "general",
                "time_range": self._plan_meta.get("time_range"),
            }
            new_steps.append(
                {
                    "step_id": discovery_step_id,
                    "title": "Research golf communities list",
                    "description": "Discover golf communities and developments in the target area.",
                    "status": "READY",
                    "tags": ["phase:execute", "phase:discover"],
                    "partition_key": "discover",
                    "priority": 6,
                    "prereq_step_ids": [step["step_id"]],
                    "cost_hint": {
                        "required_capabilities": ["structured_output"],
                        "preferred_objective": "balanced",
                        "estimated_tokens": 500,
                    },
                    "run_metadata": discovery_run_metadata,
                    "created_by": {"type": "expander", "id": step["step_id"]},
                }
            )
            new_steps.append(
                {
                    "step_id": f"verify-{discovery_step_id}",
                    "title": "Verify discovery",
                    "description": "Verify discovery outputs for golf community list.",
                    "step_type": "VERIFIER",
                    "status": "PENDING",
                    "tags": ["phase:verify", f"verifies:{discovery_step_id}"],
                    "partition_key": "discover",
                    "priority": 5,
                    "prereq_step_ids": [discovery_step_id],
                    "run_metadata": {"verify_step_id": discovery_step_id, "allow_fallback": True},
                    "cost_hint": {
                        "required_capabilities": ["structured_output"],
                        "preferred_objective": "best_quality",
                        "estimated_tokens": 200,
                    },
                    "created_by": {"type": "expander", "id": step["step_id"]},
                }
            )
        for idx, spec in enumerate(partition_specs):
            exec_step_id = str(uuid.uuid4())
            label = str(spec.get("label") or f"Partition {idx + 1}")
            queries = self._normalize_queries(spec.get("queries"))
            if needs_web and not queries and fallback_queries:
                queries = [fallback_queries[idx % len(fallback_queries)]]
            step_max_queries = int(spec.get("max_queries") or max_queries)
            if step_max_queries < 1:
                step_max_queries = 1
            step_max_results = int(spec.get("max_results") or max_results)
            if step_max_results < 1:
                step_max_results = 1
            step_search_depth = str(spec.get("search_depth") or search_depth)
            if step_search_depth not in ("basic", "advanced"):
                step_search_depth = search_depth
            step_extract_depth = str(spec.get("extract_depth") or extract_depth)
            if step_extract_depth not in ("basic", "advanced"):
                step_extract_depth = extract_depth
            run_metadata = {
                "partition_label": label,
                "queries": queries,
                "focus": spec.get("focus") or label,
                "needs_web": needs_web,
                "max_queries": step_max_queries,
                "max_results": step_max_results,
                "search_depth": step_search_depth,
                "extract_depth": step_extract_depth,
                "topic": spec.get("topic") or self._plan_meta.get("topic") or "general",
                "time_range": spec.get("time_range") or self._plan_meta.get("time_range"),
            }
            if spec.get("location"):
                run_metadata["location"] = spec.get("location")
            prereqs = [step["step_id"]]
            if discovery_step_id:
                prereqs.append(discovery_step_id)
            title = spec.get("title") or f"Execute {label}"
            description = spec.get("description") or f"Research partition: {label}."
            new_steps.append(
                {
                    "step_id": exec_step_id,
                    "title": title,
                    "description": description,
                    "status": "READY",
                    "tags": ["phase:execute", f"partition:{idx + 1}"],
                    "partition_key": str(idx + 1),
                    "priority": 5,
                    "prereq_step_ids": prereqs,
                    "cost_hint": {
                        "required_capabilities": ["structured_output"],
                        "preferred_objective": "balanced",
                        "estimated_tokens": 600,
                    },
                    "run_metadata": run_metadata,
                    "created_by": {"type": "expander", "id": step["step_id"]},
                }
            )
            new_steps.append(
                {
                    "step_id": f"verify-{exec_step_id}",
                    "title": f"Verify {label}",
                    "description": f"Verify outputs for {label}.",
                    "step_type": "VERIFIER",
                    "status": "PENDING",
                    "tags": ["phase:verify", f"verifies:{exec_step_id}"],
                    "partition_key": str(idx + 1),
                    "priority": 4,
                    "prereq_step_ids": [exec_step_id],
                    "run_metadata": {"verify_step_id": exec_step_id, "allow_fallback": True},
                    "cost_hint": {
                        "required_capabilities": ["structured_output"],
                        "preferred_objective": "best_quality",
                        "estimated_tokens": 200,
                    },
                    "created_by": {"type": "expander", "id": step["step_id"]},
                }
            )
        if new_steps:
            await self.plan_store.add_steps(plan_id, new_steps)
        ref = await self.artifact_store.put(
            None,
            {"created_steps": len(new_steps), "partitions": len(partition_specs)},
            metadata={"step_id": step["step_id"]},
            kind="json",
        )
        return [ref]

    async def _generate_partition_specs(
        self,
        question: str,
        target_count: int,
        needs_web: bool,
    ) -> List[Dict[str, Any]]:
        if not self.model_manager or not question or target_count <= 0:
            return []
        topic = self._plan_meta.get("topic") or "general"
        time_range = self._plan_meta.get("time_range")
        guidance = (
            "Return JSON with a partitions array. Each item must include label and 1-2 queries if web is needed. "
            "Queries must be runnable (no placeholders like [neighborhoods]) and include Naples or Bonita Springs. "
            "Keep queries short; do not stack every constraint into a single query."
        )
        prompt = {
            "question": question,
            "target_count": target_count,
            "needs_web": needs_web,
            "topic": topic,
            "time_range": time_range,
            "instructions": guidance,
        }
        schema = {
            "name": "partition_plan",
            "schema": {
                "type": "object",
                "properties": {
                    "partitions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "queries": {"type": "array", "items": {"type": "string"}},
                                "topic": {"type": "string"},
                                "time_range": {"type": "string"},
                                "focus": {"type": "string"},
                            },
                            "required": ["label"],
                            "additionalProperties": True,
                        },
                    }
                },
                "required": ["partitions"],
                "additionalProperties": True,
            },
        }
        try:
            response = await self._call_with_fallback(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You propose research partitions. Keep it concise and use explicit search queries."
                            ),
                        },
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 800,
                    "response_format": {"type": "json_schema", "json_schema": schema},
                    "use_responses": True,
                },
                required_capabilities=["structured_output"],
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
            parsed = json.loads(content) if content else {}
            partitions = parsed.get("partitions") if isinstance(parsed, dict) else None
            if not isinstance(partitions, list):
                return []
            cleaned: List[Dict[str, Any]] = []
            for item in partitions:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip()
                if not label:
                    continue
                queries = self._normalize_queries(item.get("queries"))
                if queries:
                    queries = [q for q in queries if "[" not in q and "]" not in q]
                    if queries:
                        item = dict(item)
                        item["queries"] = queries
                cleaned.append(item)
            return cleaned
        except Exception:
            return []

    def _real_estate_query_info(self, question: str) -> Optional[Dict[str, bool]]:
        text = (question or "").lower()
        if "golf" not in text:
            return None
        has_naples = "naples" in text
        has_bonita = "bonita" in text
        if not (has_naples or has_bonita):
            return None
        return {"has_naples": has_naples, "has_bonita": has_bonita}

    def _real_estate_discovery_queries(self, info: Dict[str, bool]) -> List[str]:
        queries: List[str] = []
        if info.get("has_naples"):
            queries.extend(
                [
                    "Naples FL golf communities list",
                    "Naples FL golf country clubs list",
                    "Naples FL golf communities new construction",
                    "Naples FL golf community homes for sale under $1M",
                ]
            )
        if info.get("has_bonita"):
            queries.extend(
                [
                    "Bonita Springs FL golf communities list",
                    "Bonita Springs FL golf country clubs list",
                    "Bonita Springs FL golf communities new construction",
                    "Bonita Springs FL golf community homes for sale under $1M",
                ]
            )
        if info.get("has_naples") and info.get("has_bonita"):
            queries.append("Naples Bonita Springs golf communities list")
        deduped: List[str] = []
        seen = set()
        for query in queries:
            key = query.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(query)
        return deduped

    def _normalize_development_name(self, name: str) -> str:
        cleaned = re.sub(r"\s+", " ", name or "").strip()
        return cleaned

    def _looks_like_list_title(self, value: str) -> bool:
        lowered = (value or "").lower()
        list_markers = (
            "top ",
            "best ",
            "list",
            "guide",
            "neighborhoods",
            "communities",
            "homes for sale",
            "real estate",
        )
        if any(marker in lowered for marker in list_markers):
            if "country club" in lowered or "golf club" in lowered or "golf & country club" in lowered:
                return False
            return True
        return False

    def _extract_development_names_from_text(self, text: str, limit: int) -> List[str]:
        if not text:
            return []
        cleaned = re.sub(r"\s+", " ", text)
        patterns = [
            r"([A-Z][-A-Za-z0-9&' ]{2,}? (?:Golf|Country) Club)",
            r"([A-Z][-A-Za-z0-9&' ]{2,}? Golf (?:&|and) Country Club)",
            r"([A-Z][-A-Za-z0-9&' ]{2,}?)\\s+(?:golf community|golf neighborhood|golf course)s?",
            r"(The Club at [A-Z][-A-Za-z0-9&' ]{2,})",
        ]
        found: List[str] = []
        seen = set()
        for pattern in patterns:
            for match in re.findall(pattern, cleaned):
                name = match[0] if isinstance(match, tuple) else match
                name = self._normalize_development_name(name)
                if not name or self._looks_like_list_title(name):
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append(name)
                if len(found) >= limit:
                    return found
        return found

    def _candidate_names_from_notes(self, text: str, limit: int) -> List[str]:
        names: List[str] = []
        seen = set()
        for line in (text or "").splitlines():
            line = line.strip()
            if not line.lower().startswith("research "):
                continue
            name = line[9:].split(":", 1)[0].strip()
            if not name or self._looks_like_list_title(name):
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
            if len(names) >= limit:
                return names
        if names:
            return names
        return self._extract_development_names_from_text(text or "", limit)

    def _real_estate_default_location(self, info: Dict[str, bool]) -> str:
        has_naples = info.get("has_naples")
        has_bonita = info.get("has_bonita")
        if has_naples and has_bonita:
            return "Naples Bonita Springs"
        if has_naples:
            return "Naples"
        if has_bonita:
            return "Bonita Springs"
        return ""

    def _real_estate_development_queries(
        self, name: str, location: str, max_queries: int
    ) -> List[str]:
        base = f"{name} {location}".strip()
        queries = [
            f"{base} golf community homes for sale under $1M",
            f"{base} single family homes villas",
            f"{base} year built 2010 or newer",
            f"{base} distance to beach",
        ]
        return queries[: max(1, max_queries)]

    def _real_estate_development_specs(
        self,
        developments: List[Dict[str, Any]],
        info: Dict[str, bool],
        extensive: bool,
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        default_location = self._real_estate_default_location(info)
        max_queries = 4 if extensive else 3
        seen: set[str] = set()
        for item in developments:
            name = self._normalize_development_name(str(item.get("name") or ""))
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            location = str(item.get("location") or "").strip() or default_location
            queries = self._real_estate_development_queries(name, location, max_queries)
            specs.append(
                {
                    "label": name,
                    "title": f"Research {name}",
                    "description": (
                        "Confirm build years (2010+), home type, price under $1M, and beach distance."
                    ),
                    "queries": queries,
                    "focus": name,
                    "location": location,
                    "max_queries": max_queries,
                }
            )
        return specs

    def _guess_developments_from_sources(
        self, sources: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for src in sources:
            title = str(src.get("title") or "").strip()
            snippet = str(src.get("snippet") or "").strip()
            extracted = str(src.get("extracted_text") or "").strip()
            candidates: List[str] = []
            for text in (title, snippet, extracted):
                if not text:
                    continue
                candidates.extend(self._extract_development_names_from_text(text, limit))
            if not candidates and title and not self._looks_like_list_title(title):
                candidates.append(title.split(" - ", 1)[0].split("|", 1)[0].strip())
            for name in candidates:
                normalized = self._normalize_development_name(name)
                if not normalized or self._looks_like_list_title(normalized):
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                results.append({"name": normalized, "location": ""})
                if len(results) >= limit:
                    return results
        return results

    async def _discover_real_estate_developments(
        self,
        question: str,
        queries: List[str],
        search_depth: str,
        max_results: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self.tavily or not self.tavily.enabled:
            return []
        normalized_queries = self._normalize_queries(queries)
        if not normalized_queries:
            return []
        sources: List[Dict[str, Any]] = []
        for query in normalized_queries[:4]:
            resp = await self.tavily.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                topic="general",
                time_range=None,
            )
            if resp.get("error"):
                continue
            for res in resp.get("results") or []:
                if not isinstance(res, dict):
                    continue
                url = str(res.get("url") or "").strip()
                if not url:
                    continue
                content = res.get("content") or res.get("raw_content") or ""
                sources.append(
                    {
                        "url": url,
                        "title": res.get("title") or "",
                        "publisher": res.get("source") or "",
                        "date_published": res.get("published_date") or "",
                        "snippet": str(content)[:400],
                        "extracted_text": content,
                    }
                )
        if not sources:
            return []
        if not self.model_manager:
            return self._guess_developments_from_sources(sources, limit)
        trimmed_sources: List[Dict[str, Any]] = []
        for src in sources[:12]:
            text = str(src.get("extracted_text") or src.get("snippet") or "")
            text = self._reduce_source_text(text, 1600)
            trimmed_sources.append(
                {
                    "url": src.get("url"),
                    "title": src.get("title"),
                    "snippet": src.get("snippet"),
                    "extracted_text": text,
                }
            )
        schema = {
            "name": "development_discovery",
            "schema": {
                "type": "object",
                "properties": {
                    "developments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "location": {"type": "string"},
                                "notes": {"type": "string"},
                            },
                            "required": ["name"],
                            "additionalProperties": True,
                        },
                    }
                },
                "required": ["developments"],
                "additionalProperties": True,
            },
        }
        prompt = {
            "question": question,
            "queries": normalized_queries,
            "sources": trimmed_sources,
            "instructions": (
                "Extract golf communities or developments in Naples or Bonita Springs. "
                "Return distinct names; include location if known."
            ),
        }
        try:
            response = await self._call_with_fallback(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Extract a clean list of golf communities. Return JSON only.",
                        },
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 700,
                    "response_format": {"type": "json_schema", "json_schema": schema},
                    "use_responses": True,
                },
                required_capabilities=["structured_output"],
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
            parsed = json.loads(content) if content else {}
        except Exception:
            return self._guess_developments_from_sources(sources, limit)
        developments = parsed.get("developments") if isinstance(parsed, dict) else None
        if not isinstance(developments, list):
            return self._guess_developments_from_sources(sources, limit)
        cleaned: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in developments:
            if not isinstance(item, dict):
                continue
            name = self._normalize_development_name(str(item.get("name") or ""))
            if not name or self._looks_like_list_title(name):
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            location = str(item.get("location") or "").strip()
            cleaned.append({"name": name, "location": location})
            if len(cleaned) >= limit:
                break
        if len(cleaned) < limit:
            for guess in self._guess_developments_from_sources(sources, limit):
                name = self._normalize_development_name(str(guess.get("name") or ""))
                if not name or self._looks_like_list_title(name):
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append({"name": name, "location": str(guess.get("location") or "").strip()})
                if len(cleaned) >= limit:
                    break
        return cleaned

    def _real_estate_partition_specs(self, question: str) -> List[Dict[str, Any]]:
        text = (question or "").lower()
        if "golf" not in text:
            return []
        has_naples = "naples" in text
        has_bonita = "bonita" in text
        if not (has_naples or has_bonita):
            return []
        specs: List[Dict[str, Any]] = []
        if has_naples:
            specs.append(
                {
                    "label": "Naples golf communities list",
                    "queries": [
                        "Naples golf communities list",
                        "Naples golf communities real estate",
                    ],
                }
            )
            specs.append(
                {
                    "label": "Naples golf community homes under $1M",
                    "queries": [
                        "Naples golf community homes for sale under $1M",
                        "Naples golf community villas for sale under $1M",
                    ],
                }
            )
            specs.append(
                {
                    "label": "Naples golf community new construction",
                    "queries": [
                        "Naples golf community new construction 2010",
                        "Naples golf community year built 2010 or newer",
                    ],
                }
            )
        if has_bonita:
            specs.append(
                {
                    "label": "Bonita Springs golf communities list",
                    "queries": [
                        "Bonita Springs golf communities list",
                        "Bonita Springs golf communities real estate",
                    ],
                }
            )
            specs.append(
                {
                    "label": "Bonita Springs golf community homes under $1M",
                    "queries": [
                        "Bonita Springs golf community homes for sale under $1M",
                        "Bonita Springs golf community villas for sale under $1M",
                    ],
                }
            )
            specs.append(
                {
                    "label": "Bonita Springs golf community new construction",
                    "queries": [
                        "Bonita Springs golf community new construction 2010",
                        "Bonita Springs golf community year built 2010 or newer",
                    ],
                }
            )
        return specs

    def _build_fallback_queries(self, question: str, limit: int) -> List[str]:
        base = (question or "").strip()
        if not base:
            return []
        base_lower = base.lower()
        queries: List[str] = []
        if "naples" in base_lower:
            queries.extend(
                [
                    "Naples golf communities list",
                    "Naples golf communities homes for sale",
                    "Naples golf community villas for sale",
                ]
            )
        if "bonita" in base_lower:
            queries.extend(
                [
                    "Bonita Springs golf communities list",
                    "Bonita Springs golf communities homes for sale",
                    "Bonita Springs golf community villas for sale",
                ]
            )
        if "golf" in base_lower:
            queries.append("Naples Bonita Springs golf communities list")
        hints = [
            "list",
            "guide",
            "official site",
            "2024",
            "2025",
            "pricing",
            "reviews",
            "map",
            "neighborhoods",
            "communities",
        ]
        variants = [base]
        for hint in hints:
            variants.append(f"{base} {hint}")
        ordered = []
        seen = set()
        for item in queries + variants:
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        if limit <= 0:
            return ordered
        if len(ordered) >= limit:
            return ordered[:limit]
        extra: List[str] = []
        idx = 0
        while len(ordered) + len(extra) < limit:
            extra.append(f"{base} {hints[idx % len(hints)]} details")
            idx += 1
        return ordered + extra

    def _fallback_partition_specs(
        self,
        question: str,
        target_count: int,
        needs_web: bool,
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        queries = self._build_fallback_queries(question, max(6, target_count))
        for idx in range(target_count):
            label = f"Partition {idx + 1}"
            spec: Dict[str, Any] = {"label": label}
            if needs_web:
                if queries:
                    spec["queries"] = [queries[idx % len(queries)]]
            else:
                spec["focus"] = label
            specs.append(spec)
        return specs

    async def _execute_partition(
        self,
        plan_id: str,
        step: Dict[str, Any],
        instance: Optional[ModelInstanceInfo] = None,
    ) -> List[Dict[str, Any]]:
        meta = step.get("run_metadata") or {}
        partition_label = str(meta.get("partition_label") or step.get("title") or step.get("partition_key") or "")
        focus = str(meta.get("focus") or partition_label or "")
        queries = self._normalize_queries(meta.get("queries"))
        needs_web = bool(meta.get("needs_web", self._plan_meta.get("needs_web", False)))
        search_depth = str(meta.get("search_depth") or self._plan_meta.get("search_depth_mode") or "basic")
        if search_depth not in ("basic", "advanced"):
            search_depth = "basic"
        extract_depth = str(meta.get("extract_depth") or self._plan_meta.get("extract_depth") or "basic")
        if extract_depth not in ("basic", "advanced"):
            extract_depth = "basic"
        max_results = int(meta.get("max_results") or self._plan_meta.get("max_results") or 6)
        if max_results < 1:
            max_results = 1
        max_results = min(max_results, 8)
        max_queries = int(meta.get("max_queries") or 1)
        if max_queries < 1:
            max_queries = 1
        topic = meta.get("topic") or self._plan_meta.get("topic") or "general"
        time_range = meta.get("time_range") or self._plan_meta.get("time_range")
        if needs_web and not queries and self._question:
            queries = [self._question]
        errors: List[str] = []
        search_sources: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        if needs_web:
            if not self.tavily or not self.tavily.enabled:
                errors.append("Tavily API key missing or disabled.")
                await self._emit(
                    "tavily_error",
                    {"step": step.get("step_id"), "message": "Tavily API key missing or disabled."},
                )
            else:
                async def _run_search(query_list: List[str]) -> None:
                    for query in query_list:
                        resp = await self.tavily.search(
                            query=query,
                            search_depth=search_depth,
                            max_results=max_results,
                            topic=topic,
                            time_range=time_range,
                        )
                        if resp.get("error"):
                            errors.append(f"search_error:{resp.get('error')}")
                            await self._emit(
                                "tavily_error",
                                {"step": step.get("step_id"), "message": resp.get("error"), "query": query},
                            )
                            continue
                        results = resp.get("results") or []
                        await self._emit(
                            "tavily_search",
                            {
                                "step": step.get("step_id"),
                                "query": query,
                                "result_count": len(results),
                                "new_sources": len(results),
                                "duplicate_sources": 0,
                            },
                        )
                        for res in results:
                            if not isinstance(res, dict):
                                continue
                            url = str(res.get("url") or "").strip()
                            if not url:
                                continue
                            content = res.get("content") or res.get("raw_content") or ""
                            search_sources.append(
                                {
                                    "url": url,
                                    "title": res.get("title") or "",
                                    "publisher": res.get("source") or "",
                                    "date_published": res.get("published_date") or "",
                                    "snippet": str(content)[:400],
                                    "extracted_text": content,
                                }
                            )

                await _run_search(queries[:max_queries])
                seen_urls: set[str] = set()
                deduped: List[Dict[str, Any]] = []
                for src in search_sources:
                    url = str(src.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    deduped.append(src)
                search_sources = deduped
                if not search_sources:
                    fallback_queries = self._build_fallback_queries(self._question, 4)
                    extra_queries = [q for q in fallback_queries if q not in queries][:1]
                    if extra_queries:
                        await _run_search(extra_queries)
                        seen_urls.clear()
                        deduped = []
                        for src in search_sources:
                            url = str(src.get("url") or "").strip()
                            if not url or url in seen_urls:
                                continue
                            seen_urls.add(url)
                            deduped.append(src)
                        search_sources = deduped
                else:
                    q_lower = self._question.lower()
                    location_keywords: List[str] = []
                    domain_keywords: List[str] = []
                    if "naples" in q_lower:
                        location_keywords.append("naples")
                    if "bonita" in q_lower:
                        location_keywords.append("bonita")
                    if "golf" in q_lower:
                        domain_keywords.append("golf")
                    relevant = 0
                    for src in search_sources:
                        text = f"{src.get('url','')} {src.get('title','')} {src.get('snippet','')}".lower()
                        has_location = not location_keywords or any(k in text for k in location_keywords)
                        has_domain = not domain_keywords or any(k in text for k in domain_keywords)
                        if has_location and has_domain:
                            relevant += 1
                    if relevant == 0:
                        fallback_queries = self._build_fallback_queries(self._question, 4)
                        extra_queries = [q for q in fallback_queries if q not in queries][:1]
                        if extra_queries:
                            await _run_search(extra_queries)
                            seen_urls.clear()
                            deduped = []
                            for src in search_sources:
                                url = str(src.get("url") or "").strip()
                                if not url or url in seen_urls:
                                    continue
                                seen_urls.add(url)
                            deduped.append(src)
                        search_sources = deduped
                if search_sources:
                    for src in search_sources[:3]:
                        await self._emit(
                            "source_found",
                            {
                                "step": step.get("step_id"),
                                "title": src.get("title") or "",
                                "publisher": src.get("publisher") or "",
                                "url": src.get("url") or "",
                            },
                        )
                sources = list(search_sources)
                urls = [s.get("url") for s in search_sources if s.get("url")]
                if urls:
                    extract_budget = int((self._plan_meta.get("tool_budget") or {}).get("tavily_extract", 0) or 0)
                    max_extracts = int(meta.get("max_extracts") or 0)
                    if max_extracts <= 0:
                        max_extracts = min(len(urls), max(2, min(extract_budget or max_results, max_results)))
                    url_slice = urls[:max_extracts]
                    await self._emit(
                        "tavily_extract",
                        {"step": step.get("step_id"), "urls": url_slice},
                    )
                    extract_resp = await self.tavily.extract(url_slice, extract_depth=extract_depth)
                    if extract_resp.get("error"):
                        errors.append(f"extract_error:{extract_resp.get('error')}")
                        await self._emit(
                            "tavily_error",
                            {"step": step.get("step_id"), "message": extract_resp.get("error")},
                        )
                    else:
                        extracted: List[Dict[str, Any]] = []
                        for res in extract_resp.get("results") or []:
                            if not isinstance(res, dict):
                                continue
                            url = str(res.get("url") or "").strip()
                            if not url:
                                continue
                            content = res.get("content") or res.get("raw_content") or ""
                            extracted.append(
                                {
                                    "url": url,
                                    "title": res.get("title") or "",
                                    "publisher": res.get("source") or "",
                                    "date_published": res.get("published_date") or "",
                                    "snippet": str(content)[:400],
                                    "extracted_text": content,
                                }
                            )
                        if extracted:
                            sources = extracted
        analysis: Dict[str, Any] = {"summary": "", "candidates": [], "gaps": []}
        if self.model_manager:
            trimmed_sources: List[Dict[str, Any]] = []
            for src in sources[:12]:
                text = str(src.get("extracted_text") or src.get("snippet") or "")
                text = self._reduce_source_text(text, 2000)
                trimmed_sources.append(
                    {
                        "url": src.get("url"),
                        "title": src.get("title"),
                        "snippet": src.get("snippet"),
                        "extracted_text": text,
                    }
                )
            schema = {
                "name": "partition_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "notes": {"type": "string"},
                                    "meets_criteria": {"type": "boolean"},
                                    "evidence_urls": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["name"],
                                "additionalProperties": True,
                            },
                        },
                        "gaps": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "candidates", "gaps"],
                    "additionalProperties": True,
                },
            }
            prompt = {
                "question": self._question,
                "partition": partition_label,
                "focus": focus,
                "queries": queries,
                "sources": trimmed_sources,
                "notes": (
                    "List any named communities or neighborhoods mentioned in the sources. "
                    "If criteria are missing, set meets_criteria=false and note what is missing."
                ),
            }
            request = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Summarize sources and extract candidate communities. "
                            "Return JSON only; do not leave candidates empty if names are present."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
                ],
                "temperature": 0.2,
                "max_tokens": 700,
                "response_format": {"type": "json_schema", "json_schema": schema},
                "use_responses": True,
            }
            required = await self._resolve_required_capabilities(["structured_output"])
            if "structured_output" not in required:
                request.pop("response_format", None)
            try:
                if instance:
                    response = await self.model_manager.backends[instance.backend_id].call_chat_completion(
                        instance,
                        request,
                    )
                else:
                    response = await self._call_with_fallback(
                        request,
                        required_capabilities=required,
                    )
                content = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
                parsed = None
                if content:
                    try:
                        parsed = json.loads(content)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict):
                    analysis.update(parsed)
                elif content:
                    analysis["summary"] = content.strip()
            except Exception as exc:
                analysis["gaps"].append(f"analysis_error:{exc}")
        else:
            if needs_web and not sources:
                analysis["gaps"].append("No sources available for this partition.")
            analysis["summary"] = f"Partition {partition_label} complete."
        if not analysis.get("summary"):
            analysis["summary"] = f"Partition {partition_label} complete."
        if not analysis.get("candidates"):
            guesses = self._guess_developments_from_sources(sources, 6)
            if guesses:
                analysis["candidates"] = [
                    {
                        "name": item.get("name") or "",
                        "notes": (item.get("location") or "Needs verification").strip() or "Needs verification",
                        "meets_criteria": False,
                    }
                    for item in guesses
                    if item.get("name")
                ]
        payload = {
            "partition": partition_label,
            "focus": focus,
            "queries": queries,
            "sources": sources,
            "analysis": analysis,
            "errors": errors,
            "timestamp_utc": self._utc_iso(),
        }
        ref = await self.artifact_store.put(
            None,
            payload,
            metadata={"step_id": step["step_id"], "partition": partition_label},
            kind="json",
        )
        await self._maybe_raise_worker_finding(plan_id, step, [ref])
        return [ref]

    def _format_partition_note(self, payload: Any, title: str) -> str:
        if payload is None:
            return ""
        summary = ""
        candidates: List[Any] = []
        gaps: List[Any] = []
        if isinstance(payload, dict):
            analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
            summary = str(analysis.get("summary") or payload.get("summary") or "").strip()
            candidates = analysis.get("candidates") or payload.get("candidates") or []
            gaps = analysis.get("gaps") or payload.get("gaps") or []
        else:
            summary = str(payload).strip()
        lines: List[str] = []
        header = title
        if summary:
            header = f"{title}: {summary}"
        lines.append(header)
        if isinstance(candidates, list) and candidates:
            for candidate in candidates[:5]:
                if isinstance(candidate, dict):
                    name = str(candidate.get("name") or "").strip()
                    notes = str(candidate.get("notes") or "").strip()
                    meets = candidate.get("meets_criteria")
                    parts = [p for p in [name, notes] if p]
                    if meets is True:
                        parts.append("meets_criteria")
                    elif meets is False:
                        parts.append("borderline")
                    line = " - " + ": ".join(parts) if parts else ""
                else:
                    line = " - " + str(candidate)
                if line.strip():
                    lines.append(line)
        if isinstance(gaps, list) and gaps:
            gap_text = "; ".join([str(g) for g in gaps[:3] if g])
            if gap_text:
                lines.append(f"Gaps: {gap_text}")
        return "\n".join(lines).strip()

    async def _draftbook(self, plan_id: str, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        prereq_ids = step.get("prereq_step_ids") or []
        step_rows = await self.plan_store.get_steps(
            plan_id,
            prereq_ids,
            fields=["step_id", "output_refs", "title"],
        )
        notes: List[str] = []
        for row in step_rows.get("steps") or []:
            title = str(row.get("title") or row.get("step_id") or "")
            for ref in row.get("output_refs") or []:
                payload = await self.artifact_store.get(ref.get("ref_id"))
                note = self._format_partition_note(payload, title)
                if note:
                    notes.append(note)
        max_chars = max(500, self.draft_token_budget * 4)
        packed: List[str] = []
        total = 0
        for note in notes:
            if total + len(note) > max_chars:
                break
            packed.append(note)
            total += len(note) + 1
        packed_text = "\n\n".join(packed)
        if self.model_manager:
            schema = {
                "name": "draftbook",
                "schema": {
                    "type": "object",
                    "properties": {
                        "draft": {"type": "string"},
                        "candidates": {"type": "array", "items": {"type": "string"}},
                        "key_points": {"type": "array", "items": {"type": "string"}},
                        "gaps": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["draft", "candidates", "key_points", "gaps"],
                    "additionalProperties": True,
                },
            }
            prompt = {
                "question": self._question,
                "partition_notes": packed_text,
                "instructions": (
                    "Summarize partition notes into a draft answer and key points. "
                    "List candidate communities mentioned, even if criteria are not confirmed."
                ),
            }
            try:
                response = await self._call_with_fallback(
                    request={
                        "messages": [
                            {"role": "system", "content": "Create a draftbook from the notes. Return JSON only."},
                            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 900,
                        "response_format": {"type": "json_schema", "json_schema": schema},
                        "use_responses": True,
                    },
                    required_capabilities=["structured_output"],
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
                parsed = None
                if content:
                    try:
                        parsed = json.loads(content)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict):
                    draft_ref = await self.artifact_store.put(
                        None,
                        parsed,
                        metadata={"step_id": step["step_id"]},
                        kind="json",
                    )
                    return [draft_ref]
                if content:
                    packed_text = content.strip()
            except Exception:
                pass
        draft_text = packed_text or "Draftbook incomplete."
        fallback_candidates = self._candidate_names_from_notes(packed_text, 12) if packed_text else []
        draft_ref = await self.artifact_store.put(
            None,
            {
                "draft": draft_text,
                "candidates": fallback_candidates,
                "key_points": [],
                "gaps": ["draftbook_unstructured"] if fallback_candidates else [],
            },
            metadata={"step_id": step["step_id"]},
            kind="json",
        )
        return [draft_ref]

    async def _finalize(self, plan_id: str, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        prereq_ids = step.get("prereq_step_ids") or []
        step_rows = await self.plan_store.get_steps(plan_id, prereq_ids, fields=["output_refs"])
        refs: List[Dict[str, Any]] = []
        for row in step_rows.get("steps") or []:
            refs.extend(row.get("output_refs") or [])
        draft_payload: Any = None
        if refs:
            draft_payload = await self.artifact_store.get(refs[0]["ref_id"])
        draft_text = ""
        candidate_text = ""
        if isinstance(draft_payload, dict):
            draft_text = str(draft_payload.get("draft") or json.dumps(draft_payload, ensure_ascii=True))
            candidates = draft_payload.get("candidates") or []
            if isinstance(candidates, list) and candidates:
                candidate_text = "\n".join([f"- {c}" for c in candidates if c])
        elif isinstance(draft_payload, str):
            draft_text = draft_payload
        if self.model_manager and draft_text:
            prompt = f"Question: {self._question}\n\nDraftbook:\n{draft_text}\n\n"
            if candidate_text:
                prompt += f"Candidate communities:\n{candidate_text}\n\n"
            prompt += (
                "Write the final answer. Include a concise candidate list with caveats and "
                "note which criteria are unverified if evidence is missing."
            )
            try:
                response = await self.model_manager.call(
                    required_capabilities=[],
                    request={
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You finalize the response. Return plain text only and include "
                                    "candidate communities with brief caveats."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 900,
                        "use_responses": True,
                    },
                )
                final_text = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
                final_text = final_text.strip() or draft_text
            except Exception:
                final_text = draft_text
        else:
            final_text = draft_text or "Final synthesis complete."
        final_text = self._sanitize_text(final_text)
        ref = await self.artifact_store.put(
            None, final_text, metadata={"step_id": step["step_id"]}, kind="text"
        )
        return [ref]

    async def _find_final_output(self, plan_id: str) -> Optional[Dict[str, Any]]:
        cursor = None
        while True:
            steps = await self.plan_store.list_steps(
                plan_id,
                status="DONE",
                cursor=cursor,
                limit=self.page_size,
                fields=["step_id", "tags", "output_refs", "title"],
            )
            for step in steps.get("steps") or []:
                tags = set(step.get("tags") or [])
                if "phase:finalize" in tags or str(step.get("title", "")).lower().startswith("final"):
                    refs = step.get("output_refs") or []
                    return refs[0] if refs else None
            cursor = steps.get("cursor_next")
            if cursor is None:
                break
        return None

    async def _fulfill_requests(self, plan_id: str) -> None:
        pending = await self.request_store.list_pending(plan_id=plan_id, limit=10)
        for req in pending:
            try:
                payload = req.get("payload") or {}
                expected_schema = req.get("expected_output_schema") or {}
                if self.model_manager:
                    prompt = payload.get("prompt") or payload.get("question") or json.dumps(payload, ensure_ascii=True)
                    messages = payload.get("messages")
                    if not isinstance(messages, list):
                        messages = [{"role": "user", "content": prompt}]
                    required_caps = ["structured_output"] if expected_schema else []
                    response = await self.model_manager.call(
                        required_capabilities=required_caps,
                        request={
                            "messages": messages,
                            "temperature": payload.get("temperature", 0.2),
                            "max_tokens": payload.get("max_tokens", 400),
                            "use_responses": True,
                        },
                    )
                    content = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
                    ref = await self.artifact_store.put(
                        None,
                        content,
                        metadata={"plan_id": plan_id, "request_id": req.get("request_id")},
                        kind="text",
                    )
                else:
                    ref = await self.artifact_store.put(
                        None,
                        {"request_type": req.get("type"), "payload": payload},
                        metadata={"plan_id": plan_id},
                        kind="json",
                    )
                await self.request_store.fulfill(req["request_id"], [ref])
            except Exception as exc:
                await self.request_store.fail(req["request_id"], str(exc))
