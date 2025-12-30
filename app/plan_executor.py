import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from .artifact_store import ArtifactStore
from .executor_state_store import ExecutorStateStore
from .plan_store import PlanStore
from .model_manager import ModelManager, ModelInstanceInfo
from .request_store import RequestStore
from .resource_manager import ResourceManager


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
        self.max_parallel = max(1, max_parallel)
        self.page_size = max(50, page_size)
        self.draft_token_budget = draft_token_budget
        self._paused_steps: set[str] = set()
        self._paused_by_finding: Dict[str, List[str]] = {}

    async def run(self, plan_id: str, stop_event: Optional[asyncio.Event] = None) -> Dict[str, Any]:
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
            try:
                output_refs = await self._execute_step(plan_id, step, instance=instance)
                await self.plan_store.mark_done(plan_id, step_id, output_refs)
                return {"ok": True, "step_id": step_id, "output_refs": output_refs}
            except Exception as exc:
                await self.plan_store.mark_failed(plan_id, step_id, str(exc), retryable=True)
                max_retries = int(step.get("max_retries") or 0)
                if attempt < max_retries:
                    await self.plan_store.update_step(plan_id, step_id, {"status": "READY"})
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
        ready = await self.plan_store.list_steps(
            plan_id,
            status="READY",
            limit=min(self.page_size, capacity),
            fields=[
                "step_id",
                "title",
                "description",
                "step_type",
                "status",
                "prereq_step_ids",
                "cost_hint",
                "max_retries",
                "attempt",
                "tags",
                "run_metadata",
            ],
        )
        steps = [s for s in (ready.get("steps") or []) if s.get("step_id") not in self._paused_steps]
        if len(steps) >= capacity:
            return steps
        pending = await self.plan_store.list_steps(
            plan_id,
            status="PENDING",
            limit=min(self.page_size, capacity - len(steps)),
            fields=[
                "step_id",
                "title",
                "description",
                "step_type",
                "status",
                "prereq_step_ids",
                "cost_hint",
                "max_retries",
                "attempt",
                "tags",
                "run_metadata",
            ],
        )
        pending_steps = [s for s in (pending.get("steps") or []) if s.get("step_id") not in self._paused_steps]
        remaining = max(0, capacity - len(steps) - len(pending_steps))
        stale = {"steps": []}
        if remaining:
            stale = await self.plan_store.list_steps(
                plan_id,
                status="STALE",
                limit=min(self.page_size, remaining),
                fields=[
                    "step_id",
                    "title",
                    "description",
                    "step_type",
                    "status",
                    "prereq_step_ids",
                    "cost_hint",
                    "max_retries",
                    "attempt",
                    "tags",
                    "run_metadata",
                ],
            )
        for step in pending_steps:
            if not step.get("prereq_step_ids"):
                await self.plan_store.update_step(plan_id, step["step_id"], {"status": "READY"})
                steps.append({**step, "status": "READY"})
            else:
                steps.append(step)
        for step in [s for s in (stale.get("steps") or []) if s.get("step_id") not in self._paused_steps]:
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
        steps.sort(
            key=lambda item: (
                int(item.get("priority") or 0),
                dependents.get(item.get("step_id"), 0),
            ),
            reverse=True,
        )
        return steps[:capacity]

    async def _acquire_instance(self, step: Dict[str, Any]) -> Optional[ModelInstanceInfo]:
        if not self.model_manager:
            return None
        cost_hint = step.get("cost_hint") or {}
        required = cost_hint.get("required_capabilities") or []
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
        if instance and instance.resource_reservation:
            for key in ("vram_mb", "ram_mb", "ram_bytes", "cpu_pct", "gpu_id"):
                if key in instance.resource_reservation and instance.resource_reservation[key] is not None:
                    budgets.setdefault(key, instance.resource_reservation[key])
        if not budgets.get("vram_mb"):
            profile = self.resource_manager.model_profile(budgets.get("model_class") or "dynamic")
            budgets.setdefault("vram_mb", profile.get("vram_est"))
            budgets.setdefault("cpu_pct", profile.get("cpu_est"))
        return budgets

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

    def _allow_fallback(self, step: Dict[str, Any]) -> bool:
        meta = step.get("run_metadata") or {}
        if meta.get("allow_fallback"):
            return True
        return self._resolve_step_type(step) in {"VERIFIER", "REPLAN_PATCH", "PATCH_VERIFY"}

    def _coerce_str_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        if value is None:
            return []
        return [str(value)]

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
        new_steps: List[Dict[str, Any]] = []
        for idx in range(expansion_count):
            exec_step_id = str(uuid.uuid4())
            new_steps.append(
                {
                    "step_id": exec_step_id,
                    "title": f"Execute partition {idx + 1}",
                    "description": f"Process partition {idx + 1}.",
                    "status": "READY",
                    "tags": ["phase:execute", f"partition:{idx + 1}"],
                    "partition_key": str(idx + 1),
                    "priority": 5,
                    "prereq_step_ids": [step["step_id"]],
                    "cost_hint": {
                        "required_capabilities": [],
                        "preferred_objective": "latency",
                        "estimated_tokens": 200,
                    },
                    "created_by": {"type": "expander", "id": step["step_id"]},
                }
            )
            new_steps.append(
                {
                    "step_id": f"verify-{exec_step_id}",
                    "title": f"Verify partition {idx + 1}",
                    "description": f"Verify outputs for partition {idx + 1}.",
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
            {"created_steps": len(new_steps)},
            metadata={"step_id": step["step_id"]},
            kind="json",
        )
        return [ref]

    async def _draftbook(self, plan_id: str, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        prereq_ids = step.get("prereq_step_ids") or []
        step_rows = await self.plan_store.get_steps(
            plan_id,
            prereq_ids,
            fields=["step_id", "output_refs"],
        )
        refs: List[Dict[str, Any]] = []
        for row in step_rows.get("steps") or []:
            refs.extend(row.get("output_refs") or [])
        draft_ref = await self.artifact_store.summarize(refs, self.draft_token_budget)
        return [draft_ref]

    async def _finalize(self, plan_id: str, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        prereq_ids = step.get("prereq_step_ids") or []
        step_rows = await self.plan_store.get_steps(plan_id, prereq_ids, fields=["output_refs"])
        refs: List[Dict[str, Any]] = []
        for row in step_rows.get("steps") or []:
            refs.extend(row.get("output_refs") or [])
        if refs:
            content = await self.artifact_store.get(refs[0]["ref_id"])
            final_text = content if isinstance(content, str) else str(content)
        else:
            final_text = "Final synthesis complete."
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
