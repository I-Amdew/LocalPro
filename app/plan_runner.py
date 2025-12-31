import asyncio
import logging
from typing import Any, Dict, Optional

from .artifact_store import ArtifactStore
from .executor_state_store import ExecutorStateStore
from .plan_executor import PlanExecutor
from .plan_store import PlanStore
from .planner_v2 import scaffold_plan
from .tavily import TavilyClient
from .request_store import RequestStore
from .resource_manager import ResourceManager


logger = logging.getLogger("uvicorn.error")


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _summarize_overview(view: Dict[str, Any]) -> str:
    counts = view.get("counts_by_status") or {}
    total = int(sum(_coerce_int(v) for v in counts.values()) or 0)
    done = int(_coerce_int(counts.get("DONE")) or 0)
    failed = int(_coerce_int(counts.get("FAILED")) or 0)
    running = int(_coerce_int(counts.get("RUNNING")) or 0)
    ready = int(_coerce_int(counts.get("READY")) or 0)
    pending = int(_coerce_int(counts.get("PENDING")) or 0) + int(_coerce_int(counts.get("CLAIMED")) or 0)
    partitions = view.get("partitions_stats") or {}
    blockers = view.get("top_blockers") or []
    blocker_summary = ""
    if blockers:
        tops = [
            f"{item.get('step_id')}({item.get('blocked_count')})"
            for item in blockers
            if isinstance(item, dict)
        ]
        if tops:
            blocker_summary = f" blockers={', '.join(tops)}"
    return (
        f"steps={done + failed}/{total} "
        f"(running={running}, ready={ready}, pending={pending}, failed={failed}) "
        f"partitions={len(partitions)} rev={view.get('revision')}{blocker_summary}"
    )
async def run_plan_pipeline(
    *,
    db_path: str,
    question: str,
    reasoning_mode: str,
    planning_mode: str,
    reasoning_level: Optional[int],
    max_parallel: int,
    ram_headroom_pct: float = 10.0,
    vram_headroom_pct: float = 10.0,
    max_concurrent_runs: Optional[int] = None,
    per_model_class_limits: Optional[Dict[str, int]] = None,
    plan_context: Optional[Dict[str, Any]] = None,
    model_manager: Optional[Any] = None,
    tavily_client: Optional[TavilyClient] = None,
    bus: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    plan_store = PlanStore(db_path)
    artifact_store = ArtifactStore(db_path)
    request_store = RequestStore(db_path)
    resource_manager = ResourceManager(
        ram_headroom_pct=ram_headroom_pct,
        vram_headroom_pct=vram_headroom_pct,
        max_concurrent_runs=max_concurrent_runs,
        per_model_class_limits=per_model_class_limits,
    )
    state_store = ExecutorStateStore(db_path)

    overrides: Dict[str, Any] = {}
    if plan_context:
        overrides["metadata"] = dict(plan_context)
    plan_id = await scaffold_plan(
        plan_store,
        question=question,
        reasoning_mode=reasoning_mode,
        planning_mode=planning_mode,
        reasoning_level=reasoning_level,
        overrides=overrides,
    )
    overview = await plan_store.get_overview(plan_id)
    updater_stop = asyncio.Event()
    updater_task: Optional[asyncio.Task] = None
    if bus and run_id:
        total_steps = sum(overview.get("counts_by_status", {}).values())
        await bus.emit(
            run_id,
            "plan_created",
            {"steps": total_steps, "expected_total_steps": total_steps, "expected_passes": 1},
        )
        async def _ui_updater() -> None:
            last_revision = -1
            while not updater_stop.is_set():
                try:
                    view = await plan_store.get_overview(plan_id)
                    revision = int(view.get("revision") or 0)
                    if revision != last_revision:
                        try:
                            summary = _summarize_overview(view)
                        except Exception as exc:
                            summary = f"rev={revision} (summary unavailable)"
                            logger.warning("Run %s plan_overview summary failed: %s", run_id, exc)
                        logger.info("Run %s plan_overview: %s", run_id, summary)
                        await bus.emit(run_id, "plan_overview", view)
                        last_revision = revision
                except Exception as exc:
                    logger.warning("Run %s plan_overview refresh failed: %s", run_id, exc)
                await asyncio.sleep(0.5)

        updater_task = asyncio.create_task(_ui_updater())
    executor = PlanExecutor(
        plan_store,
        artifact_store,
        request_store,
        resource_manager,
        state_store,
        model_manager=model_manager,
        tavily_client=tavily_client,
        bus=bus,
        run_id=run_id,
        max_parallel=max_parallel,
    )
    result = await executor.run(plan_id)
    if updater_task:
        updater_stop.set()
        await updater_task
    final_ref = result.get("final_ref")
    final_text = ""
    if final_ref and final_ref.get("ref_id"):
        content = await artifact_store.get(final_ref["ref_id"])
        if isinstance(content, str):
            final_text = content
        else:
            final_text = str(content)
    return {"plan_id": plan_id, "final_ref": final_ref, "final_text": final_text}
