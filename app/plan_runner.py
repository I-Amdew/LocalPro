import asyncio
from typing import Any, Dict, Optional

from .artifact_store import ArtifactStore
from .executor_state_store import ExecutorStateStore
from .plan_executor import PlanExecutor
from .plan_store import PlanStore
from .planner_v2 import scaffold_plan
from .request_store import RequestStore
from .resource_manager import ResourceManager


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
    model_manager: Optional[Any] = None,
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

    plan_id = await scaffold_plan(
        plan_store,
        question=question,
        reasoning_mode=reasoning_mode,
        planning_mode=planning_mode,
        reasoning_level=reasoning_level,
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
                view = await plan_store.get_overview(plan_id)
                revision = int(view.get("revision") or 0)
                if revision != last_revision:
                    await bus.emit(run_id, "plan_overview", view)
                    last_revision = revision
                await asyncio.sleep(0.5)

        updater_task = asyncio.create_task(_ui_updater())
    executor = PlanExecutor(
        plan_store,
        artifact_store,
        request_store,
        resource_manager,
        state_store,
        model_manager=model_manager,
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
