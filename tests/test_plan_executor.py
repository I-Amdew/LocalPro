import pytest

from app.artifact_store import ArtifactStore
from app.db import Database
from app.executor_state_store import ExecutorStateStore
from app.plan_executor import PlanExecutor
from app.plan_store import PlanStore
from app.planner_v2 import scaffold_plan
from app.request_store import RequestStore
from app.resource_manager import ResourceBackend, ResourceManager


class StubBackend(ResourceBackend):
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def snapshot(self):
        return self._snapshot


def roomy_resource_manager() -> ResourceManager:
    return ResourceManager(backend=StubBackend({"gpus": [], "ram": {"available_gb": 64}}))


@pytest.mark.asyncio
async def test_resource_gating(tmp_path):
    backend = StubBackend(
        {"gpus": [{"total_gb": 1.0, "free_gb": 0.5}], "ram": {"available_gb": 0.25}}
    )
    rm = ResourceManager(backend=backend, vram_headroom_pct=50.0)
    denied = rm.reserve("run1", {"vram_mb": 100, "gpu_id": 0})
    assert denied["granted"] is False
    rm = ResourceManager(backend=backend, vram_headroom_pct=5.0)
    granted = rm.reserve("run2", {"vram_mb": 100, "gpu_id": 0})
    assert granted["granted"] is True


@pytest.mark.asyncio
async def test_request_dispatch(tmp_path):
    db_path = tmp_path / "req.db"
    db = Database(str(db_path))
    await db.init()
    plan_store = PlanStore(str(db_path))
    artifact_store = ArtifactStore(str(db_path))
    request_store = RequestStore(str(db_path))
    state_store = ExecutorStateStore(str(db_path))
    plan_id = await plan_store.create({"query": "request"})
    req_id = await request_store.create({"plan_id": plan_id, "type": "echo", "payload": {"x": 1}})
    executor = PlanExecutor(
        plan_store,
        artifact_store,
        request_store,
        roomy_resource_manager(),
        state_store,
        max_parallel=1,
    )
    await executor.run(plan_id)
    assert await request_store.status(req_id) == "done"
    refs = await request_store.result(req_id)
    assert refs and refs[0]["ref_id"]


@pytest.mark.asyncio
async def test_draftbook_creation(tmp_path):
    db_path = tmp_path / "draft.db"
    db = Database(str(db_path))
    await db.init()
    plan_store = PlanStore(str(db_path))
    artifact_store = ArtifactStore(str(db_path))
    request_store = RequestStore(str(db_path))
    state_store = ExecutorStateStore(str(db_path))
    plan_id = await scaffold_plan(
        plan_store,
        question="draft",
        reasoning_mode="normal",
        planning_mode="normal",
        reasoning_level=1,
    )
    executor = PlanExecutor(
        plan_store,
        artifact_store,
        request_store,
        roomy_resource_manager(),
        state_store,
        max_parallel=2,
    )
    result = await executor.run(plan_id)
    assert result["final_ref"] is not None
    steps = await plan_store.list_steps(plan_id, status="DONE", fields=["tags", "output_refs"])
    draft_steps = [s for s in steps["steps"] if "phase:draft" in (s.get("tags") or [])]
    assert draft_steps and draft_steps[0]["output_refs"]


@pytest.mark.asyncio
async def test_extensive_plan_scaling(tmp_path):
    db_path = tmp_path / "scale.db"
    db = Database(str(db_path))
    await db.init()
    plan_store = PlanStore(str(db_path))
    artifact_store = ArtifactStore(str(db_path))
    request_store = RequestStore(str(db_path))
    state_store = ExecutorStateStore(str(db_path))
    plan_id = await plan_store.create({"query": "scale"})
    batch = []
    for idx in range(10000):
        batch.append(
            {
                "step_id": f"done-{idx}",
                "title": f"Done {idx}",
                "status": "DONE",
                "tags": ["phase:execute"],
            }
        )
        if len(batch) >= 1000:
            await plan_store.add_steps(plan_id, batch)
            batch = []
    if batch:
        await plan_store.add_steps(plan_id, batch)
    await plan_store.add_steps(
        plan_id,
        [
            {
                "step_id": "final-1",
                "title": "Finalize",
                "status": "READY",
                "tags": ["phase:finalize"],
            }
        ],
    )
    executor = PlanExecutor(
        plan_store,
        artifact_store,
        request_store,
        roomy_resource_manager(),
        state_store,
        max_parallel=1,
        page_size=200,
    )
    result = await executor.run(plan_id)
    assert result["final_ref"] is not None


@pytest.mark.asyncio
async def test_toy_end_to_end(tmp_path):
    db_path = tmp_path / "toy.db"
    db = Database(str(db_path))
    await db.init()
    plan_store = PlanStore(str(db_path))
    artifact_store = ArtifactStore(str(db_path))
    request_store = RequestStore(str(db_path))
    state_store = ExecutorStateStore(str(db_path))
    plan_id = await scaffold_plan(
        plan_store,
        question="toy",
        reasoning_mode="extensive",
        planning_mode="extensive",
        reasoning_level=1,
        overrides={"expansion_count": 5},
    )
    executor = PlanExecutor(
        plan_store,
        artifact_store,
        request_store,
        roomy_resource_manager(),
        state_store,
        max_parallel=2,
    )
    await executor.run(plan_id)
    overview = await plan_store.get_overview(plan_id)
    pending_left = sum(
        overview.get("counts_by_status", {}).get(status, 0)
        for status in ("PENDING", "READY", "CLAIMED", "RUNNING")
    )
    assert pending_left == 0
