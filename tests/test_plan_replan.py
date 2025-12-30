import pytest

from app.artifact_store import ArtifactStore
from app.db import Database
from app.executor_state_store import ExecutorStateStore
from app.plan_executor import PlanExecutor
from app.plan_store import PlanStore
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
async def test_verifier_triggers_replan(tmp_path):
    db_path = tmp_path / "replan.db"
    db = Database(str(db_path))
    await db.init()
    plan_store = PlanStore(str(db_path))
    artifact_store = ArtifactStore(str(db_path))
    request_store = RequestStore(str(db_path))
    state_store = ExecutorStateStore(str(db_path))
    plan_id = await plan_store.create({"query": "replan"})
    step_a = "step-a"
    step_b = "step-b"
    await plan_store.add_steps(
        plan_id,
        [
            {
                "step_id": step_a,
                "title": "Step A",
                "status": "READY",
                "tags": ["phase:execute"],
                "run_metadata": {"mock_output": ""},
            },
            {
                "step_id": step_b,
                "title": "Step B",
                "status": "PENDING",
                "tags": ["phase:execute"],
                "prereq_step_ids": [step_a],
            },
        ],
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
    steps = await plan_store.list_steps(plan_id, fields=["step_id", "tags", "prereq_step_ids", "status"])
    fix_steps = [s for s in steps["steps"] if "replan:fix" in (s.get("tags") or [])]
    assert fix_steps
    fix_id = fix_steps[0]["step_id"]
    step_b_row = next(s for s in steps["steps"] if s["step_id"] == step_b)
    assert fix_id in (step_b_row.get("prereq_step_ids") or [])
    assert step_b_row["status"] == "DONE"


@pytest.mark.asyncio
async def test_patch_conflict_triggers_replan(tmp_path):
    db_path = tmp_path / "conflict.db"
    db = Database(str(db_path))
    await db.init()
    plan_store = PlanStore(str(db_path))
    artifact_store = ArtifactStore(str(db_path))
    request_store = RequestStore(str(db_path))
    state_store = ExecutorStateStore(str(db_path))
    plan_id = await plan_store.create({"query": "conflict"})
    step_id = "step-1"
    await plan_store.add_steps(
        plan_id,
        [{"step_id": step_id, "title": "Step 1", "status": "READY", "tags": ["phase:execute"]}],
    )
    base_rev = (await plan_store.get_overview(plan_id))["revision"]
    patch_id = await plan_store.propose_patch(
        plan_id,
        base_rev,
        "add step",
        [],
        [
            {
                "op": "ADD_STEP",
                "step": {
                    "step_id": "step-2",
                    "title": "Step 2",
                    "status": "PENDING",
                    "tags": ["phase:execute"],
                },
            }
        ],
    )
    await plan_store.validate_patch(plan_id, patch_id)
    await plan_store.update_step(plan_id, step_id, {"title": "Step 1 updated"})

    executor = PlanExecutor(
        plan_store,
        artifact_store,
        request_store,
        roomy_resource_manager(),
        state_store,
        max_parallel=1,
    )
    await executor.run(plan_id)
    findings = await plan_store.list_findings(plan_id)
    assert any(f.get("category") == "CONTRADICTION" for f in findings.get("items") or [])
    steps = await plan_store.list_steps(plan_id, fields=["step_type"])
    assert any(s.get("step_type") == "REPLAN_PATCH" for s in steps.get("steps") or [])


@pytest.mark.asyncio
async def test_bulk_invalidation(tmp_path):
    db_path = tmp_path / "bulk.db"
    db = Database(str(db_path))
    await db.init()
    plan_store = PlanStore(str(db_path))
    plan_id = await plan_store.create({"query": "bulk"})
    steps = []
    for idx in range(5):
        steps.append(
            {
                "step_id": f"s{idx}",
                "title": f"Tile {idx}",
                "status": "DONE",
                "tags": ["phase:execute", "partition:tile-12"],
                "output_refs": [{"ref_id": f"ref-{idx}", "kind": "text", "uri": f"db://artifact/ref-{idx}"}],
            }
        )
    await plan_store.add_steps(plan_id, steps)
    base_rev = (await plan_store.get_overview(plan_id))["revision"]
    patch_id = await plan_store.propose_patch(
        plan_id,
        base_rev,
        "invalidate tile-12",
        [],
        [
            {
                "op": "BULK_OP",
                "match": {"tags": ["partition:tile-12"], "status": "DONE"},
                "action": {
                    "op": "INVALIDATE_STEP_OUTPUTS",
                    "reason": "invalid outputs",
                    "behavior": "SET_STATUS_STALE",
                },
            }
        ],
    )
    report = await plan_store.validate_patch(plan_id, patch_id)
    assert report["ok"] is True
    await plan_store.apply_patch(plan_id, patch_id, approver_id="tester")
    updated = await plan_store.list_steps(plan_id, fields=["status", "output_refs", "tags"])
    for step in updated.get("steps") or []:
        if "partition:tile-12" in (step.get("tags") or []):
            assert step["status"] == "STALE"
            assert step.get("output_refs") == []


@pytest.mark.asyncio
async def test_unaffected_work_continues(tmp_path):
    db_path = tmp_path / "parallel.db"
    db = Database(str(db_path))
    await db.init()
    plan_store = PlanStore(str(db_path))
    artifact_store = ArtifactStore(str(db_path))
    request_store = RequestStore(str(db_path))
    state_store = ExecutorStateStore(str(db_path))
    plan_id = await plan_store.create({"query": "parallel"})
    await plan_store.add_steps(
        plan_id,
        [
            {
                "step_id": "step-a",
                "title": "Partition A",
                "status": "READY",
                "tags": ["phase:execute", "partition:A"],
                "run_metadata": {"mock_output": ""},
            },
            {
                "step_id": "step-b",
                "title": "Partition B",
                "status": "READY",
                "tags": ["phase:execute", "partition:B"],
            },
        ],
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
    steps = await plan_store.list_steps(plan_id, fields=["step_id", "status"])
    step_b = next(s for s in steps.get("steps") or [] if s.get("step_id") == "step-b")
    assert step_b["status"] == "DONE"


@pytest.mark.asyncio
async def test_cycle_detection(tmp_path):
    db_path = tmp_path / "cycle.db"
    db = Database(str(db_path))
    await db.init()
    plan_store = PlanStore(str(db_path))
    plan_id = await plan_store.create({"query": "cycle"})
    await plan_store.add_steps(
        plan_id,
        [
            {"step_id": "a", "title": "A", "status": "READY"},
            {"step_id": "b", "title": "B", "status": "READY"},
        ],
    )
    base_rev = (await plan_store.get_overview(plan_id))["revision"]
    patch_id = await plan_store.propose_patch(
        plan_id,
        base_rev,
        "cycle",
        [],
        [
            {"op": "SET_PREREQS", "step_id": "a", "prereq_step_ids": ["b"]},
            {"op": "SET_PREREQS", "step_id": "b", "prereq_step_ids": ["a"]},
        ],
    )
    report = await plan_store.validate_patch(plan_id, patch_id)
    assert report["ok"] is False
    assert report["would_create_cycle"] is True
