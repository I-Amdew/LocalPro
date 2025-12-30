import asyncio

import pytest

from app.db import Database
from app.plan_store import PlanStore


@pytest.mark.asyncio
async def test_prereq_enforcement(tmp_path):
    db_path = tmp_path / "plan.db"
    db = Database(str(db_path))
    await db.init()
    store = PlanStore(str(db_path))
    plan_id = await store.create({"query": "test"})
    step_a = "step-a"
    step_b = "step-b"
    await store.add_steps(
        plan_id,
        [
            {"step_id": step_a, "title": "A", "status": "READY"},
            {"step_id": step_b, "title": "B", "status": "READY", "prereq_step_ids": [step_a]},
        ],
    )
    await store.update_step(plan_id, step_b, {"status": "CLAIMED"})
    blocked = await store.mark_done(plan_id, step_b, [])
    assert blocked["status"] == "blocked"
    await store.update_step(plan_id, step_a, {"status": "CLAIMED"})
    assert (await store.mark_done(plan_id, step_a, []))["ok"] is True
    await store.update_step(plan_id, step_b, {"status": "CLAIMED"})
    assert (await store.mark_done(plan_id, step_b, []))["ok"] is True


@pytest.mark.asyncio
async def test_claim_concurrency(tmp_path):
    db_path = tmp_path / "claim.db"
    db = Database(str(db_path))
    await db.init()
    store = PlanStore(str(db_path))
    plan_id = await store.create({"query": "claim"})
    step_id = "step-1"
    await store.add_steps(plan_id, [{"step_id": step_id, "title": "Claim", "status": "READY"}])

    results = await asyncio.gather(
        store.claim_step(plan_id, step_id, "w1"),
        store.claim_step(plan_id, step_id, "w2"),
    )
    assert sum(1 for res in results if res.get("ok")) == 1


@pytest.mark.asyncio
async def test_diff_correctness(tmp_path):
    db_path = tmp_path / "diff.db"
    db = Database(str(db_path))
    await db.init()
    store = PlanStore(str(db_path))
    plan_id = await store.create({"query": "diff"})
    step_id = "step-1"
    await store.add_steps(plan_id, [{"step_id": step_id, "title": "Step", "status": "PENDING"}])
    diff1 = await store.get_diff(plan_id, since_revision=0)
    assert diff1["changes"]
    rev1 = diff1["revision"]
    await store.update_step(plan_id, step_id, {"status": "READY"})
    diff2 = await store.get_diff(plan_id, since_revision=rev1)
    assert any("status" in change["changed_fields"] for change in diff2["changes"])


@pytest.mark.asyncio
async def test_pagination_stable_order(tmp_path):
    db_path = tmp_path / "page.db"
    db = Database(str(db_path))
    await db.init()
    store = PlanStore(str(db_path))
    plan_id = await store.create({"query": "page"})
    steps = []
    for idx in range(5):
        steps.append({"step_id": f"s{idx}", "title": f"Step {idx}", "status": "READY", "priority": idx})
    await store.add_steps(plan_id, steps)
    first = await store.list_steps(plan_id, limit=2)
    second = await store.list_steps(plan_id, cursor=first["cursor_next"], limit=2)
    first_ids = {s["step_id"] for s in first["steps"]}
    second_ids = {s["step_id"] for s in second["steps"]}
    assert not first_ids.intersection(second_ids)
