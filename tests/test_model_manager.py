import asyncio

import pytest

from app.model_manager import Autoscaler, ModelCandidate, ModelManager, ModelSelector
from tests.fakes import FakeModelBackend


def test_model_selector_prefers_required_capabilities():
    candidates = [
        ModelCandidate(
            backend_id="fake",
            model_key="fast-model",
            display_name="fast",
            capabilities={"tool_use": False, "structured_output": True},
            profile={"tps": 50, "latency_ms": 40},
        ),
        ModelCandidate(
            backend_id="fake",
            model_key="tool-model",
            display_name="tool",
            capabilities={"tool_use": True, "structured_output": True},
            profile={"tps": 20, "latency_ms": 80},
        ),
    ]
    selector = ModelSelector(candidates)
    pick_tool = selector.choose_candidate(required_capabilities=["tool_use"], objective="best_latency")
    assert pick_tool is not None
    assert pick_tool.model_key == "tool-model"
    pick_fast = selector.choose_candidate(required_capabilities=[], objective="best_latency")
    assert pick_fast is not None
    assert pick_fast.model_key == "fast-model"


def test_autoscaler_scales_with_backlog():
    scaler = Autoscaler(
        enabled=True,
        global_max_instances=10,
        per_backend_max_instances={"fake": 5},
        min_instances={"executor": 1},
    )
    desired = scaler.desired_instances("fake", backlog=4, ready=1)
    assert desired == 4


@pytest.mark.asyncio
async def test_model_manager_autoscale_spawns_instances(tmp_path):
    backend = FakeModelBackend(model_keys=["alpha"])
    manager = ModelManager(
        db_path=str(tmp_path / "mm.db"),
        backends={backend.id: backend},
        resource_manager=None,
        autoscaler=Autoscaler(enabled=True),
        routing_objective="balanced",
        tool_required_by_default=True,
    )
    candidate = ModelCandidate(
        backend_id=backend.id,
        model_key="alpha",
        display_name="alpha",
        capabilities={"tool_use": True, "structured_output": True},
        profile={"tps": 20, "latency_ms": 60},
    )
    async with manager._candidate_lock:
        manager._candidates = [candidate]
    instance = await manager.acquire_instance(required_capabilities=["tool_use"], backlog=2)
    assert instance is not None
    assert len(backend.load_calls) == 2
    await manager.release_instance(instance.instance_id)


@pytest.mark.asyncio
async def test_model_manager_concurrent_calls_use_distinct_instances(tmp_path):
    backend = FakeModelBackend(model_keys=["alpha"], delay_seconds=0.05)
    manager = ModelManager(
        db_path=str(tmp_path / "mm2.db"),
        backends={backend.id: backend},
        resource_manager=None,
        autoscaler=Autoscaler(enabled=False),
        routing_objective="balanced",
        tool_required_by_default=True,
    )
    candidate = ModelCandidate(
        backend_id=backend.id,
        model_key="alpha",
        display_name="alpha",
        capabilities={"tool_use": True, "structured_output": True},
        profile={"tps": 20, "latency_ms": 60},
    )
    async with manager._candidate_lock:
        manager._candidates = [candidate]
    inst1 = await backend.load_instance("alpha", {"identifier": "alpha-1"})
    inst2 = await backend.load_instance("alpha", {"identifier": "alpha-2"})
    await manager.worker_pool.add(inst1)
    await manager.worker_pool.add(inst2)

    async def run_call():
        response, instance = await manager.call_with_instance(
            required_capabilities=[],
            request={"messages": [{"role": "user", "content": "ping"}], "max_tokens": 10},
        )
        return response, instance

    (resp_a, inst_a), (resp_b, inst_b) = await asyncio.gather(run_call(), run_call())
    assert inst_a.instance_id != inst_b.instance_id
    assert resp_a and resp_b
