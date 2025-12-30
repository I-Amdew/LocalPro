import asyncio
import time

import pytest

from app.capacity_planner import CapacityPlanner
from app.load_profiler import LoadProfiler, build_config_signature
from app.model_manager import Autoscaler, ModelCandidate, ModelManager, ModelInstanceInfo
from app.resource_manager import ResourceManager
from app.resource_profiles import ModelResourceProfileStore, PROFILE_VERSION
from app.spawn_governor import SpawnGovernor
from tests.fakes import FakeModelBackend, FakeTelemetry


def _snapshot(ram_total_gb: float, ram_used_gb: float, vram_total_mb: float, vram_used_mb: float) -> dict:
    return {
        "ram": {
            "total_bytes": ram_total_gb * 1024**3,
            "used_bytes": ram_used_gb * 1024**3,
        },
        "gpus": [
            {
                "gpu_id": 0,
                "vram_total_mb": vram_total_mb,
                "vram_used_mb": vram_used_mb,
            }
        ],
        "captured_at": "2025-01-01T00:00:00Z",
    }


def test_capacity_planner_headroom_limits(tmp_path):
    profile = {
        "ram_instance_peak_bytes": 2 * 1024**3,
        "vram_instance_peak_mb_by_gpu": {"0": 3000.0},
        "baseline_snapshot": {
            "ram_used_bytes": 2 * 1024**3,
            "ram_total_bytes": 10 * 1024**3,
            "gpus": [{"gpu_id": 0, "vram_used_mb": 2000.0, "vram_total_mb": 10000.0}],
        },
    }
    snapshot = _snapshot(10, 2, 10000, 2000)
    telemetry = FakeTelemetry([snapshot])
    store = ModelResourceProfileStore(str(tmp_path / "cap.db"))
    planner = CapacityPlanner(telemetry=telemetry, store=store, ram_headroom_pct=10.0, vram_headroom_pct=10.0)
    capacity = planner.compute_capacity(profile, snapshot)
    assert capacity["max_instances_by_gpu"]["0"] == 2
    assert capacity["max_instances"] == 2


def test_resource_manager_headroom_denies():
    snapshot = {
        "ram": {"total_bytes": 10 * 1024**3, "used_bytes": 8.5 * 1024**3},
        "gpus": [{"gpu_id": 0, "vram_total_mb": 10000.0, "vram_used_mb": 8800.0}],
    }

    class StaticBackend:
        def snapshot(self):
            return snapshot

    manager = ResourceManager(backend=StaticBackend(), ram_headroom_pct=10.0, vram_headroom_pct=10.0)
    deny_ram = manager.reserve("run-ram", {"ram_bytes": 1.2 * 1024**3})
    assert not deny_ram.get("granted")
    deny_vram = manager.reserve("run-vram", {"vram_mb": 500.0, "gpu_id": 0})
    assert not deny_vram.get("granted")


@pytest.mark.asyncio
async def test_spawn_governor_unloads_on_headroom_violation():
    before = _snapshot(10, 5, 10000, 5000)
    after = _snapshot(10, 9.5, 10000, 9500)
    telemetry = FakeTelemetry([before, after])

    class Backend:
        def __init__(self):
            self.unload_calls = []

        async def load_instance(self, model_key, opts):
            return ModelInstanceInfo(
                backend_id="fake",
                instance_id="inst-1",
                model_key=model_key,
                api_identifier="inst-1",
                endpoint="http://fake",
            )

        async def unload_instance(self, instance_id):
            self.unload_calls.append(instance_id)

    backend = Backend()
    governor = SpawnGovernor(telemetry=telemetry, ram_headroom_pct=10.0, vram_headroom_pct=10.0)
    inst = await governor.spawn_instance(backend=backend, model_key="alpha", opts={}, budgets={})
    assert inst is None
    assert backend.unload_calls


@pytest.mark.asyncio
async def test_autoscaler_respects_capacity(tmp_path):
    snapshot = _snapshot(8, 2, 8000, 2000)
    telemetry = FakeTelemetry([snapshot, snapshot, snapshot])
    backend = FakeModelBackend(model_keys=["alpha"])
    manager = ModelManager(
        db_path=str(tmp_path / "mm.db"),
        backends={backend.id: backend},
        resource_manager=None,
        telemetry=telemetry,
        autoscaler=Autoscaler(enabled=True),
        routing_objective="balanced",
        tool_required_by_default=True,
    )
    candidate = ModelCandidate(
        backend_id=backend.id,
        model_key="alpha",
        display_name="alpha",
        capabilities={"tool_use": True, "structured_output": True},
        metadata={},
    )
    async with manager._candidate_lock:
        manager._candidates = [candidate]
    config_sig = build_config_signature(candidate, manager._profile_opts(candidate))
    profile = {
        "backend_id": backend.id,
        "model_key": "alpha",
        "config_signature": config_sig,
        "profile_version": PROFILE_VERSION,
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_snapshot": {
            "ram_used_bytes": 2 * 1024**3,
            "ram_total_bytes": 8 * 1024**3,
            "gpus": [{"gpu_id": 0, "vram_used_mb": 2000.0, "vram_total_mb": 8000.0}],
        },
        "ram_instance_peak_bytes": 3 * 1024**3,
        "vram_instance_peak_mb_by_gpu": {"0": 3000.0},
    }
    await manager.resource_profile_store.upsert(profile, ttl_seconds=3600)
    inst = await manager.acquire_instance(required_capabilities=["tool_use"], backlog=2)
    assert inst is not None
    assert len(backend.load_calls) == 1


@pytest.mark.asyncio
async def test_profile_persistence_and_config_change(tmp_path, monkeypatch):
    backend = FakeModelBackend(model_keys=["alpha"])
    telemetry = FakeTelemetry([_snapshot(8, 2, 8000, 2000)])
    manager = ModelManager(
        db_path=str(tmp_path / "mm2.db"),
        backends={backend.id: backend},
        resource_manager=None,
        telemetry=telemetry,
        autoscaler=Autoscaler(enabled=False),
        routing_objective="balanced",
        tool_required_by_default=True,
    )
    candidate = ModelCandidate(
        backend_id=backend.id,
        model_key="alpha",
        display_name="alpha",
        metadata={},
    )
    config_sig = build_config_signature(candidate, manager._profile_opts(candidate))
    profiled = {
        "backend_id": backend.id,
        "model_key": "alpha",
        "config_signature": config_sig,
        "profile_version": PROFILE_VERSION,
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_snapshot": {"ram_used_bytes": 0, "ram_total_bytes": 0, "gpus": []},
        "ram_instance_peak_bytes": 1,
        "vram_instance_peak_mb_by_gpu": {},
    }
    await manager.resource_profile_store.upsert(profiled, ttl_seconds=3600)
    manager._profile_tasks = {}

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(manager, "_profile_candidate", _noop)
    await manager._schedule_profiles([candidate])
    assert len(manager._profile_tasks) == 0

    candidate.metadata["context_length"] = 4096
    await manager._schedule_profiles([candidate])
    assert len(manager._profile_tasks) == 1


def test_load_profiler_suite_consistency(tmp_path):
    telemetry = FakeTelemetry([_snapshot(8, 2, 8000, 2000)])
    store = ModelResourceProfileStore(str(tmp_path / "prof.db"))
    profiler = LoadProfiler(telemetry=telemetry, store=store)
    names = [t["name"] for t in profiler.tests]
    assert names == [
        "sanity_short",
        "medium_completion",
        "long_completion",
        "structured_json",
        "tool_call",
    ]
    tokens = [t.get("max_tokens") for t in profiler.tests]
    assert all(isinstance(val, int) for val in tokens)


@pytest.mark.asyncio
async def test_profiling_sequential_loads(tmp_path):
    class TrackingBackend(FakeModelBackend):
        def __init__(self):
            super().__init__(model_keys=["alpha", "beta"])
            self.inflight = 0
            self.max_inflight = 0

        async def load_instance(self, model_key: str, opts: dict):
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            await asyncio.sleep(0.05)
            info = await super().load_instance(model_key, opts)
            self.inflight -= 1
            return info

    telemetry = FakeTelemetry([_snapshot(8, 2, 8000, 2000)] * 50)
    backend = TrackingBackend()
    manager = ModelManager(
        db_path=str(tmp_path / "seq.db"),
        backends={backend.id: backend},
        resource_manager=None,
        telemetry=telemetry,
        autoscaler=Autoscaler(enabled=False),
        routing_objective="balanced",
        tool_required_by_default=True,
        profiling={"enabled": True, "auto_profile": False, "max_concurrent_profiles": 1},
    )
    await manager.start_profiling(force=True)
    while manager.profile_status().get("running"):
        await asyncio.sleep(0.05)
    assert backend.max_inflight == 1
