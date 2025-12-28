import asyncio

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from tests.fakes import FakeLMStudioClient


async def wait_for_run(app, run_id: str, timeout: float = 5.0) -> None:
    task = app.state.run_tasks.get(run_id)
    if task is not None:
        await asyncio.wait_for(task, timeout=timeout)


@pytest.mark.asyncio
async def test_start_run_creates_run_and_emits_event(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/run",
                json={
                    "question": "Hello",
                    "model_tier": "fast",
                    "reasoning_mode": "auto",
                    "auto_memory": False,
                },
            )
            assert res.status_code == 200
            run_id = res.json()["run_id"]
            await wait_for_run(app, run_id)

            events = await app.state.db.list_events(run_id)
            event_types = {ev["event_type"] for ev in events}
            assert "run_started" in event_types


@pytest.mark.asyncio
async def test_stop_run_sets_status(app_factory):
    fake_lm = FakeLMStudioClient(delay_seconds=0.05)
    app, _, _, _ = app_factory(fake_lm=fake_lm)
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/run",
                json={
                    "question": "Stop test",
                    "model_tier": "fast",
                    "reasoning_mode": "auto",
                    "auto_memory": False,
                },
            )
            assert res.status_code == 200
            run_id = res.json()["run_id"]
            stop_res = await client.post(f"/api/run/{run_id}/stop")
            assert stop_res.status_code == 200
            assert stop_res.json()["status"] in ("stopping", "stopped", "completed")
            await wait_for_run(app, run_id)

            run = await app.state.db.get_run_summary(run_id)
            assert run["status"] in ("stopped", "completed") or run["status"].startswith("error")


@pytest.mark.asyncio
async def test_run_artifacts_endpoint_shape_and_evidence_dump(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/run",
                json={
                    "question": "Artifacts",
                    "model_tier": "fast",
                    "reasoning_mode": "auto",
                    "auto_memory": False,
                    "evidence_dump": True,
                },
            )
            assert res.status_code == 200
            run_id = res.json()["run_id"]
            await wait_for_run(app, run_id)

            res = await client.get(f"/api/run/{run_id}/artifacts")
            assert res.status_code == 200
            payload = res.json()
            for key in ("run", "sources", "claims", "draft", "verifier", "artifacts", "uploads"):
                assert key in payload
            dump_artifacts = [a for a in payload["artifacts"] if a["artifact_type"] == "evidence_dump"]
            assert dump_artifacts
            assert "sources" in dump_artifacts[0]["content_json"]
            assert "claims" in dump_artifacts[0]["content_json"]


@pytest.mark.asyncio
async def test_run_events_endpoint_returns_events(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/run",
                json={
                    "question": "Events check",
                    "model_tier": "fast",
                    "reasoning_mode": "auto",
                    "auto_memory": False,
                },
            )
            assert res.status_code == 200
            run_id = res.json()["run_id"]
            await wait_for_run(app, run_id)

            events_res = await client.get(f"/api/run/{run_id}/events")
            assert events_res.status_code == 200
            payload = events_res.json()
            assert payload["events"]
            assert payload["last_seq"] >= payload["events"][-1]["seq"]
