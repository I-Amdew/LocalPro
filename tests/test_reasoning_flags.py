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
async def test_reasoning_auto_forces_auto(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/run",
                json={
                    "question": "Reasoning auto",
                    "model_tier": "pro",
                    "reasoning_mode": "manual",
                    "reasoning_auto": True,
                    "manual_level": "LOW",
                },
            )
            assert res.status_code == 200
            run_id = res.json()["run_id"]
            await wait_for_run(app, run_id)
            run = await app.state.db.get_run_summary(run_id)
            assert run["reasoning_mode"] == "auto"


@pytest.mark.asyncio
async def test_reasoning_auto_false_respects_manual(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/run",
                json={
                    "question": "Reasoning manual",
                    "model_tier": "pro",
                    "reasoning_mode": "manual",
                    "reasoning_auto": False,
                    "manual_level": "LOW",
                },
            )
            assert res.status_code == 200
            run_id = res.json()["run_id"]
            await wait_for_run(app, run_id)
            run = await app.state.db.get_run_summary(run_id)
            assert run["reasoning_mode"] == "manual"
            assert run["router_decision"]["reasoning_level"] == "LOW"


@pytest.mark.asyncio
async def test_reasoning_depth_default_applies_on_router_fallback(app_factory):
    fake_lm = FakeLMStudioClient(router_response="not-json")
    app, _, _, _ = app_factory(fake_lm=fake_lm, reasoning_depth_default="HIGH")
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/run",
                json={
                    "question": "Default depth",
                    "model_tier": "pro",
                },
            )
            assert res.status_code == 200
            run_id = res.json()["run_id"]
            await wait_for_run(app, run_id)
            run = await app.state.db.get_run_summary(run_id)
            assert run["router_decision"]["reasoning_level"] == "HIGH"
