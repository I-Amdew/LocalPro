import asyncio

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from tests.fakes import FakeLMStudioClient


@pytest.mark.asyncio
async def test_prompt_state_get_and_clear(app_factory):
    fake_lm = FakeLMStudioClient(delay_seconds=0.05)
    app, _, _, _ = app_factory(fake_lm=fake_lm)
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post(
                "/api/run",
                json={
                    "question": "Prompt state check",
                    "model_tier": "fast",
                    "reasoning_mode": "auto",
                    "auto_memory": False,
                },
            )
            assert res.status_code == 200
            run_id = res.json()["run_id"]

            prompt_res = await client.get("/api/prompt")
            assert prompt_res.status_code == 200
            prompt = prompt_res.json()["prompt"]
            assert prompt is not None
            assert prompt["run_id"] == run_id

            clear_res = await client.delete("/api/prompt")
            assert clear_res.status_code == 200

            prompt_res = await client.get("/api/prompt")
            assert prompt_res.status_code == 200
            assert prompt_res.json()["prompt"] is None

            task = app.state.run_tasks.get(run_id)
            if task is not None:
                await asyncio.wait_for(task, timeout=5)
            run = await app.state.db.get_run_summary(run_id)
            assert run["status"] in ("stopped", "completed") or run["status"].startswith("error")
