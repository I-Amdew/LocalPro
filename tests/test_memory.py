import asyncio

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_memory_crud(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            convo_res = await client.post("/api/conversations", json={"title": "Memory test"})
            assert convo_res.status_code == 200
            conversation_id = convo_res.json()["conversation"]["id"]

            res = await client.post(
                "/api/memory",
                json={"conversation_id": conversation_id, "title": "Note", "content": "Remember"},
            )
            assert res.status_code == 200
            mem_id = res.json()["id"]

            res = await client.get(f"/api/memory?conversation_id={conversation_id}")
            assert res.status_code == 200
            ids = {item["id"] for item in res.json()["items"]}
            assert mem_id in ids

            res = await client.patch(f"/api/memory/{mem_id}", json={"title": "Updated"})
            assert res.status_code == 200

            res = await client.delete(f"/api/memory/{mem_id}")
            assert res.status_code == 200

            res = await client.get(f"/api/memory?conversation_id={conversation_id}")
            ids = {item["id"] for item in res.json()["items"]}
            assert mem_id not in ids


@pytest.mark.asyncio
async def test_run_auto_memory_saves_when_enabled(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {
                "question": "What is LocalPro?",
                "model_tier": "fast",
                "reasoning_mode": "auto",
                "auto_memory": True,
            }
            res = await client.post("/api/run", json=payload)
            assert res.status_code == 200
            run_id = res.json()["run_id"]
            conversation_id = res.json()["conversation_id"]
            task = app.state.run_tasks.get(run_id)
            assert task is not None
            await asyncio.wait_for(task, timeout=5)

            res = await client.get(f"/api/memory?conversation_id={conversation_id}")
            items = res.json()["items"]
            assert len(items) >= 1
