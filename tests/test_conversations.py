import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_conversation_crud(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post("/api/conversations", json={"title": "Test chat"})
            assert res.status_code == 200
            convo_id = res.json()["conversation"]["id"]

            res = await client.get("/api/conversations")
            assert res.status_code == 200
            convo_ids = {c["id"] for c in res.json()["conversations"]}
            assert convo_id in convo_ids

            res = await client.patch(f"/api/conversations/{convo_id}", json={"title": "Renamed"})
            assert res.status_code == 200
            assert res.json()["conversation"]["title"] == "Renamed"

            res = await client.delete(f"/api/conversations/{convo_id}")
            assert res.status_code == 200

            res = await client.get("/api/conversations")
            convo_ids = {c["id"] for c in res.json()["conversations"]}
            assert convo_id not in convo_ids


@pytest.mark.asyncio
async def test_messages_listing_for_conversation(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        db = app.state.db
        convo = await db.create_conversation(title="Chat")
        await db.add_message("run-1", convo["id"], "user", "hello")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get(f"/api/conversations/{convo['id']}/messages")
            assert res.status_code == 200
            messages = res.json()["messages"]
            assert any(m["content"] == "hello" for m in messages)
