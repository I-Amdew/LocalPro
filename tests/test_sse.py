import asyncio
import json

import pytest
from asgi_lifespan import LifespanManager

from app.main import stream_events, stream_global_events


@pytest.mark.asyncio
async def test_run_sse_stream_returns_past_events(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        db = app.state.db
        bus = app.state.bus
        convo = await db.create_conversation(title="Chat")
        await db.insert_run("run-sse", convo["id"], "question", "auto")
        await bus.emit("run-sse", "run_started", {"run_id": "run-sse"})
        response = await stream_events("run-sse", db=db, bus=bus)
        chunk = await asyncio.wait_for(response.body_iterator.__anext__(), timeout=1)
        line = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else chunk
        payload = json.loads(line.replace("data:", "").strip())
        assert payload["event_type"] == "run_started"
        await response.body_iterator.aclose()


@pytest.mark.asyncio
async def test_global_sse_stream_receives_new_event(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        bus = app.state.bus
        response = await stream_global_events(bus=bus)

        async def emit_event():
            await asyncio.sleep(0.01)
            await bus.emit("conversation", "conversation_created", {"conversation_id": "c1"})

        task = asyncio.create_task(emit_event())
        chunk = await asyncio.wait_for(response.body_iterator.__anext__(), timeout=1)
        line = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else chunk
        payload = json.loads(line.replace("data:", "").strip())
        assert payload["event_type"] == "conversation_created"
        await task
        await response.body_iterator.aclose()
