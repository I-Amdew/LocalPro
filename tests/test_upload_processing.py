import asyncio

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_run_processes_uploads_marks_status_and_emits_events(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("image.png", b"data", "image/png")}
            upload_res = await client.post("/api/uploads", files=files)
            assert upload_res.status_code == 200
            upload_id = upload_res.json()["id"]

            run_res = await client.post(
                "/api/run",
                json={
                    "question": "Use this upload",
                    "model_tier": "fast",
                    "reasoning_mode": "auto",
                    "auto_memory": False,
                    "upload_ids": [upload_id],
                },
            )
            assert run_res.status_code == 200
            run_id = run_res.json()["run_id"]

            task = app.state.run_tasks.get(run_id)
            if task is not None:
                await asyncio.wait_for(task, timeout=5)

            upload = await app.state.db.get_upload(upload_id)
            assert upload["status"] == "processed"
            assert upload["summary_text"]

            events = await app.state.db.list_events(run_id)
            assert any(ev["event_type"] == "upload_processed" for ev in events)

            artifacts_res = await client.get(f"/api/run/{run_id}/artifacts")
            assert artifacts_res.status_code == 200
            uploads = artifacts_res.json()["uploads"]
            assert uploads and uploads[0]["status"] == "processed"
