import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_upload_rejects_path_traversal_names(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("../evil.png", b"data", "image/png")}
            res = await client.post("/api/uploads", files=files)
            assert res.status_code == 400
            assert "Invalid filename" in res.text


@pytest.mark.asyncio
async def test_upload_rejects_oversize(app_factory):
    app, _, _, _ = app_factory(upload_max_mb=0)
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("big.png", b"x", "image/png")}
            res = await client.post("/api/uploads", files=files)
            assert res.status_code == 400
            assert "File too large" in res.text


@pytest.mark.asyncio
async def test_upload_accepts_and_downloads(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("ok.png", b"data", "image/png")}
            res = await client.post("/api/uploads", files=files)
            assert res.status_code == 200
            upload_id = res.json()["id"]
            res2 = await client.get(f"/api/uploads/{upload_id}")
            assert res2.status_code == 200
            assert res2.content == b"data"
            assert res2.headers["content-type"].startswith("image/png")
