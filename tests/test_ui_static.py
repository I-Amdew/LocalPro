import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_root_serves_static_index(app_factory):
    app, _, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/")
            assert res.status_code == 200
            assert "LocalPro Chat" in res.text
