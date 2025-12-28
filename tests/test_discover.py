import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from tests.fakes import FakeLMStudioClient


@pytest.mark.asyncio
async def test_discover_models_returns_available_ids(app_factory):
    fake_lm = FakeLMStudioClient(model_ids=["alpha", "beta"])
    app, _, _, _ = app_factory(fake_lm=fake_lm, discovery_base_urls=["http://lm.test/v1"])
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post("/api/discover", json={"base_urls": ["http://lm.test/v1"]})
            assert res.status_code == 200
            payload = res.json()
            assert payload["results"]["http://lm.test/v1"]["ok"] is True
            assert payload["results"]["http://lm.test/v1"]["models"] == ["alpha", "beta"]
