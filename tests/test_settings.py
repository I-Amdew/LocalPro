import json

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.config import load_settings


@pytest.mark.asyncio
async def test_get_settings_masks_tavily_key(app_factory):
    app, _, _, _ = app_factory(tavily_api_key="secret-key")
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/settings")
            assert res.status_code == 200
            data = res.json()
            assert data["settings"]["tavily_api_key"] == "********"


@pytest.mark.asyncio
async def test_post_settings_persists_config_and_db(app_factory):
    app, config_path, _, _ = app_factory()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            db = app.state.db
            before = await db.fetchone("SELECT COUNT(*) as cnt FROM configs")
            res = await client.post("/settings", json={"tavily_api_key": "new-key"})
            assert res.status_code == 200
            after = await db.fetchone("SELECT COUNT(*) as cnt FROM configs")
            assert after["cnt"] == before["cnt"] + 1

    saved = json.loads(config_path.read_text())
    assert saved["tavily_api_key"] == "new-key"


def test_config_precedence_configjson_wins_by_default(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"lm_studio_base_url": "http://config"}))
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://env")
    monkeypatch.delenv("LOCALPRO_ENV_OVERRIDES_CONFIG", raising=False)
    settings = load_settings(config_path=config_path)
    assert settings.lm_studio_base_url == "http://config"


def test_env_override_when_localpro_env_override_set(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"lm_studio_base_url": "http://config"}))
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://env")
    monkeypatch.setenv("LOCALPRO_ENV_OVERRIDES_CONFIG", "1")
    settings = load_settings(config_path=config_path)
    assert settings.lm_studio_base_url == "http://env"
