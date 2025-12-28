from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.config import AppSettings, EndpointConfig
from app.main import create_app
from tests.fakes import FakeLMStudioClient, FakeTavilyClient


def make_settings(tmp_path: Path, **overrides) -> AppSettings:
    base_url = overrides.pop("base_url", "http://lm.test/v1")
    settings = AppSettings(
        lm_studio_base_url=base_url,
        model_orch="test-model",
        oss_max_tokens=4096,
        model_qwen8="test-model",
        model_qwen4="test-model",
        orch_endpoint=EndpointConfig(base_url=base_url, model_id="test-model"),
        worker_a_endpoint=EndpointConfig(base_url=base_url, model_id="test-model"),
        worker_b_endpoint=EndpointConfig(base_url="", model_id=""),
        worker_c_endpoint=EndpointConfig(base_url="", model_id=""),
        fast_endpoint=EndpointConfig(base_url=base_url, model_id="test-model"),
        deep_planner_endpoint=EndpointConfig(base_url=base_url, model_id="test-model"),
        deep_orchestrator_endpoint=EndpointConfig(base_url=base_url, model_id="test-model"),
        router_endpoint=EndpointConfig(base_url=base_url, model_id="test-model"),
        summarizer_endpoint=EndpointConfig(base_url=base_url, model_id="test-model"),
        verifier_endpoint=EndpointConfig(base_url=base_url, model_id="test-model"),
        tavily_api_key=None,
        database_path=str(tmp_path / "test.db"),
        upload_dir=str(tmp_path / "uploads"),
        host="127.0.0.1",
        port=8000,
        reasoning_depth_default="HIGH",
    )
    if overrides:
        settings = settings.model_copy(update=overrides)
    return settings


@pytest.fixture
def app_factory(tmp_path: Path):
    def _factory(
        *,
        fake_lm: FakeLMStudioClient | None = None,
        fake_tavily: FakeTavilyClient | None = None,
        config_path: Path | None = None,
        **settings_overrides,
    ):
        settings = make_settings(tmp_path, **settings_overrides)
        lm_client = fake_lm or FakeLMStudioClient()
        tavily_client = fake_tavily or FakeTavilyClient(api_key=settings.tavily_api_key)
        cfg_path = config_path or (tmp_path / "config.json")
        app = create_app(settings, lm_client=lm_client, tavily_client=tavily_client, config_path=cfg_path)
        return app, cfg_path, lm_client, tavily_client

    return _factory


@pytest.fixture
async def client(app_factory):
    app, config_path, lm_client, tavily_client = app_factory()
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            http_client.app = app  # type: ignore[attr-defined]
            http_client.config_path = config_path  # type: ignore[attr-defined]
            http_client.fake_lm = lm_client  # type: ignore[attr-defined]
            http_client.fake_tavily = tavily_client  # type: ignore[attr-defined]
            yield http_client
