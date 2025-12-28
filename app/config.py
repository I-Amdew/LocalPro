import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


CONFIG_PATH = Path("config.json")
ENV_OVERRIDE_KEY = "LOCALPRO_ENV_OVERRIDES_CONFIG"
ENV_OVERRIDE_TRUE = {"1", "true", "yes", "on"}


class EndpointConfig(BaseModel):
    base_url: str
    model_id: str

    model_config = {"protected_namespaces": ()}


class AppSettings(BaseModel):
    # Legacy defaults (single endpoint)
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    model_orch: str = "qwen/qwen3-vl-4b"
    oss_max_tokens: int = 131072
    model_qwen8: str = "qwen/qwen3-vl-8b"
    model_qwen4: str = "qwen/qwen3-vl-4b"

    # Per-role endpoints/models
    orch_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-4b")
    )
    worker_a_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-8b")
    )
    worker_b_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-8b:2")
    )
    worker_c_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-8b:3")
    )
    # Tier presets
    fast_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-4b")
    )
    deep_planner_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-4b")
    )
    deep_orchestrator_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-4b")
    )
    router_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-4b")
    )
    summarizer_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-4b")
    )
    verifier_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://127.0.0.1:1234/v1", model_id="qwen/qwen3-vl-8b")
    )

    tavily_api_key: Optional[str] = None
    search_depth_mode: str = Field(default="auto")
    max_results_base: int = 6
    max_results_high: int = 10
    extract_depth: str = "basic"
    database_path: str = "app_data.db"
    host: str = "0.0.0.0"
    port: int = 8000
    strict_mode: bool = False
    reasoning_depth_default: str = "AUTO"
    discovery_base_urls: list[str] = Field(default_factory=lambda: ["http://localhost:1234/v1"])
    upload_dir: str = "uploads"
    upload_max_mb: int = 15

    def to_safe_dict(self) -> dict:
        data = self.model_dump()
        if data.get("tavily_api_key"):
            data["tavily_api_key"] = "********"
        return data

    model_config = {"protected_namespaces": ()}


def _load_from_env() -> dict:
    load_dotenv()
    env_map = {
        "lm_studio_base_url": os.getenv("LM_STUDIO_BASE_URL"),
        "model_orch": os.getenv("MODEL_ORCH"),
        "oss_max_tokens": os.getenv("OSS_MAX_TOKENS"),
        "model_qwen8": os.getenv("MODEL_QWEN8"),
        "model_qwen4": os.getenv("MODEL_QWEN4"),
        "tavily_api_key": os.getenv("TAVILY_API_KEY"),
        "search_depth_mode": os.getenv("SEARCH_DEPTH_MODE"),
        "max_results_base": os.getenv("MAX_RESULTS_BASE"),
        "max_results_high": os.getenv("MAX_RESULTS_HIGH"),
        "extract_depth": os.getenv("EXTRACT_DEPTH"),
        "database_path": os.getenv("DATABASE_PATH"),
        "host": os.getenv("HOST"),
        "port": os.getenv("PORT"),
        "strict_mode": os.getenv("STRICT_MODE"),
        "reasoning_depth_default": os.getenv("REASONING_DEPTH_DEFAULT"),
        "upload_dir": os.getenv("UPLOAD_DIR"),
        "upload_max_mb": os.getenv("UPLOAD_MAX_MB"),
    }
    cleaned = {k: v for k, v in env_map.items() if v not in (None, "")}
    if "max_results_base" in cleaned:
        cleaned["max_results_base"] = int(cleaned["max_results_base"])
    if "max_results_high" in cleaned:
        cleaned["max_results_high"] = int(cleaned["max_results_high"])
    if "oss_max_tokens" in cleaned:
        cleaned["oss_max_tokens"] = int(cleaned["oss_max_tokens"])
    if "port" in cleaned:
        cleaned["port"] = int(cleaned["port"])
    if "strict_mode" in cleaned:
        cleaned["strict_mode"] = str(cleaned["strict_mode"]).lower() in ("1", "true", "yes", "on")
    if "upload_max_mb" in cleaned:
        cleaned["upload_max_mb"] = int(cleaned["upload_max_mb"])
    return cleaned


def _env_overrides_config() -> bool:
    return str(os.getenv(ENV_OVERRIDE_KEY, "")).strip().lower() in ENV_OVERRIDE_TRUE


def _apply_legacy_endpoint_overrides(
    merged: Dict[str, Any],
    file_data: Dict[str, Any],
    env_data: Dict[str, Any],
    allow_env_overrides: bool,
) -> None:
    """Backfill/override endpoint configs from legacy env vars when config.json still has defaults."""
    if not allow_env_overrides:
        return
    defaults = AppSettings()
    base_url_override = env_data.get("lm_studio_base_url")
    legacy_base_url = file_data.get("lm_studio_base_url", defaults.lm_studio_base_url)
    legacy_orch = file_data.get("model_orch", defaults.model_orch)
    legacy_qwen8 = file_data.get("model_qwen8", defaults.model_qwen8)
    legacy_qwen4 = file_data.get("model_qwen4", defaults.model_qwen4)

    def override_endpoint(key: str, env_model: Optional[str], legacy_model: str) -> None:
        endpoint = merged.get(key)
        if not isinstance(endpoint, dict):
            endpoint = {}
        updated = False
        if env_model and (not endpoint.get("model_id") or endpoint.get("model_id") == legacy_model):
            endpoint["model_id"] = env_model
            updated = True
        if base_url_override and (
            not endpoint.get("base_url") or endpoint.get("base_url") == legacy_base_url
        ):
            endpoint["base_url"] = base_url_override
            updated = True
        if updated:
            merged[key] = endpoint

    override_endpoint("orch_endpoint", env_data.get("model_orch"), legacy_orch)
    for key in (
        "worker_a_endpoint",
        "worker_b_endpoint",
        "worker_c_endpoint",
        "fast_endpoint",
        "deep_planner_endpoint",
        "verifier_endpoint",
    ):
        override_endpoint(key, env_data.get("model_qwen8"), legacy_qwen8)
    for key in ("router_endpoint", "summarizer_endpoint", "deep_orchestrator_endpoint"):
        override_endpoint(key, env_data.get("model_qwen4"), legacy_qwen4)


def load_settings(config_path: Optional[Path] = None) -> AppSettings:
    env_data = _load_from_env()
    path = config_path or CONFIG_PATH
    file_data: Dict[str, Any] = {}
    if path.exists():
        try:
            file_data = json.loads(path.read_text())
        except Exception:
            file_data = {}
    allow_env_overrides = _env_overrides_config()
    # Config wins by default; allow env overrides only when explicitly enabled.
    if allow_env_overrides:
        merged = {**file_data, **env_data}
    else:
        merged = {**env_data, **file_data}
    _apply_legacy_endpoint_overrides(merged, file_data, env_data, allow_env_overrides)
    # Backfill endpoint fields from legacy
    if "orch_endpoint" not in merged and "lm_studio_base_url" in merged and "model_orch" in merged:
        merged["orch_endpoint"] = {"base_url": merged["lm_studio_base_url"], "model_id": merged["model_orch"]}
    return AppSettings(**merged)


def save_settings(settings: AppSettings, config_path: Optional[Path] = None) -> None:
    path = config_path or CONFIG_PATH
    path.write_text(settings.model_dump_json(indent=2))
