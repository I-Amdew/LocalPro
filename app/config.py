import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


CONFIG_PATH = Path("config.json")


class EndpointConfig(BaseModel):
    base_url: str
    model_id: str

    model_config = {"protected_namespaces": ()}


class AppSettings(BaseModel):
    # Legacy defaults (single endpoint)
    lm_studio_base_url: str = "http://localhost:1234/v1"
    model_orch: str = "openai/gpt-oss-20b"
    model_qwen8: str = "qwen/qwen3-v1-8b"
    model_qwen4: str = "qwen/qwen-4b"

    # Per-role endpoints/models
    orch_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://localhost:1234/v1", model_id="openai/gpt-oss-20b")
    )
    worker_a_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://localhost:1234/v1", model_id="qwen/qwen3-v1-8b")
    )
    worker_b_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://localhost:1234/v1", model_id="qwen/qwen3-v1-8b")
    )
    worker_c_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://localhost:1234/v1", model_id="qwen/qwen3-v1-8b")
    )
    router_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://localhost:1234/v1", model_id="qwen/qwen-4b")
    )
    summarizer_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://localhost:1234/v1", model_id="qwen/qwen-4b")
    )
    verifier_endpoint: EndpointConfig = Field(
        default_factory=lambda: EndpointConfig(base_url="http://localhost:1234/v1", model_id="qwen/qwen3-v1-8b")
    )

    tavily_api_key: Optional[str] = None
    search_depth_mode: str = Field(default="auto")
    max_results_base: int = 6
    max_results_high: int = 10
    extract_depth: str = "basic"
    database_path: str = "app_data.db"
    port: int = 8000
    strict_mode: bool = False
    reasoning_depth_default: str = "AUTO"
    discovery_base_urls: list[str] = Field(default_factory=lambda: ["http://localhost:1234/v1"])

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
        "model_qwen8": os.getenv("MODEL_QWEN8"),
        "model_qwen4": os.getenv("MODEL_QWEN4"),
        "tavily_api_key": os.getenv("TAVILY_API_KEY"),
        "search_depth_mode": os.getenv("SEARCH_DEPTH_MODE"),
        "max_results_base": os.getenv("MAX_RESULTS_BASE"),
        "max_results_high": os.getenv("MAX_RESULTS_HIGH"),
        "extract_depth": os.getenv("EXTRACT_DEPTH"),
        "database_path": os.getenv("DATABASE_PATH"),
        "port": os.getenv("PORT"),
        "strict_mode": os.getenv("STRICT_MODE"),
        "reasoning_depth_default": os.getenv("REASONING_DEPTH_DEFAULT"),
    }
    cleaned = {k: v for k, v in env_map.items() if v not in (None, "")}
    if "max_results_base" in cleaned:
        cleaned["max_results_base"] = int(cleaned["max_results_base"])
    if "max_results_high" in cleaned:
        cleaned["max_results_high"] = int(cleaned["max_results_high"])
    if "port" in cleaned:
        cleaned["port"] = int(cleaned["port"])
    if "strict_mode" in cleaned:
        cleaned["strict_mode"] = str(cleaned["strict_mode"]).lower() in ("1", "true", "yes", "on")
    return cleaned


def load_settings() -> AppSettings:
    env_data = _load_from_env()
    file_data: Dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            file_data = json.loads(CONFIG_PATH.read_text())
        except Exception:
            file_data = {}
    merged = {**env_data, **file_data}
    # Backfill endpoint fields from legacy
    if "orch_endpoint" not in merged and "lm_studio_base_url" in merged and "model_orch" in merged:
        merged["orch_endpoint"] = {"base_url": merged["lm_studio_base_url"], "model_id": merged["model_orch"]}
    return AppSettings(**merged)


def save_settings(settings: AppSettings) -> None:
    CONFIG_PATH.write_text(settings.model_dump_json(indent=2))
