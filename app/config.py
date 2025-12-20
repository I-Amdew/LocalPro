import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


CONFIG_PATH = Path("config.json")


class AppSettings(BaseModel):
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL for the LM Studio OpenAI-compatible API.",
    )
    model_orch: str = Field(default="openai/gpt-oss-20b")
    model_qwen8: str = Field(default="qwen/qwen3-v1-8b")
    model_qwen4: str = Field(default="qwen/qwen-4b")
    tavily_api_key: Optional[str] = Field(default=None)
    search_depth_mode: str = Field(default="auto")
    max_results_base: int = Field(default=6)
    max_results_high: int = Field(default=10)
    extract_depth: str = Field(default="basic")
    database_path: str = Field(default="app_data.db")
    port: int = Field(default=8000)
    strict_mode: bool = Field(default=False)

    def to_safe_dict(self) -> dict:
        data = self.model_dump()
        if data.get("tavily_api_key"):
            data["tavily_api_key"] = "********"
        return data


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
    }
    cleaned = {k: v for k, v in env_map.items() if v not in (None, "")}
    # Cast numeric/bool fields
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
    file_data = {}
    if CONFIG_PATH.exists():
        try:
            file_data = json.loads(CONFIG_PATH.read_text())
        except Exception:
            file_data = {}
    merged = {**env_data, **file_data}
    return AppSettings(**merged)


def save_settings(settings: AppSettings) -> None:
    CONFIG_PATH.write_text(settings.model_dump_json(indent=2))

