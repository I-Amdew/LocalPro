import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from dotenv import load_dotenv

CONFIG_PATH = Path("config.json")
ENV_OVERRIDE_KEY = "LOCALPRO_ENV_OVERRIDES_CONFIG"
ENV_OVERRIDE_TRUE = {"1", "true", "yes", "on"}


class EndpointConfig(BaseModel):
    base_url: str
    model_id: str

    model_config = {"protected_namespaces": ()}


class LMStudioBackendConfig(BaseModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 1234
    use_cli: bool = True
    cli_path: Optional[str] = None
    default_ttl_s: int = 600

    model_config = {"protected_namespaces": ()}


class ModelCandidateConfig(BaseModel):
    mode: str = "auto"
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    prefer: list[str] = Field(default_factory=list)

    model_config = {"protected_namespaces": ()}


class AutoscalingConfig(BaseModel):
    enabled: bool = True
    global_max_instances: Optional[int] = None
    per_backend_max_instances: Dict[str, Optional[int]] = Field(default_factory=dict)
    min_instances: Dict[str, int] = Field(default_factory=lambda: {"executor": 1})
    max_concurrent_loads: int = 1
    headroom: Dict[str, Any] = Field(
        default_factory=lambda: {"vram_free_mb_min": None, "vram_headroom_pct": 10, "ram_headroom_pct": 10}
    )

    model_config = {"protected_namespaces": ()}


class RoutingConfig(BaseModel):
    objective: str = "balanced"
    tool_required_by_default: bool = True

    model_config = {"protected_namespaces": ()}


class ProfilingConfig(BaseModel):
    enabled: bool = True
    auto_profile: bool = True
    observe_on_use: bool = True
    enforce_headroom: bool = False
    pause_execution: bool = False
    startup_wait_s: int = 120
    repeats: int = 1
    sample_interval_ms: int = 250
    test_timeout_s: int = 120
    settle_timeout_s: int = 12
    max_output_tokens: Optional[int] = None
    max_concurrent_profiles: int = 1
    context_length: Optional[int] = None
    profile_ttl_s: int = 6 * 60 * 60

    model_config = {"protected_namespaces": ()}


class LegacyCompatConfig(BaseModel):
    read_old_worker_slots_as_preferences: bool = True

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
    plan_reasoning_mode_default: str = "auto"
    planning_mode_default: str = "auto"
    reasoning_level_default: int = 2
    ram_headroom_pct: float = 10.0
    vram_headroom_pct: float = 10.0
    max_concurrent_runs: Optional[int] = None
    per_model_class_limits: Dict[str, int] = Field(default_factory=dict)
    discovery_base_urls: list[str] = Field(default_factory=lambda: ["http://localhost:1234/v1"])
    upload_dir: str = "uploads"
    upload_max_mb: int = 15
    backends: Dict[str, LMStudioBackendConfig] = Field(
        default_factory=lambda: {"lmstudio": LMStudioBackendConfig()}
    )
    model_candidates: ModelCandidateConfig = Field(default_factory=ModelCandidateConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    legacy_compat: LegacyCompatConfig = Field(default_factory=LegacyCompatConfig)

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
        "lmstudio_host": os.getenv("LM_STUDIO_HOST"),
        "lmstudio_port": os.getenv("LM_STUDIO_PORT"),
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
        "plan_reasoning_mode_default": os.getenv("PLAN_REASONING_MODE_DEFAULT"),
        "planning_mode_default": os.getenv("PLANNING_MODE_DEFAULT"),
        "reasoning_level_default": os.getenv("REASONING_LEVEL_DEFAULT"),
        "ram_headroom_pct": os.getenv("RAM_HEADROOM_PCT"),
        "vram_headroom_pct": os.getenv("VRAM_HEADROOM_PCT"),
        "max_concurrent_runs": os.getenv("MAX_CONCURRENT_RUNS"),
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
    if "lmstudio_port" in cleaned:
        cleaned["lmstudio_port"] = int(cleaned["lmstudio_port"])
    if "strict_mode" in cleaned:
        cleaned["strict_mode"] = str(cleaned["strict_mode"]).lower() in ("1", "true", "yes", "on")
    if "upload_max_mb" in cleaned:
        cleaned["upload_max_mb"] = int(cleaned["upload_max_mb"])
    if "reasoning_level_default" in cleaned:
        cleaned["reasoning_level_default"] = int(cleaned["reasoning_level_default"])
    if "ram_headroom_pct" in cleaned:
        cleaned["ram_headroom_pct"] = float(cleaned["ram_headroom_pct"])
    if "vram_headroom_pct" in cleaned:
        cleaned["vram_headroom_pct"] = float(cleaned["vram_headroom_pct"])
    if "max_concurrent_runs" in cleaned:
        cleaned["max_concurrent_runs"] = int(cleaned["max_concurrent_runs"])
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


def _extract_legacy_model_ids(merged: Dict[str, Any]) -> List[str]:
    keys = [
        "model_orch",
        "model_qwen8",
        "model_qwen4",
        "orch_endpoint",
        "worker_a_endpoint",
        "worker_b_endpoint",
        "worker_c_endpoint",
        "fast_endpoint",
        "deep_planner_endpoint",
        "deep_orchestrator_endpoint",
        "router_endpoint",
        "summarizer_endpoint",
        "verifier_endpoint",
    ]
    models: List[str] = []
    for key in keys:
        value = merged.get(key)
        if isinstance(value, str):
            if value:
                models.append(value)
            continue
        if isinstance(value, dict):
            model_id = value.get("model_id") or value.get("model")
            if model_id:
                models.append(str(model_id))
    return sorted({m for m in models if m})


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
    if not merged.get("tavily_api_key") and env_data.get("tavily_api_key"):
        merged["tavily_api_key"] = env_data["tavily_api_key"]
    _apply_legacy_endpoint_overrides(merged, file_data, env_data, allow_env_overrides)
    # Backfill endpoint fields from legacy
    if "orch_endpoint" not in merged and "lm_studio_base_url" in merged and "model_orch" in merged:
        merged["orch_endpoint"] = {"base_url": merged["lm_studio_base_url"], "model_id": merged["model_orch"]}
    # Ensure backend config exists and is synced to legacy base URL.
    backends = merged.get("backends") or {}
    if "lmstudio" not in backends:
        backends["lmstudio"] = LMStudioBackendConfig()
    lmstudio_cfg = backends.get("lmstudio") or {}
    if isinstance(lmstudio_cfg, dict):
        if merged.get("lmstudio_host"):
            lmstudio_cfg["host"] = merged["lmstudio_host"]
        if merged.get("lmstudio_port"):
            lmstudio_cfg["port"] = merged["lmstudio_port"]
        base_url = merged.get("lm_studio_base_url")
        if base_url and ("host" not in lmstudio_cfg or "port" not in lmstudio_cfg):
            try:
                from urllib.parse import urlparse

                parsed = urlparse(base_url)
                if parsed.hostname:
                    lmstudio_cfg.setdefault("host", parsed.hostname)
                if parsed.port:
                    lmstudio_cfg.setdefault("port", parsed.port)
            except Exception:
                pass
    backends["lmstudio"] = lmstudio_cfg
    merged["backends"] = backends
    if "model_candidates" not in merged:
        merged["model_candidates"] = ModelCandidateConfig().model_dump()
    if "profiling" not in merged:
        merged["profiling"] = ProfilingConfig().model_dump()
    if "legacy_compat" not in merged:
        merged["legacy_compat"] = LegacyCompatConfig().model_dump()
    compat = merged.get("legacy_compat") or {}
    if compat.get("read_old_worker_slots_as_preferences", True):
        legacy_ids = _extract_legacy_model_ids(merged)
        model_candidates = merged.get("model_candidates") or {}
        prefer = model_candidates.get("prefer") or []
        merged_pref = sorted({*prefer, *legacy_ids})
        model_candidates["prefer"] = merged_pref
        merged["model_candidates"] = model_candidates
    return AppSettings(**merged)


def save_settings(settings: AppSettings, config_path: Optional[Path] = None) -> None:
    path = config_path or CONFIG_PATH
    path.write_text(settings.model_dump_json(indent=2))
