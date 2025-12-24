from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


ReasoningLevel = Literal["LOW", "MED", "HIGH", "ULTRA"]


class RouterDecision(BaseModel):
    needs_web: bool = False
    reasoning_level: ReasoningLevel = "MED"
    topic: Literal["general", "news", "finance", "science", "tech"] = "general"
    max_results: int = 6
    extract_depth: Literal["basic", "advanced"] = "basic"
    tool_budget: Dict[str, int] = Field(default_factory=dict)
    stop_conditions: Dict[str, Any] = Field(default_factory=dict)
    expected_passes: int = 1
    notes: Optional[str] = None

    model_config = {"extra": "allow"}


class EvidenceSource(BaseModel):
    url: str
    title: Optional[str] = ""
    publisher: Optional[str] = ""
    date_published: Optional[str] = ""
    snippet: Optional[str] = ""
    extracted_text: Optional[str] = ""


class EvidencePack(BaseModel):
    lane: str
    queries: List[str] = Field(default_factory=list)
    sources: List[EvidenceSource] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    conflicts_found: bool = False

    model_config = {"extra": "allow"}


class VerifierReport(BaseModel):
    verdict: Literal["PASS", "NEEDS_REVISION"]
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    revised_answer: Optional[str] = None
    extra_queries: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}

class PlanStep(BaseModel):
    step_id: int
    name: str
    type: str
    depends_on: List[int] = Field(default_factory=list)
    agent_profile: str = "ResearchPrimary"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    on_fail: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class StepPlan(BaseModel):
    plan_id: str
    goal: str
    global_constraints: Dict[str, Any] = Field(default_factory=dict)
    steps: List[PlanStep]

    model_config = {"extra": "allow"}


class ControlCommand(BaseModel):
    control: Literal["CONTINUE", "BACKTRACK", "RERUN_STEP", "ADD_STEPS", "STOP"]
    to_step: Optional[int] = None
    step_id: Optional[int] = None
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    reason: Optional[str] = None
    new_constraints: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class Artifact(BaseModel):
    step_id: int
    key: str
    artifact_type: str
    content_text: Optional[str] = None
    content_json: Optional[dict] = None
    created_at: Optional[str] = None

    model_config = {"extra": "allow"}


class UploadRecord(BaseModel):
    id: int
    run_id: Optional[str] = None
    filename: str
    original_name: str
    mime: str
    size_bytes: int
    status: str = "received"
    summary_text: Optional[str] = ""
    created_at: Optional[str] = None

    model_config = {"extra": "allow"}


class EventPayload(BaseModel):
    seq: int
    event_type: str
    payload: Dict[str, Any]
    created_at: str


class StartRunRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    reasoning_mode: Literal["auto", "manual"] = "auto"
    manual_level: ReasoningLevel = "MED"
    evidence_dump: bool = False
    model_tier: Literal["fast", "deep", "pro", "auto"] = "pro"
    deep_mode: Literal["auto", "oss", "cluster"] = "auto"
    search_depth_mode: Literal["auto", "basic", "advanced"] = "auto"
    max_results: Optional[int] = None
    strict_mode: bool = False
    auto_memory: bool = True
    reasoning_auto: bool = True
    upload_ids: List[int] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return data
        tier_raw = str(data.get("model_tier", "") or "").lower()
        tier_map = {
            "localauto": "auto",
            "local-auto": "auto",
            "auto": "auto",
            "localfast": "fast",
            "local-fast": "fast",
            "localdeep": "deep",
            "local-deep": "deep",
            "localpro": "pro",
            "local-pro": "pro",
        }
        if tier_raw:
            data["model_tier"] = tier_map.get(tier_raw, tier_raw)

        mode_raw = str(data.get("reasoning_mode", "") or "").lower()
        if data.get("model_tier") == "auto":
            data["reasoning_mode"] = "auto"
        elif mode_raw:
            data["reasoning_mode"] = "manual" if mode_raw == "manual" else "auto"
        else:
            data["reasoning_mode"] = "auto"

        level_raw = data.get("manual_level")
        if level_raw is None:
            data.pop("manual_level", None)
        elif isinstance(level_raw, str):
            level_up = level_raw.upper()
            if level_up == "AUTO":
                data.pop("manual_level", None)
            else:
                data["manual_level"] = level_up

        if data.get("model_tier") == "auto" and not data.get("manual_level"):
            data["manual_level"] = "MED"

        deep_mode_raw = data.get("deep_mode")
        if isinstance(deep_mode_raw, str):
            data["deep_mode"] = deep_mode_raw.lower()

        return data

    model_config = {"protected_namespaces": ()}
