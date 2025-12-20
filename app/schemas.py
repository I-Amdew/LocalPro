from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ReasoningLevel = Literal["LOW", "MED", "HIGH", "ULTRA"]


class RouterDecision(BaseModel):
    needs_web: bool = False
    reasoning_level: ReasoningLevel = "MED"
    topic: Literal["general", "news", "finance", "science", "tech"] = "general"
    max_results: int = 6
    extract_depth: Literal["basic", "advanced"] = "basic"
    stop_conditions: Dict[str, Any] = Field(default_factory=dict)
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


class EventPayload(BaseModel):
    seq: int
    event_type: str
    payload: Dict[str, Any]
    created_at: str


class StartRunRequest(BaseModel):
    question: str
    reasoning_mode: Literal["auto", "manual"] = "auto"
    manual_level: ReasoningLevel = "MED"
    evidence_dump: bool = False
    search_depth_mode: Literal["auto", "basic", "advanced"] = "auto"
    max_results: Optional[int] = None
    strict_mode: bool = False

