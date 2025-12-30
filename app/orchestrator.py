import asyncio
import ast
import base64
import csv
import io
import json
import math
import operator
import os
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

try:
    from PIL import Image
except Exception:
    Image = None

from . import agents
from .db import Database
from .llm import LMStudioClient, resolve_model_id
from .model_manager import ModelManager
from .schemas import (
    Artifact,
    ControlCommand,
    PlanStep,
    RouterDecision,
    StepPlan,
    VerifierReport,
)
from .plan_runner import run_plan_pipeline
from .tavily import TavilyClient
from .system_info import get_resource_snapshot


# Plan depth defaults (initial plan granularity).
REASONING_DEPTHS = {
    "LOW": {"max_steps": 6, "research_rounds": 1, "tool_budget": {"tavily_search": 4, "tavily_extract": 6}},
    "MED": {"max_steps": 10, "research_rounds": 2, "tool_budget": {"tavily_search": 8, "tavily_extract": 10}},
    "HIGH": {"max_steps": 14, "research_rounds": 3, "tool_budget": {"tavily_search": 12, "tavily_extract": 16}, "advanced": True},
    "ULTRA": {"max_steps": 20, "research_rounds": 3, "tool_budget": {"tavily_search": 18, "tavily_extract": 24}, "advanced": True, "strict_verify": True},
}

PLAN_GRANULARITY_LEVELS = {"LOW": 1, "MED": 2, "HIGH": 3, "ULTRA": 4}
PLAN_GRANULARITY_NAMES = {v: k for k, v in PLAN_GRANULARITY_LEVELS.items()}
EXHAUSTIVE_TERMS = (
    "all ",
    "every ",
    "each ",
    "entire",
    "exhaustive",
    "complete list",
    "full list",
    "list all",
    "list every",
    "enumerate",
    "comprehensive",
)
COMPLEXITY_TERMS = (
    "compare",
    "comparison",
    "vs ",
    "versus",
    "tradeoff",
    "trade-off",
    "pros and cons",
    "advantages",
    "disadvantages",
    "strategy",
    "roadmap",
    "plan",
    "architecture",
    "design",
    "implementation",
    "migration",
    "audit",
    "security",
    "threat model",
    "debug",
    "root cause",
    "postmortem",
    "optimize",
    "optimization",
    "benchmark",
    "performance",
    "requirements",
    "policy",
    "governance",
    "compliance",
    "analysis",
    "analyze",
    "synthesize",
    "recommend",
    "prioritize",
    "rank",
    "investigate",
)


def granularity_level_from_router(value: Optional[str], fallback: int = 2) -> int:
    if not value:
        return fallback
    return PLAN_GRANULARITY_LEVELS.get(str(value).upper(), fallback)


def _question_complexity(question: str) -> Dict[str, Any]:
    text = (question or "").strip()
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    word_count = len(words)
    sentence_count = len([s for s in re.split(r"[.!?]+", lowered) if s.strip()])
    bullet_count = len(re.findall(r"(?m)^\s*(?:[-*]|\d+\.)\s+", text))
    multi_part = sentence_count > 1 or lowered.count("?") > 1 or bullet_count >= 2
    multi_part = multi_part or lowered.count(" and ") >= 2 or lowered.count(";") >= 1
    complex_hit = any(term in lowered for term in COMPLEXITY_TERMS)
    exhaustive_hit = any(term in lowered for term in EXHAUSTIVE_TERMS)
    return {
        "word_count": word_count,
        "multi_part": multi_part,
        "complex_hit": complex_hit,
        "exhaustive_hit": exhaustive_hit,
    }


def auto_reasoning_level(question: str, decision: RouterDecision) -> Tuple[str, bool]:
    if looks_like_math_expression(question):
        return "LOW", True
    info = _question_complexity(question)
    if info["exhaustive_hit"]:
        return "ULTRA", True
    needs_web = bool(decision.needs_web or needs_freshness(question) or guess_needs_web(question))
    score = 2
    if info["word_count"] <= 10 and not needs_web and not info["complex_hit"] and not info["multi_part"]:
        score = 1
    elif info["complex_hit"] or info["multi_part"] or info["word_count"] >= 28:
        score = 3
    if needs_web and score < 2:
        score = 2
    return PLAN_GRANULARITY_NAMES.get(max(1, min(score, 4)), "MED"), False


def choose_auto_reasoning_level(question: str, decision: RouterDecision) -> str:
    heuristic_level, force = auto_reasoning_level(question, decision)
    if force:
        return heuristic_level
    router_level = decision.reasoning_level or heuristic_level
    router_score = PLAN_GRANULARITY_LEVELS.get(str(router_level).upper(), 2)
    heuristic_score = PLAN_GRANULARITY_LEVELS.get(str(heuristic_level).upper(), router_score)
    if router_score < heuristic_score:
        return heuristic_level
    if router_score - heuristic_score >= 2:
        return heuristic_level
    return router_level


PLANNING_MODE_SYSTEM = """
SYSTEM (PLANNING MODE SELECTOR)
You decide planning_mode and plan_reasoning_mode.
- planning_mode controls whether to scaffold + expand into many steps.
- plan_reasoning_mode controls exhaustive interpretation (all/every/each must be enumerated).
Return JSON only: {"planning_mode":"normal|extensive","plan_reasoning_mode":"normal|extensive"}.
Use "extensive" when the question demands exhaustive coverage or when plan granularity is high.
""".strip()


def _heuristic_planning_modes(question: str, reasoning_level: int) -> Dict[str, str]:
    text = (question or "").lower()
    exhaustive = any(term in text for term in EXHAUSTIVE_TERMS) or reasoning_level >= 4
    mode = "extensive" if exhaustive else "normal"
    return {"planning_mode": mode, "plan_reasoning_mode": mode}


async def decide_planning_modes(
    lm_client: LMStudioClient,
    planner_endpoint: Optional[Dict[str, str]],
    question: str,
    reasoning_level: int,
    run_state: Optional["RunState"] = None,
) -> Dict[str, str]:
    fallback = _heuristic_planning_modes(question, reasoning_level)
    if not (run_state and run_state.model_manager) and (
        not planner_endpoint or not planner_endpoint.get("model") or not planner_endpoint.get("base_url")
    ):
        return fallback
    prompt = (
        f"Question: {question}\n"
        f"Plan granularity level: {reasoning_level}\n"
        "Return JSON only."
    )
    try:
        if run_state and run_state.model_manager:
            content = await run_worker(
                lm_client,
                "Planner",
                {},
                prompt,
                temperature=0.1,
                max_tokens=120,
                run_state=run_state,
                model_manager=run_state.model_manager,
                system_prompt_override=PLANNING_MODE_SYSTEM,
                context="plan_mode",
            )
        else:
            resp = await lm_client.chat_completion(
                model=planner_endpoint["model"],
                messages=[{"role": "system", "content": PLANNING_MODE_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=120,
                base_url=planner_endpoint["base_url"],
                run_state=run_state,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(
            content,
            lm_client,
            planner_endpoint["model"] if planner_endpoint else "",
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        if not isinstance(parsed, dict):
            return fallback
        planning_mode = str(parsed.get("planning_mode") or "").strip().lower()
        plan_reasoning_mode = str(parsed.get("plan_reasoning_mode") or "").strip().lower()
        if planning_mode not in ("normal", "extensive"):
            planning_mode = fallback["planning_mode"]
        if plan_reasoning_mode not in ("normal", "extensive"):
            plan_reasoning_mode = fallback["plan_reasoning_mode"]
        return {"planning_mode": planning_mode, "plan_reasoning_mode": plan_reasoning_mode}
    except Exception:
        return fallback


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# No fixed minimum; parallelism is driven by live resource checks.
MIN_PARALLEL_SLOTS = 1

# Cache models that LM Studio reports as unloaded or missing.
UNAVAILABLE_MODELS: Set[Tuple[str, str]] = set()


@dataclass
class RunState:
    can_chat: bool = True
    can_web: bool = False
    chat_error: Optional[str] = None
    web_error: Optional[str] = None
    freshness_required: bool = False
    question: str = ""
    work_log_flags: Set[str] = field(default_factory=set)
    narration_recent: Deque[str] = field(default_factory=lambda: deque(maxlen=6))
    narration_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    dev_trace_cb: Optional[Callable[[str, Optional[dict]], None]] = None
    model_manager: Optional[ModelManager] = None

    def mark_chat_unavailable(self, reason: str) -> None:
        self.can_chat = False
        self.chat_error = reason

    def add_dev_trace(self, message: str, detail: Optional[dict] = None) -> None:
        if self.dev_trace_cb:
            self.dev_trace_cb(message, detail)


@dataclass
class AllocationDecision:
    start_ids: List[int]
    queue_ids: List[int] = field(default_factory=list)
    target_slots: Optional[int] = None
    note: str = ""
    used_executor: bool = False


async def emit_work_log(
    bus: "EventBus",
    run_id: str,
    text: str,
    tone: str = "info",
    urls: Optional[List[str]] = None,
) -> None:
    payload = {"text": text, "tone": tone}
    if urls:
        payload["urls"] = urls
    await bus.emit(run_id, "work_log", payload)


async def maybe_emit_work_log(
    run_state: RunState,
    bus: "EventBus",
    run_id: str,
    key: str,
    text: str,
    tone: str = "info",
    urls: Optional[List[str]] = None,
) -> None:
    if key in run_state.work_log_flags:
        return
    run_state.work_log_flags.add(key)
    await emit_work_log(bus, run_id, text, tone=tone, urls=urls)


def _clip_narration_text(value: Any, max_len: int = 120) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if not text:
        return ""
    if len(text) > max_len:
        if max_len <= 3:
            return text[:max_len]
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _summarize_tool_request(detail: Optional[dict]) -> Dict[str, str]:
    if not isinstance(detail, dict):
        return {}
    requests = detail.get("requests")
    if not isinstance(requests, list) or not requests:
        return {}
    req = requests[0] if isinstance(requests[0], dict) else {}
    tool = str(req.get("tool") or req.get("type") or req.get("name") or "").strip()
    summary: Dict[str, str] = {}
    if tool:
        summary["tool"] = tool
    expr = req.get("expr") or req.get("expression") or req.get("input")
    if expr:
        summary["expr"] = _clip_narration_text(expr, 60)
    path = req.get("path") or req.get("file") or req.get("filename")
    if path:
        summary["path"] = _clip_narration_text(path, 80)
    code = req.get("code") or req.get("source")
    if code:
        summary["code"] = _clip_narration_text(code, 60)
    return summary


def _summarize_tool_result(detail: Optional[dict]) -> Dict[str, str]:
    if not isinstance(detail, dict):
        return {}
    results = detail.get("results")
    if not isinstance(results, list) or not results:
        return {}
    res = results[0] if isinstance(results[0], dict) else {}
    tool = str(res.get("tool") or res.get("type") or res.get("name") or "").strip()
    summary: Dict[str, str] = {}
    if tool:
        summary["tool"] = tool
    if "result" in res:
        result_val = res.get("result")
        if isinstance(result_val, (dict, list)):
            result_text = json.dumps(result_val, ensure_ascii=True)
        else:
            result_text = str(result_val)
        summary["result"] = _clip_narration_text(result_text, 80)
    return summary


def _summarize_step_detail(detail: Optional[dict]) -> Dict[str, Any]:
    if not isinstance(detail, dict):
        return {}
    step_id = detail.get("step_id") if detail.get("step_id") is not None else detail.get("step")
    return {
        "id": step_id,
        "name": _clip_narration_text(detail.get("name"), 60),
        "type": detail.get("type"),
        "agent_profile": detail.get("agent_profile"),
    }


def _trim_url_list(urls: Any, limit: int = 2) -> List[str]:
    if not isinstance(urls, list):
        return []
    trimmed: List[str] = []
    for url in urls:
        if not url:
            continue
        trimmed.append(_clip_narration_text(url, 120))
        if len(trimmed) >= limit:
            break
    return trimmed


def _build_narration_context(question: str, event_type: str, detail: Optional[dict]) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "question": _clip_narration_text(question, 140),
        "event": event_type,
    }
    if not isinstance(detail, dict):
        return context
    if event_type in ("step_started", "step_completed", "step_error"):
        context["step"] = _summarize_step_detail(detail)
        message = detail.get("message") or detail.get("error")
        if message:
            context["message"] = _clip_narration_text(message, 120)
        return context
    if event_type == "tavily_search":
        query = detail.get("query")
        if query:
            context["query"] = _clip_narration_text(query, 120)
        if detail.get("mode"):
            context["mode"] = detail.get("mode")
        return context
    if event_type == "search_skipped":
        query = detail.get("query")
        if query:
            context["query"] = _clip_narration_text(query, 120)
        reason = detail.get("reason")
        if reason:
            context["reason"] = _clip_narration_text(reason, 60)
        return context
    if event_type == "tavily_extract":
        urls = _trim_url_list(detail.get("urls"))
        if urls:
            context["urls"] = urls
        return context
    if event_type == "tool_request":
        tool = _summarize_tool_request(detail)
        if tool:
            context["tool_request"] = tool
        return context
    if event_type == "tool_result":
        tool = _summarize_tool_result(detail)
        if tool:
            context["tool_result"] = tool
        return context
    if event_type == "router_decision":
        context["reasoning_level"] = detail.get("reasoning_level")
        context["needs_web"] = detail.get("needs_web")
        context["model_tier"] = detail.get("model_tier")
        return context
    if event_type == "plan_created":
        context["steps"] = detail.get("steps") or detail.get("expected_total_steps")
        context["expected_passes"] = detail.get("expected_passes")
        return context
    if event_type == "plan_updated":
        context["steps"] = detail.get("steps") or detail.get("expected_total_steps")
        context["expected_passes"] = detail.get("expected_passes")
        return context
    if event_type == "allocator_decision":
        context["start_ids"] = detail.get("start_ids")
        context["target_slots"] = detail.get("target_slots")
        return context
    if event_type == "source_found":
        context["title"] = _clip_narration_text(detail.get("title"), 120)
        context["publisher"] = _clip_narration_text(detail.get("publisher"), 80)
        context["url"] = _clip_narration_text(detail.get("url"), 120)
        context["date_published"] = _clip_narration_text(detail.get("date_published"), 40)
        return context
    if event_type == "claim_found":
        context["claim"] = _clip_narration_text(detail.get("claim"), 140)
        urls = _trim_url_list(detail.get("urls"))
        if urls:
            context["urls"] = urls
        return context
    if event_type == "upload_processed":
        context["upload"] = _clip_narration_text(detail.get("name"), 80)
        return context
    if event_type == "upload_failed":
        context["upload"] = _clip_narration_text(detail.get("name"), 80)
        context["error"] = _clip_narration_text(detail.get("error"), 120)
        return context
    if event_type == "loop_iteration":
        context["iteration"] = detail.get("iteration")
        return context
    simple_detail: Dict[str, Any] = {}
    for key, value in detail.items():
        if isinstance(value, (str, int, float, bool)):
            simple_detail[key] = _clip_narration_text(value, 80)
    if simple_detail:
        context["detail"] = simple_detail
    return context


def _clean_narration_line(text: Any) -> str:
    line = _clip_narration_text(text, 200)
    if not line:
        return ""
    line = line.splitlines()[0].strip()
    if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
        line = line[1:-1].strip()
    for prefix in ("4B:", "Executor:", "Narrator:", "Narration:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix):].strip()
            break
    return _clip_narration_text(line, 160)


def _narration_endpoint(model_map: Optional[Dict[str, Dict[str, str]]]) -> Dict[str, str]:
    if not model_map:
        return {}
    return (
        model_map.get("executor")
        or model_map.get("summarizer")
        or model_map.get("router")
        or model_map.get("orch")
        or {}
    )


async def emit_narration(
    lm_client: LMStudioClient,
    model_map: Optional[Dict[str, Dict[str, str]]],
    run_state: Optional[RunState],
    bus: "EventBus",
    run_id: str,
    question: str,
    event_type: str,
    detail: Optional[dict] = None,
    tone: str = "info",
) -> None:
    if not run_state or not run_state.can_chat:
        return
    model_manager = run_state.model_manager
    endpoint = _narration_endpoint(model_map) if model_map else {}
    model = endpoint.get("model") if isinstance(endpoint, dict) else None
    context = _build_narration_context(question, event_type, detail)
    if run_state.narration_recent:
        context["recent_lines"] = list(run_state.narration_recent)
    prompt = (
        "Write one short, human line for a live UI. "
        "Sound like a real teammate narrating their own work. "
        "Focus on what you are doing or just learned, not internal mechanics. "
        "Use present tense, roughly 6-16 words. "
        "Avoid internal jargon (worker slots, step ids, allocators, tool names). "
        "If there is no user-facing update, return an empty string. "
        "Light label prefixes like \"Goal:\" or \"Plan:\" are OK when they fit. "
        "If a source or claim is provided, mention the source title or publisher. "
        "No quotes, no emojis, no chain-of-thought. "
        "Use the context fields if helpful. Output only the line.\n"
        f"Context: {json.dumps(context, ensure_ascii=True)}"
    )
    try:
        if model_manager:
            raw = await run_worker(
                lm_client,
                "Summarizer",
                model_map or {},
                prompt,
                temperature=0.4,
                max_tokens=60,
                run_state=run_state,
                model_manager=model_manager,
                system_prompt_override=agents.NARRATOR_SYSTEM,
                context="narration",
            )
            line = _clean_narration_line(raw)
        else:
            if not model:
                return
            resp = await lm_client.chat_completion(
                model=model,
                messages=[{"role": "system", "content": agents.NARRATOR_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=60,
                base_url=endpoint.get("base_url") or lm_client.base_url,
                run_state=run_state,
            )
            line = _clean_narration_line(resp["choices"][0]["message"]["content"])
        if not line:
            return
        async with run_state.narration_lock:
            if line in run_state.narration_recent:
                return
            run_state.narration_recent.append(line)
        payload = {"text": line, "tone": tone, "event": event_type}
        if isinstance(detail, dict):
            urls: List[str] = []
            seen: Set[str] = set()
            url = detail.get("url")
            if isinstance(url, str):
                cleaned = url.strip()
                if cleaned:
                    urls.append(cleaned)
                    seen.add(cleaned)
            url_list = detail.get("urls")
            if isinstance(url_list, list):
                for item in url_list:
                    if not isinstance(item, str):
                        continue
                    cleaned = item.strip()
                    if cleaned and cleaned not in seen:
                        urls.append(cleaned)
                        seen.add(cleaned)
                    if len(urls) >= 3:
                        break
            elif isinstance(url_list, str):
                cleaned = url_list.strip()
                if cleaned and cleaned not in seen:
                    urls.append(cleaned)
            if urls:
                payload["urls"] = urls
        await bus.emit(run_id, "narration", payload)
    except Exception:
        return


def queue_narration(
    lm_client: LMStudioClient,
    model_map: Optional[Dict[str, Dict[str, str]]],
    run_state: Optional[RunState],
    bus: "EventBus",
    run_id: str,
    question: str,
    event_type: str,
    detail: Optional[dict] = None,
    tone: str = "info",
) -> None:
    if not run_state or not run_state.can_chat:
        return
    if not model_map:
        return
    skip_events = {
        "allocator_decision",
        "model_selected",
        "model_error",
        "model_unavailable",
        "planner_verifier",
        "resource_budget",
        "role_map",
        "step_started",
        "step_completed",
        "tool_request",
        "tool_result",
        "worker_warmup",
        "strict_mode",
        "memory_retrieved",
    }
    if event_type in skip_events:
        return
    if not question and run_state.question:
        question = run_state.question
    if not question:
        question = ""
    asyncio.create_task(
        emit_narration(
            lm_client,
            model_map,
            run_state,
            bus,
            run_id,
            question,
            event_type,
            detail=detail,
            tone=tone,
        )
    )


def make_dev_trace_cb(bus: "EventBus", run_id: str) -> Callable[[str, Optional[dict]], None]:
    def _cb(message: str, detail: Optional[dict] = None) -> None:
        payload = {"message": message}
        if detail is not None:
            payload["detail"] = detail
        asyncio.create_task(bus.emit(run_id, "dev_trace", payload))

    return _cb


class EventBus:
    """In-memory fan-out for SSE plus persisted events."""

    def __init__(self, db: Database):
        self.db = db
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self.global_subscribers: List[asyncio.Queue] = []
        self.lock = asyncio.Lock()
        self.run_conversations: Dict[str, str] = {}

    def register_run(self, run_id: str, conversation_id: Optional[str]) -> None:
        if run_id and conversation_id:
            self.run_conversations[run_id] = conversation_id

    async def emit(self, run_id: str, event_type: str, payload: dict) -> dict:
        safe_payload = dict(payload or {})
        safe_payload.setdefault("run_id", run_id)
        if "conversation_id" not in safe_payload and run_id in self.run_conversations:
            safe_payload["conversation_id"] = self.run_conversations[run_id]
        stored = await self.db.add_event(run_id, event_type, safe_payload)
        async with self.lock:
            queues = list(self.subscribers.get(run_id, []))
            global_queues = list(self.global_subscribers)
        for q in queues:
            await q.put(stored)
        for q in global_queues:
            await q.put(stored)
        return stored

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self.lock:
            self.subscribers.setdefault(run_id, []).append(queue)
        return queue

    async def subscribe_global(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self.lock:
            self.global_subscribers.append(queue)
        return queue

    async def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        async with self.lock:
            queues = self.subscribers.get(run_id, [])
            if queue in queues:
                queues.remove(queue)
            if not queues:
                self.subscribers.pop(run_id, None)

    async def unsubscribe_global(self, queue: asyncio.Queue) -> None:
        async with self.lock:
            if queue in self.global_subscribers:
                self.global_subscribers.remove(queue)


async def safe_json_parse(
    raw: str,
    lm_client: LMStudioClient,
    fixer_model: str,
    run_state: Optional[RunState] = None,
    model_manager: Optional[ModelManager] = None,
) -> Optional[dict]:
    """Try to parse JSON, and fallback to the JSONRepair profile to fix."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    if model_manager is not None:
        try:
            resp = await model_manager.call(
                required_capabilities=["structured_output"],
                objective=model_manager.routing_objective,
                request={
                    "messages": [
                        {"role": "system", "content": agents.JSON_REPAIR_SYSTEM},
                        {"role": "user", "content": raw},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 400,
                    "use_responses": True,
                },
            )
            fixed = resp["choices"][0]["message"]["content"]
            return json.loads(fixed)
        except Exception:
            return None
    if not fixer_model:
        return None
    try:
        resp = await lm_client.chat_completion(
            model=fixer_model,
            messages=[
                {"role": "system", "content": agents.JSON_REPAIR_SYSTEM},
                {"role": "user", "content": raw},
            ],
            temperature=0.0,
            max_tokens=400,
            run_state=run_state,
        )
        fixed = resp["choices"][0]["message"]["content"]
        return json.loads(fixed)
    except Exception:
        return None


def pdf_excerpt(path: Path, max_chars: int = 4000) -> str:
    try:
        reader = PdfReader(str(path))
        parts: List[str] = []
        for page in reader.pages[:6]:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
            if sum(len(p) for p in parts) > max_chars:
                break
        return "\n".join(parts)[:max_chars]
    except Exception:
        return ""


def data_url_from_file(path: Path, mime: str) -> str:
    data = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _text_split(value: Any, sep: Optional[str] = None, maxsplit: int = -1) -> List[str]:
    return _coerce_text(value).split(sep, maxsplit)


def _text_splitlines(value: Any) -> List[str]:
    return _coerce_text(value).splitlines()


def _text_strip(value: Any, chars: Optional[str] = None) -> str:
    return _coerce_text(value).strip(chars)


def _text_lower(value: Any) -> str:
    return _coerce_text(value).lower()


def _text_upper(value: Any) -> str:
    return _coerce_text(value).upper()


def _text_replace(value: Any, old: Any, new: Any, count: int = -1) -> str:
    return _coerce_text(value).replace(str(old), str(new), count)


def _text_startswith(value: Any, prefix: Any) -> bool:
    return _coerce_text(value).startswith(str(prefix))


def _text_endswith(value: Any, suffix: Any) -> bool:
    return _coerce_text(value).endswith(str(suffix))


def _csv_rows(text: Any, delimiter: str = ",", max_rows: Optional[int] = 10000) -> List[List[str]]:
    data = _coerce_text(text)
    rows: List[List[str]] = []
    reader = csv.reader(io.StringIO(data), delimiter=delimiter)
    for row in reader:
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def _csv_dicts(text: Any, delimiter: str = ",", max_rows: Optional[int] = 10000) -> List[Dict[str, str]]:
    row_limit = None if max_rows is None else max_rows + 1
    rows = _csv_rows(text, delimiter=delimiter, max_rows=row_limit)
    if not rows:
        return []
    header = rows[0]
    output: List[Dict[str, str]] = []
    for row in rows[1:]:
        record = {}
        for idx, key in enumerate(header):
            record[str(key)] = row[idx] if idx < len(row) else ""
        output.append(record)
        if max_rows is not None and len(output) >= max_rows:
            break
    return output


def _safe_get(mapping: Any, key: Any, default: Any = None) -> Any:
    if mapping is None:
        return default
    if hasattr(mapping, "get"):
        try:
            return mapping.get(key, default)
        except Exception:
            return default
    try:
        return mapping[key]
    except Exception:
        return default


SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
SAFE_UNARY_OPS = {ast.UAdd: lambda v: v, ast.USub: lambda v: -v, ast.Not: lambda v: not v}
SAFE_NAMES: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "round": round,
    "len": len,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "range": range,
    "any": any,
    "all": all,
    "enumerate": enumerate,
    "zip": zip,
    **{name: getattr(math, name) for name in ("sqrt", "log", "log10", "sin", "cos", "tan", "exp", "ceil", "floor", "fabs")},
}
SAFE_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def safe_eval_expr(expr: str, names: Optional[Dict[str, Any]] = None) -> Any:
    """Evaluate a basic expression safely (no attribute access/imports)."""
    tree = ast.parse(expr, mode="eval")
    allowed_names = dict(SAFE_NAMES)
    if names:
        allowed_names.update(names)

    def _resolve_name(name: str, local_env: Dict[str, Any]) -> Any:
        if name in local_env:
            return local_env[name]
        if name in allowed_names:
            return allowed_names[name]
        raise ValueError("Name not allowed")

    def _assign_target(target: ast.AST, value: Any, local_env: Dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            local_env[target.id] = value
            return
        if isinstance(target, ast.Tuple):
            if not isinstance(value, (list, tuple)):
                raise ValueError("Tuple unpacking requires a list/tuple")
            if len(target.elts) != len(value):
                raise ValueError("Tuple unpack mismatch")
            for elt, item in zip(target.elts, value):
                _assign_target(elt, item, local_env)
            return
        raise ValueError("Unsupported comprehension target")

    def _eval_comprehension(node: ast.AST, local_env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.ListComp):
            results: List[Any] = []
            elt = node.elt
        elif isinstance(node, ast.GeneratorExp):
            results = []
            elt = node.elt
        elif isinstance(node, ast.SetComp):
            results = set()
            elt = node.elt
        elif isinstance(node, ast.DictComp):
            results = {}
            key_node = node.key
            value_node = node.value
            elt = None
        else:
            raise ValueError("Unsupported comprehension")

        def _walk(gen_index: int, env: Dict[str, Any]) -> None:
            if gen_index >= len(node.generators):
                if isinstance(results, list):
                    results.append(_eval(elt, env))
                elif isinstance(results, set):
                    results.add(_eval(elt, env))
                else:
                    results[_eval(key_node, env)] = _eval(value_node, env)
                return
            gen = node.generators[gen_index]
            if getattr(gen, "is_async", False):
                raise ValueError("Async comprehensions not allowed")
            iterable = _eval(gen.iter, env)
            for item in iterable:
                next_env = dict(env)
                _assign_target(gen.target, item, next_env)
                if all(_eval(cond, next_env) for cond in gen.ifs):
                    _walk(gen_index + 1, next_env)

        _walk(0, dict(local_env))
        return results

    def _eval(node: ast.AST, local_env: Optional[Dict[str, Any]] = None) -> Any:
        env = local_env or {}
        if isinstance(node, ast.Expression):
            return _eval(node.body, env)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool, str)) or node.value is None:
                return node.value
            raise ValueError("Unsupported literal")
        if isinstance(node, ast.Tuple):
            return tuple(_eval(elt, env) for elt in node.elts)
        if isinstance(node, ast.List):
            return [_eval(elt, env) for elt in node.elts]
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise ValueError("Dict unpacking not allowed")
            return {_eval(k, env): _eval(v, env) for k, v in zip(node.keys, node.values)}
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
            return _eval_comprehension(node, env)
        if isinstance(node, ast.Subscript):
            target = _eval(node.value, env)
            slice_node = node.slice
            if isinstance(slice_node, ast.Slice):
                lower = _eval(slice_node.lower, env) if slice_node.lower else None
                upper = _eval(slice_node.upper, env) if slice_node.upper else None
                step = _eval(slice_node.step, env) if slice_node.step else None
                return target[slice(lower, upper, step)]
            if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
                slice_node = slice_node.value
            if isinstance(slice_node, ast.Tuple):
                index = tuple(_eval(elt, env) for elt in slice_node.elts)
            else:
                index = _eval(slice_node, env)
            return target[index]
        if isinstance(node, ast.Compare):
            left = _eval(node.left, env)
            for op, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator, env)
                op_type = type(op)
                if op_type in SAFE_COMPARE_OPS:
                    if not SAFE_COMPARE_OPS[op_type](left, right):
                        return False
                elif isinstance(op, ast.In):
                    if left not in right:
                        return False
                elif isinstance(op, ast.NotIn):
                    if left in right:
                        return False
                elif isinstance(op, ast.Is):
                    if left is not right:
                        return False
                elif isinstance(op, ast.IsNot):
                    if left is right:
                        return False
                else:
                    raise ValueError("Comparison not allowed")
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value in node.values:
                    if not _eval(value, env):
                        return False
                return True
            if isinstance(node.op, ast.Or):
                for value in node.values:
                    if _eval(value, env):
                        return True
                return False
            raise ValueError("Boolean op not allowed")
        if isinstance(node, ast.IfExp):
            return _eval(node.body, env) if _eval(node.test, env) else _eval(node.orelse, env)
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BIN_OPS:
            return SAFE_BIN_OPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPS:
            return SAFE_UNARY_OPS[type(node.op)](_eval(node.operand, env))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            func = _resolve_name(func_name, env)
            args = [_eval(arg, env) for arg in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise ValueError("Keyword splat not allowed")
                kwargs[kw.arg] = _eval(kw.value, env)
            return func(*args, **kwargs)
        if isinstance(node, ast.Name):
            return _resolve_name(node.id, env)
        raise ValueError("Disallowed expression")

    return _eval(tree, {})


MATH_QUERY_PREFIXES = (
    "calculate",
    "compute",
    "solve",
    "evaluate",
    "simplify",
    "approximate",
    "approx",
)
MATH_TEXT_HINTS = {
    "what",
    "whats",
    "what's",
    "is",
    "the",
    "result",
    "of",
    "equals",
    "equal",
    "plus",
    "minus",
    "times",
    "divided",
    "by",
    "calculate",
    "compute",
    "solve",
    "evaluate",
    "simplify",
    "approximate",
    "approx",
}
MATH_FUNC_TOKENS = {"sin", "cos", "tan", "sqrt", "log", "log10", "exp", "ceil", "floor", "fabs", "pi", "e"}


def normalize_math_expression(text: str) -> str:
    base = strip_search_filler(text)
    if not base:
        base = " ".join((text or "").strip().split())
    base = base.strip(" .?!,;:")
    lower = base.lower()
    for prefix in MATH_QUERY_PREFIXES:
        if lower.startswith(prefix + " "):
            base = base[len(prefix):].strip()
            break
    if "^" in base and "**" not in base:
        base = base.replace("^", "**")
    if base:
        match = re.search(r"[0-9][0-9\\s\\+\\-\\*\\/\\.\\(\\)\\^]+", base)
        if match:
            base = match.group(0).strip()
    return base


def is_safe_math_expression(expr: str) -> bool:
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return False

    allowed_names = set(SAFE_NAMES.keys())

    def _check(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return _check(node.body)
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPS:
            return _check(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BIN_OPS:
            return _check(node.left) and _check(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in allowed_names:
                return False
            if any(kw.arg is None for kw in node.keywords):
                return False
            return all(_check(arg) for arg in node.args) and all(_check(kw.value) for kw in node.keywords)
        if isinstance(node, ast.Name):
            return node.id in allowed_names
        return False

    return _check(tree)


def looks_like_math_expression(text: str) -> bool:
    expr = normalize_math_expression(text)
    if not expr:
        return False
    lowered = (text or "").lower()
    tokens = re.findall(r"[a-z]+", lowered)
    if tokens:
        for token in tokens:
            if token in MATH_TEXT_HINTS or token in MATH_FUNC_TOKENS:
                continue
            return False
    return is_safe_math_expression(expr)


_CODING_BLOCK_RE = re.compile(r"(?m)^\s{4,}\S")
_CODING_EXT_RE = re.compile(
    r"\b[\w./-]+\.(py|js|ts|jsx|tsx|java|cpp|c|h|hpp|rs|go|rb|php|cs|swift|kt|sql|html|css|sh|ps1|bat|cmd|yml|yaml|json|toml|ini|md)\b",
    re.IGNORECASE,
)
_CODING_HINT_RE = re.compile(
    r"\b("
    r"code|coding|script|program|function|api|endpoint|refactor|debug|compile|build|runtime|syntax|"
    r"stack trace|traceback|exception|regex|regular expression|sql|query|database|schema|json|yaml|toml|ini|"
    r"python|javascript|typescript|java|rust|golang|ruby|php|swift|kotlin|bash|powershell|shell|"
    r"dockerfile|makefile|pip|npm|yarn|pnpm|gradle|maven|cargo|dotnet|node"
    r")\b",
    re.IGNORECASE,
)


def looks_like_coding_task(text: str) -> bool:
    if not text:
        return False
    if "```" in text:
        return True
    if _CODING_BLOCK_RE.search(text):
        return True
    if _CODING_EXT_RE.search(text):
        return True
    return bool(_CODING_HINT_RE.search(text))


def build_math_tool_request(question: str) -> Optional[dict]:
    if not looks_like_math_expression(question):
        return None
    expr = normalize_math_expression(question)
    return {"tool": "local_code", "code": expr}


TOOL_TEXT_MAX_CHARS = 4000
TOOL_BYTES_MAX = 200000
TOOL_LIST_MAX = 60
TOOL_IMAGE_MAX_SIZE = 1024
TOOL_IMAGE_MAX_LIMIT = 2048
TOOL_MODEL_DEFAULT_TOKENS = 400
TOOL_MODEL_MAX_TOKENS = 1200
TOOL_MODEL_MAX_CHARS = 4000
TOOL_MODEL_TIMEOUT_SECS = 45
RESOURCE_REFRESH_SECS = 3.0
ALLOCATOR_MAX_READY = 20
ALLOCATOR_MAX_RUNNING = 12
VALIDATION_PROMPT_MAX_CHARS = 1800
VALIDATION_SUMMARY_MAX_CHARS = 600


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def _normalize_tool_roots(upload_dir: Optional[Path]) -> List[Path]:
    base = Path(upload_dir) if upload_dir else Path("uploads")
    roots = [base, base / "snapshots"]
    resolved: List[Path] = []
    for root in roots:
        try:
            resolved_root = root.resolve()
        except Exception:
            resolved_root = root.absolute()
        if resolved_root not in resolved:
            resolved.append(resolved_root)
    return resolved


def _resolve_tool_path(path_value: str, roots: List[Path]) -> Path:
    if path_value is None:
        raise ValueError("Missing path")
    raw = str(path_value).strip()
    if not raw:
        raise ValueError("Missing path")
    path = Path(raw)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        for root in roots:
            candidates.append(root / path)
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand.absolute()
        if any(resolved == root or root in resolved.parents for root in roots):
            return resolved
    raise ValueError("Path not allowed")


def _tool_read_text(path_value: str, roots: List[Path], max_chars: int = TOOL_TEXT_MAX_CHARS) -> str:
    resolved = _resolve_tool_path(path_value, roots)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("File not found")
    data = resolved.read_text(encoding="utf-8", errors="ignore")
    if max_chars and len(data) > max_chars:
        return data[:max_chars]
    return data


def _tool_read_bytes(path_value: str, roots: List[Path], max_bytes: int = TOOL_BYTES_MAX) -> Dict[str, Any]:
    resolved = _resolve_tool_path(path_value, roots)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("File not found")
    data = resolved.read_bytes()
    if max_bytes and len(data) > max_bytes:
        data = data[:max_bytes]
    return {
        "path": _display_path(resolved),
        "bytes": len(data),
        "base64": base64.b64encode(data).decode("ascii"),
    }


def _tool_list_files(path_value: Optional[str], roots: List[Path], max_entries: int = TOOL_LIST_MAX) -> List[str]:
    if path_value:
        resolved = _resolve_tool_path(path_value, roots)
        if not resolved.exists():
            raise ValueError("Path not found")
        base = resolved
    else:
        return [_display_path(root) for root in roots]
    if base.is_file():
        return [base.name]
    entries: List[str] = []
    for entry in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        if len(entries) >= max_entries:
            break
        name = entry.name + ("/" if entry.is_dir() else "")
        entries.append(name)
    return entries


def _tool_http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
    max_bytes: int = TOOL_BYTES_MAX,
) -> Dict[str, Any]:
    if not url or not str(url).strip():
        raise ValueError("Missing url")
    parsed = httpx.URL(str(url))
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")

    def _coerce_str_dict(value: Optional[Dict[str, Any]], label: str) -> Optional[Dict[str, str]]:
        if value is None or value == "":
            return None
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a dict")
        return {str(k): str(v) for k, v in value.items()}

    params = _coerce_str_dict(params, "params")
    headers = _coerce_str_dict(headers, "headers")
    try:
        timeout_value = float(timeout)
    except Exception:
        timeout_value = 10.0
    if timeout_value <= 0:
        timeout_value = 10.0
    try:
        max_bytes_value = int(max_bytes)
    except Exception:
        max_bytes_value = TOOL_BYTES_MAX
    if max_bytes_value <= 0:
        max_bytes_value = TOOL_BYTES_MAX
    max_bytes_value = min(max_bytes_value, TOOL_BYTES_MAX)

    data = bytearray()
    truncated = False
    with httpx.Client(timeout=timeout_value, follow_redirects=True) as client:
        with client.stream("GET", str(parsed), params=params, headers=headers) as resp:
            status = resp.status_code
            content_type = resp.headers.get("content-type", "")
            for chunk in resp.iter_bytes():
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) >= max_bytes_value:
                    data = data[:max_bytes_value]
                    truncated = True
                    break

    text = data.decode("utf-8", errors="replace")
    json_data = None
    if "json" in content_type.lower() or text.lstrip().startswith(("{", "[")):
        try:
            json_data = json.loads(text)
        except Exception:
            json_data = None
    return {
        "url": str(parsed),
        "status": status,
        "content_type": content_type,
        "bytes": len(data),
        "truncated": truncated,
        "text": text,
        "json": json_data,
    }


def _require_image() -> None:
    if Image is None:
        raise ValueError("Pillow not installed")


def _normalize_image_format(fmt: str) -> str:
    if not fmt:
        return "PNG"
    cleaned = str(fmt).strip().upper()
    if cleaned == "JPG":
        cleaned = "JPEG"
    if cleaned not in ("PNG", "JPEG", "WEBP"):
        return "PNG"
    return cleaned


def _clamp_image_size(value: Any) -> int:
    try:
        size = int(value)
    except Exception:
        size = TOOL_IMAGE_MAX_SIZE
    if size <= 0:
        size = TOOL_IMAGE_MAX_SIZE
    return min(size, TOOL_IMAGE_MAX_LIMIT)


def _image_to_data_url(img: "Image.Image", fmt: str) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{data}"


def _tool_image_info(path_value: str, roots: List[Path]) -> Dict[str, Any]:
    _require_image()
    resolved = _resolve_tool_path(path_value, roots)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("File not found")
    with Image.open(resolved) as img:
        return {
            "path": _display_path(resolved),
            "format": img.format,
            "mode": img.mode,
            "size": list(img.size),
        }


def _tool_image_load(
    path_value: str,
    roots: List[Path],
    max_size: int = TOOL_IMAGE_MAX_SIZE,
    format: str = "PNG",
) -> Dict[str, Any]:
    _require_image()
    resolved = _resolve_tool_path(path_value, roots)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("File not found")
    fmt = _normalize_image_format(format)
    max_size = _clamp_image_size(max_size)
    with Image.open(resolved) as img:
        img = img.copy()
        if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        if max_size and max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        data_url = _image_to_data_url(img, fmt)
        return {
            "path": _display_path(resolved),
            "format": fmt,
            "size": list(img.size),
            "data_url": data_url,
        }


def _tool_image_zoom(
    path_value: str,
    roots: List[Path],
    box: Optional[Any] = None,
    left: Optional[Any] = None,
    top: Optional[Any] = None,
    right: Optional[Any] = None,
    bottom: Optional[Any] = None,
    scale: float = 2.0,
    max_size: int = TOOL_IMAGE_MAX_SIZE,
    format: str = "PNG",
) -> Dict[str, Any]:
    _require_image()
    resolved = _resolve_tool_path(path_value, roots)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("File not found")
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Box must be [left, top, right, bottom]")
        left, top, right, bottom = box
    if None in (left, top, right, bottom):
        raise ValueError("Missing box coordinates")
    fmt = _normalize_image_format(format)
    max_size = _clamp_image_size(max_size)
    try:
        scale_value = float(scale)
    except Exception:
        scale_value = 1.0
    if scale_value <= 0:
        scale_value = 1.0
    crop_box = (int(left), int(top), int(right), int(bottom))
    with Image.open(resolved) as img:
        cropped = img.crop(crop_box)
        if scale_value != 1.0:
            new_w = max(1, int(cropped.size[0] * scale_value))
            new_h = max(1, int(cropped.size[1] * scale_value))
            cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
        if max_size and max(cropped.size) > max_size:
            cropped.thumbnail((max_size, max_size), Image.LANCZOS)
        if fmt == "JPEG" and cropped.mode in ("RGBA", "LA", "P"):
            cropped = cropped.convert("RGB")
        data_url = _image_to_data_url(cropped, fmt)
        return {
            "path": _display_path(resolved),
            "box": list(crop_box),
            "scale": scale_value,
            "format": fmt,
            "size": list(cropped.size),
            "data_url": data_url,
        }


def _tool_image_adjust(
    path_value: str,
    roots: List[Path],
    rotate: Optional[Any] = None,
    flip: Optional[Any] = None,
    flop: Optional[Any] = None,
    grayscale: Optional[Any] = None,
    brightness: Optional[Any] = None,
    contrast: Optional[Any] = None,
    color: Optional[Any] = None,
    sharpness: Optional[Any] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    max_size: int = TOOL_IMAGE_MAX_SIZE,
    format: str = "PNG",
) -> Dict[str, Any]:
    _require_image()
    resolved = _resolve_tool_path(path_value, roots)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("File not found")
    fmt = _normalize_image_format(format)
    max_size = _clamp_image_size(max_size)
    try:
        from PIL import ImageEnhance
    except Exception:
        raise ValueError("Pillow not installed")
    with Image.open(resolved) as img:
        img = img.copy()
        angle = _coerce_float(rotate)
        if angle is not None and angle != 0:
            img = img.rotate(angle, expand=True)
        if flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if flop:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        def _apply_enhance(enhancer: Any, value: Optional[Any]) -> None:
            nonlocal img
            factor = _coerce_float(value)
            if factor is None:
                return
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img = enhancer(img).enhance(factor)

        _apply_enhance(ImageEnhance.Brightness, brightness)
        _apply_enhance(ImageEnhance.Contrast, contrast)
        _apply_enhance(ImageEnhance.Color, color)
        _apply_enhance(ImageEnhance.Sharpness, sharpness)

        if grayscale:
            img = img.convert("L")

        width_value = _coerce_int(width)
        height_value = _coerce_int(height)
        if width_value or height_value:
            if width_value is None or width_value <= 0:
                width_value = None
            if height_value is None or height_value <= 0:
                height_value = None
            if width_value and height_value:
                img = img.resize((width_value, height_value), Image.LANCZOS)
            elif width_value:
                ratio = width_value / max(1, img.size[0])
                img = img.resize((width_value, max(1, int(img.size[1] * ratio))), Image.LANCZOS)
            elif height_value:
                ratio = height_value / max(1, img.size[1])
                img = img.resize((max(1, int(img.size[0] * ratio)), height_value), Image.LANCZOS)

        if max_size and max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        data_url = _image_to_data_url(img, fmt)
        return {
            "path": _display_path(resolved),
            "format": fmt,
            "size": list(img.size),
            "data_url": data_url,
        }


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _coerce_float_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    coerced: List[float] = []
    for item in values:
        num = _coerce_float(item)
        if num is None:
            continue
        coerced.append(num)
    return coerced


def _tool_plot_chart(req: Dict[str, Any]) -> Dict[str, Any]:
    _require_image()
    try:
        from PIL import ImageDraw, ImageFont
    except Exception:
        raise ValueError("Pillow not installed")
    chart_type = str(req.get("chart_type") or req.get("type") or "bar").strip().lower()
    width = _coerce_int(req.get("width") or 800) or 800
    height = _coerce_int(req.get("height") or 480) or 480
    width = max(320, min(width, TOOL_IMAGE_MAX_LIMIT))
    height = max(240, min(height, TOOL_IMAGE_MAX_LIMIT))
    fmt = _normalize_image_format(str(req.get("format") or "PNG"))
    title = str(req.get("title") or "").strip()
    labels = req.get("labels") or req.get("x") or []
    if not isinstance(labels, list):
        labels = []
    series_input = req.get("series")
    series: List[Tuple[str, List[float]]] = []
    if isinstance(series_input, list):
        for idx, item in enumerate(series_input):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or f"Series {idx + 1}")
            values = _coerce_float_list(item.get("values") or item.get("y") or [])
            if values:
                series.append((name, values))
    if not series:
        values = _coerce_float_list(req.get("values") or req.get("y") or [])
        if not values:
            raise ValueError("Missing values for chart")
        series = [("Series 1", values)]
    max_len = max(len(vals) for _, vals in series)
    if not labels:
        labels = [str(i + 1) for i in range(max_len)]
    labels = [str(lbl) for lbl in labels][:max_len]

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    margin_left = 60
    margin_right = 20
    margin_top = 20 + (18 if title else 0)
    margin_bottom = 60
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    if title:
        draw.text((margin_left, 10), title, fill=(20, 20, 20), font=font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(60, 60, 60), width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(60, 60, 60), width=1)

    palette = [
        (52, 102, 164),
        (219, 83, 23),
        (68, 144, 88),
        (141, 80, 141),
    ]
    max_value = max(max(vals) for _, vals in series if vals) or 1.0
    if chart_type == "line":
        for idx, (_, values) in enumerate(series):
            if not values:
                continue
            color = palette[idx % len(palette)]
            points: List[Tuple[int, int]] = []
            for i, val in enumerate(values[:max_len]):
                x = plot_left + int((plot_right - plot_left) * i / max(1, max_len - 1))
                y = plot_bottom - int((plot_bottom - plot_top) * (val / max_value))
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)
            for x, y in points:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline=color)
    else:
        group_width = (plot_right - plot_left) / max(1, max_len)
        bar_gap = max(2, int(group_width * 0.1))
        bar_width = (group_width - bar_gap) / max(1, len(series))
        for idx, (_, values) in enumerate(series):
            color = palette[idx % len(palette)]
            for i, val in enumerate(values[:max_len]):
                x0 = plot_left + int(group_width * i + bar_gap / 2 + idx * bar_width)
                x1 = x0 + int(bar_width)
                y1 = plot_bottom
                y0 = plot_bottom - int((plot_bottom - plot_top) * (val / max_value))
                draw.rectangle((x0, y0, x1, y1), fill=color, outline=color)

    for i, label in enumerate(labels):
        x = plot_left + int((plot_right - plot_left) * (i + 0.5) / max_len)
        y = plot_bottom + 8
        draw.text((x - 8, y), label[:8], fill=(60, 60, 60), font=font)

    data_url = _image_to_data_url(img, fmt)
    return {
        "chart_type": chart_type,
        "format": fmt,
        "size": [width, height],
        "labels": labels,
        "series": [{"name": name, "values": values} for name, values in series],
        "data_url": data_url,
    }


def _parse_box_spec(value: Any) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    if isinstance(value, dict):
        left = _coerce_int(value.get("left") or value.get("x"))
        top = _coerce_int(value.get("top") or value.get("y"))
        right = _coerce_int(value.get("right") or value.get("x2"))
        bottom = _coerce_int(value.get("bottom") or value.get("y2"))
        if None not in (left, top, right, bottom):
            return left, top, right, bottom
    if isinstance(value, (list, tuple)) and len(value) == 4:
        coords = [_coerce_int(v) for v in value]
        if any(v is None for v in coords):
            return None
        return coords[0], coords[1], coords[2], coords[3]
    if isinstance(value, str):
        raw = value.replace(",", " ")
        parts = [p for p in raw.split() if p]
        if len(parts) == 4:
            coords = [_coerce_int(p) for p in parts]
            if any(v is None for v in coords):
                return None
            return coords[0], coords[1], coords[2], coords[3]
    return None


def _parse_page_spec(value: Any) -> List[int]:
    pages: List[int] = []
    if value is None:
        return pages
    if isinstance(value, (list, tuple, set)):
        for item in value:
            pages.extend(_parse_page_spec(item))
        return pages
    if not isinstance(value, str):
        num = _coerce_int(value)
        if num is not None:
            return [num]
        return pages
    text = value.strip()
    if not text:
        return pages
    for part in text.replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = _coerce_int(start_s)
            end = _coerce_int(end_s)
            if start is None or end is None:
                continue
            step = 1 if end >= start else -1
            pages.extend(range(start, end + step, step))
            continue
        single = _coerce_int(token)
        if single is not None:
            pages.append(single)
    return pages


def _tool_pdf_scan(
    path_value: str,
    roots: List[Path],
    pages: Optional[Any] = None,
    page_start: Optional[Any] = None,
    page_end: Optional[Any] = None,
    max_chars: int = TOOL_TEXT_MAX_CHARS,
) -> Dict[str, Any]:
    resolved = _resolve_tool_path(path_value, roots)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("File not found")
    reader = PdfReader(str(resolved))
    total_pages = len(reader.pages)
    page_list = _parse_page_spec(pages)
    if not page_list and (page_start is not None or page_end is not None):
        start = _coerce_int(page_start) or 1
        end = _coerce_int(page_end) or start
        step = 1 if end >= start else -1
        page_list = list(range(start, end + step, step))
    if not page_list:
        page_list = list(range(1, min(total_pages, 6) + 1))
    normalized_pages: List[int] = []
    for page in page_list:
        page_num = _coerce_int(page)
        if page_num is None or page_num < 1 or page_num > total_pages:
            continue
        if page_num not in normalized_pages:
            normalized_pages.append(page_num)
    if not normalized_pages and total_pages:
        normalized_pages = [1]
    parts: List[str] = []
    char_count = 0
    for page_num in normalized_pages:
        try:
            text = reader.pages[page_num - 1].extract_text() or ""
        except Exception:
            text = ""
        if text:
            parts.append(text)
            char_count += len(text)
            if max_chars and char_count >= max_chars:
                break
    combined = "\n".join(parts)
    if max_chars and len(combined) > max_chars:
        combined = combined[:max_chars]
    return {
        "path": _display_path(resolved),
        "pages": normalized_pages,
        "text": combined,
    }


def build_exec_helpers(roots: List[Path]) -> Dict[str, Any]:
    def read_text(path: str, max_chars: int = TOOL_TEXT_MAX_CHARS) -> str:
        return _tool_read_text(path, roots, max_chars=max_chars)

    def read_bytes(path: str, max_bytes: int = TOOL_BYTES_MAX) -> Dict[str, Any]:
        return _tool_read_bytes(path, roots, max_bytes=max_bytes)

    def list_files(path: Optional[str] = None) -> List[str]:
        return _tool_list_files(path, roots)

    def image_info(path: str) -> Dict[str, Any]:
        return _tool_image_info(path, roots)

    def image_load(path: str, max_size: int = TOOL_IMAGE_MAX_SIZE, format: str = "PNG") -> Dict[str, Any]:
        return _tool_image_load(path, roots, max_size=max_size, format=format)

    def image_zoom(
        path: str,
        box: Optional[Any] = None,
        left: Optional[Any] = None,
        top: Optional[Any] = None,
        right: Optional[Any] = None,
        bottom: Optional[Any] = None,
        scale: float = 2.0,
        max_size: int = TOOL_IMAGE_MAX_SIZE,
        format: str = "PNG",
    ) -> Dict[str, Any]:
        return _tool_image_zoom(
            path,
            roots,
            box=box,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            scale=scale,
            max_size=max_size,
            format=format,
        )

    def image_adjust(
        path: str,
        rotate: Optional[Any] = None,
        flip: Optional[Any] = None,
        flop: Optional[Any] = None,
        grayscale: Optional[Any] = None,
        brightness: Optional[Any] = None,
        contrast: Optional[Any] = None,
        color: Optional[Any] = None,
        sharpness: Optional[Any] = None,
        width: Optional[Any] = None,
        height: Optional[Any] = None,
        max_size: int = TOOL_IMAGE_MAX_SIZE,
        format: str = "PNG",
    ) -> Dict[str, Any]:
        return _tool_image_adjust(
            path,
            roots,
            rotate=rotate,
            flip=flip,
            flop=flop,
            grayscale=grayscale,
            brightness=brightness,
            contrast=contrast,
            color=color,
            sharpness=sharpness,
            width=width,
            height=height,
            max_size=max_size,
            format=format,
        )

    def split(text: Any, sep: Optional[str] = None, maxsplit: int = -1) -> List[str]:
        return _text_split(text, sep=sep, maxsplit=maxsplit)

    def splitlines(text: Any) -> List[str]:
        return _text_splitlines(text)

    def strip(text: Any, chars: Optional[str] = None) -> str:
        return _text_strip(text, chars=chars)

    def lower(text: Any) -> str:
        return _text_lower(text)

    def upper(text: Any) -> str:
        return _text_upper(text)

    def replace(text: Any, old: Any, new: Any, count: int = -1) -> str:
        return _text_replace(text, old=old, new=new, count=count)

    def startswith(text: Any, prefix: Any) -> bool:
        return _text_startswith(text, prefix=prefix)

    def endswith(text: Any, suffix: Any) -> bool:
        return _text_endswith(text, suffix=suffix)

    def csv_rows(text: Any, delimiter: str = ",", max_rows: Optional[int] = 10000) -> List[List[str]]:
        return _csv_rows(text, delimiter=delimiter, max_rows=max_rows)

    def csv_dicts(text: Any, delimiter: str = ",", max_rows: Optional[int] = 10000) -> List[Dict[str, str]]:
        return _csv_dicts(text, delimiter=delimiter, max_rows=max_rows)

    def get(mapping: Any, key: Any, default: Any = None) -> Any:
        return _safe_get(mapping, key, default=default)

    def http_get_text(
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
        max_bytes: int = TOOL_BYTES_MAX,
    ) -> Dict[str, Any]:
        payload = _tool_http_get(url, params=params, headers=headers, timeout=timeout, max_bytes=max_bytes)
        return {
            "url": payload.get("url"),
            "status": payload.get("status"),
            "content_type": payload.get("content_type"),
            "bytes": payload.get("bytes"),
            "truncated": payload.get("truncated"),
            "text": payload.get("text"),
        }

    def http_get_json(
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
        max_bytes: int = TOOL_BYTES_MAX,
    ) -> Dict[str, Any]:
        payload = _tool_http_get(url, params=params, headers=headers, timeout=timeout, max_bytes=max_bytes)
        if payload.get("json") is None:
            raise ValueError("Response was not valid JSON")
        return {
            "url": payload.get("url"),
            "status": payload.get("status"),
            "content_type": payload.get("content_type"),
            "bytes": payload.get("bytes"),
            "truncated": payload.get("truncated"),
            "data": payload.get("json"),
        }

    return {
        "read_text": read_text,
        "read_bytes": read_bytes,
        "list_files": list_files,
        "image_info": image_info,
        "image_load": image_load,
        "image_zoom": image_zoom,
        "image_crop": image_zoom,
        "image_adjust": image_adjust,
        "image_transform": image_adjust,
        "split": split,
        "splitlines": splitlines,
        "strip": strip,
        "lower": lower,
        "upper": upper,
        "replace": replace,
        "startswith": startswith,
        "endswith": endswith,
        "csv_rows": csv_rows,
        "csv_dicts": csv_dicts,
        "get": get,
        "http_get_text": http_get_text,
        "http_get_json": http_get_json,
        "UPLOADS_DIR": _display_path(roots[0]) if roots else "uploads",
        "SNAPSHOTS_DIR": _display_path(roots[-1]) if roots else "uploads/snapshots",
    }


def _sanitize_tool_result(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_tool_result(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_tool_result(v) for v in value]
    return str(value)


PLOT_TOOL_NAMES = {"plot_chart", "plot_graph", "chart", "graph"}


def strip_data_urls(item: Any, allow_plot: bool = False) -> Any:
    if isinstance(item, dict):
        tool_name = item.get("tool") or item.get("type") or item.get("name")
        allow_here = allow_plot
        if tool_name and str(tool_name).lower() in PLOT_TOOL_NAMES:
            allow_here = True
        cleaned: Dict[str, Any] = {}
        for key, value in item.items():
            if key == "data_url":
                cleaned[key] = value if allow_here else "<omitted>"
            elif isinstance(value, (dict, list)):
                cleaned[key] = strip_data_urls(value, allow_plot=allow_here)
            else:
                cleaned[key] = value
        return cleaned
    if isinstance(item, list):
        return [strip_data_urls(v, allow_plot=allow_plot) for v in item]
    return item


def _normalize_profile_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


_MODEL_CALL_PROFILE_ALIASES_RAW = {
    "orchestrator": "Orchestrator",
    "orch": "Orchestrator",
    "executor": "Executor",
    "researchprimary": "ResearchPrimary",
    "research": "ResearchPrimary",
    "primary": "ResearchPrimary",
    "researchrecency": "ResearchRecency",
    "recency": "ResearchRecency",
    "researchadversarial": "ResearchAdversarial",
    "adversarial": "ResearchAdversarial",
    "evidencesynth": "EvidenceSynth",
    "evidence": "EvidenceSynth",
    "math": "Math",
    "critic": "Critic",
    "summarizer": "Summarizer",
    "writer": "Writer",
    "finalizer": "Finalizer",
    "jsonrepair": "JSONRepair",
    "verifier": "Verifier",
}
MODEL_CALL_PROFILE_ALIASES = {_normalize_profile_token(k): v for k, v in _MODEL_CALL_PROFILE_ALIASES_RAW.items()}
for _name in set(MODEL_CALL_PROFILE_ALIASES.values()):
    MODEL_CALL_PROFILE_ALIASES.setdefault(_normalize_profile_token(_name), _name)

MODEL_CALL_TOOL_NAMES = {
    "model_call",
    "call_model",
    "agent_call",
    "delegate_model",
    "delegate",
}

FINALIZE_TOOL_NAMES = {"finalize_answer"}


def _resolve_model_profile(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    token = _normalize_profile_token(raw)
    return MODEL_CALL_PROFILE_ALIASES.get(token)


def _trim_tool_text(text: Any, max_chars: int) -> Tuple[str, bool]:
    raw = "" if text is None else str(text)
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw, False
    if max_chars <= 3:
        return raw[:max_chars], True
    return raw[: max_chars - 3].rstrip() + "...", True


async def _tool_model_call(
    req: dict,
    lm_client: Optional[LMStudioClient],
    model_map: Optional[Dict[str, Dict[str, str]]],
    run_id: Optional[str] = None,
    bus: Optional["EventBus"] = None,
    step_id: Optional[int] = None,
    run_state: Optional[RunState] = None,
) -> Dict[str, Any]:
    if lm_client is None or model_map is None:
        raise ValueError("model_call requires lm_client and model_map")
    profile = _resolve_model_profile(
        req.get("profile")
        or req.get("agent_profile")
        or req.get("agent")
        or req.get("role")
        or req.get("model")
    )
    if not profile:
        raise ValueError("Missing or unknown profile")
    prompt = str(req.get("prompt") or req.get("message") or req.get("input") or req.get("content") or "").strip()
    if not prompt:
        raise ValueError("Missing prompt")
    temperature = _coerce_float(req.get("temperature") or req.get("temp"))
    if temperature is None:
        temperature = 0.2
    temperature = max(0.0, min(1.5, temperature))
    max_tokens = _coerce_int(req.get("max_tokens") or req.get("max_output_tokens") or req.get("tokens"))
    if not max_tokens or max_tokens <= 0:
        max_tokens = TOOL_MODEL_DEFAULT_TOKENS
    max_tokens = min(max_tokens, TOOL_MODEL_MAX_TOKENS)
    max_chars = _coerce_int(req.get("max_chars") or req.get("limit")) or TOOL_MODEL_MAX_CHARS
    output = await run_worker(
        lm_client,
        profile,
        model_map,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        run_id=run_id,
        bus=bus,
        step_id=step_id,
        context="tool_model_call",
        run_state=run_state,
        model_manager=run_state.model_manager if run_state else None,
    )
    trimmed_prompt, prompt_truncated = _trim_tool_text(prompt, TOOL_TEXT_MAX_CHARS)
    trimmed_output, output_truncated = _trim_tool_text(output, max_chars)
    payload: Dict[str, Any] = {
        "profile": profile,
        "prompt": trimmed_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "result": _sanitize_tool_result(trimmed_output),
        "status": "ok",
    }
    if prompt_truncated:
        payload["prompt_truncated"] = True
    if output_truncated:
        payload["truncated"] = True
    return payload


def _looks_like_tool_markup(text: str) -> bool:
    lowered = text.lower()
    if "<|channel|" in lowered or "<|message|" in lowered or "<|constrain|" in lowered:
        return True
    if "to=local_code" in lowered or "to=execute_code" in lowered or "to=code_eval" in lowered:
        return True
    return False


async def resolve_tool_requests(
    tool_requests: List[dict],
    upload_dir: Optional[Path] = None,
    *,
    db: Optional[Database] = None,
    conversation_id: Optional[str] = None,
    lm_client: Optional[LMStudioClient] = None,
    model_map: Optional[Dict[str, Dict[str, str]]] = None,
    run_id: Optional[str] = None,
    bus: Optional["EventBus"] = None,
    step_id: Optional[int] = None,
    run_state: Optional[RunState] = None,
) -> List[dict]:
    """Resolve tool requests locally or via model-to-model calls."""
    resolved: List[dict] = []
    model_tasks: List[Tuple[Dict[str, Any], asyncio.Task]] = []
    now_iso = utc_iso()
    tool_roots = _normalize_tool_roots(upload_dir)
    exec_helpers: Optional[Dict[str, Any]] = None
    for raw_req in tool_requests or []:
        if isinstance(raw_req, dict):
            req = raw_req
        elif isinstance(raw_req, str):
            req = {"tool": raw_req}
        else:
            req = {"tool": str(raw_req)}
        tool = str(req.get("tool") or req.get("type") or req.get("name") or "").lower()
        entry: Dict[str, Any] = {"tool": tool or req.get("tool") or req.get("type") or req.get("name")}
        if tool in MODEL_CALL_TOOL_NAMES:
            resolved.append(entry)
            if lm_client is None or model_map is None:
                entry["status"] = "error"
                entry["error"] = "model_call requires lm_client and model_map"
            else:
                task = asyncio.create_task(
                    asyncio.wait_for(
                        _tool_model_call(
                            req,
                            lm_client,
                            model_map,
                            run_id=run_id,
                            bus=bus,
                            step_id=step_id,
                            run_state=run_state,
                        ),
                        timeout=TOOL_MODEL_TIMEOUT_SECS,
                    )
                )
                model_tasks.append((entry, task))
            continue
        try:
            if tool in ("live_date", "time_now", "now", "date"):
                entry["result"] = now_iso
                entry["status"] = "ok"
            elif tool in ("calculator", "calc", "math"):
                expr = str(req.get("expr") or req.get("expression") or req.get("input") or "").strip()
                if not expr:
                    raise ValueError("Missing expression")
                entry["expr"] = expr
                entry["result"] = _sanitize_tool_result(safe_eval_expr(expr))
                entry["status"] = "ok"
            elif tool in ("code_eval", "code", "python"):
                code = str(req.get("code") or req.get("expr") or req.get("source") or "").strip()
                if not code:
                    raise ValueError("Missing code")
                entry["code"] = code
                entry["result"] = _sanitize_tool_result(safe_eval_expr(code))
                entry["status"] = "ok"
            elif tool in ("execute_code", "exec_code", "code_exec", "execute", "python_exec", "local_code"):
                code = str(req.get("code") or req.get("expr") or req.get("expression") or "").strip()
                path = str(req.get("path") or req.get("file") or "").strip()
                if path and not code:
                    code = _tool_read_text(path, tool_roots, max_chars=TOOL_TEXT_MAX_CHARS)
                    entry["path"] = path
                if not code:
                    raise ValueError("Missing code")
                if exec_helpers is None:
                    exec_helpers = build_exec_helpers(tool_roots)
                entry["code"] = code
                entry["result"] = _sanitize_tool_result(safe_eval_expr(code, names=exec_helpers))
                entry["status"] = "ok"
            elif tool in FINALIZE_TOOL_NAMES:
                approved = req.get("approved")
                if approved is not None and not bool(approved):
                    raise ValueError("Final answer not approved")
                final_text = req.get("final_text") or req.get("answer") or req.get("text") or req.get("content")
                if final_text is None:
                    raise ValueError("Missing final_text")
                final_text = str(final_text).strip()
                if not final_text:
                    raise ValueError("Empty final_text")
                if _looks_like_tool_markup(final_text):
                    raise ValueError("Final text looks like tool markup")
                entry["result"] = _sanitize_tool_result(final_text)
                entry["status"] = "ok"
            elif tool in ("memory_search", "memory_query", "memory_lookup", "memory_list", "memory_recall"):
                if db is None:
                    raise ValueError("memory_search requires db")
                if not conversation_id:
                    raise ValueError("memory_search requires conversation_id")
                query = str(req.get("query") or req.get("q") or req.get("text") or "").strip()
                limit = _coerce_int(req.get("limit") or req.get("top") or req.get("max_results")) or 10
                limit = max(1, min(50, limit))
                entry["query"] = query
                entry["limit"] = limit
                if query:
                    items = await db.search_memory(query, conversation_id=conversation_id, limit=limit)
                else:
                    items = await db.list_memory(conversation_id=conversation_id, limit=limit)
                entry["result"] = _sanitize_tool_result(items)
                entry["status"] = "ok"
            elif tool in ("memory_save", "memory_store", "memory_add", "memory_fact"):
                if db is None:
                    raise ValueError("memory_save requires db")
                if not conversation_id:
                    raise ValueError("memory_save requires conversation_id")
                raw_items = req.get("items") or req.get("facts") or req.get("memory_notes")
                normalized: List[Dict[str, Any]] = []

                def _push_item(content_val: Any, title_val: Any = None, tags_val: Any = None) -> None:
                    text_val = str(content_val or "").strip()
                    if not text_val:
                        return
                    title_text = str(title_val or "").strip() or text_val[:80]
                    tags_list: List[str] = []
                    if isinstance(tags_val, list):
                        tags_list = [str(t).strip() for t in tags_val if str(t).strip()]
                    elif isinstance(tags_val, str) and tags_val.strip():
                        tags_list = [t.strip() for t in tags_val.split(",") if t.strip()]
                    normalized.append({"title": title_text, "content": text_val, "tags": tags_list})

                if isinstance(raw_items, list):
                    for item in raw_items:
                        if isinstance(item, dict):
                            _push_item(
                                item.get("content") or item.get("text") or item.get("fact"),
                                item.get("title"),
                                item.get("tags"),
                            )
                        else:
                            _push_item(item)
                elif isinstance(raw_items, dict):
                    _push_item(
                        raw_items.get("content") or raw_items.get("text") or raw_items.get("fact"),
                        raw_items.get("title"),
                        raw_items.get("tags"),
                    )
                elif raw_items:
                    _push_item(raw_items)
                else:
                    _push_item(
                        req.get("content") or req.get("text") or req.get("fact"),
                        req.get("title"),
                        req.get("tags"),
                    )
                if not normalized:
                    raise ValueError("No memory items to save")
                saved_ids: List[int] = []
                for item in normalized[:10]:
                    content = str(item.get("content") or "").strip()
                    if not content:
                        continue
                    title = str(item.get("title") or "").strip() or content[:80]
                    if len(content) > 400:
                        content = content[:400].rstrip() + "..."
                    tags = [t for t in (item.get("tags") or []) if t]
                    if "fact" not in [t.lower() for t in tags]:
                        tags.insert(0, "fact")
                    mem_id = await db.add_memory_item(
                        conversation_id,
                        kind="fact",
                        title=title[:80],
                        content=content,
                        tags=tags,
                        pinned=False,
                        relevance_score=1.0,
                    )
                    if run_id:
                        await db.link_memory_to_run(run_id, mem_id, "tool")
                    saved_ids.append(mem_id)
                entry["result"] = _sanitize_tool_result({"saved": len(saved_ids), "ids": saved_ids})
                entry["status"] = "ok"
            elif tool in ("memory_delete", "memory_remove", "memory_forget"):
                if db is None:
                    raise ValueError("memory_delete requires db")
                if not conversation_id:
                    raise ValueError("memory_delete requires conversation_id")
                raw_ids = req.get("ids") or req.get("items") or req.get("id")
                ids: List[int] = []
                if isinstance(raw_ids, list):
                    for item in raw_ids:
                        try:
                            ids.append(int(item))
                        except Exception:
                            continue
                elif raw_ids is not None:
                    try:
                        ids.append(int(raw_ids))
                    except Exception:
                        ids = []
                if not ids:
                    raise ValueError("No memory ids provided")
                deleted_ids: List[int] = []
                for mem_id in ids:
                    deleted = await db.delete_memory_item_for_conversation(conversation_id, mem_id)
                    if deleted:
                        deleted_ids.append(mem_id)
                entry["result"] = _sanitize_tool_result({"deleted": len(deleted_ids), "ids": deleted_ids})
                entry["status"] = "ok"
            elif tool in ("read_text", "read_file", "file_read", "text_read"):
                path = str(req.get("path") or req.get("file") or req.get("filename") or "").strip()
                if not path:
                    raise ValueError("Missing path")
                max_chars = _coerce_int(req.get("max_chars") or req.get("limit")) or TOOL_TEXT_MAX_CHARS
                entry["path"] = path
                entry["result"] = _sanitize_tool_result(_tool_read_text(path, tool_roots, max_chars=max_chars))
                entry["status"] = "ok"
            elif tool in ("read_bytes", "file_bytes", "read_file_bytes"):
                path = str(req.get("path") or req.get("file") or req.get("filename") or "").strip()
                if not path:
                    raise ValueError("Missing path")
                max_bytes = _coerce_int(req.get("max_bytes") or req.get("limit")) or TOOL_BYTES_MAX
                entry["path"] = path
                entry["result"] = _sanitize_tool_result(_tool_read_bytes(path, tool_roots, max_bytes=max_bytes))
                entry["status"] = "ok"
            elif tool in ("list_files", "list_dir", "list_directory", "ls"):
                path = str(req.get("path") or req.get("dir") or "").strip() or None
                entry["path"] = path or ""
                entry["result"] = _sanitize_tool_result(_tool_list_files(path, tool_roots))
                entry["status"] = "ok"
            elif tool in ("image_info", "image_metadata"):
                path = str(req.get("path") or req.get("file") or req.get("image") or "").strip()
                if not path:
                    raise ValueError("Missing path")
                entry["path"] = path
                entry["result"] = _sanitize_tool_result(_tool_image_info(path, tool_roots))
                entry["status"] = "ok"
            elif tool in ("image_load", "image_open"):
                path = str(req.get("path") or req.get("file") or req.get("image") or "").strip()
                if not path:
                    raise ValueError("Missing path")
                max_size = req.get("max_size") or req.get("size")
                fmt = req.get("format") or req.get("fmt") or "PNG"
                entry["path"] = path
                entry["result"] = _sanitize_tool_result(
                    _tool_image_load(path, tool_roots, max_size=max_size, format=fmt)
                )
                entry["status"] = "ok"
            elif tool in ("image_zoom", "image_crop", "image_eval"):
                path = str(req.get("path") or req.get("file") or req.get("image") or "").strip()
                if not path:
                    raise ValueError("Missing path")
                box = _parse_box_spec(req.get("box") or req.get("crop") or req.get("bbox") or req.get("region"))
                left = _coerce_int(req.get("left") or req.get("x"))
                top = _coerce_int(req.get("top") or req.get("y"))
                right = _coerce_int(req.get("right") or req.get("x2"))
                bottom = _coerce_int(req.get("bottom") or req.get("y2"))
                width = _coerce_int(req.get("width") or req.get("w"))
                height = _coerce_int(req.get("height") or req.get("h"))
                if box is None and left is not None and top is not None:
                    if right is None and width is not None:
                        right = left + width
                    if bottom is None and height is not None:
                        bottom = top + height
                scale = req.get("scale") or req.get("zoom") or 2.0
                max_size = req.get("max_size") or req.get("size")
                fmt = req.get("format") or req.get("fmt") or "PNG"
                entry["path"] = path
                entry["result"] = _sanitize_tool_result(
                    _tool_image_zoom(
                        path,
                        tool_roots,
                        box=box,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                        scale=scale,
                        max_size=max_size,
                        format=fmt,
                    )
                )
                entry["status"] = "ok"
            elif tool in ("plot_chart", "plot_graph", "chart", "graph"):
                entry["result"] = _sanitize_tool_result(_tool_plot_chart(req))
                entry["status"] = "ok"
            elif tool in ("pdf_scan", "pdf_read", "pdf_inspect"):
                path = str(req.get("path") or req.get("file") or req.get("pdf") or "").strip()
                if not path:
                    raise ValueError("Missing path")
                pages = req.get("pages") or req.get("page")
                page_start = req.get("page_start") or req.get("start_page") or req.get("from_page")
                page_end = req.get("page_end") or req.get("end_page") or req.get("to_page")
                max_chars = _coerce_int(req.get("max_chars") or req.get("limit")) or TOOL_TEXT_MAX_CHARS
                entry["path"] = path
                entry["result"] = _sanitize_tool_result(
                    _tool_pdf_scan(
                        path,
                        tool_roots,
                        pages=pages,
                        page_start=page_start,
                        page_end=page_end,
                        max_chars=max_chars,
                    )
                )
                entry["status"] = "ok"
            else:
                entry["status"] = "unknown_tool"
                entry["result"] = ""
            resolved.append(entry)
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            resolved.append(entry)
    if model_tasks:
        results = await asyncio.gather(*(task for _, task in model_tasks), return_exceptions=True)
        for (entry, _task), result in zip(model_tasks, results):
            if isinstance(result, Exception):
                entry["status"] = "error"
                if isinstance(result, asyncio.TimeoutError):
                    entry["error"] = f"timeout after {TOOL_MODEL_TIMEOUT_SECS}s"
                else:
                    entry["error"] = str(result)
            else:
                entry.update(result)
    return resolved


def reasoning_to_search_depth(level: str, preferred: str, depth_profile: Optional[dict] = None) -> str:
    if preferred != "auto":
        return preferred
    if depth_profile and depth_profile.get("advanced"):
        return "advanced"
    if level in ("HIGH", "ULTRA"):
        return "advanced"
    return "basic"


def guess_needs_web(question: str) -> bool:
    """Lightweight heuristic to decide if web research is needed when the router is unsure."""
    if looks_like_math_expression(question):
        return False
    q = (question or "").lower()
    recency_tokens = RECENCY_HINTS + (
        "update",
        "press release",
        "announcement",
        "changelog",
        "release notes",
    )
    if any(token in q for token in recency_tokens):
        return True
    citation_tokens = ("source", "sources", "citation", "cite", "reference", "references", "link", "links")
    if any(token in q for token in citation_tokens):
        return True
    data_signals = (
        "percent",
        "percentage",
        "share",
        "rate",
        "price",
        "cost",
        "net worth",
        "worth",
        "market",
        "market cap",
        "revenue",
        "growth",
        "forecast",
        "population",
        "household",
        "median",
        "average",
        "top",
        "rank",
        "list",
        "survey",
        "report",
        "study",
        "statistic",
        "statistics",
        "benchmark",
    )
    if any(token in q for token in data_signals):
        return True
    return False


def needs_freshness(question: str) -> bool:
    """Detect explicit freshness/verification requests that need live sources."""
    q = (question or "").lower()
    tokens = (
        "verify",
        "verified",
        "confirm",
        "confirmed",
        "is it true",
        "did it happen",
        "did this happen",
        "happened",
        "latest",
        "today",
        "current",
        "as of",
        "right now",
        "breaking",
        "this week",
        "this month",
        "this year",
        "up to date",
        "up-to-date",
        "recent",
        "news",
    )
    return any(token in q for token in tokens)


SEARCH_FILLER_PREFIXES = (
    "please",
    "can you",
    "could you",
    "would you",
    "tell me",
    "show me",
    "get me",
    "give me",
    "find",
    "search for",
    "look up",
    "what is",
    "what are",
    "what's",
    "whats",
    "latest on",
    "update on",
    "updates on",
    "info on",
    "information on",
)


def strip_search_filler(text: str) -> str:
    base = " ".join((text or "").strip().split())
    base = base.strip(" .?!,;:")
    if not base:
        return ""
    lower = base.lower()
    for prefix in SEARCH_FILLER_PREFIXES:
        if lower.startswith(prefix + " "):
            base = base[len(prefix):].strip()
            lower = base.lower()
            break
    if lower.startswith("please "):
        base = base[7:].strip()
    return base


def split_query_text(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    if "\n" in raw or "\r" in raw:
        parts = [p.strip() for p in raw.replace("\r", "\n").split("\n")]
    else:
        parts = [p.strip() for p in raw.split(",")]
    cleaned: List[str] = []
    for item in parts:
        item = item.strip(" \t-0123456789.)(")
        if item:
            cleaned.append(item)
    return cleaned


def build_fallback_queries(
    question: str,
    prompt: str = "",
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> List[str]:
    base = strip_search_filler(question)
    if not base:
        base = strip_search_filler(prompt)
    if not base and topic == "news":
        base = "news"
    if not base:
        return []
    lower = base.lower()
    news_mode = topic == "news" or any(token in lower for token in ("news", "headline", "headlines", "breaking"))
    recency = any(token in lower for token in RECENCY_HINTS)
    if time_range in ("day", "week"):
        recency = True
    variants: List[str] = []
    if base:
        variants.append(base)
        if news_mode and "news" not in lower:
            variants.append(f"{base} news")
        if news_mode and "latest" not in lower and "current" not in lower:
            variants.append(f"{base} latest")
    if news_mode:
        variants.extend(
            [
                "latest news headlines",
                "breaking news today",
                "top news stories",
                "world news headlines",
            ]
        )
    else:
        if recency and base:
            variants.append(f"{base} latest")
            variants.append(f"{base} headlines")
            variants.append(f"{base} this week")
        if base:
            if "official" not in lower:
                variants.append(f"{base} official")
            if "data" not in lower and "statistics" not in lower:
                variants.append(f"{base} data")
            if "report" not in lower and "study" not in lower:
                variants.append(f"{base} report")
            if "site:" not in lower:
                variants.append(f"{base} site:.gov")
    queries: List[str] = []
    seen: Set[str] = set()
    for q in variants:
        q = q.strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(q)
    return queries[:6]


def normalize_research_payload(parsed: Any) -> Tuple[Dict[str, Any], bool]:
    """Coerce research outputs into a dict with list-based queries/tool_requests."""
    coerced = False
    if isinstance(parsed, dict):
        payload: Dict[str, Any] = dict(parsed)
    elif isinstance(parsed, list):
        payload = {"queries": parsed}
        coerced = True
    elif isinstance(parsed, str):
        payload = {"queries": split_query_text(parsed)}
        coerced = True
    else:
        return {"queries": [], "tool_requests": []}, True

    queries = payload.get("queries", [])
    if isinstance(queries, str):
        queries = split_query_text(queries)
        coerced = True
    elif not isinstance(queries, list):
        queries = []
        coerced = True
    normalized_queries: List[str] = []
    for item in queries:
        value = item
        if isinstance(item, dict):
            value = item.get("query") or item.get("text") or item.get("q")
            coerced = True
        if value is None:
            continue
        text = str(value).strip()
        if text:
            normalized_queries.append(text)
    payload["queries"] = normalized_queries

    tool_requests = payload.get("tool_requests", [])
    if not isinstance(tool_requests, list):
        tool_requests = []
        coerced = True
    payload["tool_requests"] = tool_requests

    if "time_range" in payload and payload["time_range"] is not None:
        payload["time_range"] = str(payload["time_range"]).strip()
    if "topic" in payload and payload["topic"] is not None:
        payload["topic"] = str(payload["topic"]).strip()

    return payload, coerced


async def resolve_model_map(
    model_map: Dict[str, Dict[str, str]],
    lm_client: LMStudioClient,
    run_state: Optional[RunState] = None,
) -> Dict[str, Dict[str, str]]:
    resolved_map: Dict[str, Dict[str, str]] = {}
    cached: Dict[str, List[str]] = {}
    for role, cfg in model_map.items():
        base_url = cfg.get("base_url")
        model = cfg.get("model")
        if not base_url or not model:
            resolved_map[role] = cfg
            continue
        if base_url not in cached:
            try:
                cached[base_url] = await lm_client.list_models_cached(base_url)
            except Exception as exc:
                if run_state:
                    run_state.add_dev_trace(
                        "Model list lookup failed",
                        {"base_url": base_url, "error": str(exc)},
                    )
                cached[base_url] = []
        resolved = resolve_model_id(model, cached.get(base_url, []))
        if resolved and resolved != model:
            resolved_map[role] = {**cfg, "model": resolved}
        else:
            resolved_map[role] = cfg
    return resolved_map


_MODEL_SIZE_HINT_RE = re.compile(r"(\d+)\s*[bB]\b")


def _model_size_hint(model_id: str) -> Optional[int]:
    if not model_id:
        return None
    match = _MODEL_SIZE_HINT_RE.search(model_id)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_coder_model(model_id: str) -> bool:
    return "coder" in (model_id or "").lower()


def _is_heavy_coder_model(model_id: str) -> bool:
    if not _is_coder_model(model_id):
        return False
    size = _model_size_hint(model_id)
    if size is not None:
        return size >= 20
    lowered = (model_id or "").lower()
    return any(token in lowered for token in ("30b", "32b", "33b", "34b"))


def _sort_models_by_size(models: List[str], prefer_large: bool = False) -> List[str]:
    def _key(mid: str) -> Tuple[int, int]:
        size = _model_size_hint(mid)
        if size is None:
            return (1, 0)
        return (0, size)

    ordered = sorted(models, key=_key, reverse=prefer_large)
    return ordered


def _role_size_preference(role: str) -> str:
    role_key = (role or "").lower()
    if role_key in {"orch", "deep_orch", "deep_planner"}:
        return "large"
    if role_key == "worker":
        return "small"
    if role_key in {"worker_b", "worker_c", "verifier"}:
        return "medium"
    if role_key in {"router", "summarizer", "executor", "fast"}:
        return "medium"
    return "medium"


def _should_override_role_model(role_pref: str, model_id: str) -> bool:
    if _is_heavy_coder_model(model_id) and role_pref in ("small", "medium"):
        return True
    size = _model_size_hint(model_id)
    if size is None:
        return True
    if role_pref == "small":
        return size >= 8
    if role_pref == "medium":
        return size >= 16
    if role_pref == "large":
        return size <= 8
    return False


async def apply_runtime_model_overrides(
    model_map: Dict[str, Dict[str, str]],
    lm_client: LMStudioClient,
    run_state: Optional[RunState] = None,
    run_id: Optional[str] = None,
    bus: Optional["EventBus"] = None,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    """Prefer models that respond to a minimal chat check and report role assignments."""
    role_configs = {
        role: cfg
        for role, cfg in model_map.items()
        if isinstance(cfg, dict) and cfg.get("base_url") and cfg.get("model")
    }
    by_base: Dict[str, List[str]] = {}
    for role, cfg in role_configs.items():
        base_url = str(cfg.get("base_url") or "")
        by_base.setdefault(base_url, []).append(role)

    check_cache: Dict[Tuple[str, str], Optional[bool]] = {}
    available_cache: Dict[str, List[str]] = {}
    prefer_non_coder = False
    if run_state and run_state.question:
        prefer_non_coder = not looks_like_coding_task(run_state.question)

    async def _check_model(base_url: str, model: str) -> bool:
        key = (base_url, model)
        cached = check_cache.get(key)
        if cached is not None:
            return cached
        ok = False
        try:
            ok, _ = await lm_client.check_chat(base_url=base_url, model=model, run_state=run_state)
        except Exception:
            ok = False
        check_cache[key] = ok
        return ok

    for base_url, roles in by_base.items():
        if base_url not in available_cache:
            try:
                available = await lm_client.list_models_cached(base_url)
            except Exception as exc:
                if run_state:
                    run_state.add_dev_trace("Model list lookup failed", {"base_url": base_url, "error": str(exc)})
                available = []
            available_cache[base_url] = [m for m in available if m and "embed" not in m.lower()]
        preferred: List[str] = []
        for role in roles:
            model = str(role_configs[role].get("model") or "")
            if model and model not in preferred:
                preferred.append(model)
        # Check each preferred model so per-role availability is accurate.
        for model in preferred:
            await _check_model(base_url, model)

    updated: Dict[str, Dict[str, str]] = {}
    role_report: Dict[str, Any] = {}
    for role, cfg in model_map.items():
        base_url = cfg.get("base_url")
        model = cfg.get("model")
        status = "unset"
        selected = model
        if base_url and model:
            ok = check_cache.get((base_url, model))
            if ok is None:
                ok = await _check_model(base_url, model)
            if ok is True:
                role_pref = _role_size_preference(role)
                if (_is_heavy_coder_model(model) and role_pref in ("small", "medium")) or (
                    prefer_non_coder and _is_coder_model(model)
                ):
                    ok = False
                else:
                    status = "ok"
            if ok is not True:
                available = available_cache.get(base_url, [])
                available = [m for m in available if (base_url, m) not in UNAVAILABLE_MODELS]
                if prefer_non_coder:
                    without_coder = [m for m in available if not _is_coder_model(m)]
                    if without_coder:
                        available = without_coder
                role_pref = _role_size_preference(role)
                if role_pref in ("small", "medium"):
                    without_heavy = [m for m in available if not _is_heavy_coder_model(m)]
                    if without_heavy:
                        available = without_heavy
                prefer_large = role_pref == "large"
                ordered = _sort_models_by_size(available, prefer_large=prefer_large)
                if role_pref == "small":
                    ordered = [m for m in ordered if (_model_size_hint(m) or 0) <= 6] or ordered
                elif role_pref == "large":
                    ordered = [m for m in ordered if (_model_size_hint(m) or 99) >= 16] or ordered
                if model in ordered and not _should_override_role_model(role_pref, model):
                    ordered = [model] + [m for m in ordered if m != model]
                for candidate in ordered:
                    if await _check_model(base_url, candidate):
                        selected = candidate
                        status = "fallback" if candidate != model else "ok"
                        break
                if status == "unset":
                    status = "unavailable" if ok is False else "unknown"
                if selected and selected != model:
                    cfg = {**cfg, "model": selected}
        updated[role] = cfg
        role_report[role] = {
            "model": updated[role].get("model"),
            "base_url": updated[role].get("base_url"),
            "status": status,
            "preferred": model,
        }

    if run_id and bus:
        await bus.emit(run_id, "role_map", {"roles": role_report})
    return updated, {"roles": role_report, "check_cache": check_cache}


WEB_PROBE_TIMEOUT_S = 5
TAVILY_TIMEOUT_S = 20


async def check_web_access(tavily: TavilyClient) -> Tuple[bool, Optional[str]]:
    """Return (can_web, error). Non-auth errors are reported but do not disable web."""
    if not tavily.enabled:
        return False, "missing_api_key"
    try:
        resp = await asyncio.wait_for(
            tavily.search(query="ping", search_depth="basic", max_results=1),
            timeout=WEB_PROBE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return False, "timeout"
    resp = coerce_tavily_response(resp)
    if resp.get("error"):
        status = resp.get("status_code")
        error_msg = format_tavily_error(resp)
        if status in (401, 403) or resp.get("error") == "missing_api_key":
            return False, error_msg
        return True, error_msg
    return True, None


RECENCY_HINTS = (
    "today",
    "current",
    "latest",
    "recent",
    "breaking",
    "news",
    "headline",
    "headlines",
    "this week",
    "this month",
    "this year",
)

ALLOWED_TOPICS = {"general", "news", "finance", "science", "tech"}

TIME_RANGE_ALIASES = {
    "today": "day",
    "day": "day",
    "24h": "day",
    "last_24_hours": "day",
    "week": "week",
    "7d": "week",
    "last_7_days": "week",
    "month": "month",
    "30d": "month",
    "last_30_days": "month",
    "year": "year",
    "12m": "year",
    "last_12_months": "year",
}


def normalize_time_range(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = str(value).strip().lower().replace(" ", "_")
    if cleaned in TIME_RANGE_ALIASES:
        return TIME_RANGE_ALIASES[cleaned]
    if "day" in cleaned or "24" in cleaned:
        return "day"
    if "week" in cleaned or "7" in cleaned:
        return "week"
    if "month" in cleaned or "30" in cleaned:
        return "month"
    if "year" in cleaned or "12" in cleaned or "annual" in cleaned:
        return "year"
    return None


def infer_time_range(question: str) -> Optional[str]:
    text = (question or "").lower()
    if any(token in text for token in ("today", "current", "latest", "breaking")):
        return "day"
    if "this week" in text or "past week" in text or "last week" in text:
        return "week"
    if "this month" in text or "past month" in text or "last month" in text:
        return "month"
    if "this year" in text or "past year" in text or "last year" in text or "annual" in text:
        return "year"
    if any(token in text for token in ("news", "headline", "headlines")):
        return "week"
    return None


def widen_time_range(value: Optional[str]) -> Optional[str]:
    order = ["day", "week", "month", "year"]
    if value not in order:
        return None
    idx = order.index(value)
    if idx + 1 < len(order):
        return order[idx + 1]
    return None


def compact_sources_for_synthesis(
    sources: List[dict], max_sources: int = 6, max_chars: int = 900
) -> List[dict]:
    compact: List[dict] = []
    for src in sources[:max_sources]:
        excerpt = (src.get("extracted_text") or src.get("snippet") or "").strip()
        if excerpt:
            excerpt = " ".join(excerpt.split())
        if max_chars and len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars] + "..."
        compact.append(
            {
                "url": src.get("url"),
                "title": src.get("title"),
                "publisher": src.get("publisher"),
                "date_published": src.get("date_published"),
                "excerpt": excerpt,
            }
        )
    return compact


EXPLORATORY_PHRASES = (
    "idea",
    "ideas",
    "brainstorm",
    "explore",
    "exploratory",
    "options",
    "approach",
    "strategy",
    "possibility",
    "creative",
    "scenario",
    "what if",
)


def is_exploratory_question(question: str, decision: RouterDecision) -> bool:
    """Detect open-ended prompts so LocalDeep can route them to the Mini Pro lane."""
    text = (question or "").strip().lower()
    if any(phrase in text for phrase in EXPLORATORY_PHRASES):
        return True
    if decision.reasoning_level == "LOW" and not decision.needs_web:
        budget = decision.tool_budget or {}
        if budget.get("tavily_search", 0) <= 2 and budget.get("tavily_extract", 0) <= 1:
            return True
    return False


def compute_progress_meta(step_plan: StepPlan, expected_passes: int) -> Dict[str, int]:
    base_steps = len(step_plan.steps)
    analysis_steps = sum(1 for s in step_plan.steps if s.type == "analysis")
    per_pass_rerun = max(base_steps - analysis_steps, 0)
    counted_passes = max(expected_passes, 1)
    total_steps = base_steps + max(0, counted_passes - 1) * per_pass_rerun
    return {
        "base_steps": base_steps,
        "analysis_steps": analysis_steps,
        "per_pass_rerun": per_pass_rerun,
        "counted_passes": counted_passes,
        "total_steps": total_steps,
    }


def response_guidance_text(question: str, reasoning_level: str, progress_meta: Dict[str, int]) -> str:
    total = progress_meta.get("total_steps", 0)
    passes = progress_meta.get("counted_passes", 1)
    q_len = len(question or "")
    if total <= 6 and q_len < 120:
        style = "Very concise (<=120 words) aimed directly at the ask."
    elif total <= 12:
        style = "Concise (<=200 words) with tight bullets plus a one-line takeaway."
    else:
        style = "Compact but complete (<=350 words) with short sections and crisp bullets."
    if passes > 1:
        style += f" Note progress and clarify if another pass ({passes}) is in flight or expected."
    if reasoning_level in ("ULTRA", "HIGH"):
        style += " Note any remaining risks."
    return style


def desired_parallelism(
    decision: RouterDecision,
    worker_budget: Dict[str, Any],
    strict_mode: bool = False,
) -> int:
    """Choose how many worker slots to actively fill based on available capacity."""
    max_parallel = int(worker_budget.get("max_parallel") or 1)
    if max_parallel <= 1:
        return 1
    return max_parallel


def compute_pool_budget(
    resource_snapshot: Dict[str, Any],
    model_tier: str,
    ready_count: int,
    candidate_count: int,
) -> Dict[str, Any]:
    ram_headroom = 2.0 if model_tier == "fast" else 2.5 if model_tier == "deep" else 3.5
    vram_headroom = 2.0 if model_tier == "fast" else 3.0 if model_tier == "deep" else 4.5
    ram_slots: Optional[int] = None
    vram_slots: Optional[int] = None
    ram_pressure = False
    vram_pressure = False
    ram_info = resource_snapshot.get("ram") or {}
    if ram_info.get("available_gb") is not None:
        available_gb = float(ram_info.get("available_gb") or 0.0)
        ram_slots = max(1, int(available_gb // ram_headroom))
        percent = ram_info.get("percent")
        if percent is not None and float(percent) >= 90.0:
            ram_pressure = True
        if available_gb < ram_headroom:
            ram_pressure = True
    gpu_list = [
        g for g in resource_snapshot.get("gpus") or [] if isinstance(g, dict) and g.get("free_gb") is not None
    ]
    if gpu_list:
        free_gpu = max((g.get("free_gb") or 0.0 for g in gpu_list), default=0.0)
        vram_slots = max(1, int(float(free_gpu) // vram_headroom))
        if float(free_gpu) < vram_headroom:
            vram_pressure = True
    base_slots = max(1, ready_count)
    if ram_slots is None and vram_slots is None:
        max_parallel = base_slots
    else:
        if ram_slots is None:
            ram_slots = base_slots
        if vram_slots is None:
            vram_slots = ram_slots
        max_parallel = max(1, min(ram_slots, vram_slots))
    return {
        "max_parallel": max_parallel,
        "configured": max(1, candidate_count),
        "variants": max(1, candidate_count),
        "ram_slots": ram_slots or base_slots,
        "vram_slots": vram_slots or base_slots,
        "ram_headroom_gb": ram_headroom,
        "vram_headroom_gb": vram_headroom,
        "ram_pressure": ram_pressure,
        "vram_pressure": vram_pressure,
    }


def ensure_parallel_research(
    step_plan: StepPlan,
    desired_slots: int,
    decision: RouterDecision,
) -> StepPlan:
    """Ensure the plan has enough parallel research lanes to keep workers busy."""
    steps = step_plan.steps
    if not steps:
        return step_plan
    analysis_ids = [s.step_id for s in steps if s.type == "analysis"]
    analysis_anchor = max(analysis_ids) if analysis_ids else None
    analysis_set = set(analysis_ids)

    def is_parallel_research(step: PlanStep) -> bool:
        return step.type == "research" and set(step.depends_on or []).issubset(analysis_set)

    parallel_research = [s for s in steps if is_parallel_research(s)]
    next_id = max(s.step_id for s in steps) + 1
    changed = False
    if decision.needs_web and not parallel_research:
        new_step = PlanStep(
            step_id=next_id,
            name="Research primary",
            type="research",
            depends_on=[analysis_anchor] if analysis_anchor else [],
            agent_profile="ResearchPrimary",
            inputs={"use_web": decision.needs_web},
            outputs=[{"artifact_type": "evidence", "key": f"lane_extra_{next_id}"}],
            acceptance_criteria=["notes ready"],
            on_fail={"action": "rerun_step"},
        )
        steps.append(new_step)
        parallel_research.append(new_step)
        next_id += 1
        changed = True
    if desired_slots <= 1:
        if not changed:
            return step_plan
        # Ensure downstream steps depend on the added research step.
        merge_steps = sorted((s for s in steps if s.type == "merge"), key=lambda s: s.step_id)
        if merge_steps:
            merge_step = merge_steps[0]
            deps = set(merge_step.depends_on or [])
            deps.update(s.step_id for s in parallel_research)
            merge_step.depends_on = sorted(deps)
            return step_plan
        draft_steps = sorted((s for s in steps if s.type == "draft"), key=lambda s: s.step_id)
        if not draft_steps:
            return step_plan
        deps = set(draft_steps[0].depends_on or [])
        deps.update(s.step_id for s in parallel_research)
        draft_steps[0].depends_on = sorted(deps)
        return step_plan
    missing = desired_slots - len(parallel_research)
    if missing <= 0 and not changed:
        return step_plan
    profiles = ["ResearchPrimary", "ResearchRecency", "ResearchAdversarial"]
    for idx in range(missing):
        profile = profiles[(len(parallel_research) + idx) % len(profiles)]
        name = f"Research lane {len(parallel_research) + idx + 1}"
        new_step = PlanStep(
            step_id=next_id,
            name=name,
            type="research",
            depends_on=[analysis_anchor] if analysis_anchor else [],
            agent_profile=profile,
            inputs={"use_web": decision.needs_web},
            outputs=[{"artifact_type": "evidence", "key": f"lane_extra_{next_id}"}],
            acceptance_criteria=["notes ready"],
            on_fail={"action": "rerun_step"},
        )
        steps.append(new_step)
        parallel_research.append(new_step)
        next_id += 1

    merge_steps = sorted((s for s in steps if s.type == "merge"), key=lambda s: s.step_id)
    if merge_steps:
        merge_step = merge_steps[0]
        deps = set(merge_step.depends_on or [])
        deps.update(s.step_id for s in parallel_research)
        merge_step.depends_on = sorted(deps)
        return step_plan

    draft_steps = sorted((s for s in steps if s.type == "draft"), key=lambda s: s.step_id)
    if not draft_steps:
        return step_plan
    if len(parallel_research) > 1:
        merge_step = PlanStep(
            step_id=next_id,
            name="Merge notes",
            type="merge",
            depends_on=[s.step_id for s in parallel_research],
            agent_profile="Summarizer",
            inputs={},
            outputs=[{"artifact_type": "ledger", "key": "claims_ledger"}],
            acceptance_criteria=["ledger_ready"],
            on_fail={"action": "revise_step"},
        )
        steps.append(merge_step)
        draft_steps[0].depends_on = [merge_step.step_id]
    else:
        deps = set(draft_steps[0].depends_on or [])
        deps.update(s.step_id for s in parallel_research)
        draft_steps[0].depends_on = sorted(deps)
    return step_plan


def ensure_parallel_analysis(step_plan: StepPlan, desired_slots: int) -> StepPlan:
    """Add lightweight analysis lanes to use parallel workers when web research is off."""
    if desired_slots <= 1:
        return step_plan
    steps = step_plan.steps
    if not steps:
        return step_plan
    analysis_steps = [s for s in steps if s.type == "analysis"]
    target_total = min(max(desired_slots + 1, 2), 4)
    if len(analysis_steps) >= target_total:
        return step_plan

    analysis_anchor = max((s.step_id for s in analysis_steps), default=None)
    next_id = max(s.step_id for s in steps) + 1
    prompts = [
        "List key requirements, constraints, and success criteria in short bullets.",
        "Identify risks, edge cases, and mitigations in short bullets.",
        "Propose a clean outline/structure for the final answer (short bullets).",
        "Draft a short TODO list of missing info or follow-up checks.",
    ]
    added: List[PlanStep] = []
    for idx in range(target_total - len(analysis_steps)):
        lane_index = len(analysis_steps) + idx + 1
        prompt_hint = prompts[(lane_index - 1) % len(prompts)]
        new_step = PlanStep(
            step_id=next_id,
            name=f"Analysis lane {lane_index}",
            type="analysis",
            depends_on=[analysis_anchor] if analysis_anchor else [],
            agent_profile="Summarizer",
            inputs={"prompt_hint": prompt_hint},
            outputs=[{"artifact_type": "note", "key": f"analysis_lane_{lane_index}"}],
            acceptance_criteria=["notes ready"],
            on_fail={"action": "rerun_step"},
        )
        steps.append(new_step)
        added.append(new_step)
        next_id += 1

    draft_steps = sorted((s for s in steps if s.type == "draft"), key=lambda s: s.step_id)
    if draft_steps and added:
        deps = set(draft_steps[0].depends_on or [])
        deps.update(s.step_id for s in added)
        draft_steps[0].depends_on = sorted(deps)
    return step_plan

def strip_research_steps(step_plan: StepPlan) -> StepPlan:
    """Remove web research steps and clean dependencies when web access is unavailable."""
    web_types = {"research", "tavily_search", "tavily_extract", "search", "extract"}
    removed = {s.step_id for s in step_plan.steps if s.type in web_types}
    if not removed:
        return step_plan
    step_plan.steps = [s for s in step_plan.steps if s.step_id not in removed]
    for step in step_plan.steps:
        if step.depends_on:
            step.depends_on = [d for d in step.depends_on if d not in removed]
    return step_plan


def loosen_parallel_dependencies(step_plan: StepPlan, allow_parallel: bool, desired_slots: int) -> StepPlan:
    """Let parallel lanes start without waiting on analysis-only deps."""
    if not allow_parallel or desired_slots <= 1:
        return step_plan
    analysis_ids = {s.step_id for s in step_plan.steps if s.type == "analysis"}
    if not analysis_ids:
        return step_plan
    parallel_types = {"research", "tavily_search", "search"}
    for step in step_plan.steps:
        if step.type not in parallel_types or not step.depends_on:
            continue
        deps = [d for d in step.depends_on if d not in analysis_ids]
        if deps != step.depends_on:
            step.depends_on = deps
    return step_plan


def ensure_finalize_step(step_plan: StepPlan) -> StepPlan:
    steps = step_plan.steps
    if not steps:
        return step_plan
    if any(s.type in ("finalize", "final") for s in steps):
        return step_plan
    verify_steps = [s for s in steps if s.type in ("verify", "verifier", "verifier_worker")]
    draft_steps = [s for s in steps if s.type == "draft"]
    depends_on: Set[int] = set()
    requires_verifier = False
    if verify_steps:
        verify_step = max(verify_steps, key=lambda s: s.step_id)
        depends_on.add(verify_step.step_id)
        requires_verifier = True
    elif draft_steps:
        draft_step = max(draft_steps, key=lambda s: s.step_id)
        depends_on.add(draft_step.step_id)
    else:
        depends_on.add(max(s.step_id for s in steps))

    dependent_ids = {dep for s in steps for dep in (s.depends_on or [])}
    terminal_ids = {s.step_id for s in steps if s.step_id not in dependent_ids}
    depends_on.update(terminal_ids)
    next_id = max(s.step_id for s in steps) + 1
    steps.append(
        PlanStep(
            step_id=next_id,
            name="Finalize answer",
            type="finalize",
            depends_on=sorted(depends_on),
            agent_profile="Finalizer",
            inputs={"requires_verifier": requires_verifier},
            outputs=[{"artifact_type": "final", "key": "final_answer"}],
            acceptance_criteria=["finalized_answer"],
            on_fail={"action": "rerun_step"},
        )
    )
    return step_plan


def trim_step_plan(step_plan: StepPlan, max_steps: int) -> StepPlan:
    steps = step_plan.steps
    if not steps or len(steps) <= max_steps:
        return step_plan
    finalize_step = next((s for s in reversed(steps) if s.type in ("finalize", "final")), None)
    if not finalize_step:
        step_plan.steps = steps[:max_steps]
        return step_plan

    requires_verifier = False
    if isinstance(finalize_step.inputs, dict):
        requires_verifier = bool(finalize_step.inputs.get("requires_verifier"))
    draft_step = next((s for s in reversed(steps) if s.type == "draft"), None)
    verify_step = next((s for s in reversed(steps) if s.type in ("verify", "verifier", "verifier_worker")), None)
    required_ids = {finalize_step.step_id}
    if draft_step:
        required_ids.add(draft_step.step_id)
    if requires_verifier and verify_step:
        required_ids.add(verify_step.step_id)

    if len(required_ids) > max_steps:
        max_steps = len(required_ids)

    remaining_required = set(required_ids)
    trimmed: List[PlanStep] = []
    for step in steps:
        is_required = step.step_id in required_ids
        if is_required:
            trimmed.append(step)
            remaining_required.discard(step.step_id)
            continue
        slots_left = max_steps - len(trimmed)
        if slots_left <= len(remaining_required):
            continue
        trimmed.append(step)

    if len(trimmed) > max_steps:
        trimmed = trimmed[:max_steps]

    if trimmed and trimmed[-1].step_id != finalize_step.step_id:
        trimmed = [s for s in trimmed if s.step_id != finalize_step.step_id]
        if len(trimmed) >= max_steps:
            trimmed = trimmed[: max_steps - 1]
        trimmed.append(finalize_step)

    kept_ids = {s.step_id for s in trimmed}
    for step in trimmed:
        if step.depends_on:
            step.depends_on = [d for d in step.depends_on if d in kept_ids]
        if step.type in ("finalize", "final") and not step.depends_on and len(trimmed) > 1:
            step.depends_on = [trimmed[-2].step_id]

    step_plan.steps = trimmed
    return step_plan


def profile_system(profile: str) -> str:
    return {
        "Orchestrator": agents.MICROMANAGER_SYSTEM,
        "ResearchPrimary": agents.RESEARCH_PRIMARY_SYSTEM,
        "ResearchRecency": agents.RESEARCH_RECENCY_SYSTEM,
        "ResearchAdversarial": agents.RESEARCH_ADVERSARIAL_SYSTEM,
        "EvidenceSynth": agents.EVIDENCE_SYNTH_SYSTEM,
        "Math": agents.MATH_SYSTEM,
        "Critic": agents.CRITIC_SYSTEM,
        "Summarizer": agents.SUMMARIZER_SYSTEM,
        "Writer": agents.WRITER_SYSTEM,
        "Finalizer": agents.FINALIZER_SYSTEM,
        "Executor": agents.EXECUTOR_SYSTEM,
        "JSONRepair": agents.JSON_REPAIR_SYSTEM,
        "Verifier": agents.VERIFIER_SYSTEM,
    }.get(profile, agents.RESEARCH_PRIMARY_SYSTEM)


_PROFILE_CAPABILITIES = {
    "Orchestrator": ["structured_output"],
    "Executor": ["structured_output", "tool_use"],
    "Router": ["structured_output"],
    "Planner": ["structured_output"],
    "Summarizer": [],
    "Writer": [],
    "Finalizer": ["structured_output"],
    "Verifier": ["structured_output"],
    "Critic": ["structured_output"],
    "Math": ["structured_output"],
    "EvidenceSynth": ["structured_output"],
    "JSONRepair": ["structured_output"],
    "ResearchPrimary": [],
    "ResearchRecency": [],
    "ResearchAdversarial": [],
    "VisionAnalyst": ["vision"],
    "UploadSecretary": ["vision", "structured_output"],
}

_PROFILE_TIMEOUTS = {
    "Router": 30,
    "Planner": 60,
    "Orchestrator": 60,
    "Summarizer": 45,
    "JSONRepair": 25,
    "Critic": 60,
    "Verifier": 90,
    "EvidenceSynth": 60,
    "ResearchPrimary": 90,
    "ResearchRecency": 90,
    "ResearchAdversarial": 90,
}

_LATENCY_OBJECTIVE_PROFILES = {"Router", "Planner", "Summarizer", "JSONRepair"}


def profile_requirements(profile: str, tool_required_default: bool = True) -> List[str]:
    caps = list(_PROFILE_CAPABILITIES.get(profile, []))
    if tool_required_default and "tool_use" not in caps:
        if profile in {"Executor", "Planner"}:
            caps.append("tool_use")
    return caps


def profile_model(profile: str, model_map: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    """Return (base_url, model_id) for a given profile."""
    if profile == "Orchestrator":
        cfg = model_map.get("orch")
    elif profile == "Executor":
        cfg = model_map.get("executor") or model_map.get("summarizer") or model_map.get("router")
    elif profile == "Writer":
        cfg = model_map.get("orch") or model_map.get("worker")
    elif profile == "Finalizer":
        cfg = model_map.get("orch") or model_map.get("worker")
    elif profile in ("Summarizer", "Critic", "JSONRepair"):
        cfg = model_map.get("summarizer") or model_map.get("router") or model_map.get("worker")
    elif profile == "Verifier":
        cfg = model_map.get("verifier") or model_map.get("worker")
    elif profile == "ResearchRecency":
        cfg = model_map.get("worker_b") or model_map.get("worker")
    elif profile == "ResearchAdversarial":
        cfg = model_map.get("worker_c") or model_map.get("worker")
    elif profile == "EvidenceSynth":
        cfg = model_map.get("worker") or model_map.get("summarizer")
    else:
        cfg = model_map.get("worker")
    if not cfg:
        cfg = {"base_url": "", "model": ""}
    return cfg.get("base_url"), cfg.get("model")


def candidate_endpoints(profile: str, model_map: Dict[str, Dict[str, str]]) -> List[Tuple[str, str]]:
    """Return ordered (base_url, model) tuples with fallbacks for a profile."""
    if profile == "Executor":
        order = ["executor", "summarizer", "router", "deep_orch", "orch", "worker"]
    elif profile == "ResearchRecency":
        order = ["worker_b", "worker", "worker_c", "orch", "summarizer", "router"]
    elif profile == "ResearchAdversarial":
        order = ["worker_c", "worker", "worker_b", "orch", "summarizer", "router"]
    elif profile == "ResearchPrimary":
        order = ["worker", "worker_b", "worker_c", "orch", "summarizer", "router"]
    elif profile == "EvidenceSynth":
        order = ["worker", "worker_b", "worker_c", "orch", "summarizer", "router"]
    elif profile == "Orchestrator":
        order = ["orch", "worker", "summarizer", "router"]
    elif profile == "Writer":
        order = ["orch", "worker", "summarizer", "router"]
    elif profile == "Finalizer":
        order = ["orch", "worker", "summarizer", "router"]
    elif profile == "Verifier":
        order = ["verifier", "worker", "orch", "summarizer"]
    elif profile in ("Summarizer", "Critic", "JSONRepair"):
        order = ["summarizer", "worker", "router", "orch"]
    else:
        order = ["worker", "orch", "summarizer", "router", "verifier"]
    seen: Set[Tuple[str, str]] = set()
    candidates: List[Tuple[str, str]] = []
    for key in order:
        cfg = model_map.get(key) or {}
        base_url = cfg.get("base_url")
        model = cfg.get("model")
        if not base_url or not model:
            continue
        pair = (base_url, model)
        if pair in UNAVAILABLE_MODELS:
            continue
        if pair in seen:
            continue
        seen.add(pair)
        candidates.append(pair)
    # Ensure the explicit profile mapping is included even if it's non-standard.
    primary = profile_model(profile, model_map)
    if primary[0] and primary[1] and primary not in seen:
        candidates.insert(0, primary)
    return candidates


def select_model_suite(
    base_map: Dict[str, Dict[str, str]], tier: str, deep_route: str
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, str], bool, str]:
    """
    Return (model_map, planner_endpoint, executor_endpoint, allow_parallel, execution_mode).
    - planner_endpoint: slow/accurate planner (OSS).
    - executor_endpoint: fast executor for scheduling and control gating (4B).
    """
    def has_cfg(cfg: Dict[str, str]) -> bool:
        return bool(cfg and cfg.get("base_url") and cfg.get("model"))

    orch = base_map.get("orch") or base_map.get("worker") or {}
    planner = orch
    router = base_map.get("router") or {}
    summarizer = base_map.get("summarizer") or {}
    worker = base_map.get("worker") or orch
    worker_b = base_map.get("worker_b") or worker
    worker_c = base_map.get("worker_c") or worker
    verifier = base_map.get("verifier") or worker
    fast_cfg = base_map.get("fast") or {}
    deep_planner = base_map.get("deep_planner") or {}
    deep_orch = base_map.get("deep_orch") or {}

    if tier == "fast" and has_cfg(fast_cfg):
        orch = fast_cfg
        planner = fast_cfg
        router = fast_cfg
        summarizer = fast_cfg
        worker = fast_cfg
        worker_b = fast_cfg
        worker_c = fast_cfg
        verifier = fast_cfg
    elif tier == "deep":
        if has_cfg(deep_orch):
            orch = deep_orch
        if has_cfg(deep_planner):
            planner = deep_planner

    executor = base_map.get("executor") or {}
    if not has_cfg(executor):
        if tier == "deep" and has_cfg(deep_orch):
            executor = deep_orch
        else:
            executor = summarizer or router or deep_orch or orch
    suite = {
        "orch": orch,
        "worker": worker,
        "worker_b": worker_b,
        "worker_c": worker_c,
        "router": router or executor,
        "summarizer": summarizer or executor,
        "verifier": verifier,
        "deep_planner": deep_planner or planner,
        "deep_orch": deep_orch or executor,
        "fast": fast_cfg or worker,
    }
    allow_parallel = True
    if tier == "fast":
        execution_mode = "fast_direct"
        allow_parallel = False
    elif tier == "deep":
        execution_mode = "oss_team" if deep_route == "oss" else "deep_cluster"
    else:
        execution_mode = "pro_full"
    return suite, planner, executor, allow_parallel, execution_mode


def resolve_auto_tier(decision: RouterDecision) -> str:
    """Choose the lightest tier that can still satisfy routing needs."""
    tool_budget = decision.tool_budget or {}
    heavy_web = tool_budget.get("tavily_search", 0) > 8 or tool_budget.get("tavily_extract", 0) > 12
    if decision.extract_depth == "advanced":
        return "deep"
    if decision.needs_web:
        return "deep" if heavy_web else "pro"
    if (decision.expected_passes or 0) > 1:
        return "pro"
    if decision.reasoning_level in ("MED", "HIGH", "ULTRA"):
        return "pro"
    if tool_budget.get("tavily_search", 0) > 0 or tool_budget.get("tavily_extract", 0) > 0:
        return "pro"
    return "fast"


def normalize_step_plan_payload(plan: Dict[str, Any], question: str) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}

    def _coerce_step_id(value: Any) -> Optional[int]:
        step_id = _coerce_int(value)
        if step_id is not None:
            return step_id
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                try:
                    return int(match.group(0))
                except Exception:
                    return None
        return None

    if not plan.get("plan_id"):
        plan["plan_id"] = str(uuid.uuid4())
    if not plan.get("goal"):
        plan["goal"] = question
    if not isinstance(plan.get("global_constraints"), dict):
        plan["global_constraints"] = {}

    steps = plan.get("steps")
    if isinstance(steps, dict):
        steps = [
            {"step_id": key, **value} if isinstance(value, dict) else {"step_id": key, "name": str(value)}
            for key, value in steps.items()
        ]
    if not isinstance(steps, list):
        steps = []

    normalized: List[dict] = []
    seen_ids: Set[int] = set()
    next_id = 1
    for step in steps:
        if not isinstance(step, dict):
            continue
        raw_id = step.get("step_id") or step.get("id") or step.get("step")
        step_id = _coerce_step_id(raw_id)
        if step_id is None or step_id in seen_ids:
            while next_id in seen_ids:
                next_id += 1
            step_id = next_id
        seen_ids.add(step_id)
        if step_id >= next_id:
            next_id = step_id + 1

        depends = step.get("depends_on") or step.get("depends") or step.get("deps") or []
        if isinstance(depends, (str, int)):
            depends = [depends]
        dep_ids: List[int] = []
        if isinstance(depends, list):
            for dep in depends:
                dep_id = _coerce_step_id(dep)
                if dep_id is None:
                    continue
                dep_ids.append(dep_id)
        step["step_id"] = step_id
        step["depends_on"] = sorted(set(dep_ids))
        raw_type = str(step.get("type") or "analysis").strip()
        type_token = raw_type.lower()
        type_aliases = {
            "writer": "draft",
            "drafting": "draft",
            "final": "finalize",
            "finalizer": "finalize",
            "finalization": "finalize",
            "verifier": "verify",
            "verification": "verify",
            "researchprimary": "research",
            "researchrecency": "research",
            "researchadversarial": "research",
            "tavilysearch": "tavily_search",
            "tavilyextract": "tavily_extract",
        }
        step_type = type_aliases.get(type_token, type_token or "analysis")
        if step_type:
            step["type"] = step_type
        step_name = str(step.get("name") or f"Step {step_id}").strip()
        step["name"] = step_name or f"Step {step_id}"
        profile = str(step.get("agent_profile") or "").strip()
        if not profile:
            default_profiles = {
                "analysis": "Summarizer",
                "draft": "Writer",
                "finalize": "Finalizer",
                "verify": "Verifier",
                "research": "ResearchPrimary",
                "merge": "Summarizer",
            }
            profile = default_profiles.get(step_type.lower(), "Summarizer")
        step["agent_profile"] = profile
        inputs = step.get("inputs")
        if not isinstance(inputs, dict):
            step["inputs"] = {}
        outputs = step.get("outputs") or []
        if isinstance(outputs, dict):
            outputs = [outputs]
        elif isinstance(outputs, str):
            outputs = [{"artifact_type": "note", "key": outputs}]
        if not isinstance(outputs, list):
            outputs = []
        step["outputs"] = outputs
        criteria = step.get("acceptance_criteria") or step.get("acceptance") or step.get("criteria") or []
        if isinstance(criteria, str):
            criteria = [criteria]
        if isinstance(criteria, list):
            criteria = [str(item).strip() for item in criteria if str(item).strip()]
        else:
            criteria = []
        step["acceptance_criteria"] = criteria
        on_fail = step.get("on_fail")
        if not isinstance(on_fail, dict):
            on_fail = {}
        step["on_fail"] = on_fail
        normalized.append(step)

    plan["steps"] = normalized
    return plan


def build_linear_plan(
    question: str,
    decision: RouterDecision,
    depth_profile: dict,
    needs_verify: bool = True,
    worker_slots: int = 1,
    prefer_parallel: bool = False,
) -> StepPlan:
    """Deterministic lightweight plan for fast/oss-linear modes with optional parallel research lanes."""
    steps: List[dict] = [
        {
            "step_id": 1,
            "name": "Clarify task",
            "type": "analysis",
            "depends_on": [],
            "agent_profile": "Summarizer",
            "inputs": {"from_user": True},
            "outputs": [{"artifact_type": "criteria", "key": "success_criteria"}],
            "acceptance_criteria": ["criteria captured"],
            "on_fail": {"action": "rerun_step"},
        },
    ]
    next_step_id = 2
    draft_dep = 1
    if decision.needs_web:
        research_steps: List[dict] = []
        parallel_research = prefer_parallel and worker_slots > 1
        research_defs = [
            ("ResearchPrimary", "Research primary", "lane_primary"),
            ("ResearchRecency", "Research recency", "lane_recency"),
            ("ResearchAdversarial", "Research adversarial", "lane_adversarial"),
        ]
        max_research = 1
        if parallel_research:
            max_research = worker_slots
            max_steps = int(depth_profile.get("max_steps") or 0)
            if max_steps:
                # Cap lanes so the linear plan fits within the step budget.
                reserved = 1 + 1 + 1 + (1 if needs_verify else 0)
                max_allow = max_steps - reserved
                if max_allow <= 1:
                    max_research = 1
                else:
                    max_research = min(max_research, max_allow - 1)
            if max_research <= 1:
                parallel_research = False
        if parallel_research:
            for idx in range(max_research):
                profile, name, key = research_defs[idx % len(research_defs)]
                name_suffix = "" if idx < len(research_defs) else f" {idx + 1}"
                key_suffix = "" if idx < len(research_defs) else f"_{idx + 1}"
                research_steps.append(
                    {
                        "step_id": next_step_id,
                        "name": f"{name}{name_suffix}",
                        "type": "research",
                        "depends_on": [1],
                        "agent_profile": profile,
                        "inputs": {"use_web": decision.needs_web},
                        "outputs": [{"artifact_type": "evidence", "key": f"{key}{key_suffix}"}],
                        "acceptance_criteria": ["notes ready"],
                        "on_fail": {"action": "rerun_step"},
                    }
                )
                next_step_id += 1
            steps.extend(research_steps)
            if len(research_steps) > 1:
                merge_id = next_step_id
                steps.append(
                    {
                        "step_id": merge_id,
                        "name": "Merge notes",
                        "type": "merge",
                        "depends_on": [s["step_id"] for s in research_steps],
                        "agent_profile": "Summarizer",
                        "inputs": {},
                        "outputs": [{"artifact_type": "ledger", "key": "claims_ledger"}],
                        "acceptance_criteria": ["ledger_ready"],
                        "on_fail": {"action": "revise_step"},
                    }
                )
                next_step_id += 1
                draft_dep = merge_id
            else:
                draft_dep = research_steps[0]["step_id"]
        else:
            steps.append(
                {
                    "step_id": next_step_id,
                    "name": "Gather notes",
                    "type": "research",
                    "depends_on": [1],
                    "agent_profile": "ResearchPrimary",
                    "inputs": {"use_web": decision.needs_web},
                    "outputs": [{"artifact_type": "evidence", "key": "lane_primary"}],
                    "acceptance_criteria": ["notes ready"],
                    "on_fail": {"action": "rerun_step"},
                }
            )
            draft_dep = next_step_id
            next_step_id += 1
    draft_step_id = next_step_id
    steps.append(
        {
            "step_id": draft_step_id,
            "name": "Draft answer",
            "type": "draft",
            "depends_on": [draft_dep],
            "agent_profile": "Writer",
            "inputs": {},
            "outputs": [{"artifact_type": "draft", "key": "draft_answer"}],
            "acceptance_criteria": ["draft_complete"],
            "on_fail": {"action": "revise_step"},
        }
    )
    next_step_id += 1
    verify_step_id = None
    if needs_verify:
        steps.append(
            {
                "step_id": next_step_id,
                "name": "Verify",
                "type": "verify",
                "depends_on": [draft_step_id],
                "agent_profile": "Verifier",
                "inputs": {},
                "outputs": [{"artifact_type": "verifier", "key": "verifier_report"}],
                "acceptance_criteria": ["verdict_ready"],
                "on_fail": {"action": "rerun_step"},
            }
        )
        verify_step_id = next_step_id
        next_step_id += 1
    finalize_dep = verify_step_id or draft_step_id
    steps.append(
        {
            "step_id": next_step_id,
            "name": "Finalize answer",
            "type": "finalize",
            "depends_on": [finalize_dep],
            "agent_profile": "Finalizer",
            "inputs": {"requires_verifier": bool(verify_step_id)},
            "outputs": [{"artifact_type": "final", "key": "final_answer"}],
            "acceptance_criteria": ["finalized_answer"],
            "on_fail": {"action": "rerun_step"},
        }
    )
    plan = {
        "plan_id": str(uuid.uuid4()),
        "goal": question,
        "global_constraints": {
            "needs_web": decision.needs_web,
            "reasoning_level": decision.reasoning_level,
            "max_loops": depth_profile.get("max_loops", 1),
            "tool_budget": depth_profile.get("tool_budget", {"tavily_search": 6, "tavily_extract": 6}),
        },
        "steps": steps,
    }
    return StepPlan(**plan)


def build_fast_plan(
    question: str,
    decision: RouterDecision,
    needs_verify: bool = False,
) -> StepPlan:
    """Single-step plan for fast tier responses."""
    steps: List[dict] = [
        {
            "step_id": 1,
            "name": "Draft answer",
            "type": "draft",
            "depends_on": [],
            "agent_profile": "Writer",
            "inputs": {},
            "outputs": [{"artifact_type": "draft", "key": "draft_answer"}],
            "acceptance_criteria": ["draft_complete"],
            "on_fail": {"action": "revise_step"},
        },
    ]
    next_step_id = 2
    verify_step_id = None
    if needs_verify:
        steps.append(
            {
                "step_id": next_step_id,
                "name": "Verify",
                "type": "verify",
                "depends_on": [1],
                "agent_profile": "Verifier",
                "inputs": {},
                "outputs": [{"artifact_type": "verifier", "key": "verifier_report"}],
                "acceptance_criteria": ["verdict_ready"],
                "on_fail": {"action": "rerun_step"},
            }
        )
        verify_step_id = next_step_id
        next_step_id += 1
    finalize_dep = verify_step_id or 1
    steps.append(
        {
            "step_id": next_step_id,
            "name": "Finalize answer",
            "type": "finalize",
            "depends_on": [finalize_dep],
            "agent_profile": "Finalizer",
            "inputs": {"requires_verifier": bool(verify_step_id)},
            "outputs": [{"artifact_type": "final", "key": "final_answer"}],
            "acceptance_criteria": ["finalized_answer"],
            "on_fail": {"action": "rerun_step"},
        }
    )
    plan = {
        "plan_id": str(uuid.uuid4()),
        "goal": question,
        "global_constraints": {
            "needs_web": False,
            "reasoning_level": decision.reasoning_level,
            "max_loops": 0,
            "tool_budget": decision.tool_budget or {"tavily_search": 0, "tavily_extract": 0},
        },
        "steps": steps,
    }
    return StepPlan(**plan)


async def choose_deep_route(
    lm_client: LMStudioClient,
    router_endpoint: Dict[str, str],
    question: str,
    preference: str,
    router_decision: Optional[RouterDecision] = None,
    run_state: Optional[RunState] = None,
) -> str:
    """Router for LocalDeep between OSS linear vs. mini-cluster."""
    if preference in ("oss", "cluster"):
        return preference
    coding_task = looks_like_coding_task(question)
    if router_decision:
        try:
            needs_web = bool(router_decision.needs_web)
            extract_depth = (router_decision.extract_depth or "").lower()
            budget = router_decision.tool_budget or {}
            heavy_web = budget.get("tavily_search", 0) > 0 or budget.get("tavily_extract", 0) > 0
            if is_exploratory_question(question, router_decision):
                return "cluster"
            if needs_web or extract_depth == "advanced" or heavy_web:
                return "cluster"
            if not coding_task:
                return "oss"
            if not needs_web and router_decision.reasoning_level in ("LOW", "MED"):
                return "oss"
        except Exception:
            pass
    prompt = (
        "Choose the best execution lane for this question.\n"
        "- Use route 'oss' when the OSS model's internal knowledge should be enough (fact lookup, summarization, no current-events or web search needed).\n"
        "- Prefer 'oss' for non-coding tasks; coder-focused models are resource heavy and best reserved for coding.\n"
        "- Use route 'cluster' when current data, web search, multi-source research, or cross-checking is likely required.\n"
        "Return JSON only: {\"route\": \"oss\" | \"cluster\"}."
        f"\nQuestion: {question}"
    )
    try:
        if run_state and run_state.model_manager:
            content = await run_worker(
                lm_client,
                "Router",
                {},
                prompt,
                temperature=0.0,
                max_tokens=80,
                run_state=run_state,
                model_manager=run_state.model_manager,
                system_prompt_override=agents.SUMMARIZER_SYSTEM,
                context="deep_route",
            )
        else:
            resp = await lm_client.chat_completion(
                model=router_endpoint["model"],
                messages=[{"role": "system", "content": agents.SUMMARIZER_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
                base_url=router_endpoint["base_url"],
                run_state=run_state,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(
            content,
            lm_client,
            router_endpoint["model"],
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        if parsed and parsed.get("route") in ("oss", "cluster"):
            return parsed["route"]
    except Exception:
        pass
    # Fallback heuristic: shorter questions lean oss, else cluster
    return "oss" if len(question) < 120 else "cluster"


async def call_router(
    lm_client: LMStudioClient,
    endpoint: Dict[str, str],
    question: str,
    manual_level: Optional[str] = None,
    default_level: Optional[str] = None,
    strict_mode: bool = False,
    run_state: Optional[RunState] = None,
) -> RouterDecision:
    user_msg = f"User question: {question}\nReturn JSON only."
    parsed = None
    fallback_used = False
    needs_web_guess = guess_needs_web(question)
    try:
        if run_state and run_state.model_manager:
            content = await run_worker(
                lm_client,
                "Router",
                {},
                user_msg,
                temperature=0.1,
                max_tokens=300,
                run_state=run_state,
                model_manager=run_state.model_manager,
                system_prompt_override=agents.ROUTER_SYSTEM,
                context="router",
            )
        else:
            resp = await lm_client.chat_completion(
                model=endpoint["model"],
                messages=[{"role": "system", "content": agents.ROUTER_SYSTEM}, {"role": "user", "content": user_msg}],
                temperature=0.1,
                max_tokens=300,
                base_url=endpoint["base_url"],
                run_state=run_state,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(
            content,
            lm_client,
            endpoint["model"],
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
    except Exception:
        parsed = None
    if not parsed:
        fallback_used = True
        expected_passes = 2 if strict_mode else 1
        parsed = {
            "needs_web": needs_web_guess,
            "reasoning_level": manual_level or default_level or "MED",
            "topic": "general",
            "max_results": 6,
            "extract_depth": "basic",
            "expected_passes": expected_passes,
            "stop_conditions": {},
        }
    decision = RouterDecision(**parsed)
    if manual_level is not None:
        decision.reasoning_level = manual_level
    # If the router was unsure, lean toward web for data-heavy questions.
    decision.needs_web = decision.needs_web or needs_web_guess
    decision.expected_passes = max(1, decision.expected_passes or 1)
    if manual_level is None and not fallback_used:
        decision.reasoning_level = choose_auto_reasoning_level(question, decision)
    return decision


async def build_step_plan(
    lm_client: LMStudioClient,
    orch_endpoint: Dict[str, str],
    question: str,
    decision: RouterDecision,
    depth_profile: dict,
    memory_context: str = "",
    planner_endpoint: Optional[Dict[str, str]] = None,
    desired_parallel: int = 1,
    run_state: Optional[RunState] = None,
) -> StepPlan:
    plan_prompt = (
        "Produce a JSON step plan for answering the question. "
        "Include step_id, name, type, depends_on (list of ids), agent_profile, acceptance_criteria. "
        "Keep 6-12 steps for typical questions and use agent_profile 'Writer' for the draft step. "
        "Add global_constraints.expected_passes (1-3) if a verifier rerun is likely, and response_guidance describing how long the final answer should be based on task complexity. "
        "Use available worker slots for parallel research lanes when useful; rotate ResearchPrimary/ResearchRecency/ResearchAdversarial and add multiple lanes per profile if slots exceed profiles."
    )
    user_content = (
        f"Question: {question}\nNeeds web: {decision.needs_web}\nReasoning level: {decision.reasoning_level}\nExpected passes: {decision.expected_passes}\n"
        f"Available worker slots: {max(desired_parallel, 1)} (aim to keep them busy in parallel)\n"
        f"Chat facts (this conversation): {memory_context}\n"
        "Return JSON only as {\"plan_id\": \"...\", \"goal\": \"...\", \"global_constraints\": {...}, \"steps\": [...]}"
    )
    parsed = None
    plan_ep = planner_endpoint or orch_endpoint
    try:
        if run_state and run_state.model_manager:
            content = await run_worker(
                lm_client,
                "Orchestrator",
                {},
                user_content,
                temperature=0.25,
                max_tokens=900,
                run_state=run_state,
                model_manager=run_state.model_manager,
                system_prompt_override=agents.MICROMANAGER_SYSTEM + plan_prompt,
                context="step_plan",
            )
        else:
            resp = await lm_client.chat_completion(
                model=plan_ep["model"],
                messages=[
                    {"role": "system", "content": agents.MICROMANAGER_SYSTEM + plan_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.25,
                max_tokens=900,
                base_url=plan_ep["base_url"],
                run_state=run_state,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(
            content,
            lm_client,
            plan_ep["model"],
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        parsed = {}
    if parsed:
        parsed = normalize_step_plan_payload(parsed, question)
    if not parsed or not parsed.get("steps"):
        fallback = build_linear_plan(
            question,
            decision,
            depth_profile,
            needs_verify=True,
            worker_slots=max(desired_parallel, 1),
            prefer_parallel=desired_parallel > 1,
        )
        fallback.global_constraints.setdefault("expected_passes", decision.expected_passes)
        fallback.global_constraints.setdefault(
            "response_guidance",
            "Keep the answer concise and sized to the question; expand only when evidence is complex.",
        )
        return fallback
    try:
        return StepPlan(**parsed)
    except Exception as exc:
        if run_state:
            run_state.add_dev_trace("Plan validation failed; using fallback.", {"error": str(exc)})
        fallback = build_linear_plan(
            question,
            decision,
            depth_profile,
            needs_verify=True,
            worker_slots=max(desired_parallel, 1),
            prefer_parallel=desired_parallel > 1,
        )
        fallback.global_constraints.setdefault("expected_passes", decision.expected_passes)
        fallback.global_constraints.setdefault(
            "response_guidance",
            "Keep the answer concise and sized to the question; expand only when evidence is complex.",
        )
        return fallback


async def run_worker(
    lm_client: LMStudioClient,
    profile: str,
    model_map: Dict[str, Dict[str, str]],
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 700,
    run_id: Optional[str] = None,
    bus: Optional["EventBus"] = None,
    step_id: Optional[int] = None,
    context: str = "",
    run_state: Optional[RunState] = None,
    model_manager: Optional[ModelManager] = None,
    system_prompt_override: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> str:
    if run_state and not run_state.can_chat:
        raise RuntimeError("Local model unavailable.")
    system_prompt = system_prompt_override or profile_system(profile)
    avoid_coder = False
    if run_state and run_state.question:
        avoid_coder = not looks_like_coding_task(run_state.question)
    if model_manager is not None:
        required = profile_requirements(profile, model_manager.tool_required_by_default)
        request = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_responses": True,
        }
        effective_objective = model_manager.routing_objective
        if profile in _LATENCY_OBJECTIVE_PROFILES:
            effective_objective = "best_latency"
        call_timeout = timeout_s if timeout_s is not None else _PROFILE_TIMEOUTS.get(profile)
        try:
            if call_timeout:
                resp, instance = await asyncio.wait_for(
                    model_manager.call_with_instance(
                        required_capabilities=required,
                        objective=effective_objective,
                        request=request,
                        avoid_coder=avoid_coder,
                    ),
                    timeout=call_timeout,
                )
            else:
                resp, instance = await model_manager.call_with_instance(
                    required_capabilities=required,
                    objective=effective_objective,
                    request=request,
                    avoid_coder=avoid_coder,
                )
        except asyncio.TimeoutError as exc:
            if run_state:
                run_state.add_dev_trace(
                    "Model call timed out.",
                    {"profile": profile, "timeout_s": call_timeout or 0},
                )
            raise RuntimeError("Model call timed out.") from exc
        if run_id and bus:
            await bus.emit(
                run_id,
                "model_selected",
                {
                    "profile": profile,
                    "model": instance.model_key,
                    "instance": instance.api_identifier,
                    "backend_id": instance.backend_id,
                    "step_id": step_id,
                    "context": context,
                },
            )
        return resp["choices"][0]["message"]["content"]
    last_error: Optional[Exception] = None
    last_base_url: Optional[str] = None
    last_detail: str = ""
    saw_unavailable = False
    for base_url, model in candidate_endpoints(profile, model_map):
        if run_state and not run_state.can_chat:
            break
        last_base_url = base_url
        try:
            resp = await lm_client.chat_completion(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url,
                run_state=run_state,
            )
            if run_id and bus:
                model_used = model
                if isinstance(resp, dict):
                    model_used = str(resp.get("_model_used") or model)
                await bus.emit(
                    run_id,
                    "model_selected",
                    {
                        "profile": profile,
                        "model": model_used,
                        "base_url": base_url,
                        "step_id": step_id,
                        "context": context,
                        **({"requested_model": model} if model_used != model else {}),
                    },
                )
            return resp["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if run_state and not run_state.can_chat:
                break
            detail = ""
            try:
                if exc.response is not None:
                    detail = lm_client._extract_error_detail(exc.response)
            except Exception:
                detail = ""
            if detail:
                last_detail = detail
            detail_lower = detail.lower()
            async def _mark_unavailable() -> None:
                nonlocal saw_unavailable
                saw_unavailable = True
                UNAVAILABLE_MODELS.add((base_url, model))
                lm_client.mark_model_unavailable(base_url, model)
                if run_id and bus:
                    unavailable_payload = {
                        "profile": profile,
                        "model": model,
                        "base_url": base_url,
                        "step_id": step_id,
                        "context": context,
                    }
                    await bus.emit(run_id, "model_unavailable", unavailable_payload)
                    queue_narration(
                        lm_client,
                        model_map,
                        run_state,
                        bus,
                        run_id,
                        run_state.question if run_state else "",
                        "model_unavailable",
                        unavailable_payload,
                        tone="warn",
                    )
            # Retry with fallback models if the current model is unavailable or unloaded.
            if exc.response is not None and exc.response.status_code in (400, 404):
                if (
                    "failed to load model" in detail_lower
                    or "operation canceled" in detail_lower
                    or "out of memory" in detail_lower
                    or "insufficient memory" in detail_lower
                    or "model is unloaded" in detail_lower
                    or "model unloaded" in detail_lower
                    or "model not found" in detail_lower
                    or "invalid model identifier" in detail_lower
                    or "valid downloaded model" in detail_lower
                ):
                    await _mark_unavailable()
                    continue
                if not detail:
                    try:
                        ok, _ = await lm_client.check_chat(base_url=base_url, model=model, run_state=run_state)
                    except Exception:
                        ok = False
                    if not ok:
                        await _mark_unavailable()
                        continue
                # Avoid repeating the same rejected payload against other endpoints.
                break
            # For other status errors, try fallbacks but keep the last error.
            if run_id and bus:
                error_payload = {
                    "profile": profile,
                    "model": model,
                    "base_url": base_url,
                    "step_id": step_id,
                    "context": context,
                    "error": str(exc),
                }
                if detail:
                    error_payload["detail"] = detail
                await bus.emit(run_id, "model_error", error_payload)
                queue_narration(
                    lm_client,
                    model_map,
                    run_state,
                    bus,
                    run_id,
                    run_state.question if run_state else "",
                    "model_error",
                    error_payload,
                    tone="warn",
                )
            continue
        except httpx.RequestError as exc:
            last_error = exc
            if run_state and not run_state.can_chat:
                break
            if run_id and bus:
                error_payload = {
                    "profile": profile,
                    "model": model,
                    "base_url": base_url,
                    "step_id": step_id,
                    "context": context,
                    "error": str(exc),
                }
                await bus.emit(run_id, "model_error", error_payload)
                queue_narration(
                    lm_client,
                    model_map,
                    run_state,
                    bus,
                    run_id,
                    run_state.question if run_state else "",
                    "model_error",
                    error_payload,
                    tone="warn",
                )
            continue
    if saw_unavailable and last_base_url:
        try:
            available = await lm_client.list_models_cached(last_base_url)
            filtered = []
            for candidate in available:
                if not candidate:
                    continue
                lowered = candidate.lower()
                if "embed" in lowered:
                    continue
                if (last_base_url, candidate) in UNAVAILABLE_MODELS:
                    continue
                filtered.append(candidate)
            fallback_any = [c for c in available if c and "embed" not in c.lower()]
            seen: Set[str] = set()
            for pass_candidates in (filtered, fallback_any) if filtered else (fallback_any,):
                for candidate in pass_candidates:
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    try:
                        resp = await lm_client.chat_completion(
                            model=candidate,
                            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            base_url=last_base_url,
                            run_state=run_state,
                        )
                        if run_id and bus:
                            await bus.emit(
                                run_id,
                                "model_selected",
                                {
                                    "profile": profile,
                                    "model": candidate,
                                    "base_url": last_base_url,
                                    "step_id": step_id,
                                    "context": context,
                                    "fallback": True,
                                },
                            )
                        return resp["choices"][0]["message"]["content"]
                    except httpx.HTTPStatusError as exc:
                        last_error = exc
                        try:
                            if exc.response is not None:
                                last_detail = lm_client._extract_error_detail(exc.response) or last_detail
                        except Exception:
                            pass
                        continue
                    except httpx.RequestError as exc:
                        last_error = exc
                        continue
        except Exception as exc:
            last_error = exc
    if saw_unavailable and run_state:
        run_state.mark_chat_unavailable(last_detail or "No available chat model is loaded.")
    if last_error:
        raise last_error
    raise RuntimeError("No available model endpoint for profile.")


def _default_research_prompt_hint(profile: str) -> str:
    profile_key = (profile or "").strip().lower()
    if profile_key == "researchrecency":
        return (
            "Focus on the most recent updates, stats, or announcements. "
            "Prefer sources from the last week/month and include recency cues in queries."
        )
    if profile_key == "researchadversarial":
        return (
            "Look for critiques, risks, counterexamples, and conflicting claims. "
            "Include skeptical or independent sources."
        )
    return "Prioritize primary/official sources, definitions, and baseline facts."


async def seed_research_prompts(
    lm_client: LMStudioClient,
    executor_endpoint: Optional[Dict[str, str]],
    question: str,
    decision: RouterDecision,
    step_plan: StepPlan,
    run_state: Optional[RunState] = None,
) -> None:
    if not executor_endpoint:
        return
    model = executor_endpoint.get("model")
    base_url = executor_endpoint.get("base_url")
    if not model or not base_url:
        return
    research_steps = [s for s in step_plan.steps if s.type == "research"]
    if not research_steps:
        return
    if all(
        isinstance(s.inputs, dict) and (s.inputs.get("prompt_hint") or s.inputs.get("prompt"))
        for s in research_steps
    ):
        return
    steps_payload = [
        {
            "step_id": s.step_id,
            "name": s.name,
            "agent_profile": s.agent_profile,
            "depends_on": s.depends_on,
        }
        for s in research_steps
    ]
    prompt = (
        "You are the executor. Assign each research step a distinct angle so lanes do not overlap. "
        "Return JSON only as {\"prompts\": [{\"step_id\": 1, \"prompt_hint\": \"...\"}]}.\n"
        "Each prompt_hint should be concise (1-3 sentences) and mention any specific subtopics or source types to target. "
        "Do not include chain-of-thought.\n"
        f"Question: {question}\nReasoning level: {decision.reasoning_level}\n"
        f"Research steps: {json.dumps(steps_payload, ensure_ascii=True)}"
    )
    parsed = None
    try:
        if run_state and run_state.model_manager:
            content = await run_worker(
                lm_client,
                "Executor",
                {},
                prompt,
                temperature=0.2,
                max_tokens=350,
                run_state=run_state,
                model_manager=run_state.model_manager,
                system_prompt_override=agents.EXECUTOR_SYSTEM,
                context="seed_research",
            )
        else:
            resp = await lm_client.chat_completion(
                model=model,
                messages=[{"role": "system", "content": agents.EXECUTOR_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=350,
                base_url=base_url,
                run_state=run_state,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(
            content,
            lm_client,
            model,
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
    except Exception:
        parsed = None

    hints_by_id: Dict[int, str] = {}

    def _add_hint(step_id_val: Any, hint_val: Any) -> None:
        try:
            step_id = int(step_id_val)
        except Exception:
            return
        hint = str(hint_val or "").strip()
        if not hint:
            return
        if len(hint) > 800:
            hint = hint[:800].rstrip() + "..."
        hints_by_id[step_id] = hint

    prompts = None
    if isinstance(parsed, dict):
        prompts = parsed.get("prompts") or parsed.get("prompt_hints") or parsed.get("steps")
        if isinstance(prompts, dict):
            for key, value in prompts.items():
                _add_hint(key, value)
        elif isinstance(prompts, list):
            for item in prompts:
                if not isinstance(item, dict):
                    continue
                _add_hint(
                    item.get("step_id") or item.get("id"),
                    item.get("prompt_hint") or item.get("prompt") or item.get("hint"),
                )
    elif isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            _add_hint(
                item.get("step_id") or item.get("id"),
                item.get("prompt_hint") or item.get("prompt") or item.get("hint"),
            )

    for step in research_steps:
        if not isinstance(step.inputs, dict):
            step.inputs = {}
        if step.inputs.get("prompt_hint") or step.inputs.get("prompt"):
            continue
        hint = hints_by_id.get(step.step_id) or _default_research_prompt_hint(step.agent_profile)
        step.inputs["prompt_hint"] = hint


async def warm_worker_pool(
    lm_client: LMStudioClient,
    model_map: Dict[str, Dict[str, str]],
    max_workers: int,
    ready_models: Optional[Set[Tuple[str, str]]] = None,
    run_state: Optional[RunState] = None,
    run_id: Optional[str] = None,
    bus: Optional["EventBus"] = None,
    model_manager: Optional[ModelManager] = None,
) -> None:
    if max_workers <= 0:
        return
    if model_manager:
        for _ in range(max_workers):
            instance = await model_manager.acquire_instance(required_capabilities=["tool_use"], backlog=max_workers)
            if instance:
                await model_manager.release_instance(instance.instance_id)
        return
    seen: Set[Tuple[str, str]] = set()
    targets: List[Tuple[str, str]] = []
    for role in ("worker", "worker_b", "worker_c"):
        cfg = model_map.get(role) or {}
        base_url = cfg.get("base_url")
        model = cfg.get("model")
        if not base_url or not model:
            continue
        pair = (base_url, model)
        if ready_models is not None and pair not in ready_models:
            continue
        if pair in UNAVAILABLE_MODELS:
            continue
        if pair in seen:
            continue
        seen.add(pair)
        targets.append(pair)
        if len(targets) >= max_workers:
            break
    if not targets:
        return

    if bus and run_id:
        try:
            await bus.emit(
                run_id,
                "worker_warmup",
                {"count": len(targets), "models": [m for _, m in targets]},
            )
        except Exception:
            pass

    async def _ping(base_url: str, model: str) -> None:
        try:
            await lm_client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": "warmup"}],
                temperature=0.0,
                max_tokens=1,
                base_url=base_url,
                run_state=None,
                allow_minimal_retry=False,
            )
        except Exception as exc:
            if run_state:
                run_state.add_dev_trace(
                    "Worker warmup failed",
                    {"model": model, "base_url": base_url, "error": str(exc)},
                )

    await asyncio.gather(*(_ping(base_url, model) for base_url, model in targets), return_exceptions=True)


async def generate_step_prompt(
    lm_client: LMStudioClient,
    orch_model: str,
    question: str,
    step: PlanStep,
    artifacts: List[Artifact],
    answer_guidance: str = "",
    toolbox_hint: str = "",
) -> str:
    context_parts: List[str] = []
    recent_artifacts = [art for art in artifacts if art.artifact_type != "validation"][-5:]
    for art in recent_artifacts:
        text = (art.content_text or "").strip()
        if not text and art.content_json:
            data = art.content_json
            if art.artifact_type in ("evidence", "ledger"):
                sources = []
                for src in data.get("sources", []):
                    sources.append(
                        {
                            "url": src.get("url"),
                            "title": src.get("title"),
                            "publisher": src.get("publisher"),
                            "date_published": src.get("date_published"),
                            "snippet": src.get("snippet"),
                        }
                    )
                slim = {
                    "sources": sources,
                    "claims": data.get("claims", []),
                }
                tool_results = data.get("tool_results") or []
                if tool_results:
                    slim["tool_results"] = strip_data_urls(tool_results, allow_plot=False)
                if art.artifact_type == "ledger":
                    slim["conflicts"] = data.get("conflicts", [])
                else:
                    slim["conflicts_found"] = data.get("conflicts_found", False)
                text = json.dumps(slim, ensure_ascii=True)
            else:
                text = json.dumps(data, ensure_ascii=True)
        if len(text) > 1200:
            text = text[:1200] + "..."
        if text:
            context_parts.append(f"{art.key} ({art.artifact_type}): {text}")
    context = "\n".join(context_parts) if context_parts else "None"
    prompt = (
        f"User question: {question}\n"
        f"Step: {step.step_id} - {step.name} ({step.type})\n"
        f"Acceptance: {step.acceptance_criteria}\n"
        f"Recent artifacts:\n{context}\n"
        f"Produce the needed output for this step."
    )
    prompt_hint = ""
    if isinstance(step.inputs, dict):
        hint_val = step.inputs.get("prompt_hint") or step.inputs.get("prompt") or step.inputs.get("focus")
        if hint_val:
            prompt_hint = str(hint_val).strip()
    if prompt_hint:
        if len(prompt_hint) > 800:
            prompt_hint = prompt_hint[:800].rstrip() + "..."
        prompt += f"\nExecutor guidance: {prompt_hint}"
    if toolbox_hint:
        prompt += f"\nTooling you can request (tool_requests[]): {toolbox_hint}"
    # For most steps this generic prompt suffices; for research we include instruction.
    if step.type == "research":
        prompt += (
            "\nReturn JSON with queries (3-6 specific Tavily web searches), optional time_range/topic, tool_requests[] if needed. "
            "Queries are executed by the backend; include variations and recency hints when relevant. Do not provide sources, claims, or a final answer."
        )
        prompt += f"\n{agents.SEARCH_GUIDE.strip()}"
    if step.type == "draft":
        prompt += "\nReturn the final answer only (plain text). Do not output JSON or tool-call markup."
        prompt += "\nDo not include citations or a Sources section; sources are shown separately."
    if step.type == "finalize":
        requires_verifier = False
        if isinstance(step.inputs, dict):
            requires_verifier = bool(step.inputs.get("requires_verifier"))
        prompt += "\nReturn JSON only with tool_requests (include exactly one finalize_answer tool call)."
        if requires_verifier:
            prompt += "\nOnly call finalize_answer if the verifier verdict is PASS; otherwise return tool_requests as an empty list."
    if answer_guidance and step.type in {"draft", "analysis", "merge", "verify"}:
        prompt += f"\nAnswer guidance: {answer_guidance}"
    return prompt


def merge_evidence_artifacts(artifacts: List[Artifact]) -> Dict[str, Any]:
    sources_by_url: Dict[str, dict] = {}
    claims: List[dict] = []
    tool_results: List[dict] = []
    tool_requests: List[dict] = []
    for art in artifacts:
        if art.artifact_type != "evidence":
            continue
        data = art.content_json or {}
        for src in data.get("sources", []):
            url = src.get("url")
            if url and url not in sources_by_url:
                sources_by_url[url] = src
        for claim in data.get("claims", []):
            claims.append(claim)
        tool_results.extend(data.get("tool_results") or [])
        tool_requests.extend(data.get("tool_requests") or [])
    conflicts = [c for c in claims if c.get("conflict")]
    return {
        "sources": list(sources_by_url.values()),
        "claims": claims,
        "conflicts": conflicts,
        "tool_results": tool_results,
        "tool_requests": tool_requests,
    }


def _clip_payload_text(value: Any, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True)
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def _extract_draft_text(output: Any) -> str:
    if isinstance(output, dict):
        for key in ("draft", "answer", "final_answer", "final", "text", "content"):
            val = output.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    if isinstance(output, str) and output.strip():
        return output.strip()
    return ""


def _latest_artifact(artifacts: List[Artifact], artifact_type: str) -> Optional[Artifact]:
    for art in reversed(artifacts):
        if art.artifact_type == artifact_type:
            return art
    return None


def _approved_final_text(artifacts: List[Artifact]) -> Tuple[str, Optional[str]]:
    verifier_art = _latest_artifact(artifacts, "verifier")
    verdict = None
    revised = None
    if verifier_art and isinstance(verifier_art.content_json, dict):
        verdict = verifier_art.content_json.get("verdict")
        revised = verifier_art.content_json.get("revised_answer")
    draft_art = _latest_artifact(artifacts, "draft")
    draft_text = (draft_art.content_text or "").strip() if draft_art else ""
    if verdict == "PASS" and revised and str(revised).strip():
        return str(revised).strip(), verdict
    if verdict == "PASS" and draft_text:
        return draft_text, verdict
    if verdict is None:
        return draft_text, verdict
    return "", verdict


def _output_has_numbers(output: Any) -> bool:
    snippet = _clip_payload_text(output, 800)
    return any(ch.isdigit() for ch in snippet)


def compute_validation_slots(worker_budget: Dict[str, Any], strict_mode: bool) -> int:
    max_parallel = int(worker_budget.get("max_parallel") or 1)
    if max_parallel <= 1:
        return 1
    if worker_budget.get("ram_pressure") or worker_budget.get("vram_pressure"):
        return 1
    if strict_mode:
        return min(4, max_parallel)
    return min(3, max_parallel)


def build_step_validation_requests(
    step: PlanStep,
    question: str,
    output: Any,
    artifacts: List[Artifact],
    max_checks: int,
) -> List[dict]:
    if max_checks <= 0:
        return []
    step_summary = {
        "id": step.step_id,
        "name": step.name,
        "type": step.type,
        "agent_profile": step.agent_profile,
        "acceptance": step.acceptance_criteria,
    }
    output_text = _clip_payload_text(output, VALIDATION_PROMPT_MAX_CHARS)
    critic_prompt = (
        "Review the step output against acceptance criteria. "
        "Return JSON only with issues[] and suggested_fix_steps[].\n"
        f"Question: {question}\nStep: {json.dumps(step_summary, ensure_ascii=True)}\nOutput: {output_text}"
    )
    summarizer_prompt = (
        "Summarize the step output for a quick sanity check. "
        "Return JSON only: {\"activity_lines\": [...], \"memory_notes\": [...], \"candidate_memory\": []}.\n"
        f"Step: {json.dumps(step_summary, ensure_ascii=True)}\nOutput: {output_text}"
    )
    requests: List[dict] = []
    needs_verifier = step.type in ("draft", "merge")
    if needs_verifier:
        ledger = merge_evidence_artifacts(artifacts)
        draft_text = _extract_draft_text(output) or output_text
        verifier_prompt = (
            f"Question: {question}\nDraft: {draft_text}\nClaims ledger: {_clip_payload_text(ledger, 2800)}\n"
            "Return JSON verdict: PASS/NEEDS_REVISION, issues[], revised_answer?, extra_steps[]."
        )
        requests.append(
            {
                "tool": "model_call",
                "profile": "Verifier",
                "prompt": verifier_prompt,
                "temperature": 0.0,
                "max_tokens": 420,
            }
        )
    requests.append(
        {
            "tool": "model_call",
            "profile": "Critic",
            "prompt": critic_prompt,
            "temperature": 0.0,
            "max_tokens": 320,
        }
    )
    requests.append(
        {
            "tool": "model_call",
            "profile": "Summarizer",
            "prompt": summarizer_prompt,
            "temperature": 0.2,
            "max_tokens": 220,
        }
    )
    if _output_has_numbers(output):
        math_prompt = (
            "Verify any numeric reasoning in the output. "
            "Return JSON with steps and result.\n"
            f"Output: {output_text}"
        )
        requests.append(
            {
                "tool": "model_call",
                "profile": "Math",
                "prompt": math_prompt,
                "temperature": 0.0,
                "max_tokens": 240,
            }
        )
    if len(requests) > max_checks:
        requests = requests[:max_checks]
    return requests


def _parse_json_maybe(text: str) -> Optional[Any]:
    raw = text.strip()
    if not raw or raw[0] not in "{[":
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def summarize_validation_results(tool_results: List[dict]) -> str:
    summaries: List[str] = []
    for res in tool_results or []:
        if not isinstance(res, dict):
            continue
        if res.get("status") not in ("ok", None, ""):
            continue
        profile = res.get("profile") or res.get("tool") or "checker"
        result = res.get("result")
        parsed = None
        if isinstance(result, dict):
            parsed = result
        elif isinstance(result, str):
            parsed = _parse_json_maybe(result)
        if isinstance(parsed, dict):
            if "verdict" in parsed:
                issues = parsed.get("issues") or []
                summaries.append(f"{profile}: {parsed.get('verdict')} ({len(issues)} issues)")
                continue
            if "issues" in parsed:
                issues = parsed.get("issues") or []
                summaries.append(f"{profile}: {len(issues)} issues")
                continue
            if "activity_lines" in parsed:
                lines = parsed.get("activity_lines") or []
                if lines:
                    summaries.append(f"{profile}: {str(lines[0])[:80]}")
                    continue
        if isinstance(result, str):
            snippet = result.strip().replace("\n", " ")
            if snippet:
                summaries.append(f"{profile}: {snippet[:80]}")
                continue
    summary = "; ".join(summaries)
    if len(summary) > VALIDATION_SUMMARY_MAX_CHARS:
        summary = summary[:VALIDATION_SUMMARY_MAX_CHARS].rstrip() + "..."
    return summary


def extract_validation_todos(tool_results: List[dict]) -> List[str]:
    todos: List[str] = []
    issues: List[str] = []
    seen: Set[str] = set()

    def _add_item(target: List[str], value: Any) -> None:
        text = _clip_narration_text(value, 90)
        if not text or text in seen:
            return
        seen.add(text)
        target.append(text)

    def _extract_text(item: Any) -> Optional[str]:
        if isinstance(item, dict):
            for key in ("name", "description", "step", "fix", "issue", "note"):
                if key in item and item.get(key):
                    return str(item.get(key))
            return json.dumps(item, ensure_ascii=True)
        return str(item) if item is not None else None

    for res in tool_results or []:
        if not isinstance(res, dict):
            continue
        result = res.get("result")
        parsed: Optional[dict] = None
        if isinstance(result, dict):
            parsed = result
        elif isinstance(result, str):
            parsed = _parse_json_maybe(result)
        if not isinstance(parsed, dict):
            continue
        for key in ("suggested_fix_steps", "extra_steps"):
            items = parsed.get(key) or []
            if isinstance(items, list):
                for item in items:
                    text = _extract_text(item)
                    if text:
                        _add_item(todos, text)
        if not todos:
            items = parsed.get("issues") or []
            if isinstance(items, list):
                for item in items:
                    text = _extract_text(item)
                    if text:
                        _add_item(issues, text)

    if todos:
        return todos
    return issues


def normalize_verifier_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    verdict = parsed.get("verdict")
    if not verdict:
        if "PASS/NEEDS_REVISION" in parsed:
            verdict = parsed.get("PASS/NEEDS_REVISION")
        elif "PASS" in parsed:
            verdict = parsed.get("PASS")
    if isinstance(verdict, bool):
        verdict = "PASS" if verdict else "NEEDS_REVISION"
    if isinstance(verdict, str):
        verdict_norm = verdict.strip().upper().replace(" ", "_")
        if "NEEDS" in verdict_norm:
            verdict = "NEEDS_REVISION"
        else:
            verdict = "PASS"
    issues = parsed.get("issues") or []
    revised_answer = parsed.get("revised_answer")
    if revised_answer is None:
        revised_answer = parsed.get("revised_answer?") or parsed.get("revised")
    extra_steps = parsed.get("extra_steps") or []
    return {
        **parsed,
        "verdict": verdict or "PASS",
        "issues": issues,
        "revised_answer": revised_answer,
        "extra_steps": extra_steps,
    }


async def run_step_double_checks(
    lm_client: LMStudioClient,
    model_map: Dict[str, Dict[str, str]],
    step: PlanStep,
    question: str,
    output: Any,
    artifacts: List[Artifact],
    worker_budget: Dict[str, Any],
    strict_mode: bool,
    run_id: str,
    bus: "EventBus",
    db: Optional[Database] = None,
    conversation_id: Optional[str] = None,
    upload_dir: Optional[Path] = None,
    run_state: Optional[RunState] = None,
) -> Tuple[List[dict], List[dict], str]:
    max_checks = compute_validation_slots(worker_budget, strict_mode)
    tool_requests = build_step_validation_requests(step, question, output, artifacts, max_checks)
    if not tool_requests:
        return [], [], ""
    await bus.emit(
        run_id,
        "tool_request",
        {"step": step.step_id, "requests": tool_requests, "context": "double_check"},
    )
    queue_narration(
        lm_client,
        model_map,
        run_state,
        bus,
        run_id,
        question,
        "tool_request",
        {"step": step.step_id, "requests": tool_requests, "context": "double_check"},
    )
    tool_results = await resolve_tool_requests(
        tool_requests,
        upload_dir=upload_dir,
        db=db,
        conversation_id=conversation_id,
        lm_client=lm_client,
        model_map=model_map,
        run_id=run_id,
        bus=bus,
        step_id=step.step_id,
        run_state=run_state,
    )
    if tool_results:
        safe_results = strip_data_urls(tool_results, allow_plot=True)
        await bus.emit(
            run_id,
            "tool_result",
            {"step": step.step_id, "results": safe_results, "context": "double_check"},
        )
        queue_narration(
            lm_client,
            model_map,
            run_state,
            bus,
            run_id,
            question,
            "tool_result",
            {"step": step.step_id, "results": safe_results, "context": "double_check"},
        )
    if tool_results and run_state:
        todos = extract_validation_todos(tool_results)
        if todos:
            todo_line = "Possible follow-ups: " + "; ".join(todos[:3])
            await maybe_emit_work_log(
                run_state,
                bus,
                run_id,
                f"todo_step_{step.step_id}",
                todo_line,
            )
    summary = summarize_validation_results(tool_results)
    return tool_requests, tool_results, summary


async def evaluate_control(
    lm_client: LMStudioClient,
    orch_endpoint: Dict[str, str],
    step: PlanStep,
    step_output: Dict[str, Any],
    validation_summary: str = "",
    run_state: Optional[RunState] = None,
) -> ControlCommand:
    prompt = (
        "Evaluate the step output against acceptance criteria. "
        "If fine, respond {\"control\":\"CONTINUE\"}. "
        "Otherwise choose: BACKTRACK, RERUN_STEP, ADD_STEPS, STOP. "
        "If evidence is missing, prefer ADD_STEPS with concrete new steps and dependencies. "
        "You may also include new_constraints to update plan guidance without interrupting the run. "
        f"Step: {step.model_dump()}\nOutput: {json.dumps(step_output)[:1500]}"
    )
    if validation_summary:
        prompt += f"\nDouble-check notes: {validation_summary[:600]}"
    try:
        if run_state and run_state.model_manager:
            content = await run_worker(
                lm_client,
                "Orchestrator",
                {},
                prompt,
                temperature=0.1,
                max_tokens=300,
                run_state=run_state,
                model_manager=run_state.model_manager,
                system_prompt_override=agents.MICROMANAGER_SYSTEM,
                context="evaluate_control",
            )
        else:
            resp = await lm_client.chat_completion(
                model=orch_endpoint["model"],
                messages=[{"role": "system", "content": agents.MICROMANAGER_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
                base_url=orch_endpoint["base_url"],
                run_state=run_state,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(
            content,
            lm_client,
            orch_endpoint["model"],
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    if not parsed:
        parsed = {"control": "CONTINUE"}
    allowed = {"CONTINUE", "BACKTRACK", "RERUN_STEP", "ADD_STEPS", "STOP"}
    control_val = str(
        parsed.get("control")
        or parsed.get("action")
        or parsed.get("decision")
        or parsed.get("command")
        or "CONTINUE"
    ).upper()
    if control_val not in allowed:
        control_val = "CONTINUE"
    parsed["control"] = control_val
    if control_val == "BACKTRACK" and not parsed.get("to_step"):
        deps = step.depends_on or []
        parsed["to_step"] = max(deps) if deps else max(1, step.step_id - 1)
    if control_val == "RERUN_STEP" and not parsed.get("step_id"):
        parsed["step_id"] = step.step_id
    try:
        return ControlCommand(**{k: v for k, v in parsed.items() if k in ControlCommand.model_fields})
    except Exception:
        return ControlCommand(control="CONTINUE")


async def evaluate_control_fast(
    lm_client: LMStudioClient,
    fast_endpoint: Dict[str, str],
    step: PlanStep,
    step_output: Dict[str, Any],
    validation_summary: str = "",
    run_state: Optional[RunState] = None,
) -> Tuple[ControlCommand, bool]:
    """
    Lightweight guardrail using the faster 4B endpoint.
    Returns (control_command, escalate) where escalate means defer to the OSS orchestrator.
    """
    prompt = (
        "Quick gate the step output. If it clearly meets acceptance criteria, CONTINUE. "
        "If minor issues, RERUN_STEP. For missing dependencies or wrong direction, BACKTRACK. "
        "If unsure, respond ESCALATE to punt to the main orchestrator. "
        "Return JSON only."
        f"Step: {step.model_dump()}\nOutput: {json.dumps(step_output)[:1200]}"
    )
    if validation_summary:
        prompt += f"\nDouble-check notes: {validation_summary[:400]}"
    try:
        if run_state and run_state.model_manager:
            content = await run_worker(
                lm_client,
                "Executor",
                {},
                prompt,
                temperature=0.0,
                max_tokens=200,
                run_state=run_state,
                model_manager=run_state.model_manager,
                system_prompt_override=agents.EXECUTOR_SYSTEM,
                context="evaluate_control_fast",
            )
        else:
            resp = await lm_client.chat_completion(
                model=fast_endpoint["model"],
                messages=[{"role": "system", "content": agents.EXECUTOR_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                base_url=fast_endpoint["base_url"],
                run_state=run_state,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(
            content,
            lm_client,
            fast_endpoint["model"],
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
    except Exception:
        parsed = None
    if not parsed:
        parsed = {"control": "CONTINUE"}
    allowed = {"CONTINUE", "BACKTRACK", "RERUN_STEP", "ADD_STEPS", "STOP"}
    control_val = parsed.get("control", "CONTINUE")
    if control_val not in allowed:
        control_val = "CONTINUE"
    # Only trust the fast gate for a green light; any other signal escalates to the main orchestrator.
    escalate = control_val != "CONTINUE"
    if escalate:
        control_val = "CONTINUE"
    parsed["control"] = control_val
    try:
        cmd = ControlCommand(**{k: v for k, v in parsed.items() if k in ControlCommand.model_fields})
    except Exception:
        cmd = ControlCommand(control="CONTINUE")
    return cmd, escalate


async def allocate_ready_steps(
    lm_client: LMStudioClient,
    executor_endpoint: Dict[str, str],
    ready_steps: List[PlanStep],
    artifacts: List[Artifact],
    running_count: int,
    target_slots: int,
    question: str = "",
    resource_budget: Optional[Dict[str, Any]] = None,
    running_steps: Optional[List[PlanStep]] = None,
    run_state: Optional[RunState] = None,
) -> AllocationDecision:
    """Ask the executor to allocate ready steps up to available slot capacity."""
    if not ready_steps:
        return AllocationDecision(start_ids=[])
    capacity = max(0, target_slots - running_count)
    if capacity <= 0:
        return AllocationDecision(start_ids=[])
    ready_ids = [s.step_id for s in ready_steps]
    default_ids = ready_ids[:capacity]
    decision = AllocationDecision(start_ids=default_ids, queue_ids=ready_ids[capacity:])
    if not executor_endpoint or not executor_endpoint.get("model"):
        return decision
    ready_summary = [
        {
            "id": s.step_id,
            "name": s.name,
            "type": s.type,
            "agent_profile": s.agent_profile,
            "depends_on": s.depends_on,
        }
        for s in ready_steps[:ALLOCATOR_MAX_READY]
    ]
    running_summary = []
    if running_steps:
        running_summary = [
            {
                "id": s.step_id,
                "name": s.name,
                "type": s.type,
                "agent_profile": s.agent_profile,
            }
            for s in running_steps[:ALLOCATOR_MAX_RUNNING]
        ]
    prompt = (
        "You are the executor. Decide which ready steps to start now to keep parallel work moving and avoid bottlenecks. "
        "Return JSON only: {\"start_ids\": [...], \"target_slots\": N, \"queue_ids\": [...], \"note\": \"...\"}. "
        "The note is user-facing: keep it short, plainspoken, and free of internal jargon. "
        "Use only IDs from ready_ids. Keep start_ids length <= capacity. "
        "If resource_budget.elastic_parallel is true and there is headroom, you may raise target_slots to add parallel workers.\n"
        f"Question: {question}\nCapacity: {capacity}\nCurrent target slots: {target_slots}\n"
        f"Ready steps: {json.dumps(ready_summary, ensure_ascii=True)}\n"
        f"Running steps: {json.dumps(running_summary, ensure_ascii=True)}\n"
        f"Resource budget: {json.dumps(resource_budget or {}, ensure_ascii=True)}\n"
        f"Ready ids: {ready_ids}"
    )
    try:
        if run_state and run_state.model_manager:
            content = await run_worker(
                lm_client,
                "Executor",
                {},
                prompt,
                temperature=0.0,
                max_tokens=220,
                run_state=run_state,
                model_manager=run_state.model_manager,
                system_prompt_override=agents.EXECUTOR_SYSTEM,
                context="allocator",
            )
        else:
            resp = await lm_client.chat_completion(
                model=executor_endpoint["model"],
                messages=[{"role": "system", "content": agents.EXECUTOR_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=220,
                base_url=executor_endpoint.get("base_url"),
                run_state=run_state,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(
            content,
            lm_client,
            executor_endpoint["model"],
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        return decision
    start_ids_raw = parsed.get("start_ids") or parsed.get("start") or parsed.get("start_steps") or []
    if not isinstance(start_ids_raw, list):
        start_ids_raw = []
    ready_set = set(ready_ids)
    start_ids: List[int] = []
    for item in start_ids_raw:
        try:
            sid = int(item)
        except Exception:
            continue
        if sid in ready_set and sid not in start_ids:
            start_ids.append(sid)
        if len(start_ids) >= capacity:
            break
    if not start_ids:
        start_ids = list(default_ids)
    if len(start_ids) < capacity:
        for sid in ready_ids:
            if sid in start_ids:
                continue
            start_ids.append(sid)
            if len(start_ids) >= capacity:
                break
    queue_raw = parsed.get("queue_ids") or parsed.get("queue") or parsed.get("queued") or []
    queue_ids: List[int] = []
    if isinstance(queue_raw, list):
        for item in queue_raw:
            try:
                sid = int(item)
            except Exception:
                continue
            if sid in ready_set and sid not in start_ids and sid not in queue_ids:
                queue_ids.append(sid)
    if not queue_ids:
        queue_ids = [sid for sid in ready_ids if sid not in start_ids]
    target_slots_raw = parsed.get("target_slots") or parsed.get("parallel_slots")
    target_slots_val: Optional[int] = None
    if target_slots_raw is not None:
        try:
            target_slots_val = int(target_slots_raw)
        except Exception:
            target_slots_val = None
    decision = AllocationDecision(
        start_ids=start_ids,
        queue_ids=queue_ids,
        target_slots=target_slots_val,
        note=str(parsed.get("note") or "").strip(),
        used_executor=True,
    )
    return decision


async def build_executor_brief(
    lm_client: LMStudioClient,
    executor_endpoint: Dict[str, str],
    question: str,
    step_plan: StepPlan,
    target_slots: int,
    run_state: Optional[RunState] = None,
) -> Optional[Dict[str, Any]]:
    if not executor_endpoint or not executor_endpoint.get("model"):
        return None
    steps_summary = [
        {
            "id": s.step_id,
            "name": s.name,
            "type": s.type,
            "depends_on": s.depends_on,
            "agent_profile": s.agent_profile,
        }
        for s in step_plan.steps
    ]
    prompt = (
        "You are the executor. Summarize how you will run this plan."
        " Return JSON only: {\"note\": \"...\", \"focus_steps\": [ids], \"parallel_slots\": N, \"risks\": [..]}."
        " The note is user-facing: keep it short, plainspoken, and free of internal jargon."
        f"\nQuestion: {question}\nParallel slots: {target_slots}\nSteps: {json.dumps(steps_summary)[:1500]}"
    )
    try:
        if run_state and run_state.model_manager:
            content = await run_worker(
                lm_client,
                "Executor",
                {},
                prompt,
                temperature=0.0,
                max_tokens=220,
                run_state=run_state,
                model_manager=run_state.model_manager,
                system_prompt_override=agents.EXECUTOR_SYSTEM,
                context="executor_brief",
            )
        else:
            resp = await lm_client.chat_completion(
                model=executor_endpoint["model"],
                messages=[{"role": "system", "content": agents.EXECUTOR_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=220,
                base_url=executor_endpoint["base_url"],
                run_state=run_state,
            )
            content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(
            content,
            lm_client,
            executor_endpoint["model"],
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def coerce_tavily_response(resp: Any) -> Dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, list):
        return {"results": resp}
    if resp is None:
        return {"error": "invalid_response", "detail": "empty response"}
    return {"error": "invalid_response", "detail": str(resp)}


def format_tavily_error(resp: Any) -> str:
    resp = coerce_tavily_response(resp)
    if not resp:
        return "tavily_error"
    message = str(resp.get("error") or "tavily_error")
    detail = resp.get("detail")
    if isinstance(detail, dict):
        detail_msg = (
            detail.get("detail", {}).get("error")
            or detail.get("error")
            or detail.get("message")
        )
        detail = detail_msg or json.dumps(detail)
    if detail:
        message = f"{message}: {detail}"
    status = resp.get("status_code")
    if status:
        message = f"{status} {message}"
    return message


async def synthesize_evidence_from_sources(
    lm_client: LMStudioClient,
    model_map: Dict[str, Dict[str, str]],
    question: str,
    sources: List[dict],
    run_id: Optional[str] = None,
    bus: Optional["EventBus"] = None,
    step_id: Optional[int] = None,
    run_state: Optional[RunState] = None,
) -> Dict[str, Any]:
    if not sources:
        return {"claims": [], "gaps": [], "conflicts_found": False}
    compact = compact_sources_for_synthesis(sources, max_chars=700)
    prompt = (
        f"Question: {question}\n"
        f"Sources: {json.dumps(compact)}\n"
        "Return JSON only: {\"claims\": [{\"claim\": \"...\", \"urls\": [\"...\"]}], "
        "\"gaps\": [\"...\"], \"conflicts_found\": false}."
    )
    try:
        raw = await run_worker(
            lm_client,
            "EvidenceSynth",
            model_map,
            prompt,
            temperature=0.2,
            max_tokens=600,
            run_id=run_id,
            bus=bus,
            step_id=step_id,
            context="evidence_synth",
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
    except Exception:
        return {
            "claims": [],
            "gaps": ["Evidence synthesis failed; review sources directly."],
            "conflicts_found": False,
        }
    fixer_model = (
        (model_map.get("summarizer") or {}).get("model")
        or (model_map.get("worker") or {}).get("model")
        or (model_map.get("orch") or {}).get("model")
        or ""
    )
    parsed = await safe_json_parse(
        raw,
        lm_client,
        fixer_model,
        run_state=run_state,
        model_manager=run_state.model_manager if run_state else None,
    )
    if not isinstance(parsed, dict):
        return {"claims": [], "gaps": [], "conflicts_found": False}
    claims = parsed.get("claims")
    gaps = parsed.get("gaps")
    conflicts = parsed.get("conflicts_found")
    if not isinstance(claims, list):
        claims = []
    if not isinstance(gaps, list):
        gaps = []
    return {
        "claims": claims,
        "gaps": gaps,
        "conflicts_found": bool(conflicts),
    }


SEARCH_STATS_FIELDS = ("queries", "skipped_queries", "duplicate_sources", "new_sources", "results")


def normalize_search_key(text: str) -> str:
    cleaned = str(text or "").strip().lower()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[-_/]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .,:;!?\"'`")
    return cleaned


def normalize_source_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw.rstrip("/")
    if not parsed.scheme or not parsed.netloc:
        return raw.rstrip("/")
    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()
    path = parsed.path or ""
    if path.endswith("/") and path != "/":
        path = path[:-1]
    normalized = f"{scheme}://{host}{path}"
    if parsed.query:
        normalized = f"{normalized}?{parsed.query}"
    return normalized


def _empty_search_stats() -> Dict[str, int]:
    return {field: 0 for field in SEARCH_STATS_FIELDS}


def _merge_search_stats(base: Dict[str, int], add: Dict[str, int]) -> Dict[str, int]:
    for field in SEARCH_STATS_FIELDS:
        try:
            base[field] = int(base.get(field, 0)) + int(add.get(field, 0))
        except Exception:
            base[field] = int(base.get(field, 0))
    return base


async def run_tavily_queries(
    run_id: str,
    step: PlanStep,
    queries: List[str],
    search_depth: str,
    per_query_max: int,
    tavily: TavilyClient,
    db: Database,
    bus: EventBus,
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
    mode: Optional[str] = None,
    executed: Optional[Set[str]] = None,
    executed_order: Optional[List[str]] = None,
    *,
    question: str = "",
    lm_client: Optional[LMStudioClient] = None,
    model_map: Optional[Dict[str, Dict[str, str]]] = None,
    run_state: Optional[RunState] = None,
) -> Tuple[List[dict], List[str], Dict[str, int]]:
    gathered_sources: List[dict] = []
    tavily_errors: List[str] = []
    stats = _empty_search_stats()
    executed_set = executed if executed is not None else set()
    seen_urls: Set[str] = set()
    for query in queries:
        q = str(query or "").strip()
        if not q:
            continue
        key = normalize_search_key(q)
        if not key:
            continue
        if key in executed_set:
            stats["skipped_queries"] += 1
            payload = {"step": step.step_id, "query": q, "reason": "duplicate_query"}
            await bus.emit(run_id, "search_skipped", payload)
            if lm_client and model_map and run_state:
                queue_narration(
                    lm_client,
                    model_map,
                    run_state,
                    bus,
                    run_id,
                    question,
                    "search_skipped",
                    dict(payload),
                )
            continue
        executed_set.add(key)
        stats["queries"] += 1
        if executed_order is not None:
            executed_order.append(q)
        payload = {"step": step.step_id, "query": q}
        if mode:
            payload["mode"] = mode
        try:
            search_resp = await asyncio.wait_for(
                tavily.search(
                    query=q,
                    search_depth=search_depth,
                    max_results=per_query_max,
                    topic=topic,
                    time_range=time_range,
                ),
                timeout=TAVILY_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            search_resp = {"error": "timeout"}
        search_resp = coerce_tavily_response(search_resp)
        results = search_resp.get("results") or []
        if not isinstance(results, list):
            results = []
        result_count = len(results)
        stats["results"] += result_count
        urls = [res.get("url") for res in results if isinstance(res, dict) and res.get("url")]
        if urls:
            payload["urls"] = urls[: min(5, per_query_max)]
        new_sources = 0
        duplicate_sources = 0
        if not search_resp.get("error"):
            for res in results[:per_query_max]:
                if not isinstance(res, dict):
                    continue
                url_raw = res.get("url") or ""
                url_key = normalize_source_url(url_raw)
                if url_key and url_key in seen_urls:
                    duplicate_sources += 1
                    continue
                if url_key:
                    seen_urls.add(url_key)
                src = {
                    "url": url_raw,
                    "title": res.get("title"),
                    "publisher": res.get("source"),
                    "date_published": res.get("published_date"),
                    "snippet": res.get("content", "")[:400],
                    "extracted_text": res.get("content", ""),
                }
                gathered_sources.append(src)
                new_sources += 1
                await db.add_source(
                    run_id,
                    f"Step{step.step_id}",
                    src["url"] or "",
                    src["title"] or "",
                    src["publisher"] or "",
                    src["date_published"] or "",
                    src["snippet"] or "",
                    src["extracted_text"] or "",
                )
        stats["new_sources"] += new_sources
        stats["duplicate_sources"] += duplicate_sources
        payload["result_count"] = result_count
        payload["new_sources"] = new_sources
        payload["duplicate_sources"] = duplicate_sources
        if lm_client and model_map and run_state:
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "tavily_search",
                dict(payload),
            )
        await bus.emit(run_id, "tavily_search", payload)
        await db.add_search(run_id, f"Step{step.step_id}", q, search_depth, per_query_max, search_resp)
        if search_resp.get("error"):
            error_msg = format_tavily_error(search_resp)
            await bus.emit(
                run_id,
                "tavily_error",
                {"step": step.step_id, "message": error_msg},
            )
            if lm_client and model_map and run_state:
                queue_narration(
                    lm_client,
                    model_map,
                    run_state,
                    bus,
                    run_id,
                    question,
                    "tavily_error",
                    {"step": step.step_id, "message": error_msg},
                    tone="warn",
                )
            tavily_errors.append(error_msg)
            continue
    return gathered_sources, tavily_errors, stats


async def execute_research_step(
    run_id: str,
    question: str,
    step: PlanStep,
    prompt: str,
    decision: RouterDecision,
    search_depth_mode: str,
    depth_profile: dict,
    lm_client: LMStudioClient,
    tavily: TavilyClient,
    db: Database,
    bus: EventBus,
    model_map: Dict[str, Dict[str, str]],
    conversation_id: Optional[str] = None,
    upload_dir: Optional[Path] = None,
    run_state: Optional[RunState] = None,
) -> Tuple[Dict[str, Any], List[Artifact], str]:
    offline_web = bool(run_state and not run_state.can_web)
    raw = None
    try:
        raw = await run_worker(
            lm_client,
            step.agent_profile,
            model_map,
            prompt,
            temperature=0.4,
            max_tokens=700,
            run_id=run_id,
            bus=bus,
            step_id=step.step_id,
            context="research",
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
    except Exception as exc:
        await bus.emit(
            run_id,
            "client_note",
            {"note": f"Research model failed; using fallback queries. ({type(exc).__name__})"},
        )
    fixer_model = (
        (model_map.get("worker") or {}).get("model")
        or (model_map.get("summarizer") or {}).get("model")
        or (model_map.get("orch") or {}).get("model")
        or ""
    )
    parsed_raw = (
        await safe_json_parse(
            raw,
            lm_client,
            fixer_model,
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        if raw
        else None
    )
    parsed, coerced = normalize_research_payload(parsed_raw)
    if coerced and run_state:
        run_state.add_dev_trace(
            "Normalized research output",
            {"step": step.step_id, "profile": step.agent_profile},
        )
    queries = parsed.get("queries", [])
    inputs = step.inputs if isinstance(step.inputs, dict) else {}
    requested_time_range = normalize_time_range(parsed.get("time_range"))
    if not requested_time_range:
        requested_time_range = normalize_time_range(inputs.get("time_range"))
    if not requested_time_range:
        requested_time_range = infer_time_range(question)
    requested_topic = parsed.get("topic") or inputs.get("topic") or decision.topic
    if requested_topic not in ALLOWED_TOPICS:
        requested_topic = decision.topic or "general"
    fallback_used = False
    forced_queries: List[str] = []
    input_queries = inputs.get("queries")
    if isinstance(input_queries, list):
        forced_queries = [str(q).strip() for q in input_queries if str(q).strip()]
    input_query = inputs.get("query")
    if not forced_queries and input_query:
        forced_queries = [str(input_query).strip()]
    if forced_queries:
        queries = forced_queries
        parsed["queries"] = queries
    if not queries:
        queries = build_fallback_queries(
            question,
            prompt,
            topic=requested_topic,
            time_range=requested_time_range,
        )
        parsed["queries"] = queries
        fallback_used = True
    artifacts: List[Artifact] = []
    use_web = decision.needs_web
    if "use_web" in inputs:
        use_web = bool(inputs.get("use_web", decision.needs_web))
    # If the router under-called web, backstop with a heuristic so Tavily still runs for data questions.
    if not use_web and guess_needs_web(question):
        use_web = True
    if run_state and not run_state.can_web:
        use_web = False
    # Honor explicit research queries even when the router was conservative.
    profile = (step.agent_profile or "").strip()
    if not use_web and tavily.enabled:
        if (queries and not fallback_used) or profile in ("ResearchRecency", "ResearchAdversarial"):
            use_web = True
    search_depth = reasoning_to_search_depth(decision.reasoning_level, search_depth_mode, depth_profile)
    override_depth = str(inputs.get("search_depth") or "").strip().lower()
    if override_depth in ("basic", "advanced"):
        search_depth = override_depth
    search_budget = depth_profile.get("tool_budget", {}).get("tavily_search", decision.max_results or 6)
    extract_budget = depth_profile.get("tool_budget", {}).get("tavily_extract", 6)
    extract_depth = decision.extract_depth if decision else "basic"
    override_extract = str(inputs.get("extract_depth") or "").strip().lower()
    if override_extract in ("basic", "advanced"):
        extract_depth = override_extract
    gathered_sources: List[dict] = []
    executed_queries: Set[str] = set()
    executed_order: List[str] = []
    tavily_errors: List[str] = []
    search_stats = _empty_search_stats()
    search_stats = _empty_search_stats()
    if offline_web:
        tavily_errors.append("Web browsing unavailable")
    tool_requests = parsed.get("tool_requests", [])
    if not isinstance(tool_requests, list):
        tool_requests = []
    if use_web:
        override_max = inputs.get("max_results")
        max_results = decision.max_results if decision else 6
        if override_max is not None:
            try:
                max_results = max(1, int(override_max))
            except Exception:
                pass
        per_query_max = max(3, min(search_budget, max_results))
        if not tavily.enabled:
            await bus.emit(run_id, "tavily_error", {"step": step.step_id, "message": "Tavily API key missing"})
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "tavily_error",
                {"step": step.step_id, "message": "Tavily API key missing"},
                tone="warn",
            )
            tavily_errors.append("Tavily API key missing")
        else:
            if not queries and question:
                queries = [question]
            primary_queries = queries[:5] if queries else []
            if primary_queries:
                sources, errors, stats = await run_tavily_queries(
                    run_id,
                    step,
                    primary_queries,
                    search_depth,
                    per_query_max,
                    tavily,
                    db,
                    bus,
                    topic=requested_topic,
                    time_range=requested_time_range,
                    executed=executed_queries,
                    executed_order=executed_order,
                    question=question,
                    lm_client=lm_client,
                    model_map=model_map,
                    run_state=run_state,
                )
                gathered_sources.extend(sources)
                tavily_errors.extend(errors)
                search_stats = _merge_search_stats(search_stats, stats)
            if not gathered_sources and question:
                fallback_query = question.strip()
                sources, errors, stats = await run_tavily_queries(
                    run_id,
                    step,
                    [fallback_query],
                    search_depth,
                    per_query_max,
                    tavily,
                    db,
                    bus,
                    topic=requested_topic,
                    time_range=requested_time_range,
                    mode="fallback",
                    executed=executed_queries,
                    executed_order=executed_order,
                    question=question,
                    lm_client=lm_client,
                    model_map=model_map,
                    run_state=run_state,
                )
                gathered_sources.extend(sources)
                tavily_errors.extend(errors)
                search_stats = _merge_search_stats(search_stats, stats)
            if not gathered_sources:
                retry_time = widen_time_range(requested_time_range)
                if not retry_time and requested_topic == "news":
                    retry_time = "week"
                retry_queries = build_fallback_queries(
                    question,
                    prompt,
                    topic=requested_topic,
                    time_range=retry_time,
                )
                if retry_queries:
                    sources, errors, stats = await run_tavily_queries(
                        run_id,
                        step,
                        retry_queries,
                        search_depth,
                        per_query_max,
                        tavily,
                        db,
                        bus,
                        topic=None,
                        time_range=retry_time,
                        mode="retry",
                        executed=executed_queries,
                        executed_order=executed_order,
                        question=question,
                        lm_client=lm_client,
                        model_map=model_map,
                        run_state=run_state,
                    )
                    gathered_sources.extend(sources)
                    tavily_errors.extend(errors)
                    search_stats = _merge_search_stats(search_stats, stats)
            urls = [s["url"] for s in gathered_sources if s.get("url")]
            if urls:
                url_slice = urls[: max(3, min(extract_budget, len(urls)))]
                await bus.emit(run_id, "tavily_extract", {"step": step.step_id, "urls": url_slice})
                queue_narration(
                    lm_client,
                    model_map,
                    run_state,
                    bus,
                    run_id,
                    question,
                    "tavily_extract",
                    {"step": step.step_id, "urls": url_slice},
                )
                try:
                    extract_resp = await asyncio.wait_for(
                        tavily.extract(url_slice, extract_depth=extract_depth),
                        timeout=TAVILY_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    extract_resp = {"error": "timeout"}
                extract_resp = coerce_tavily_response(extract_resp)
                await db.add_extract(run_id, f"Step{step.step_id}", ",".join(url_slice), extract_depth, extract_resp)
                if extract_resp.get("error"):
                    error_msg = format_tavily_error(extract_resp)
                    await bus.emit(
                        run_id,
                        "tavily_error",
                        {"step": step.step_id, "message": error_msg},
                    )
                    queue_narration(
                        lm_client,
                        model_map,
                        run_state,
                        bus,
                        run_id,
                        question,
                        "tavily_error",
                        {"step": step.step_id, "message": error_msg},
                        tone="warn",
                    )
                    tavily_errors.append(error_msg)
                extract_results = extract_resp.get("results") or []
                if not isinstance(extract_results, list):
                    extract_results = []
                if extract_results:
                    gathered_sources = []
                    for res in extract_results:
                        if not isinstance(res, dict):
                            continue
                        src = {
                            "url": res.get("url", ""),
                            "title": res.get("title", ""),
                            "publisher": res.get("source", ""),
                            "date_published": res.get("published_date", ""),
                            "snippet": res.get("content", "")[:400],
                            "extracted_text": res.get("content", ""),
                        }
                        gathered_sources.append(src)
                        await db.add_source(
                            run_id,
                            f"Step{step.step_id}",
                            src["url"],
                            src["title"],
                            src["publisher"],
                            src["date_published"],
                            src["snippet"],
                            src["extracted_text"],
                        )
            if gathered_sources and run_state:
                source_urls: List[str] = []
                seen_urls: Set[str] = set()
                for src in gathered_sources:
                    url = str(src.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    source_urls.append(url)
                    if len(source_urls) >= 3:
                        break
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "read_sources",
                    "Reading sources and pulling key details.",
                    urls=source_urls,
                )
    if tavily_errors and run_state:
        await maybe_emit_work_log(
            run_state,
            bus,
            run_id,
            "web_tool_error",
            "Web search isn't reachable, so I'll proceed with what I have.",
            tone="warn",
        )
    if tool_requests:
        await bus.emit(run_id, "tool_request", {"step": step.step_id, "requests": tool_requests})
        queue_narration(
            lm_client,
            model_map,
            run_state,
            bus,
            run_id,
            question,
            "tool_request",
            {"step": step.step_id, "requests": tool_requests},
        )
    tool_results = await resolve_tool_requests(
        tool_requests,
        upload_dir=upload_dir,
        db=db,
        conversation_id=conversation_id,
        lm_client=lm_client,
        model_map=model_map,
        run_id=run_id,
        bus=bus,
        step_id=step.step_id,
        run_state=run_state,
    )
    if tool_results:
        safe_results = strip_data_urls(tool_results, allow_plot=True)
        await bus.emit(
            run_id,
            "tool_result",
            {"step": step.step_id, "results": safe_results},
        )
        queue_narration(
            lm_client,
            model_map,
            run_state,
            bus,
            run_id,
            question,
            "tool_result",
            {"step": step.step_id, "results": safe_results},
        )
    if gathered_sources:
        seen_urls: Set[str] = set()
        emitted = 0
        for src in gathered_sources:
            url = str(src.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            payload = {
                "step": step.step_id,
                "lane": step.agent_profile,
                "title": src.get("title") or "",
                "publisher": src.get("publisher") or "",
                "date_published": src.get("date_published") or "",
                "url": url,
            }
            await bus.emit(run_id, "source_found", payload)
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "source_found",
                payload,
            )
            emitted += 1
            if emitted >= 3:
                break
    synth = {"claims": [], "gaps": [], "conflicts_found": False}
    if gathered_sources:
        synth = await synthesize_evidence_from_sources(
            lm_client,
            model_map,
            question,
            gathered_sources,
            run_id=run_id,
            bus=bus,
            step_id=step.step_id,
            run_state=run_state,
        )
    claims = synth.get("claims") if gathered_sources else parsed.get("claims", [])
    gaps = synth.get("gaps") if gathered_sources else parsed.get("gaps", [])
    conflicts_found = synth.get("conflicts_found") if gathered_sources else parsed.get("conflicts_found", False)
    if gathered_sources and isinstance(claims, list):
        emitted = 0
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("claim") or "").strip()
            if not text:
                continue
            payload = {
                "step": step.step_id,
                "lane": step.agent_profile,
                "claim": text,
                "urls": claim.get("urls") or [],
            }
            await bus.emit(run_id, "claim_found", payload)
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "claim_found",
                payload,
            )
            emitted += 1
            if emitted >= 3:
                break
    if not isinstance(claims, list):
        claims = []
    if not isinstance(gaps, list):
        gaps = []
    if use_web and not gathered_sources:
        gaps.append("No sources returned for the search queries.")
    for err in tavily_errors:
        gaps.append(f"Search error: {err}")
    evidence_queries = executed_order if executed_order else parsed.get("queries", [])
    evidence = {
        "lane": step.agent_profile,
        "queries": evidence_queries,
        "sources": gathered_sources,
        "claims": claims,
        "gaps": gaps,
        "conflicts_found": conflicts_found,
        "tool_requests": tool_requests,
        "tool_results": tool_results,
        "time_range": requested_time_range,
        "topic": requested_topic,
        "search_stats": search_stats,
        "timestamp_utc": utc_iso(),
    }
    for claim in evidence["claims"]:
        if isinstance(claim, dict) and "claim" in claim:
            claim_text = str(claim.get("claim"))
        else:
            claim_text = claim if isinstance(claim, str) else json.dumps(claim)
        await db.add_claim(
            run_id,
            claim_text,
            [s.get("url", "") for s in gathered_sources],
            confidence="MED",
            notes=step.agent_profile,
        )
    artifacts.append(
        Artifact(
            step_id=step.step_id,
            key=f"evidence_step_{step.step_id}",
            artifact_type="evidence",
            content_text="",
            content_json=evidence,
        )
    )
    return evidence, artifacts, prompt


async def execute_tavily_search_step(
    run_id: str,
    question: str,
    step: PlanStep,
    decision: RouterDecision,
    search_depth_mode: str,
    depth_profile: dict,
    tavily: TavilyClient,
    db: Database,
    bus: EventBus,
    lm_client: LMStudioClient,
    model_map: Dict[str, Dict[str, str]],
    prompt: str,
    run_state: Optional[RunState] = None,
) -> Tuple[Dict[str, Any], List[Artifact], str]:
    if run_state and not run_state.can_web:
        evidence = {
            "lane": step.agent_profile,
            "queries": [],
            "sources": [],
            "claims": [],
            "gaps": ["web browsing unavailable"],
            "conflicts_found": False,
            "timestamp_utc": utc_iso(),
        }
        return evidence, [Artifact(step_id=step.step_id, key=f"evidence_step_{step.step_id}", artifact_type="evidence", content_json=evidence)], prompt
    inputs = step.inputs if isinstance(step.inputs, dict) else {}
    queries: List[str] = []
    input_queries = inputs.get("queries")
    if isinstance(input_queries, list):
        queries = [str(q).strip() for q in input_queries if str(q).strip()]
    input_query = inputs.get("query")
    if not queries and input_query:
        queries = [str(input_query).strip()]
    search_depth = reasoning_to_search_depth(decision.reasoning_level, search_depth_mode, depth_profile)
    override_depth = str(inputs.get("search_depth") or "").strip().lower()
    if override_depth in ("basic", "advanced"):
        search_depth = override_depth
    time_range = normalize_time_range(inputs.get("time_range")) or infer_time_range(question)
    topic = inputs.get("topic") or decision.topic
    if topic not in ALLOWED_TOPICS:
        topic = decision.topic or "general"
    if not queries:
        queries = build_fallback_queries(
            question,
            prompt,
            topic=topic,
            time_range=time_range,
        )
    max_results = decision.max_results or 6
    override_max = inputs.get("max_results")
    if override_max is not None:
        try:
            max_results = max(1, int(override_max))
        except Exception:
            pass
    per_query_max = max(3, min(depth_profile.get("tool_budget", {}).get("tavily_search", max_results), max_results))
    gathered_sources: List[dict] = []
    executed_queries: Set[str] = set()
    executed_order: List[str] = []
    tavily_errors: List[str] = []
    search_stats = _empty_search_stats()
    if not tavily.enabled:
        await bus.emit(run_id, "tavily_error", {"step": step.step_id, "message": "Tavily API key missing"})
        queue_narration(
            lm_client,
            model_map,
            run_state,
            bus,
            run_id,
            question,
            "tavily_error",
            {"step": step.step_id, "message": "Tavily API key missing"},
            tone="warn",
        )
        tavily_errors.append("Tavily API key missing")
    else:
        primary_queries = queries[:5] if queries else []
        if primary_queries:
            sources, errors, stats = await run_tavily_queries(
                run_id,
                step,
                primary_queries,
                search_depth,
                per_query_max,
                tavily,
                db,
                bus,
                topic=topic,
                time_range=time_range,
                executed=executed_queries,
                executed_order=executed_order,
                question=question,
                lm_client=lm_client,
                model_map=model_map,
                run_state=run_state,
            )
            gathered_sources.extend(sources)
            tavily_errors.extend(errors)
            search_stats = _merge_search_stats(search_stats, stats)
        if not gathered_sources:
            retry_time = widen_time_range(time_range)
            if not retry_time and topic == "news":
                retry_time = "week"
            retry_queries = build_fallback_queries(
                question,
                prompt,
                topic=topic,
                time_range=retry_time,
            )
            if retry_queries:
                sources, errors, stats = await run_tavily_queries(
                    run_id,
                    step,
                    retry_queries,
                    search_depth,
                    per_query_max,
                    tavily,
                    db,
                    bus,
                    topic=None,
                    time_range=retry_time,
                    mode="retry",
                    executed=executed_queries,
                    executed_order=executed_order,
                    question=question,
                    lm_client=lm_client,
                    model_map=model_map,
                    run_state=run_state,
                )
                gathered_sources.extend(sources)
                tavily_errors.extend(errors)
                search_stats = _merge_search_stats(search_stats, stats)
    synth = await synthesize_evidence_from_sources(
        lm_client,
        model_map,
        question,
        gathered_sources,
        run_id=run_id,
        bus=bus,
        step_id=step.step_id,
        run_state=run_state,
    )
    claims = synth.get("claims", [])
    gaps = synth.get("gaps", [])
    if not gathered_sources:
        gaps.append("No sources returned for the search queries.")
    for err in tavily_errors:
        gaps.append(f"Search error: {err}")
    evidence_queries = executed_order if executed_order else queries
    evidence = {
        "lane": step.agent_profile,
        "queries": evidence_queries,
        "sources": gathered_sources,
        "claims": claims if isinstance(claims, list) else [],
        "gaps": gaps if isinstance(gaps, list) else [],
        "conflicts_found": bool(synth.get("conflicts_found")),
        "tool_requests": [],
        "tool_results": [],
        "time_range": time_range,
        "topic": topic,
        "search_stats": search_stats,
        "timestamp_utc": utc_iso(),
    }
    artifacts = [
        Artifact(
            step_id=step.step_id,
            key=f"evidence_step_{step.step_id}",
            artifact_type="evidence",
            content_text="",
            content_json=evidence,
        )
    ]
    return evidence, artifacts, prompt


async def execute_tavily_extract_step(
    run_id: str,
    question: str,
    step: PlanStep,
    decision: RouterDecision,
    tavily: TavilyClient,
    db: Database,
    bus: EventBus,
    lm_client: LMStudioClient,
    model_map: Dict[str, Dict[str, str]],
    prompt: str,
    run_state: Optional[RunState] = None,
) -> Tuple[Dict[str, Any], List[Artifact], str]:
    if run_state and not run_state.can_web:
        evidence = {
            "lane": step.agent_profile,
            "queries": [],
            "sources": [],
            "claims": [],
            "gaps": ["web browsing unavailable"],
            "conflicts_found": False,
            "timestamp_utc": utc_iso(),
        }
        return evidence, [Artifact(step_id=step.step_id, key=f"evidence_step_{step.step_id}", artifact_type="evidence", content_json=evidence)], prompt
    inputs = step.inputs if isinstance(step.inputs, dict) else {}
    urls: List[str] = []
    input_urls = inputs.get("urls")
    if isinstance(input_urls, list):
        urls = [str(u).strip() for u in input_urls if str(u).strip()]
    elif isinstance(input_urls, str) and input_urls.strip():
        urls = [u.strip() for u in input_urls.split(",") if u.strip()]
    extract_depth = decision.extract_depth if decision else "basic"
    override_extract = str(inputs.get("extract_depth") or "").strip().lower()
    if override_extract in ("basic", "advanced"):
        extract_depth = override_extract
    gathered_sources: List[dict] = []
    tavily_errors: List[str] = []
    if not tavily.enabled:
        await bus.emit(run_id, "tavily_error", {"step": step.step_id, "message": "Tavily API key missing"})
        queue_narration(
            lm_client,
            model_map,
            run_state,
            bus,
            run_id,
            question,
            "tavily_error",
            {"step": step.step_id, "message": "Tavily API key missing"},
            tone="warn",
        )
        tavily_errors.append("Tavily API key missing")
    if urls:
        await bus.emit(run_id, "tavily_extract", {"step": step.step_id, "urls": urls})
        queue_narration(
            lm_client,
            model_map,
            run_state,
            bus,
            run_id,
            question,
            "tavily_extract",
            {"step": step.step_id, "urls": urls},
        )
        try:
            extract_resp = await asyncio.wait_for(
                tavily.extract(urls, extract_depth=extract_depth),
                timeout=TAVILY_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            extract_resp = {"error": "timeout"}
        extract_resp = coerce_tavily_response(extract_resp)
        await db.add_extract(run_id, f"Step{step.step_id}", ",".join(urls), extract_depth, extract_resp)
        if extract_resp.get("error"):
            error_msg = format_tavily_error(extract_resp)
            await bus.emit(run_id, "tavily_error", {"step": step.step_id, "message": error_msg})
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "tavily_error",
                {"step": step.step_id, "message": error_msg},
                tone="warn",
            )
            tavily_errors.append(error_msg)
        extract_results = extract_resp.get("results") or []
        if not isinstance(extract_results, list):
            extract_results = []
        for res in extract_results:
            if not isinstance(res, dict):
                continue
            src = {
                "url": res.get("url", ""),
                "title": res.get("title", ""),
                "publisher": res.get("source", ""),
                "date_published": res.get("published_date", ""),
                "snippet": res.get("content", "")[:400],
                "extracted_text": res.get("content", ""),
            }
            gathered_sources.append(src)
            await db.add_source(
                run_id,
                f"Step{step.step_id}",
                src["url"],
                src["title"],
                src["publisher"],
                src["date_published"],
                src["snippet"],
                src["extracted_text"],
            )
    synth = await synthesize_evidence_from_sources(
        lm_client,
        model_map,
        question,
        gathered_sources,
        run_id=run_id,
        bus=bus,
        step_id=step.step_id,
        run_state=run_state,
    )
    claims = synth.get("claims", [])
    gaps = synth.get("gaps", [])
    if not urls:
        gaps.append("No URLs provided for extraction.")
    if urls and not gathered_sources:
        gaps.append("No sources returned from extract.")
    for err in tavily_errors:
        gaps.append(f"Search error: {err}")
    evidence = {
        "lane": step.agent_profile,
        "queries": [],
        "sources": gathered_sources,
        "claims": claims if isinstance(claims, list) else [],
        "gaps": gaps if isinstance(gaps, list) else [],
        "conflicts_found": bool(synth.get("conflicts_found")),
        "tool_requests": [],
        "tool_results": [],
        "timestamp_utc": utc_iso(),
    }
    artifacts = [
        Artifact(
            step_id=step.step_id,
            key=f"evidence_step_{step.step_id}",
            artifact_type="evidence",
            content_text="",
            content_json=evidence,
        )
    ]
    return evidence, artifacts, prompt


async def execute_step(
    run_id: str,
    question: str,
    step: PlanStep,
    decision: RouterDecision,
    search_depth_mode: str,
    depth_profile: dict,
    artifacts: List[Artifact],
    progress_meta: Dict[str, int],
    response_guidance: str,
    lm_client: LMStudioClient,
    tavily: TavilyClient,
    db: Database,
    bus: EventBus,
    model_map: Dict[str, Dict[str, str]],
    conversation_id: Optional[str] = None,
    upload_dir: Optional[Path] = None,
    run_state: Optional[RunState] = None,
) -> Tuple[Dict[str, Any], List[Artifact], str]:
    answer_hint = ""
    if step.type == "draft":
        answer_hint = response_guidance or response_guidance_text(question, decision.reasoning_level, progress_meta)
        if decision.needs_web and (run_state is None or run_state.can_web):
            ledger = merge_evidence_artifacts(artifacts)
            if not ledger.get("sources"):
                answer_hint = (
                    f"{answer_hint} No sources were retrieved from web search; "
                    "state that clearly and ask for a narrower query or links."
                )
    prompt = await generate_step_prompt(
        lm_client,
        model_map["orch"],
        question,
        step,
        artifacts,
        answer_guidance=answer_hint,
        toolbox_hint=(agents.TOOLBOX_GUIDE if hasattr(agents, "TOOLBOX_GUIDE") else ""),
    )
    if step.type == "research":
        return await execute_research_step(
            run_id,
            question,
            step,
            prompt,
            decision,
            search_depth_mode,
            depth_profile,
            lm_client,
            tavily,
            db,
            bus,
            model_map,
            conversation_id=conversation_id,
            upload_dir=upload_dir,
            run_state=run_state,
        )
    if step.type in ("tavily_search", "search"):
        return await execute_tavily_search_step(
            run_id,
            question,
            step,
            decision,
            search_depth_mode,
            depth_profile,
            tavily,
            db,
            bus,
            lm_client,
            model_map,
            prompt,
            run_state=run_state,
        )
    if step.type in ("tavily_extract", "extract"):
        return await execute_tavily_extract_step(
            run_id,
            question,
            step,
            decision,
            tavily,
            db,
            bus,
            lm_client,
            model_map,
            prompt,
            run_state=run_state,
        )
    elif step.type == "merge":
        merged = merge_evidence_artifacts(artifacts)
        artifact = Artifact(
            step_id=step.step_id,
            key="claims_ledger",
            artifact_type="ledger",
            content_text="",
            content_json=merged,
        )
        return merged, [artifact], prompt
    elif step.type == "draft":
        draft_profile = (step.agent_profile or "").strip()
        draft_lower = draft_profile.lower()
        if draft_lower != "writer":
            draft_profile = "Writer"
        draft_resp = await run_worker(
            lm_client,
            draft_profile,
            model_map,
            prompt,
            temperature=0.3,
            max_tokens=800,
            run_id=run_id,
            bus=bus,
            step_id=step.step_id,
            context="draft",
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        artifact = Artifact(
            step_id=step.step_id,
            key="draft_answer",
            artifact_type="draft",
            content_text=draft_resp,
            content_json={"draft": draft_resp},
        )
        return {"draft": draft_resp}, [artifact], prompt
    elif step.type in ("verify", "verifier", "verifier_worker"):
        # use verifier worker (Qwen8) but with verifier system
        ledger = merge_evidence_artifacts(artifacts)
        draft = next((a.content_text for a in artifacts if a.artifact_type == "draft"), "")
        verifier_prompt = (
            f"Question: {question}\nDraft: {draft}\nClaims ledger: {json.dumps(ledger)[:3000]}\n"
            "Return JSON verdict: PASS/NEEDS_REVISION, issues[], revised_answer?, extra_steps[]."
        )
        verifier_profile = (step.agent_profile or "").strip()
        if verifier_profile.lower() != "verifier":
            verifier_profile = "Verifier"
        try:
            report = await run_worker(
                lm_client,
                verifier_profile,
                model_map,
                verifier_prompt,
                temperature=0.0,
                max_tokens=700,
                run_id=run_id,
                bus=bus,
                step_id=step.step_id,
                context="verify",
                run_state=run_state,
                model_manager=run_state.model_manager if run_state else None,
            )
        except Exception as exc:
            if run_state:
                run_state.add_dev_trace(
                    "Verifier failed; bypassing verification.",
                    {"error": str(exc)},
                )
            report = json.dumps({"verdict": "PASS", "issues": [], "extra_steps": []})
        verifier_model = (
            (model_map.get("verifier") or {}).get("model")
            or (model_map.get("worker") or {}).get("model")
            or (model_map.get("orch") or {}).get("model")
            or ""
        )
        parsed = await safe_json_parse(
            report,
            lm_client,
            verifier_model,
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        if not isinstance(parsed, dict):
            parsed = {}
        if not parsed:
            parsed = {"issues": [], "verdict": "PASS", "extra_steps": []}
        parsed = normalize_verifier_payload(parsed)
        if decision.expected_passes > 1 or decision.needs_web:
            planner_model = (model_map.get("orch") or {}).get("model")
            planner_url = (model_map.get("orch") or {}).get("base_url")
            if planner_model and planner_url and planner_model != verifier_model:
                try:
                    if run_state and run_state.model_manager:
                        planner_content = await run_worker(
                            lm_client,
                            "Verifier",
                            {},
                            verifier_prompt,
                            temperature=0.0,
                            max_tokens=700,
                            run_state=run_state,
                            model_manager=run_state.model_manager,
                            system_prompt_override=agents.VERIFIER_SYSTEM,
                            context="planner_verify",
                        )
                    else:
                        planner_resp = await lm_client.chat_completion(
                            model=planner_model,
                            messages=[
                                {"role": "system", "content": agents.VERIFIER_SYSTEM},
                                {"role": "user", "content": verifier_prompt},
                            ],
                            temperature=0.0,
                            max_tokens=700,
                            base_url=planner_url,
                            run_state=run_state,
                        )
                        planner_content = planner_resp["choices"][0]["message"]["content"]
                    planner_parsed = await safe_json_parse(
                        planner_content,
                        lm_client,
                        planner_model,
                        run_state=run_state,
                        model_manager=run_state.model_manager if run_state else None,
                    )
                    if planner_parsed:
                        if not isinstance(planner_parsed, dict):
                            planner_parsed = {}
                        planner_parsed = normalize_verifier_payload(planner_parsed)
                        await bus.emit(
                            run_id,
                            "planner_verifier",
                            {
                                "step": step.step_id,
                                "verdict": planner_parsed.get("verdict"),
                                "issues": planner_parsed.get("issues", []),
                            },
                        )
                        if planner_parsed.get("verdict") == "NEEDS_REVISION":
                            planner_parsed["planner_override"] = True
                            parsed = planner_parsed
                except Exception:
                    pass
        artifact = Artifact(
            step_id=step.step_id,
            key="verifier_report",
            artifact_type="verifier",
            content_text=json.dumps(parsed),
            content_json=parsed,
        )
        return parsed, [artifact], prompt
    elif step.type == "finalize":
        requires_verifier = False
        if isinstance(step.inputs, dict):
            requires_verifier = bool(step.inputs.get("requires_verifier"))
        approved_text, verdict = _approved_final_text(artifacts)
        if requires_verifier and verdict != "PASS":
            raise ValueError("Final draft not approved")
        if not approved_text:
            draft_art = _latest_artifact(artifacts, "draft")
            draft_text = (draft_art.content_text or "").strip() if draft_art else ""
            if draft_text:
                approved_text = draft_text
                await bus.emit(
                    run_id,
                    "dev_trace",
                    {"text": "Finalizer fallback used latest draft text."},
                )
            else:
                raise ValueError("Missing approved draft text")
        final_profile = (step.agent_profile or "").strip()
        if final_profile.lower() != "finalizer":
            final_profile = "Finalizer"
        final_raw = await run_worker(
            lm_client,
            final_profile,
            model_map,
            prompt,
            temperature=0.0,
            max_tokens=300,
            run_id=run_id,
            bus=bus,
            step_id=step.step_id,
            context="finalize",
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        fixer_model = (
            (model_map.get("summarizer") or {}).get("model")
            or (model_map.get("orch") or {}).get("model")
            or ""
        )
        parsed = await safe_json_parse(
            final_raw,
            lm_client,
            fixer_model,
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        if not isinstance(parsed, dict):
            await bus.emit(
                run_id,
                "dev_trace",
                {"text": "Finalizer JSON parse failed; forcing finalize_answer fallback."},
            )
            parsed = {
                "tool_requests": [
                    {"tool": "finalize_answer", "final_text": approved_text, "approved": True, "fallback": True}
                ]
            }
        tool_requests = parsed.get("tool_requests") or []
        if not isinstance(tool_requests, list):
            tool_requests = []
        has_finalize = False
        for req in tool_requests:
            if not isinstance(req, dict):
                continue
            tool_name = str(req.get("tool") or req.get("type") or req.get("name") or "").lower()
            if tool_name in FINALIZE_TOOL_NAMES:
                req["final_text"] = approved_text
                req.setdefault("approved", True)
                has_finalize = True
        if not has_finalize:
            await bus.emit(
                run_id,
                "dev_trace",
                {"text": "Finalizer skipped finalize_answer; adding fallback tool call."},
            )
            tool_requests.append(
                {"tool": "finalize_answer", "final_text": approved_text, "approved": True, "fallback": True}
            )
            has_finalize = True
        if tool_requests:
            await bus.emit(run_id, "tool_request", {"step": step.step_id, "requests": tool_requests})
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "tool_request",
                {"step": step.step_id, "requests": tool_requests},
            )
        tool_results = await resolve_tool_requests(
            tool_requests,
            upload_dir=upload_dir,
            db=db,
            conversation_id=conversation_id,
            lm_client=lm_client,
            model_map=model_map,
            run_id=run_id,
            bus=bus,
            step_id=step.step_id,
            run_state=run_state,
        )
        if tool_results:
            safe_results = strip_data_urls(tool_results, allow_plot=True)
            await bus.emit(run_id, "tool_result", {"step": step.step_id, "results": safe_results})
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "tool_result",
                {"step": step.step_id, "results": safe_results},
            )
        final_text = ""
        for res in tool_results:
            if not isinstance(res, dict):
                continue
            tool_name = str(res.get("tool") or "").lower()
            if res.get("status") == "ok" and tool_name in FINALIZE_TOOL_NAMES and res.get("result"):
                final_text = str(res.get("result")).strip()
                break
        if not final_text:
            raise ValueError("Finalization failed")
        artifact = Artifact(
            step_id=step.step_id,
            key="final_answer",
            artifact_type="final",
            content_text=final_text,
            content_json={"final": final_text, "tool_requests": tool_requests, "tool_results": tool_results},
        )
        return {"final": final_text}, [artifact], prompt
    elif step.type == "analysis":
        analysis_profile = step.agent_profile or "Summarizer"
        summary_raw = await run_worker(
            lm_client,
            analysis_profile,
            model_map,
            prompt,
            temperature=0.2,
            max_tokens=400,
            run_id=run_id,
            bus=bus,
            step_id=step.step_id,
            context="analysis",
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        summary_text = summary_raw
        tool_requests = []
        tool_results = []
        fixer_model = (
            (model_map.get("summarizer") or {}).get("model")
            or (model_map.get("orch") or {}).get("model")
            or ""
        )
        parsed = await safe_json_parse(
            summary_raw,
            lm_client,
            fixer_model,
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        if isinstance(parsed, dict):
            lines = parsed.get("activity_lines") or parsed.get("memory_notes") or parsed.get("criteria")
            if isinstance(lines, list):
                summary_text = " ".join(str(x) for x in lines if str(x).strip()).strip() or summary_raw
            elif isinstance(lines, str) and lines.strip():
                summary_text = lines.strip()
            tool_requests = parsed.get("tool_requests") or []
            if not isinstance(tool_requests, list):
                tool_requests = []
        if not tool_requests:
            math_request = build_math_tool_request(question)
            if math_request:
                tool_requests = [math_request]
                if not summary_text.strip():
                    summary_text = f"Compute locally: {math_request.get('code')}"
        if tool_requests:
            await bus.emit(run_id, "tool_request", {"step": step.step_id, "requests": tool_requests})
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "tool_request",
                {"step": step.step_id, "requests": tool_requests},
            )
            tool_results = await resolve_tool_requests(
                tool_requests,
                upload_dir=upload_dir,
                db=db,
                conversation_id=conversation_id,
                lm_client=lm_client,
                model_map=model_map,
                run_id=run_id,
                bus=bus,
                step_id=step.step_id,
                run_state=run_state,
            )
            if tool_results:
                safe_results = strip_data_urls(tool_results, allow_plot=True)
                await bus.emit(
                    run_id,
                    "tool_result",
                    {"step": step.step_id, "results": safe_results},
                )
                queue_narration(
                    lm_client,
                    model_map,
                    run_state,
                    bus,
                    run_id,
                    question,
                    "tool_result",
                    {"step": step.step_id, "results": safe_results},
                )
                if looks_like_math_expression(question):
                    local_result = next(
                        (
                            r
                            for r in tool_results
                            if r.get("status") == "ok"
                            and str(r.get("tool") or "").lower() in ("local_code", "execute_code", "code_exec", "exec_code")
                        ),
                        None,
                    )
                    if local_result and "result" in local_result:
                        result_text = str(local_result.get("result"))
                        if summary_text.strip():
                            summary_text = f"{summary_text} Local result: {result_text}"
                        else:
                            summary_text = f"Local result: {result_text}"
        output_key = "success_criteria"
        output_type = "criteria"
        if isinstance(step.outputs, list) and step.outputs:
            first = step.outputs[0]
            if isinstance(first, dict):
                output_key = str(first.get("key") or output_key)
                output_type = str(first.get("artifact_type") or output_type)
        content_json = {
            "text": summary_text,
            "tool_requests": tool_requests,
            "tool_results": tool_results,
        }
        if output_type == "criteria" or output_key == "success_criteria":
            content_json["criteria"] = summary_text
        else:
            content_json[output_type] = summary_text
        artifact = Artifact(
            step_id=step.step_id,
            key=output_key,
            artifact_type=output_type,
            content_text=summary_text,
            content_json=content_json,
        )
        return {output_key or output_type or "text": summary_text}, [artifact], prompt
    else:
        generic = await run_worker(
            lm_client,
            step.agent_profile,
            model_map,
            prompt,
            temperature=0.2,
            max_tokens=500,
            run_id=run_id,
            bus=bus,
            step_id=step.step_id,
            context=step.type,
            run_state=run_state,
            model_manager=run_state.model_manager if run_state else None,
        )
        artifact = Artifact(
            step_id=step.step_id,
            key=f"step_{step.step_id}_output",
            artifact_type="note",
            content_text=generic,
            content_json={"text": generic},
        )
        return {"text": generic}, [artifact], prompt


async def process_uploads(
    run_id: str,
    question: str,
    upload_ids: List[int],
    db: Database,
    bus: EventBus,
    lm_client: LMStudioClient,
    model_map: Dict[str, Dict[str, str]],
    run_state: Optional[RunState] = None,
) -> Tuple[List[Artifact], str]:
    """Analyze uploads with vision (8B) and secretary (4B) models."""
    if not upload_ids:
        return [], ""
    artifacts: List[Artifact] = []
    summaries: List[str] = []
    vision_endpoint = model_map.get("worker") or model_map.get("worker_a") or model_map.get("orch")
    secretary_endpoint = model_map.get("summarizer") or model_map.get("router") or model_map.get("worker")
    avoid_coder = False
    if question:
        avoid_coder = not looks_like_coding_task(question)
    for uid in upload_ids:
        record = await db.get_upload(uid)
        if not record:
            continue
        await bus.emit(
            run_id,
            "upload_received",
            {
                "upload_id": record["id"],
                "name": record["original_name"],
                "mime": record["mime"],
                "size": record["size_bytes"],
            },
        )
        try:
            path = Path(record["storage_path"])
            vision_json: Dict[str, Any] = {}
            if record["mime"].startswith("image/"):
                image_block = [
                    {
                        "type": "text",
                        "text": f"User question: {question}\nDescribe the image, objects, and any text. Return JSON only.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url_from_file(path, record["mime"])}},
                ]
                if run_state and run_state.model_manager:
                    resp, instance = await run_state.model_manager.call_with_instance(
                        required_capabilities=["vision"],
                        objective=run_state.model_manager.routing_objective,
                        request={
                            "messages": [
                                {"role": "system", "content": agents.VISION_ANALYST_SYSTEM},
                                {"role": "user", "content": image_block},
                            ],
                            "temperature": 0.2,
                            "max_tokens": 600,
                            "use_responses": True,
                        },
                        avoid_coder=avoid_coder,
                    )
                    content = resp["choices"][0]["message"]["content"]
                    await bus.emit(
                        run_id,
                        "model_selected",
                        {
                            "profile": "VisionAnalyst",
                            "model": instance.model_key,
                            "instance": instance.api_identifier,
                            "backend_id": instance.backend_id,
                            "context": "vision",
                        },
                    )
                else:
                    resp = await lm_client.chat_completion(
                        model=vision_endpoint["model"],
                        messages=[
                            {"role": "system", "content": agents.VISION_ANALYST_SYSTEM},
                            {"role": "user", "content": image_block},
                        ],
                        temperature=0.2,
                        max_tokens=600,
                        base_url=vision_endpoint["base_url"],
                        run_state=run_state,
                    )
                    content = resp["choices"][0]["message"]["content"]
                vision_json = (
                    await safe_json_parse(
                        content,
                        lm_client,
                        vision_endpoint["model"],
                        run_state=run_state,
                        model_manager=run_state.model_manager if run_state else None,
                    )
                    or {"caption": content}
                )
            elif record["mime"] == "application/pdf":
                excerpt = pdf_excerpt(path)
                vision_json = {"text_excerpt": excerpt, "note": "PDF excerpt (first pages)"}
            else:
                raise ValueError("Unsupported upload type")

            secretary_prompt = (
                f"Question: {question}\n"
                f"Upload: {record['original_name']} ({record['mime']}, {record['size_bytes']} bytes)\n"
                f"Vision analysis: {json.dumps(vision_json)[:3500]}"
            )
            if run_state and run_state.model_manager:
                sec_resp, instance = await run_state.model_manager.call_with_instance(
                    required_capabilities=["structured_output"],
                    objective=run_state.model_manager.routing_objective,
                    request={
                        "messages": [
                            {"role": "system", "content": agents.UPLOAD_SECRETARY_SYSTEM},
                            {"role": "user", "content": secretary_prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 320,
                        "use_responses": True,
                    },
                    avoid_coder=avoid_coder,
                )
                sec_content = sec_resp["choices"][0]["message"]["content"]
                await bus.emit(
                    run_id,
                    "model_selected",
                    {
                        "profile": "UploadSecretary",
                        "model": instance.model_key,
                        "instance": instance.api_identifier,
                        "backend_id": instance.backend_id,
                        "context": "upload_secretary",
                    },
                )
            else:
                sec_resp = await lm_client.chat_completion(
                    model=secretary_endpoint["model"],
                    messages=[
                        {"role": "system", "content": agents.UPLOAD_SECRETARY_SYSTEM},
                        {"role": "user", "content": secretary_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=320,
                    base_url=secretary_endpoint["base_url"],
                    run_state=run_state,
                )
                sec_content = sec_resp["choices"][0]["message"]["content"]
            sec_json = (
                await safe_json_parse(
                    sec_content,
                    lm_client,
                    secretary_endpoint["model"],
                    run_state=run_state,
                    model_manager=run_state.model_manager if run_state else None,
                )
                or {"summary": sec_content}
            )
            summary_text = sec_json.get("summary") or sec_content
            artifact = Artifact(
                step_id=0,
                key=f"upload_{record['id']}",
                artifact_type="upload_summary",
                content_text=summary_text,
                content_json={
                    "upload_id": record["id"],
                    "name": record["original_name"],
                    "mime": record["mime"],
                    "vision": vision_json,
                    "secretary": sec_json,
                },
            )
            artifacts.append(artifact)
            summaries.append(f"{record['original_name']}: {summary_text}")
            await db.update_upload_status(
                record["id"],
                "processed",
                summary_text=summary_text,
                summary_json={"vision": vision_json, "secretary": sec_json},
            )
            await bus.emit(
                run_id,
                "upload_processed",
                {"upload_id": record["id"], "name": record["original_name"], "summary": summary_text},
            )
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "upload_processed",
                {"upload_id": record["id"], "name": record["original_name"]},
            )
        except Exception as exc:
            await db.update_upload_status(
                record["id"], "failed", summary_text=str(exc), summary_json={"error": str(exc)}
            )
            await bus.emit(
                run_id,
                "upload_failed",
                {"upload_id": record["id"], "name": record["original_name"], "error": str(exc)},
            )
            queue_narration(
                lm_client,
                model_map,
                run_state,
                bus,
                run_id,
                question,
                "upload_failed",
                {"upload_id": record["id"], "name": record["original_name"], "error": str(exc)},
                tone="warn",
            )
    summary_line = "; ".join(summaries)
    return artifacts, summary_line


async def run_question(
    run_id: str,
    conversation_id: str,
    question: str,
    decision_mode: str,
    manual_level: Optional[str],
    model_tier: str,
    deep_mode: str,
    search_depth_mode: str,
    max_results_override: int,
    strict_mode: bool,
    auto_memory: bool,
    db: Database,
    bus: EventBus,
    lm_client: LMStudioClient,
    model_manager: Optional[ModelManager],
    tavily: TavilyClient,
    settings_models: Dict[str, Dict[str, str]],
    model_availability: Optional[Dict[str, Any]] = None,
    upload_ids: Optional[List[int]] = None,
    upload_dir: Optional[Path] = None,
    stop_event: Optional[asyncio.Event] = None,
    control_queue: Optional[asyncio.Queue] = None,
    default_reasoning_level: Optional[str] = None,
    evidence_dump: bool = False,
    plan_reasoning_mode: str = "normal",
    planning_mode: str = "normal",
    reasoning_level: Optional[int] = None,
    ram_headroom_pct: float = 10.0,
    vram_headroom_pct: float = 10.0,
    max_concurrent_runs: Optional[int] = None,
    per_model_class_limits: Optional[Dict[str, int]] = None,
) -> None:
    """Main orchestration loop for a single run (now with parallel step execution)."""
    active_models: Dict[str, Dict[str, str]] = settings_models
    try:
        if upload_dir is None:
            upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
        await db.insert_run(run_id, conversation_id, question=question, reasoning_mode=decision_mode)
        user_msg = await db.add_message(run_id, conversation_id, "user", question)
        await bus.emit(
            run_id,
            "message_added",
            {"id": user_msg.get("id"), "role": "user", "content": question, "run_id": run_id, "created_at": user_msg.get("created_at")},
        )
        await bus.emit(run_id, "run_started", {"question": question})

        run_state = RunState()
        run_state.question = question
        run_state.freshness_required = needs_freshness(question)
        run_state.dev_trace_cb = make_dev_trace_cb(bus, run_id)
        run_state.model_manager = model_manager
        if model_manager:
            await maybe_emit_work_log(
                run_state,
                bus,
                run_id,
                "model_discovery",
                "Discovering available local models.",
            )
            await model_manager.refresh()
            bootstrap_instance = await model_manager.bootstrap()
            candidates = await model_manager.get_candidates()
            if not candidates:
                server_down = False
                backend = model_manager.backends.get("lmstudio")
                if backend and hasattr(backend, "is_server_reachable"):
                    try:
                        server_down = not await backend.is_server_reachable()  # type: ignore[call-arg]
                    except Exception:
                        server_down = False
                run_state.mark_chat_unavailable("No local models discovered")
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "no_models",
                    (
                        "LM Studio server is not running. Open LM Studio and start the local server, then retry."
                        if server_down
                        else "No local models are available yet. Load a tool-capable model in LM Studio and retry."
                    ),
                    tone="warn",
                )
                guidance = (
                    "I couldn't find any local models to run this request.\n\n"
                    "Open LM Studio, start the local API server, and load at least one tool-capable model "
                    "(default http://127.0.0.1:1234). Then retry this prompt."
                )
                assistant_msg = await db.add_message(run_id, conversation_id, "assistant", guidance)
                await bus.emit(
                    run_id,
                    "message_added",
                    {
                        "id": assistant_msg.get("id"),
                        "role": "assistant",
                        "content": guidance,
                        "run_id": run_id,
                        "created_at": assistant_msg.get("created_at"),
                    },
                )
                await db.finalize_run(run_id, guidance, "LOW")
                await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "LOW"})
                return
            if not bootstrap_instance:
                run_state.mark_chat_unavailable("Unable to load a local model")
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "model_bootstrap",
                    "Unable to load a local model instance. Check LM Studio resources and retry.",
                    tone="warn",
                )
                guidance = (
                    "I found local models but couldn't load an instance to run the request.\n\n"
                    "Please confirm LM Studio has enough resources to load a model instance, then retry."
                )
                assistant_msg = await db.add_message(run_id, conversation_id, "assistant", guidance)
                await bus.emit(
                    run_id,
                    "message_added",
                    {
                        "id": assistant_msg.get("id"),
                        "role": "assistant",
                        "content": guidance,
                        "run_id": run_id,
                        "created_at": assistant_msg.get("created_at"),
                    },
                )
                await db.finalize_run(run_id, guidance, "LOW")
                await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "LOW"})
                return
        plan_reasoning_mode = (plan_reasoning_mode or "auto").lower()
        planning_mode = (planning_mode or "auto").lower()
        valid_plan_modes = {"auto", "normal", "extensive"}
        if plan_reasoning_mode not in valid_plan_modes:
            plan_reasoning_mode = "auto"
        if planning_mode not in valid_plan_modes:
            planning_mode = "auto"
        await maybe_emit_work_log(run_state, bus, run_id, "goal", f"Here's the ask: {question}")
        await maybe_emit_work_log(run_state, bus, run_id, "access_check", "Quick check on available sources.")

        run_state.can_web, run_state.web_error = await check_web_access(tavily)
        if run_state.web_error and run_state.can_web:
            run_state.add_dev_trace("Web probe failed; continuing.", {"error": run_state.web_error})
            await maybe_emit_work_log(
                run_state,
                bus,
                run_id,
                "web_warning",
                f"Web search looks a bit flaky ({run_state.web_error}); I'll still try to pull sources.",
                tone="warn",
            )
        if not run_state.can_web:
            await maybe_emit_work_log(
                run_state,
                bus,
                run_id,
                "no_web",
                "Web browsing is off here, so I'll lean on what you provided and flag assumptions.",
                tone="warn",
            )
            if run_state.freshness_required:
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "freshness",
                    "If you need up-to-date verification, share links or switch to a browsing-enabled lane.",
                    tone="warn",
                )

        requested_tier = model_tier
        requested_tier_original = requested_tier
        force_parallel_requested = MIN_PARALLEL_SLOTS > 1
        if force_parallel_requested and requested_tier == "fast":
            requested_tier = "pro"
        chat_fallback_model: Optional[str] = None
        if not model_manager:
            if requested_tier == "fast":
                base_check = (
                    settings_models.get("fast")
                    or settings_models.get("worker")
                    or settings_models.get("summarizer")
                    or settings_models.get("orch")
                    or {}
                )
            else:
                base_check = settings_models.get("orch") or settings_models.get("router") or settings_models.get("summarizer") or {}
            check_url = base_check.get("base_url") or lm_client.base_url
            check_model = base_check.get("model") or ""
            can_chat, chat_detail = await lm_client.check_chat(base_url=check_url, model=check_model, run_state=run_state)
            if not can_chat:
                try:
                    available = await lm_client.list_models_cached(check_url)
                    for candidate in available:
                        if not candidate:
                            continue
                        lowered = candidate.lower()
                        if "embed" in lowered:
                            continue
                        ok, _ = await lm_client.check_chat(base_url=check_url, model=candidate, run_state=run_state)
                        if ok:
                            can_chat = True
                            chat_fallback_model = candidate
                            if run_state:
                                run_state.add_dev_trace(
                                    "Chat preflight fallback succeeded.",
                                    {"model": candidate, "base_url": check_url},
                                )
                            break
                except Exception:
                    pass
            if not can_chat:
                run_state.mark_chat_unavailable(chat_detail or "Local model unavailable")
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "no_chat",
                    "Local model isn't reachable right now, so I'll stop and explain how to fix it.",
                    tone="warn",
                )
                guidance = (
                    "I can't reach the local model right now, so I can't complete this request.\n\n"
                    "Please check that the configured model name exists in `/v1/models`, and that the request payload only "
                    "includes standard fields (model, messages, temperature, max_tokens, stream). "
                    "If you're using LM Studio, verify the model is loaded and reachable at the configured base URL."
                )
                assistant_msg = await db.add_message(run_id, conversation_id, "assistant", guidance)
                await bus.emit(
                    run_id,
                    "message_added",
                    {"id": assistant_msg.get("id"), "role": "assistant", "content": guidance, "run_id": run_id, "created_at": assistant_msg.get("created_at")},
                )
                await db.finalize_run(run_id, guidance, "LOW")
                await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "LOW"})
                return

            settings_models = await resolve_model_map(settings_models, lm_client, run_state=run_state)

        if requested_tier == "deep" and not model_manager:
            missing_roles: List[str] = []
            missing_models: List[str] = []
            if isinstance(model_availability, dict):
                for role in ("deep_orch", "deep_planner"):
                    info = model_availability.get(role)
                    if not isinstance(info, dict) or info.get("ok") is False:
                        missing_roles.append(role)
                        for mid in (info.get("missing") or []) if isinstance(info, dict) else []:
                            if mid:
                                missing_models.append(str(mid))
            else:
                for role in ("deep_orch", "deep_planner"):
                    cfg = settings_models.get(role) or {}
                    if not cfg.get("base_url") or not cfg.get("model"):
                        missing_roles.append(role)
            if missing_roles:
                missing_label = ", ".join(sorted(set(missing_models))) if missing_models else ", ".join(sorted(set(missing_roles)))
                guidance = (
                    "Deep tier selected, but required deep models are unavailable.\n\n"
                    f"Missing: {missing_label}\n"
                    "Load the deep planner/orchestrator models in LM Studio or update Settings, then retry."
                )
                assistant_msg = await db.add_message(run_id, conversation_id, "assistant", guidance)
                await bus.emit(
                    run_id,
                    "message_added",
                    {"id": assistant_msg.get("id"), "role": "assistant", "content": guidance, "run_id": run_id, "created_at": assistant_msg.get("created_at")},
                )
                await db.finalize_run(run_id, guidance, "LOW")
                await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "LOW"})
                return

        if model_manager:
            runtime_models = {"check_cache": {}}
        else:
            settings_models, runtime_models = await apply_runtime_model_overrides(
                settings_models,
                lm_client,
                run_state=run_state,
                run_id=run_id,
                bus=bus,
            )
        ready_worker_models: Set[Tuple[str, str]] = set()
        ready_worker_count = 0
        candidate_count = 0
        if model_manager:
            instances = await model_manager.worker_pool.list_instances()
            ready_instances = [inst for inst in instances if inst.status != "busy"]
            ready_worker_count = len(ready_instances)
            ready_worker_models = {(inst.endpoint, inst.model_key) for inst in ready_instances}
            candidate_count = len(await model_manager.get_candidates())
        else:
            ready_worker_roles: List[str] = []
            for role in ("worker", "worker_b", "worker_c"):
                cfg = settings_models.get(role) or {}
                base_url = cfg.get("base_url")
                model = cfg.get("model")
                if not base_url or not model:
                    continue
                if runtime_models.get("check_cache", {}).get((base_url, model)) is True:
                    ready_worker_models.add((base_url, model))
                    ready_worker_roles.append(role)
            ready_worker_count = len(ready_worker_roles) or len(ready_worker_models)
        min_parallel_slots = 1
        force_parallel = False

        base_router_endpoint = settings_models.get("router") or settings_models.get("summarizer") or settings_models["orch"]
        if requested_tier == "fast":
            router_decision = RouterDecision(
                needs_web=False,
                reasoning_level="LOW",
                topic="general",
                max_results=0,
                extract_depth="basic",
                tool_budget={"tavily_search": 0, "tavily_extract": 0},
                stop_conditions={},
                expected_passes=1,
            )
        else:
            router_decision = await call_router(
                lm_client,
                base_router_endpoint,
                question,
                manual_level if decision_mode == "manual" else None,
                default_level=default_reasoning_level,
                strict_mode=strict_mode,
                run_state=run_state,
            )
            if not run_state.can_chat:
                guidance = (
                    "Local model rejected the request; check model name, /v1/models, and strip unsupported fields. "
                    "If you're using LM Studio, confirm the model is loaded and the base URL is correct."
                )
                assistant_msg = await db.add_message(run_id, conversation_id, "assistant", guidance)
                await bus.emit(
                    run_id,
                    "message_added",
                    {
                        "id": assistant_msg.get("id"),
                        "role": "assistant",
                        "content": guidance,
                        "run_id": run_id,
                        "created_at": assistant_msg.get("created_at"),
                    },
                )
                await db.finalize_run(run_id, guidance, "LOW")
                await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "LOW"})
                return
        if requested_tier == "fast":
            router_decision.reasoning_level = "LOW"
            if decision_mode == "manual":
                router_decision.reasoning_level = manual_level
            router_decision.expected_passes = 1
            router_decision.needs_web = False
            router_decision.tool_budget = {"tavily_search": 0, "tavily_extract": 0}
            router_decision.max_results = 0
        else:
            if decision_mode == "manual":
                router_decision.reasoning_level = manual_level
            if max_results_override:
                router_decision.max_results = max_results_override
            if strict_mode:
                router_decision.reasoning_level = "HIGH" if router_decision.reasoning_level in ("LOW", "MED") else router_decision.reasoning_level
                router_decision.extract_depth = "advanced"
                router_decision.max_results = max(router_decision.max_results, 10)
            if strict_mode:
                router_decision.expected_passes = max(router_decision.expected_passes or 1, 2)
            if run_state.can_web and (guess_needs_web(question) or run_state.freshness_required):
                router_decision.needs_web = True
            if not run_state.can_web:
                router_decision.needs_web = False
                router_decision.tool_budget = {**(router_decision.tool_budget or {}), "tavily_search": 0, "tavily_extract": 0}
            if looks_like_math_expression(question):
                router_decision.needs_web = False
                router_decision.tool_budget = {**(router_decision.tool_budget or {}), "tavily_search": 0, "tavily_extract": 0}
        effective_tier = requested_tier
        if requested_tier == "auto":
            effective_tier = resolve_auto_tier(router_decision)
        if force_parallel and effective_tier == "fast":
            effective_tier = "pro"
        model_tier = effective_tier
        if model_tier == "fast":
            router_decision.reasoning_level = "LOW"
            router_decision.expected_passes = 1
            router_decision.needs_web = False
            router_decision.tool_budget = {"tavily_search": 0, "tavily_extract": 0}
            router_decision.max_results = 0
        plan_granularity_level = (
            int(reasoning_level) if reasoning_level is not None else granularity_level_from_router(router_decision.reasoning_level)
        )
        if plan_granularity_level >= 4:
            plan_reasoning_mode = "extensive"
            planning_mode = "extensive"
        else:
            plan_reasoning_mode = "auto"
            planning_mode = "auto"
        depth_profile = REASONING_DEPTHS.get(router_decision.reasoning_level, REASONING_DEPTHS["MED"])
        if model_tier == "fast":
            depth_profile = REASONING_DEPTHS["LOW"]
        if model_tier != "fast" and depth_profile.get("tool_budget", {}).get("tavily_extract"):
            router_decision.max_results = max(router_decision.max_results, depth_profile["tool_budget"]["tavily_extract"] // 2)
        if search_depth_mode == "auto" and depth_profile.get("advanced"):
            search_depth_mode = "advanced"
        if not router_decision.tool_budget:
            router_decision.tool_budget = depth_profile.get("tool_budget", {})
        deep_route_used = deep_mode
        if model_tier == "deep":
            deep_route_used = await choose_deep_route(
                lm_client,
                base_router_endpoint,
                question,
                deep_mode,
                router_decision,
                run_state=run_state,
            )
        active_models, planner_endpoint, executor_endpoint, allow_parallel, execution_mode = select_model_suite(
            settings_models, model_tier, deep_route_used
        )
        # Copy so we can safely adjust per-run without mutating global settings
        active_models = {k: (v.copy() if isinstance(v, dict) else v) for k, v in active_models.items()}
        if not executor_endpoint:
            executor_endpoint = active_models.get("summarizer") or active_models.get("router") or active_models.get("orch") or {}
        if executor_endpoint:
            active_models["executor"] = executor_endpoint
        if plan_reasoning_mode == "auto" or planning_mode == "auto":
            planner_decider = planner_endpoint or active_models.get("orch") or active_models.get("summarizer") or {}
            auto_modes = await decide_planning_modes(
                lm_client,
                planner_decider,
                question,
                plan_granularity_level,
                run_state=run_state,
            )
            if plan_reasoning_mode == "auto":
                plan_reasoning_mode = auto_modes.get("plan_reasoning_mode", "normal")
            if planning_mode == "auto":
                planning_mode = auto_modes.get("planning_mode", "normal")
        queue_narration(
            lm_client,
            active_models,
            run_state,
            bus,
            run_id,
            question,
            "run_started",
            {"question": question},
        )
        decision_payload = router_decision.model_dump()
        decision_payload.update(
            {
                "model_tier": model_tier,
                "requested_tier": requested_tier_original,
                "deep_route": deep_route_used,
                "execution_mode": execution_mode,
                "web_available": run_state.can_web,
                "freshness_required": run_state.freshness_required,
                "plan_reasoning_mode": plan_reasoning_mode,
                "planning_mode": planning_mode,
                "plan_reasoning_level": plan_granularity_level,
            }
        )
        try:
            resource_snapshot = get_resource_snapshot()
        except Exception:
            resource_snapshot = {}
        worker_budget = compute_pool_budget(resource_snapshot, model_tier, ready_worker_count, candidate_count)
        max_parallel_slots = worker_budget.get("max_parallel", 1)
        pressure_limit = bool(worker_budget.get("ram_pressure") or worker_budget.get("vram_pressure"))
        elastic_parallel = not pressure_limit
        if pressure_limit:
            min_parallel_slots = 1
            force_parallel = False
            allow_parallel = False
            if run_state:
                run_state.add_dev_trace(
                    "Memory pressure detected; reducing parallelism.",
                    {
                        "ram_pressure": worker_budget.get("ram_pressure"),
                        "vram_pressure": worker_budget.get("vram_pressure"),
                    },
                )
        worker_budget["ready_workers"] = ready_worker_count
        worker_budget["ready_variants"] = len(ready_worker_models)
        worker_budget["min_parallel"] = min_parallel_slots
        worker_budget["elastic_parallel"] = elastic_parallel
        if ready_worker_count <= 0 and not model_manager:
            max_parallel_slots = 1
            worker_budget["max_parallel"] = 1
            allow_parallel = False
        else:
            worker_budget["max_parallel"] = max_parallel_slots
        if force_parallel and max_parallel_slots < min_parallel_slots:
            max_parallel_slots = min_parallel_slots
            worker_budget["max_parallel"] = max_parallel_slots
        desired_slots = desired_parallelism(router_decision, worker_budget, strict_mode=strict_mode)
        desired_slots = max(min_parallel_slots, desired_slots) if force_parallel else desired_slots
        if pressure_limit:
            desired_slots = max(1, min(max_parallel_slots, desired_slots))
        else:
            desired_slots = max(1, desired_slots)
        if chat_fallback_model:
            max_parallel_slots = 1
            desired_slots = 1
            allow_parallel = False
            worker_budget["max_parallel"] = 1
            if run_state:
                run_state.add_dev_trace(
                    "Limiting parallelism due to chat fallback.",
                    {"model": chat_fallback_model},
                )
        worker_budget["desired_parallel"] = desired_slots
        if (
            max_parallel_slots > 1
            and not allow_parallel
            and model_tier != "fast"
            and (ready_worker_count > 0 or model_manager)
            and not pressure_limit
        ):
            allow_parallel = True
        if not allow_parallel:
            if force_parallel:
                allow_parallel = True
            else:
                max_parallel_slots = 1
                desired_slots = 1
        if pressure_limit:
            target_parallel_slots = max(1, min(max_parallel_slots, desired_slots))
        else:
            target_parallel_slots = max(1, desired_slots)
        if force_parallel:
            target_parallel_slots = max(min_parallel_slots, target_parallel_slots)
        loop = asyncio.get_running_loop()
        last_resource_refresh = loop.time()
        if target_parallel_slots > 1:
            await maybe_emit_work_log(
                run_state,
                bus,
                run_id,
                "warm_workers",
                "Warming up worker models in parallel.",
            )
            asyncio.create_task(
                warm_worker_pool(
                    lm_client,
                    active_models,
                    target_parallel_slots,
                    ready_models=ready_worker_models,
                    run_state=run_state,
                    run_id=run_id,
                    bus=bus,
                    model_manager=model_manager,
                )
            )
        decision_payload["resource_budget"] = worker_budget
        decision_payload["desired_parallel"] = target_parallel_slots
        worker_pool_models: List[str] = []
        worker_pool_instances = 0
        if model_manager:
            instances = await model_manager.worker_pool.list_instances()
            worker_pool_instances = len(instances)
            worker_pool_models = sorted(
                {inst.model_key for inst in instances if inst.model_key}
            )
        else:
            worker_pool_models = [
                m
                for m in [
                    (active_models.get("worker") or {}).get("model"),
                    (active_models.get("worker_b") or {}).get("model"),
                    (active_models.get("worker_c") or {}).get("model"),
                ]
                if m
            ]
            worker_pool_instances = len(worker_pool_models)
        planner_label = (planner_endpoint or {}).get("model")
        executor_label = (executor_endpoint or {}).get("model")
        verifier_label = (active_models.get("verifier") or {}).get("model")
        if model_manager:
            planner_label = "AUTO"
            executor_label = "AUTO"
            verifier_label = "AUTO"
        team_roster = {
            "planner": planner_label,
            "executor": executor_label,
            "worker_pool": {
                "models": worker_pool_models,
                "instances": worker_pool_instances,
            },
            "verifier": verifier_label,
        }
        decision_payload["team"] = team_roster
        await db.update_run_router(run_id, decision_payload)
        await bus.emit(run_id, "router_decision", decision_payload)
        queue_narration(
            lm_client,
            active_models,
            run_state,
            bus,
            run_id,
            question,
            "router_decision",
            decision_payload,
        )
        await bus.emit(
            run_id,
            "resource_budget",
            {
                "budget": worker_budget,
                "resources": resource_snapshot,
                "allow_parallel": allow_parallel,
                "desired_parallel": target_parallel_slots,
            },
        )
        budget_signature = (
            worker_budget.get("max_parallel"),
            worker_budget.get("ram_slots"),
            worker_budget.get("vram_slots"),
            worker_budget.get("ram_pressure"),
            worker_budget.get("vram_pressure"),
            target_parallel_slots,
        )
        await bus.emit(run_id, "team_roster", team_roster)
        if strict_mode:
            await bus.emit(run_id, "strict_mode", {"enabled": True})
        tier_note = model_tier.upper()
        if requested_tier_original != model_tier:
            tier_note = f"{requested_tier_original.upper()}->{model_tier.upper()}"
        await bus.emit(run_id, "client_note", {"note": f"{tier_note} mode: {execution_mode} (route {deep_route_used})"})

        # Conversation memory retrieval (facts only within this chat).
        mem_hits = await db.search_memory(question, conversation_id=conversation_id, limit=5)
        memory_lines: List[str] = []
        for item in mem_hits:
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            if not content and title:
                memory_lines.append(title)
            elif content and (not title or title.lower() == content.lower()):
                memory_lines.append(content)
            elif content:
                memory_lines.append(f"{title}: {content}")
        memory_context = "; ".join([line for line in memory_lines if line])
        artifacts: List[Artifact] = []
        if mem_hits:
            mem_art = Artifact(step_id=0, key="memory_context", artifact_type="memory", content_text=memory_context, content_json={"items": mem_hits})
            artifacts.append(mem_art)
        await bus.emit(run_id, "memory_retrieved", {"count": len(mem_hits)})
        queue_narration(
            lm_client,
            active_models,
            run_state,
            bus,
            run_id,
            question,
            "memory_retrieved",
            {"count": len(mem_hits)},
        )

        upload_id_list = upload_ids or [u["id"] for u in await db.list_uploads(run_id)]
        if upload_id_list:
            upload_artifacts, upload_summary = await process_uploads(
                run_id, question, upload_id_list, db, bus, lm_client, active_models, run_state=run_state
            )
            if upload_artifacts:
                artifacts.extend(upload_artifacts)
                if upload_summary:
                    memory_context = (memory_context + "; " if memory_context else "") + f"Uploads: {upload_summary}"
                    await maybe_emit_work_log(
                        run_state,
                        bus,
                        run_id,
                        "uploads",
                        "Reviewed your uploads and noted key details.",
                    )

        use_plan_pipeline = planning_mode == "extensive" or plan_reasoning_mode == "extensive"
        if use_plan_pipeline:
            plan_result = await run_plan_pipeline(
                db_path=db.path,
                question=question,
                reasoning_mode=plan_reasoning_mode,
                planning_mode=planning_mode,
                reasoning_level=plan_granularity_level,
                max_parallel=target_parallel_slots,
                ram_headroom_pct=ram_headroom_pct,
                vram_headroom_pct=vram_headroom_pct,
                max_concurrent_runs=max_concurrent_runs,
                per_model_class_limits=per_model_class_limits or {},
                model_manager=model_manager,
                bus=bus,
                run_id=run_id,
            )
            final_answer = plan_result.get("final_text") or "Plan completed without a final draft."
            assistant_msg = await db.add_message(run_id, conversation_id, "assistant", final_answer)
            await bus.emit(
                run_id,
                "message_added",
                {
                    "id": assistant_msg.get("id"),
                    "role": "assistant",
                    "content": final_answer,
                    "run_id": run_id,
                    "created_at": assistant_msg.get("created_at"),
                },
            )
            await db.finalize_run(run_id, final_answer, "MED")
            await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "MED"})
            return

        is_math_only = build_math_tool_request(question) is not None
        if is_math_only:
            step_plan = build_linear_plan(
                question,
                router_decision,
                depth_profile,
                needs_verify=bool(strict_mode),
                worker_slots=target_parallel_slots,
                prefer_parallel=allow_parallel,
            )
        elif model_tier == "fast":
            step_plan = build_fast_plan(
                question,
                router_decision,
                needs_verify=bool(strict_mode),
            )
        elif execution_mode == "oss_team":
            step_plan = build_linear_plan(
                question,
                router_decision,
                depth_profile,
                needs_verify=True,
                worker_slots=target_parallel_slots,
                prefer_parallel=allow_parallel,
            )
        else:
            step_plan = await build_step_plan(
                lm_client,
                active_models["orch"],
                question,
                router_decision,
                depth_profile,
                memory_context,
                planner_endpoint=planner_endpoint,
                desired_parallel=target_parallel_slots,
                run_state=run_state,
            )
        if run_state.can_web and router_decision.needs_web:
            step_plan = ensure_parallel_research(step_plan, target_parallel_slots, router_decision)
        if is_math_only:
            step_plan = strip_research_steps(step_plan)
            if run_state:
                run_state.add_dev_trace("Skipping research steps for math-only task.")
        if not router_decision.needs_web:
            step_plan = strip_research_steps(step_plan)
        if not run_state.can_web:
            step_plan = strip_research_steps(step_plan)
            if run_state:
                run_state.add_dev_trace("Web unavailable; stripped research steps.")
        if not is_math_only and target_parallel_slots > 1 and (not router_decision.needs_web or not run_state.can_web):
            step_plan = ensure_parallel_analysis(step_plan, target_parallel_slots)
        step_plan = loosen_parallel_dependencies(step_plan, allow_parallel, target_parallel_slots)
        step_plan = ensure_finalize_step(step_plan)
        for step in step_plan.steps:
            profile = (step.agent_profile or "").strip()
            profile_lower = profile.lower()
            if step.type == "draft":
                if profile_lower != "writer":
                    step.agent_profile = "Writer"
            elif step.type == "finalize":
                if profile_lower != "finalizer":
                    step.agent_profile = "Finalizer"
            elif step.type in ("verify", "verifier", "verifier_worker"):
                if profile_lower != "verifier":
                    step.agent_profile = "Verifier"
        max_steps = depth_profile.get("max_steps", len(step_plan.steps))
        if any(s.type == "finalize" for s in step_plan.steps):
            max_steps = max_steps + 1
        step_plan = trim_step_plan(step_plan, max_steps)
        step_plan.global_constraints.setdefault("expected_passes", router_decision.expected_passes)
        step_plan.global_constraints.setdefault("model_tier", model_tier)
        step_plan.global_constraints.setdefault("route", deep_route_used)
        progress_meta = compute_progress_meta(step_plan, step_plan.global_constraints.get("expected_passes", 1))
        default_response_guidance = response_guidance_text(question, router_decision.reasoning_level, progress_meta)
        step_plan.global_constraints.setdefault("response_guidance", default_response_guidance)
        step_plan.global_constraints["max_loops"] = max(
            step_plan.global_constraints.get("max_loops", 1), progress_meta["counted_passes"] - 1
        )
        response_guidance = step_plan.global_constraints.get("response_guidance", default_response_guidance)
        if not run_state.can_web:
            response_guidance += (
                " Web browsing is unavailable. Clearly label what is verified from provided materials vs assumptions, "
                "and invite the user to share links for verification."
            )
            if run_state.freshness_required:
                response_guidance += " The user asked for up-to-date verification; request sources or suggest a browsing-enabled lane."
            step_plan.global_constraints["response_guidance"] = response_guidance
        progress_meta["response_guidance"] = response_guidance
        await seed_research_prompts(
            lm_client,
            executor_endpoint
            or active_models.get("fast")
            or active_models.get("summarizer")
            or active_models.get("router")
            or active_models.get("orch"),
            question,
            router_decision,
            step_plan,
            run_state=run_state,
        )
        await db.add_step_plan(run_id, step_plan.model_dump())
        plan_payload = {
            "steps": len(step_plan.steps),
            "expected_total_steps": progress_meta["total_steps"],
            "expected_passes": progress_meta["counted_passes"],
        }
        await bus.emit(run_id, "plan_created", plan_payload)
        queue_narration(
            lm_client,
            active_models,
            run_state,
            bus,
            run_id,
            question,
            "plan_created",
            plan_payload,
        )
        if is_math_only:
            plan_note = "Game plan: compute locally, then answer directly."
        elif run_state.can_web and router_decision.needs_web:
            plan_note = "Game plan: gather sources, compare findings, then draft a clear answer."
        elif run_state.can_web:
            plan_note = "Game plan: use local knowledge/tools, then draft a clear answer."
        else:
            plan_note = "Game plan: use what you provided (plus local context), then draft a best-effort answer and flag uncertainties."
        if upload_id_list:
            plan_note = "Game plan: review your uploads first, then " + plan_note.split("Game plan: ", 1)[-1]
        await maybe_emit_work_log(run_state, bus, run_id, "plan", plan_note)
        executor_brief = None
        if model_tier != "fast":
            executor_brief = await build_executor_brief(
                lm_client, executor_endpoint, question, step_plan, target_parallel_slots, run_state=run_state
            )
            if executor_brief:
                await bus.emit(run_id, "executor_brief", executor_brief)
        # Build lookup for dependency-aware scheduling
        step_lookup: Dict[int, PlanStep] = {s.step_id: s for s in step_plan.steps}
        completed_steps: Set[int] = set()
        running_tasks: Dict[int, asyncio.Task] = {}
        infinite_retries = step_plan.global_constraints.get("infinite_retries")
        if infinite_retries is None:
            infinite_retries = True
        if infinite_retries:
            max_loops = float("inf")
        else:
            max_loops = max(step_plan.global_constraints.get("max_loops", 1), progress_meta["counted_passes"] - 1)
        loops = 0
        backtrack_counts: Dict[int, int] = {}
        max_backtracks_per_step = (
            0
            if infinite_retries
            else int(step_plan.global_constraints.get("max_backtracks_per_step", 2) or 2)
        )
        max_backtracks_total = (
            0
            if infinite_retries
            else int(
                step_plan.global_constraints.get("max_backtracks_total", max_backtracks_per_step * 2)
                or max_backtracks_per_step * 2
            )
        )
        backtrack_total = 0
        stop_requested = False
        user_stop_requested = False
        fast_endpoint = executor_endpoint or active_models.get("summarizer") or active_models.get("router") or active_models["orch"]

        async def cancel_running_tasks() -> None:
            if not running_tasks:
                return
            for t in running_tasks.values():
                t.cancel()
            await asyncio.gather(*running_tasks.values(), return_exceptions=True)
            running_tasks.clear()

        def normalize_control_steps(raw_steps: List[Dict[str, Any]]) -> List[PlanStep]:
            used_ids = set(step_lookup.keys())
            next_id = max(used_ids, default=0) + 1
            normalized: List[PlanStep] = []
            default_profiles = {
                "analysis": "Summarizer",
                "draft": "Writer",
                "finalize": "Finalizer",
                "verify": "Verifier",
                "research": "ResearchPrimary",
                "merge": "Summarizer",
            }
            for raw in raw_steps:
                if not isinstance(raw, dict):
                    continue
                raw_id = raw.get("step_id") or raw.get("id") or raw.get("step")
                step_id = _coerce_int(raw_id)
                if step_id is None or step_id in used_ids:
                    step_id = next_id
                    next_id += 1
                used_ids.add(step_id)
                depends = raw.get("depends_on") or raw.get("depends") or raw.get("deps") or []
                if isinstance(depends, (str, int)):
                    depends = [depends]
                dep_ids: List[int] = []
                if isinstance(depends, list):
                    for dep in depends:
                        dep_id = _coerce_int(dep)
                        if dep_id is None:
                            continue
                        dep_ids.append(dep_id)
                step_type = str(raw.get("type") or "analysis").strip() or "analysis"
                name = str(raw.get("name") or f"Step {step_id}").strip() or f"Step {step_id}"
                profile = str(raw.get("agent_profile") or "").strip()
                if not profile:
                    profile = default_profiles.get(step_type.lower(), "Summarizer")
                inputs = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {}
                outputs = raw.get("outputs") if isinstance(raw.get("outputs"), list) else []
                acceptance = raw.get("acceptance_criteria") if isinstance(raw.get("acceptance_criteria"), list) else []
                on_fail = raw.get("on_fail") if isinstance(raw.get("on_fail"), dict) else {}
                try:
                    normalized.append(
                        PlanStep(
                            step_id=step_id,
                            name=name,
                            type=step_type,
                            depends_on=sorted(set(dep_ids)),
                            agent_profile=profile,
                            inputs=inputs,
                            outputs=outputs,
                            acceptance_criteria=acceptance,
                            on_fail=on_fail,
                        )
                    )
                except Exception:
                    continue
            return normalized

        async def apply_control_action(control: ControlCommand, origin: str) -> None:
            nonlocal stop_requested, progress_meta
            if control.control == "CONTINUE" and not control.new_constraints and not control.steps:
                return
            plan_changed = False
            if control.new_constraints:
                step_plan.global_constraints.update(control.new_constraints)
                plan_changed = True
            if control.control == "ADD_STEPS" and control.steps:
                new_steps = normalize_control_steps(control.steps)
                if new_steps:
                    insertion = len(step_plan.steps)
                    for offset, ps in enumerate(new_steps):
                        step_plan.steps.insert(insertion + offset, ps)
                        step_lookup[ps.step_id] = ps
                    plan_changed = True
            elif control.control == "BACKTRACK" and control.to_step:
                nonlocal backtrack_total
                if origin == "system" and max_backtracks_per_step and max_backtracks_total:
                    count = backtrack_counts.get(control.to_step, 0) + 1
                    backtrack_counts[control.to_step] = count
                    backtrack_total += 1
                    if count > max_backtracks_per_step or backtrack_total > max_backtracks_total:
                        if run_state:
                            await maybe_emit_work_log(
                                run_state,
                                bus,
                                run_id,
                                "backtrack_limit",
                                "Backtrack limit reached; continuing with the best available evidence.",
                                tone="warn",
                            )
                        control = ControlCommand(control="CONTINUE", reason="backtrack_limit")
                        plan_changed = False
                        await db.add_control_action(run_id, control.model_dump())
                        control_payload = control.model_dump()
                        control_payload["origin"] = origin
                        await bus.emit(run_id, "control_action", control_payload)
                        queue_narration(
                            lm_client,
                            active_models,
                            run_state,
                            bus,
                            run_id,
                            question,
                            "control_action",
                            control_payload,
                        )
                        return
                await cancel_running_tasks()
                allowed = {sid for sid in completed_steps if sid < control.to_step}
                completed_steps.clear()
                completed_steps.update(allowed)
            elif control.control == "RERUN_STEP" and control.step_id:
                await cancel_running_tasks()
                completed_steps.discard(control.step_id)
            elif control.control == "STOP":
                stop_requested = True
                await cancel_running_tasks()
            if plan_changed:
                await db.add_step_plan(run_id, step_plan.model_dump())
                progress_meta = compute_progress_meta(step_plan, progress_meta.get("counted_passes", 1))
                plan_payload = {
                    "steps": len(step_plan.steps),
                    "expected_total_steps": progress_meta.get("total_steps"),
                    "expected_passes": progress_meta.get("counted_passes"),
                    "origin": origin,
                }
                await bus.emit(run_id, "plan_updated", plan_payload)
                queue_narration(
                    lm_client,
                    active_models,
                    run_state,
                    bus,
                    run_id,
                    question,
                    "plan_updated",
                    plan_payload,
                )
            await db.add_control_action(run_id, control.model_dump())
            control_payload = control.model_dump()
            control_payload["origin"] = origin
            await bus.emit(run_id, "control_action", control_payload)
            queue_narration(
                lm_client,
                active_models,
                run_state,
                bus,
                run_id,
                question,
                "control_action",
                control_payload,
            )

        async def drain_control_queue() -> None:
            if control_queue is None:
                return
            while True:
                try:
                    control = control_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(control, ControlCommand):
                    await apply_control_action(control, origin="user")

        async def start_step(step: PlanStep, snapshot: List[Artifact]) -> asyncio.Task:
            if step.type in ("research", "tavily_search", "search") and run_state.can_web:
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "search",
                    "Pulling sources to back up the answer.",
                )
            elif step.type in ("tavily_extract", "extract") and run_state.can_web:
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "read_sources",
                    "Reading sources and pulling out the key details.",
                )
            elif step.type == "merge":
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "compare",
                    "Cross-checking notes across sources for conflicts.",
                )
            elif step.type == "verify":
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "verify",
                    "Giving the draft a quick consistency check.",
                )
            elif step.type == "draft":
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "draft",
                    "Putting together the answer and flagging caveats.",
                )
            elif step.type == "finalize":
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "finalize",
                    "Polishing the final response.",
                )
            step_payload = {
                "step_id": step.step_id,
                "name": step.name,
                "type": step.type,
                "agent_profile": step.agent_profile,
            }
            await bus.emit(run_id, "step_started", step_payload)
            queue_narration(
                lm_client,
                active_models,
                run_state,
                bus,
                run_id,
                question,
                "step_started",
                step_payload,
            )
            step_run_id = await db.add_step_run(
                run_id,
                step.step_id,
                status="running",
                agent_profile=step.agent_profile,
                prompt_text="",
            )

            async def runner() -> Dict[str, Any]:
                try:
                    output, new_artifacts, prompt_used = await execute_step(
                        run_id,
                        question,
                        step,
                        router_decision,
                        search_depth_mode,
                        depth_profile,
                        snapshot,
                        progress_meta,
                        response_guidance,
                        lm_client,
                        tavily,
                        db,
                        bus,
                        active_models,
                        conversation_id=conversation_id,
                        upload_dir=upload_dir,
                        run_state=run_state,
                    )
                    await db.update_step_run(step_run_id, status="completed", output_json=output)
                    await db.execute("UPDATE step_runs SET prompt_text=? WHERE id=?", (prompt_used, step_run_id))
                    for art in new_artifacts:
                        await db.add_artifact(run_id, art)
                        if art.artifact_type == "draft":
                            await db.add_draft(run_id, art.content_text or "")
                        if art.artifact_type == "verifier" and art.content_json:
                            await db.add_verifier_report(
                                run_id,
                                art.content_json.get("verdict", ""),
                                art.content_json.get("issues", []),
                                art.content_json.get("revised_answer"),
                            )
                    return {
                        "status": "completed",
                        "step": step,
                        "artifacts": new_artifacts,
                        "output": output,
                    }
                except Exception as exc:
                    await db.update_step_run(step_run_id, status="error", error_text=str(exc))
                    error_payload = {
                        "step": step.step_id,
                        "name": step.name,
                        "type": step.type,
                        "agent_profile": step.agent_profile,
                        "message": str(exc),
                    }
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                        try:
                            detail = lm_client._extract_error_detail(exc.response)
                        except Exception:
                            detail = ""
                        if detail:
                            error_payload["detail"] = detail
                    await bus.emit(run_id, "step_error", error_payload)
                    queue_narration(
                        lm_client,
                        active_models,
                        run_state,
                        bus,
                        run_id,
                        question,
                        "step_error",
                        error_payload,
                        tone="warn",
                    )
                    return {"status": "error", "step": step, "error": str(exc)}

            return asyncio.create_task(runner())

        def deps_satisfied(step: PlanStep) -> bool:
            return all(dep in completed_steps for dep in step.depends_on)

        while len(completed_steps) < len(step_plan.steps) and not stop_requested:
            if stop_event and stop_event.is_set():
                user_stop_requested = True
                stop_requested = True
                await cancel_running_tasks()
                break
            if not run_state.can_chat:
                await cancel_running_tasks()
                stop_requested = True
                break
            await drain_control_queue()
            if stop_requested:
                break
            if loop.time() - last_resource_refresh >= RESOURCE_REFRESH_SECS:
                last_resource_refresh = loop.time()
                try:
                    resource_snapshot = get_resource_snapshot()
                except Exception:
                    resource_snapshot = {}
                if model_manager:
                    instances = await model_manager.worker_pool.list_instances()
                    ready_instances = [inst for inst in instances if inst.status != "busy"]
                    ready_worker_count = len(ready_instances)
                    ready_worker_models = {(inst.endpoint, inst.model_key) for inst in ready_instances}
                    candidate_count = len(await model_manager.get_candidates())
                worker_budget = compute_pool_budget(
                    resource_snapshot, model_tier, ready_worker_count, candidate_count or len(ready_worker_models)
                )
                max_parallel_slots = worker_budget.get("max_parallel", 1)
                pressure_limit = bool(worker_budget.get("ram_pressure") or worker_budget.get("vram_pressure"))
                elastic_parallel = not pressure_limit
                if pressure_limit:
                    min_parallel_slots = 1
                    force_parallel = False
                    allow_parallel = False
                worker_budget["ready_workers"] = ready_worker_count
                worker_budget["ready_variants"] = len(ready_worker_models)
                worker_budget["min_parallel"] = min_parallel_slots
                worker_budget["elastic_parallel"] = elastic_parallel
                if ready_worker_count <= 0 and not model_manager:
                    max_parallel_slots = 1
                    worker_budget["max_parallel"] = 1
                    allow_parallel = False
                else:
                    worker_budget["max_parallel"] = max_parallel_slots
                if force_parallel and max_parallel_slots < min_parallel_slots:
                    max_parallel_slots = min_parallel_slots
                    worker_budget["max_parallel"] = max_parallel_slots
                desired_slots = desired_parallelism(router_decision, worker_budget, strict_mode=strict_mode)
                if force_parallel:
                    desired_slots = max(min_parallel_slots, desired_slots)
                if pressure_limit:
                    desired_slots = max(1, min(max_parallel_slots, desired_slots))
                else:
                    desired_slots = max(1, desired_slots)
                worker_budget["desired_parallel"] = desired_slots
                if (
                    max_parallel_slots > 1
                    and not allow_parallel
                    and not pressure_limit
                    and (ready_worker_count > 0 or model_manager)
                    and not chat_fallback_model
                ):
                    allow_parallel = True
                if not allow_parallel:
                    if force_parallel and not pressure_limit:
                        allow_parallel = True
                    else:
                        max_parallel_slots = 1
                        desired_slots = 1
                if pressure_limit:
                    new_target = max(1, min(max_parallel_slots, desired_slots))
                else:
                    new_target = max(1, desired_slots)
                if force_parallel and not pressure_limit:
                    new_target = max(min_parallel_slots, new_target)
                if new_target > target_parallel_slots:
                    asyncio.create_task(
                        warm_worker_pool(
                            lm_client,
                            active_models,
                            new_target,
                            ready_models=ready_worker_models,
                            run_state=run_state,
                            run_id=run_id,
                            bus=bus,
                            model_manager=model_manager,
                        )
                    )
                target_parallel_slots = new_target
                new_signature = (
                    worker_budget.get("max_parallel"),
                    worker_budget.get("ram_slots"),
                    worker_budget.get("vram_slots"),
                    worker_budget.get("ram_pressure"),
                    worker_budget.get("vram_pressure"),
                    target_parallel_slots,
                )
                if new_signature != budget_signature:
                    budget_signature = new_signature
                    await bus.emit(
                        run_id,
                        "resource_budget",
                        {
                            "budget": worker_budget,
                            "resources": resource_snapshot,
                            "allow_parallel": allow_parallel,
                            "desired_parallel": target_parallel_slots,
                        },
                    )
            ready_steps = [
                s for s in step_plan.steps if s.step_id not in completed_steps and s.step_id not in running_tasks and deps_satisfied(s)
            ]
            capacity = target_parallel_slots - len(running_tasks)
            if ready_steps and capacity > 0:
                running_steps = [step_lookup[sid] for sid in running_tasks.keys() if sid in step_lookup]
                allocation = await allocate_ready_steps(
                    lm_client,
                    executor_endpoint,
                    ready_steps,
                    artifacts,
                    len(running_tasks),
                    target_parallel_slots,
                    question=question,
                    resource_budget=worker_budget,
                    running_steps=running_steps,
                    run_state=run_state,
                )
                start_ids = allocation.start_ids
                if allocation.target_slots is not None:
                    suggested_slots = max(1, int(allocation.target_slots))
                    if pressure_limit:
                        suggested_slots = min(max_parallel_slots, suggested_slots)
                    if not allow_parallel:
                        suggested_slots = 1
                    elif target_parallel_slots > 1:
                        # Keep target slots at capacity unless resources force a reduction.
                        suggested_slots = max(target_parallel_slots, suggested_slots)
                    target_parallel_slots = suggested_slots
                if allow_parallel:
                    queued_ids = allocation.queue_ids or [s.step_id for s in ready_steps if s.step_id not in start_ids]
                    alloc_payload = {
                        "start_ids": start_ids,
                        "ready_ids": [s.step_id for s in ready_steps],
                        "queue_ids": queued_ids,
                        "target_slots": target_parallel_slots,
                        "note": allocation.note,
                        "used_executor": allocation.used_executor,
                    }
                    await bus.emit(run_id, "allocator_decision", alloc_payload)
                    queue_narration(
                        lm_client,
                        active_models,
                        run_state,
                        bus,
                        run_id,
                        question,
                        "allocator_decision",
                        alloc_payload,
                    )
                for step in ready_steps:
                    if step.step_id in start_ids and step.step_id not in running_tasks:
                        running_tasks[step.step_id] = await start_step(step, list(artifacts))

            if not running_tasks:
                # No runnable steps left; avoid deadlock
                break

            stop_waiter = None
            wait_tasks = list(running_tasks.values())
            if stop_event and not stop_event.is_set():
                stop_waiter = asyncio.create_task(stop_event.wait())
                wait_tasks.append(stop_waiter)
            done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            if stop_waiter and stop_waiter in done:
                user_stop_requested = True
                stop_requested = True
                await cancel_running_tasks()
                break
            if stop_waiter:
                stop_waiter.cancel()
                await asyncio.gather(stop_waiter, return_exceptions=True)
            for task in done:
                if stop_waiter and task is stop_waiter:
                    continue
                result = task.result()
                step_id = result["step"].step_id
                running_tasks.pop(step_id, None)
                if result["status"] != "completed":
                    completed_steps.add(step_id)  # prevent deadlock; move on if a step errors
                    continue
                completed_steps.add(step_id)
                artifacts.extend(result["artifacts"])
                step_done_payload = {
                    "step_id": step_id,
                    "name": result["step"].name,
                    "type": result["step"].type,
                    "agent_profile": result["step"].agent_profile,
                }
                await bus.emit(run_id, "step_completed", step_done_payload)
                queue_narration(
                    lm_client,
                    active_models,
                    run_state,
                    bus,
                    run_id,
                    question,
                    "step_completed",
                    step_done_payload,
                )
                tool_requests = []
                tool_results = []
                validation_summary = ""
                try:
                    tool_requests, tool_results, validation_summary = await run_step_double_checks(
                        lm_client,
                        active_models,
                        result["step"],
                        question,
                        result["output"],
                        artifacts,
                        worker_budget,
                        strict_mode,
                        run_id,
                        bus,
                        db=db,
                        conversation_id=conversation_id,
                        upload_dir=upload_dir,
                        run_state=run_state,
                    )
                except Exception:
                    tool_requests = []
                    tool_results = []
                    validation_summary = ""
                if tool_requests or tool_results:
                    validation_artifact = Artifact(
                        step_id=step_id,
                        key=f"validation_step_{step_id}",
                        artifact_type="validation",
                        content_text=validation_summary,
                        content_json={
                            "summary": validation_summary,
                            "tool_requests": tool_requests,
                            "tool_results": tool_results,
                        },
                    )
                    await db.add_artifact(run_id, validation_artifact)
                    artifacts.append(validation_artifact)

                if model_tier != "fast":
                    fast_control, escalate = await evaluate_control_fast(
                        lm_client,
                        fast_endpoint,
                        result["step"],
                        result["output"],
                        validation_summary=validation_summary,
                        run_state=run_state,
                    )
                    control: ControlCommand = fast_control
                    # Only pull in the OSS orchestrator for heavyweight checkpoints or when escalation is requested.
                    heavy_types = {"merge", "draft", "verify", "analysis"}
                    needs_oss = (
                        escalate
                        or result["step"].type in heavy_types
                        or (strict_mode and fast_control.control != "CONTINUE")
                    )
                    if needs_oss:
                        control = await evaluate_control(
                            lm_client,
                            planner_endpoint,
                            result["step"],
                            result["output"],
                            validation_summary=validation_summary,
                            run_state=run_state,
                        )
                    if control.control != "CONTINUE" or control.new_constraints:
                        await apply_control_action(control, origin="system")
                        if stop_requested:
                            break

            # If all steps finished but verifier asked for a loop, reset to research/merge phase.
            if not running_tasks and len(completed_steps) >= len(step_plan.steps) and loops < max_loops:
                verifier_art = next((a for a in artifacts if a.artifact_type == "verifier"), None)
                if verifier_art and verifier_art.content_json and verifier_art.content_json.get("verdict") == "NEEDS_REVISION":
                    loops += 1
                    actual_passes = loops + 1
                    if actual_passes > progress_meta.get("counted_passes", 1):
                        progress_meta["counted_passes"] = actual_passes
                        progress_meta["total_steps"] += progress_meta.get("per_pass_rerun", 0)
                    completed_reset_to = len([s for s in step_plan.steps if s.type == "analysis"])
                    loop_payload = {
                        "iteration": loops,
                        "expected_total_steps": progress_meta.get("total_steps"),
                        "completed_reset_to": completed_reset_to,
                        "counted_passes": progress_meta.get("counted_passes"),
                    }
                    await bus.emit(run_id, "loop_iteration", loop_payload)
                    queue_narration(
                        lm_client,
                        active_models,
                        run_state,
                        bus,
                        run_id,
                        question,
                        "loop_iteration",
                        loop_payload,
                        tone="warn",
                    )
                    completed_steps = {s.step_id for s in step_plan.steps if s.type == "analysis"}  # keep upfront steps
                    # keep artifacts but rerun research+draft+verify
                    await cancel_running_tasks()
                    continue

        if user_stop_requested:
            await db.update_run_status(run_id, "stopped")
            await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "LOW", "stopped": True})
            return
        # finalize
        if not run_state.can_chat:
            guidance = (
                "Local model rejected the request; check model name, /v1/models, and strip unsupported fields. "
                "If you're using LM Studio, confirm the model is loaded and the base URL is correct."
            )
            assistant_msg = await db.add_message(run_id, conversation_id, "assistant", guidance)
            await bus.emit(
                run_id,
                "message_added",
                {
                    "id": assistant_msg.get("id"),
                    "role": "assistant",
                    "content": guidance,
                    "run_id": run_id,
                    "created_at": assistant_msg.get("created_at"),
                },
            )
            await db.finalize_run(run_id, guidance, "LOW")
            await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "LOW"})
            return
        final_art = _latest_artifact(artifacts, "final")
        draft_art = _latest_artifact(artifacts, "draft")
        verifier_art = _latest_artifact(artifacts, "verifier")
        ledger_art = _latest_artifact(artifacts, "ledger")
        final_step_present = any(s.type == "finalize" for s in step_plan.steps)
        final_answer = ""
        if final_art and final_art.content_text:
            final_answer = final_art.content_text
        elif not final_step_present:
            final_answer = (verifier_art.content_json.get("revised_answer") if verifier_art and verifier_art.content_json else None) or (
                draft_art.content_text if draft_art else ""
            )
        if not final_answer and not final_step_present:
            # Fallback: force a concise answer from available context + model knowledge.
            ledger_json = ledger_art.content_json if ledger_art else merge_evidence_artifacts(artifacts)
            fallback_prompt = (
                f"Question: {question}\n"
                f"Evidence (may be partial): {json.dumps(ledger_json)[:2800]}\n"
                "Provide the best direct answer you can. If evidence is light, rely on your own knowledge but flag any uncertainty.\n"
                "Do not include citations or a Sources section.\n"
                "Return a short, clear answer without chain-of-thought."
            )
            try:
                final_answer = await run_worker(
                    lm_client,
                    "Writer",
                    active_models,
                    fallback_prompt,
                    temperature=0.25,
                    max_tokens=700,
                    run_id=run_id,
                    bus=bus,
                    context="fallback_answer",
                    run_state=run_state,
                    model_manager=run_state.model_manager if run_state else None,
                )
            except Exception:
                final_answer = final_answer or "Unable to produce an answer with the available context."
        if not final_answer and final_step_present:
            final_answer = "Unable to finalize the response; please retry."
        if final_answer and _looks_like_tool_markup(final_answer):
            final_answer = "Unable to finalize the response; please retry."
        if not run_state.can_web:
            notes = [
                "Verification note: I couldn't browse the web here, so I relied on the prompt and any provided materials; anything beyond that is unverified.",
            ]
            if run_state.freshness_required:
                notes.append("If you need up-to-date verification, share links or switch to a browsing-enabled lane.")
            final_answer = (final_answer or "").rstrip() + "\n\n" + "\n".join(notes)
        confidence = "MED"
        if verifier_art and verifier_art.content_json:
            verdict = verifier_art.content_json.get("verdict", "PASS")
            confidence = "HIGH" if verdict == "PASS" else "LOW"
        assistant_msg = await db.add_message(run_id, conversation_id, "assistant", final_answer)
        await bus.emit(
            run_id,
            "message_added",
            {"id": assistant_msg.get("id"), "role": "assistant", "content": final_answer, "run_id": run_id, "created_at": assistant_msg.get("created_at")},
        )
        await db.finalize_run(run_id, final_answer, confidence)
        if evidence_dump:
            try:
                sources = await db.get_sources(run_id)
                claims = await db.get_claims(run_id)
                verifier_report = await db.get_verifier_report(run_id)
                dump_payload = {
                    "sources": sources,
                    "claims": claims,
                    "verifier": verifier_report,
                }
                dump_artifact = Artifact(
                    step_id=0,
                    key="evidence_dump",
                    artifact_type="evidence_dump",
                    content_text="",
                    content_json=dump_payload,
                )
                await db.add_artifact(run_id, dump_artifact)
            except Exception:
                pass
        existing_run_memory = await db.get_run_memory(run_id)
        if auto_memory and final_answer and not existing_run_memory:
            facts: List[str] = []
            try:
                facts_prompt = (
                    "Extract up to 6 concise factual statements learned from this chat. "
                    "Only include facts explicitly stated or confirmed here (no advice, no questions). "
                    "Return JSON only as {\"activity_lines\": [], \"memory_notes\": [...], \"candidate_memory\": []}.\n"
                    f"User question: {question}\nAssistant answer: {final_answer}"
                )
                facts_raw = await run_worker(
                    lm_client,
                    "Summarizer",
                    active_models,
                    facts_prompt,
                    temperature=0.2,
                    max_tokens=220,
                    run_id=run_id,
                    bus=bus,
                    context="memory_facts",
                    run_state=run_state,
                    model_manager=run_state.model_manager if run_state else None,
                )
                fixer_model = (
                    (active_models.get("summarizer") or {}).get("model")
                    or (active_models.get("router") or {}).get("model")
                    or (active_models.get("orch") or {}).get("model")
                    or ""
                )
                parsed = await safe_json_parse(
                    facts_raw,
                    lm_client,
                    fixer_model,
                    run_state=run_state,
                    model_manager=run_state.model_manager if run_state else None,
                )
                if isinstance(parsed, dict):
                    notes = parsed.get("memory_notes") or parsed.get("facts") or parsed.get("candidate_memory") or []
                    if isinstance(notes, list):
                        facts = [str(note).strip() for note in notes if str(note).strip()]
                    elif isinstance(notes, str) and notes.strip():
                        facts = [notes.strip()]
            except Exception:
                facts = []
            if not facts:
                fallback_fact = (final_answer or question or "").strip()
                if fallback_fact:
                    facts = [fallback_fact]
            seen_facts: Set[str] = set()
            cleaned_facts: List[str] = []
            for fact in facts:
                cleaned = str(fact).strip()
                if not cleaned or cleaned in seen_facts:
                    continue
                seen_facts.add(cleaned)
                if len(cleaned) > 400:
                    cleaned = cleaned[:400].rstrip() + "..."
                cleaned_facts.append(cleaned)
                if len(cleaned_facts) >= 6:
                    break
            if cleaned_facts:
                for fact in cleaned_facts:
                    title = fact[:80]
                    mem_id = await db.add_memory_item(
                        conversation_id,
                        kind="fact",
                        title=title,
                        content=fact,
                        tags=["fact"],
                        pinned=False,
                        relevance_score=1.0,
                    )
                    await db.link_memory_to_run(run_id, mem_id, "auto_fact")
                    existing_run_memory.append({"id": mem_id})
                await bus.emit(run_id, "memory_saved", {"count": len(cleaned_facts)})
                queue_narration(
                    lm_client,
                    active_models,
                    run_state,
                    bus,
                    run_id,
                    question,
                    "memory_saved",
                    {"count": len(cleaned_facts)},
                )
        try:
            ui_note_prompt = (
                f"Question: {question}\nAnswer: {final_answer[:320]}\n"
                f"Tier: {model_tier}, Route: {deep_route_used}, Confidence: {confidence}\n"
                "Summarize in one short status line for the UI ticker."
            )
            ui_note = await run_worker(
                lm_client,
                "Summarizer",
                active_models,
                ui_note_prompt,
                temperature=0.1,
                max_tokens=120,
                run_id=run_id,
                bus=bus,
                context="ui_note",
                run_state=run_state,
                model_manager=run_state.model_manager if run_state else None,
            )
            await bus.emit(run_id, "client_note", {"note": ui_note, "tier": model_tier, "route": deep_route_used})
        except Exception:
            pass
        archived_payload = {"run_id": run_id, "confidence": confidence}
        await bus.emit(run_id, "archived", archived_payload)
        queue_narration(
            lm_client,
            active_models,
            run_state,
            bus,
            run_id,
            question,
            "archived",
            archived_payload,
        )
    except asyncio.CancelledError:
        await db.update_run_status(run_id, "stopped")
        stopped_payload = {"run_id": run_id, "confidence": "LOW", "stopped": True}
        await bus.emit(run_id, "archived", stopped_payload)
        queue_narration(
            lm_client,
            active_models,
            run_state if "run_state" in locals() else None,
            bus,
            run_id,
            question,
            "archived",
            stopped_payload,
            tone="warn",
        )
        raise
    except Exception as exc:
        await db.update_run_status(run_id, f"error: {exc}")
        error_payload = {"message": str(exc), "fatal": True}
        await bus.emit(run_id, "error", error_payload)
        queue_narration(
            lm_client,
            active_models,
            run_state if "run_state" in locals() else None,
            bus,
            run_id,
            question,
            "error",
            error_payload,
            tone="error",
        )


def new_run_id() -> str:
    return str(uuid.uuid4())
