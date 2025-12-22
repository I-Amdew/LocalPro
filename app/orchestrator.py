import asyncio
import ast
import base64
import json
import math
import operator
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pypdf import PdfReader

from . import agents
from .db import Database
from .llm import LMStudioClient
from .schemas import (
    Artifact,
    ControlCommand,
    PlanStep,
    RouterDecision,
    StepPlan,
    VerifierReport,
)
from .tavily import TavilyClient


# Reasoning depth mapping
REASONING_DEPTHS = {
    "LOW": {"max_steps": 6, "research_rounds": 1, "tool_budget": {"tavily_search": 4, "tavily_extract": 6}},
    "MED": {"max_steps": 10, "research_rounds": 2, "tool_budget": {"tavily_search": 8, "tavily_extract": 10}},
    "HIGH": {"max_steps": 14, "research_rounds": 3, "tool_budget": {"tavily_search": 12, "tavily_extract": 16}, "advanced": True},
    "ULTRA": {"max_steps": 20, "research_rounds": 3, "tool_budget": {"tavily_search": 18, "tavily_extract": 24}, "advanced": True, "strict_verify": True},
}


class EventBus:
    """In-memory fan-out for SSE plus persisted events."""

    def __init__(self, db: Database):
        self.db = db
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self.lock = asyncio.Lock()

    async def emit(self, run_id: str, event_type: str, payload: dict) -> dict:
        stored = await self.db.add_event(run_id, event_type, payload)
        async with self.lock:
            queues = list(self.subscribers.get(run_id, []))
        for q in queues:
            await q.put(stored)
        return stored

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self.lock:
            self.subscribers.setdefault(run_id, []).append(queue)
        return queue

    async def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        async with self.lock:
            queues = self.subscribers.get(run_id, [])
            if queue in queues:
                queues.remove(queue)
            if not queues:
                self.subscribers.pop(run_id, None)


async def safe_json_parse(raw: str, lm_client: LMStudioClient, fixer_model: str) -> Optional[dict]:
    """Try to parse JSON, and fallback to the JSONRepair profile to fix."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        resp = await lm_client.chat_completion(
            model=fixer_model,
            messages=[
                {"role": "system", "content": agents.JSON_REPAIR_SYSTEM},
                {"role": "user", "content": raw},
            ],
            temperature=0.0,
            max_tokens=400,
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


SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
SAFE_UNARY_OPS = {ast.UAdd: lambda v: v, ast.USub: lambda v: -v}
SAFE_NAMES: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "round": round,
    **{name: getattr(math, name) for name in ("sqrt", "log", "log10", "sin", "cos", "tan", "exp", "ceil", "floor", "fabs")},
}


def safe_eval_expr(expr: str) -> Any:
    """Evaluate a basic math/Python expression safely (no attribute access/imports)."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ValueError("Unsupported literal")
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BIN_OPS:
            return SAFE_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPS:
            return SAFE_UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in SAFE_NAMES:
                raise ValueError("Function not allowed")
            args = [_eval(arg) for arg in node.args]
            return SAFE_NAMES[func_name](*args)
        if isinstance(node, ast.Name) and node.id in SAFE_NAMES:
            return SAFE_NAMES[node.id]
        raise ValueError("Disallowed expression")

    return _eval(tree)


def resolve_tool_requests(tool_requests: List[dict]) -> List[dict]:
    """Resolve lightweight tool requests locally (date, calculator, code_eval, image/pdf hints)."""
    resolved: List[dict] = []
    now_iso = datetime.utcnow().isoformat()
    for req in tool_requests or []:
        tool = str(req.get("tool") or req.get("type") or "").lower()
        entry: Dict[str, Any] = {"tool": tool or req.get("tool")}
        try:
            if tool in ("live_date", "time_now", "now", "date"):
                entry["result"] = now_iso
                entry["status"] = "ok"
            elif tool in ("calculator", "calc", "math"):
                expr = str(req.get("expr") or req.get("expression") or req.get("input") or "").strip()
                if not expr:
                    raise ValueError("Missing expression")
                entry["expr"] = expr
                entry["result"] = safe_eval_expr(expr)
                entry["status"] = "ok"
            elif tool in ("code_eval", "code", "python"):
                code = str(req.get("code") or req.get("expr") or req.get("source") or "").strip()
                if not code:
                    raise ValueError("Missing code")
                entry["code"] = code
                entry["result"] = safe_eval_expr(code)
                entry["status"] = "ok"
            elif tool in ("image_zoom", "image_eval", "pdf_scan", "pdf_read", "pdf_inspect"):
                entry["target"] = req.get("target") or req.get("hint") or ""
                entry["result"] = entry["target"] or "Vision/PDF helper available; specify region/pages to inspect."
                entry["status"] = "ok"
            else:
                entry["status"] = "unknown_tool"
                entry["result"] = ""
            resolved.append(entry)
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            resolved.append(entry)
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
    q = (question or "").lower()
    if any(token in q for token in ("today", "current", "latest", "recent", "breaking", "update")):
        return True
    data_signals = ("percent", "percentage", "share", "rate", "price", "cost", "net worth", "worth", "market", "market cap", "revenue", "growth", "forecast", "population", "household", "median", "average", "top", "rank", "list", "survey", "report", "study")
    if any(token in q for token in data_signals):
        return True
    if any(ch.isdigit() for ch in q):
        return True
    return False


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
        style = "Compact but complete (<=350 words) with short sections and sourced bullets."
    if passes > 1:
        style += f" Note progress and clarify if another pass ({passes}) is in flight or expected."
    if reasoning_level in ("ULTRA", "HIGH"):
        style += " Keep sources prominent and state any remaining risks."
    return style


def profile_system(profile: str) -> str:
    return {
        "Orchestrator": agents.MICROMANAGER_SYSTEM,
        "ResearchPrimary": agents.RESEARCH_PRIMARY_SYSTEM,
        "ResearchRecency": agents.RESEARCH_RECENCY_SYSTEM,
        "ResearchAdversarial": agents.RESEARCH_ADVERSARIAL_SYSTEM,
        "Math": agents.MATH_SYSTEM,
        "Critic": agents.CRITIC_SYSTEM,
        "Summarizer": agents.SUMMARIZER_SYSTEM,
        "JSONRepair": agents.JSON_REPAIR_SYSTEM,
        "Verifier": agents.VERIFIER_SYSTEM,
    }.get(profile, agents.RESEARCH_PRIMARY_SYSTEM)


def profile_model(profile: str, model_map: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    """Return (base_url, model_id) for a given profile."""
    if profile == "Orchestrator":
        cfg = model_map.get("orch")
    elif profile in ("Summarizer", "Critic", "JSONRepair"):
        cfg = model_map.get("summarizer") or model_map.get("router") or model_map.get("worker")
    elif profile == "Verifier":
        cfg = model_map.get("verifier") or model_map.get("worker")
    elif profile == "ResearchRecency":
        cfg = model_map.get("worker_b") or model_map.get("worker")
    elif profile == "ResearchAdversarial":
        cfg = model_map.get("worker_c") or model_map.get("worker")
    else:
        cfg = model_map.get("worker")
    if not cfg:
        cfg = {"base_url": "", "model": ""}
    return cfg.get("base_url"), cfg.get("model")


def select_model_suite(
    base_map: Dict[str, Dict[str, str]], tier: str, deep_route: str
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, str], bool, str]:
    """
    Return (model_map, planner_endpoint, control_endpoint, allow_parallel, execution_mode).
    - planner_endpoint is used for plan creation/drafting.
    - control_endpoint is used for control/evaluation (often 4B mini-orchestrator).
    """
    summarizer = base_map.get("summarizer") or base_map.get("router") or base_map.get("orch", {})
    verifier = base_map.get("verifier") or base_map.get("worker") or base_map.get("orch", {})
    if tier == "fast":
        fast_ep = base_map.get("fast") or base_map.get("worker") or base_map.get("orch", {})
        suite = {
            "orch": fast_ep,
            "worker": fast_ep,
            "worker_b": fast_ep,
            "worker_c": fast_ep,
            "router": summarizer,
            "summarizer": summarizer,
            "verifier": verifier,
        }
        return suite, fast_ep, summarizer, False, "fast_linear"
    if tier == "deep":
        if deep_route == "oss":
            oss_ep = base_map.get("orch") or base_map.get("worker") or {}
            suite = {
                "orch": oss_ep,
                "worker": oss_ep,
                "worker_b": oss_ep,
                "worker_c": oss_ep,
                "router": summarizer,
                "summarizer": summarizer,
                "verifier": verifier,
            }
            return suite, oss_ep, summarizer, False, "oss_linear"
        planner = base_map.get("deep_planner") or base_map.get("worker") or base_map.get("orch", {})
        control = base_map.get("deep_orch") or summarizer
        suite = {
            "orch": planner,  # planner/drafter 8B
            "worker": base_map.get("worker") or planner,
            "worker_b": base_map.get("worker_b") or base_map.get("worker") or planner,
            "worker_c": base_map.get("worker_c") or base_map.get("worker") or planner,
            "router": summarizer,
            "summarizer": summarizer,
            "verifier": verifier,
        }
        return suite, planner, control, True, "deep_cluster"
    # default to Pro (full OSS orchestrator)
    orch = base_map.get("orch") or base_map.get("worker") or {}
    return base_map, orch, orch, True, "pro_full"


def resolve_auto_tier(decision: RouterDecision) -> str:
    """Map the router's reasoning depth to the most suitable tier."""
    level = decision.reasoning_level
    if level in ("HIGH", "ULTRA") or (decision.expected_passes or 0) > 1:
        return "pro"
    if level == "MED" or decision.needs_web or decision.extract_depth == "advanced":
        return "deep"
    tool_budget = decision.tool_budget or {}
    if tool_budget.get("tavily_search", 0) > 8:
        return "deep"
    return "fast"


def build_linear_plan(question: str, decision: RouterDecision, depth_profile: dict, needs_verify: bool = True) -> StepPlan:
    """Deterministic lightweight plan for fast/oss-linear modes."""
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
        {
            "step_id": 2,
            "name": "Gather notes",
            "type": "research",
            "depends_on": [1],
            "agent_profile": "ResearchPrimary",
            "inputs": {"use_web": decision.needs_web},
            "outputs": [{"artifact_type": "evidence", "key": "lane_primary"}],
            "acceptance_criteria": ["notes ready"],
            "on_fail": {"action": "rerun_step"},
        },
        {
            "step_id": 3,
            "name": "Draft answer",
            "type": "draft",
            "depends_on": [2],
            "agent_profile": "Orchestrator",
            "inputs": {},
            "outputs": [{"artifact_type": "draft", "key": "draft_answer"}],
            "acceptance_criteria": ["draft_complete"],
            "on_fail": {"action": "revise_step"},
        },
    ]
    if needs_verify:
        steps.append(
            {
                "step_id": 4,
                "name": "Verify",
                "type": "verify",
                "depends_on": [3],
                "agent_profile": "Verifier",
                "inputs": {},
                "outputs": [{"artifact_type": "verifier", "key": "verifier_report"}],
                "acceptance_criteria": ["verdict_ready"],
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


async def choose_deep_route(
    lm_client: LMStudioClient,
    router_endpoint: Dict[str, str],
    question: str,
    preference: str,
    router_decision: Optional[RouterDecision] = None,
) -> str:
    """Router for LocalDeep between OSS linear vs. mini-cluster."""
    if preference in ("oss", "cluster"):
        return preference
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
            if not needs_web and router_decision.reasoning_level in ("LOW", "MED"):
                return "oss"
        except Exception:
            pass
    prompt = (
        "Choose the best execution lane for this question.\n"
        "- Use route 'oss' when the OSS model's internal knowledge should be enough (fact lookup, summarization, no current-events or web search needed).\n"
        "- Use route 'cluster' when current data, web search, multi-source research, or cross-checking is likely required.\n"
        "Return JSON only: {\"route\": \"oss\" | \"cluster\"}."
        f"\nQuestion: {question}"
    )
    try:
        resp = await lm_client.chat_completion(
            model=router_endpoint["model"],
            messages=[{"role": "system", "content": agents.SUMMARIZER_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80,
            base_url=router_endpoint["base_url"],
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, router_endpoint["model"])
        if parsed and parsed.get("route") in ("oss", "cluster"):
            return parsed["route"]
    except Exception:
        pass
    # Fallback heuristic: shorter questions lean oss, else cluster
    return "oss" if len(question) < 120 else "cluster"


async def call_router(
    lm_client: LMStudioClient, endpoint: Dict[str, str], question: str, manual_level: Optional[str] = None, strict_mode: bool = False
) -> RouterDecision:
    user_msg = f"User question: {question}\nReturn JSON only."
    parsed = None
    needs_web_guess = guess_needs_web(question)
    try:
        resp = await lm_client.chat_completion(
            model=endpoint["model"],
            messages=[{"role": "system", "content": agents.ROUTER_SYSTEM}, {"role": "user", "content": user_msg}],
            temperature=0.1,
            max_tokens=300,
            base_url=endpoint["base_url"],
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, endpoint["model"])
    except Exception:
        parsed = None
    if not parsed:
        expected_passes = 2 if strict_mode else 1
        parsed = {
            "needs_web": needs_web_guess,
            "reasoning_level": manual_level or "MED",
            "topic": "general",
            "max_results": 6,
            "extract_depth": "basic",
            "expected_passes": expected_passes,
            "stop_conditions": {},
        }
    decision = RouterDecision(**parsed)
    if manual_level:
        decision.reasoning_level = manual_level
    # If the router was unsure, lean toward web for data-heavy questions.
    decision.needs_web = decision.needs_web or needs_web_guess
    decision.expected_passes = max(1, decision.expected_passes or 1)
    return decision


async def build_step_plan(
    lm_client: LMStudioClient,
    orch_endpoint: Dict[str, str],
    question: str,
    decision: RouterDecision,
    depth_profile: dict,
    memory_context: str = "",
    planner_endpoint: Optional[Dict[str, str]] = None,
) -> StepPlan:
    plan_prompt = (
        "Produce a JSON step plan for answering the question. "
        "Include step_id, name, type, depends_on (list of ids), agent_profile, acceptance_criteria. "
        "Keep 6-12 steps for typical questions. "
        "Add global_constraints.expected_passes (1-3) if a verifier rerun is likely, and response_guidance describing how long the final answer should be based on task complexity."
    )
    user_content = (
        f"Question: {question}\nNeeds web: {decision.needs_web}\nReasoning level: {decision.reasoning_level}\nExpected passes: {decision.expected_passes}\n"
        f"Memory context: {memory_context}\n"
        "Return JSON only as {\"plan_id\": \"...\", \"goal\": \"...\", \"global_constraints\": {...}, \"steps\": [...]}"
    )
    parsed = None
    plan_ep = planner_endpoint or orch_endpoint
    try:
        resp = await lm_client.chat_completion(
            model=plan_ep["model"],
            messages=[
                {"role": "system", "content": agents.MICROMANAGER_SYSTEM + plan_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.25,
            max_tokens=900,
            base_url=plan_ep["base_url"],
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, plan_ep["model"])
    except Exception:
        parsed = None
    if not parsed:
        parsed = {
            "plan_id": str(uuid.uuid4()),
            "goal": question,
                "global_constraints": {
                    "needs_web": decision.needs_web,
                    "reasoning_level": decision.reasoning_level,
                    "max_loops": depth_profile.get("max_loops", 2),
                    "tool_budget": depth_profile.get("tool_budget", {"tavily_search": 12, "tavily_extract": 18}),
                    "expected_passes": decision.expected_passes,
                    "response_guidance": "Keep the answer concise and sized to the question; expand only when evidence is complex.",
                },
            "steps": [
                {
                    "step_id": 1,
                    "name": "Clarify goal",
                    "type": "analysis",
                    "depends_on": [],
                    "agent_profile": "Summarizer",
                    "inputs": {"from_user": True},
                    "outputs": [{"artifact_type": "criteria", "key": "success_criteria"}],
                    "acceptance_criteria": ["criteria defined"],
                    "on_fail": {"action": "revise_step"},
                },
                {
                    "step_id": 2,
                    "name": "Research primary",
                    "type": "research",
                    "depends_on": [1],
                    "agent_profile": "ResearchPrimary",
                    "inputs": {"use_web": decision.needs_web},
                    "outputs": [{"artifact_type": "evidence", "key": "lane_primary"}],
                    "acceptance_criteria": ["has_sources"],
                    "on_fail": {"action": "rerun_step"},
                },
                {
                    "step_id": 3,
                    "name": "Research recency",
                    "type": "research",
                    "depends_on": [1],
                    "agent_profile": "ResearchRecency",
                    "inputs": {"use_web": decision.needs_web},
                    "outputs": [{"artifact_type": "evidence", "key": "lane_recency"}],
                    "acceptance_criteria": ["has_sources"],
                    "on_fail": {"action": "rerun_step"},
                },
                {
                    "step_id": 4,
                    "name": "Research adversarial",
                    "type": "research",
                    "depends_on": [1],
                    "agent_profile": "ResearchAdversarial",
                    "inputs": {"use_web": decision.needs_web},
                    "outputs": [{"artifact_type": "evidence", "key": "lane_adversarial"}],
                    "acceptance_criteria": ["has_conflicts_checked"],
                    "on_fail": {"action": "rerun_step"},
                },
                {
                    "step_id": 5,
                    "name": "Merge evidence",
                    "type": "merge",
                    "depends_on": [2, 3, 4],
                    "agent_profile": "Summarizer",
                    "inputs": {},
                    "outputs": [{"artifact_type": "ledger", "key": "claims_ledger"}],
                    "acceptance_criteria": ["ledger_ready"],
                    "on_fail": {"action": "revise_step"},
                },
                {
                    "step_id": 6,
                    "name": "Draft answer",
                    "type": "draft",
                    "depends_on": [5],
                    "agent_profile": "Orchestrator",
                    "inputs": {},
                    "outputs": [{"artifact_type": "draft", "key": "draft_answer"}],
                    "acceptance_criteria": ["draft_complete"],
                    "on_fail": {"action": "revise_step"},
                },
                {
                    "step_id": 7,
                    "name": "Verify",
                    "type": "verify",
                    "depends_on": [6],
                    "agent_profile": "Verifier",
                    "inputs": {},
                    "outputs": [{"artifact_type": "verifier", "key": "verifier_report"}],
                    "acceptance_criteria": ["verdict_pass_or_fix"],
                    "on_fail": {"action": "backtrack", "backtrack_to_step": 5},
                },
            ],
        }
    plan_obj = StepPlan(**parsed)
    return plan_obj


async def run_worker(
    lm_client: LMStudioClient,
    profile: str,
    model_map: Dict[str, Dict[str, str]],
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> str:
    base_url, model = profile_model(profile, model_map)
    system_prompt = profile_system(profile)
    resp = await lm_client.chat_completion(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )
    return resp["choices"][0]["message"]["content"]


async def generate_step_prompt(
    lm_client: LMStudioClient,
    orch_model: str,
    question: str,
    step: PlanStep,
    artifacts: List[Artifact],
    answer_guidance: str = "",
    toolbox_hint: str = "",
) -> str:
    context = "\n".join([f"{a.key}: {a.content_text or ''}" for a in artifacts[-5:]])
    prompt = (
        f"User question: {question}\n"
        f"Step: {step.step_id} - {step.name} ({step.type})\n"
        f"Acceptance: {step.acceptance_criteria}\n"
        f"Recent artifacts:\n{context}\n"
        f"Produce the needed output for this step."
    )
    if toolbox_hint:
        prompt += f"\nTooling you can request (tool_requests[]): {toolbox_hint}"
    # For most steps this generic prompt suffices; for research we include instruction.
    if step.type == "research":
        prompt += (
            "\nReturn JSON with queries (3-6 specific Tavily web searches), sources (url,title,snippet), claims, gaps, tool_requests[] if needed. "
            "Queries are executed by the backend; include variations and recency hints when relevant. Do not provide a final answer."
        )
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


async def evaluate_control(
    lm_client: LMStudioClient,
    orch_endpoint: Dict[str, str],
    step: PlanStep,
    step_output: Dict[str, Any],
) -> ControlCommand:
    prompt = (
        "Evaluate the step output against acceptance criteria. "
        "If fine, respond {\"control\":\"CONTINUE\"}. "
        "Otherwise choose: BACKTRACK, RERUN_STEP, ADD_STEPS, STOP. "
        f"Step: {step.model_dump()}\nOutput: {json.dumps(step_output)[:1500]}"
    )
    try:
        resp = await lm_client.chat_completion(
            model=orch_endpoint["model"],
            messages=[{"role": "system", "content": agents.MICROMANAGER_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            base_url=orch_endpoint["base_url"],
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, orch_endpoint["model"])
    except Exception:
        parsed = None
    if not parsed:
        parsed = {"control": "CONTINUE"}
    return ControlCommand(**parsed)


async def evaluate_control_fast(
    lm_client: LMStudioClient,
    fast_endpoint: Dict[str, str],
    step: PlanStep,
    step_output: Dict[str, Any],
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
    try:
        resp = await lm_client.chat_completion(
            model=fast_endpoint["model"],
            messages=[{"role": "system", "content": agents.SUMMARIZER_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            base_url=fast_endpoint["base_url"],
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, fast_endpoint["model"])
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
    cmd = ControlCommand(**{k: v for k, v in parsed.items() if k in ControlCommand.model_fields})
    return cmd, escalate


async def allocate_ready_steps(
    lm_client: LMStudioClient,
    fast_endpoint: Dict[str, str],
    ready_steps: List[PlanStep],
    artifacts: List[Artifact],
    running_count: int,
) -> List[int]:
    """
    Ask the fast 4B allocator which ready steps to launch next to keep agents busy.
    Falls back to launching all ready steps if parsing fails.
    """
    if not ready_steps:
        return []
    ready_desc = ", ".join([f"{s.step_id}:{s.name}({s.type})" for s in ready_steps])
    recent = [a.key for a in artifacts[-5:]]
    prompt = (
        "You are the step allocator. Choose which ready steps to start now to keep all worker slots busy."
        " Return JSON {\"start_ids\":[step_ids...]}. Prefer research steps in parallel; keep drafts/verify after research."
        f" Ready: {ready_desc}. Running now: {running_count}. Recent artifacts: {recent}."
    )
    try:
        resp = await lm_client.chat_completion(
            model=fast_endpoint["model"],
            messages=[{"role": "system", "content": agents.SUMMARIZER_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            base_url=fast_endpoint["base_url"],
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, fast_endpoint["model"])
        if parsed and isinstance(parsed.get("start_ids"), list):
            allowed = {s.step_id for s in ready_steps}
            filtered = [sid for sid in parsed["start_ids"] if sid in allowed]
            if filtered:
                return filtered
    except Exception:
        pass
    return [s.step_id for s in ready_steps]


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
) -> Tuple[Dict[str, Any], List[Artifact], str]:
    raw = await run_worker(lm_client, step.agent_profile, model_map, prompt, temperature=0.4, max_tokens=700)
    parsed = await safe_json_parse(raw, lm_client, model_map["worker"])
    if not parsed:
        parsed = {"queries": [prompt], "sources": [], "claims": [], "gaps": [], "tool_requests": []}
    queries = parsed.get("queries", [])
    if not isinstance(queries, list):
        queries = []
    queries = [str(q).strip() for q in queries if str(q).strip()]
    if not queries:
        queries = [question] if question else [prompt]
        parsed["queries"] = queries
    artifacts: List[Artifact] = []
    use_web = decision.needs_web
    if isinstance(step.inputs, dict) and "use_web" in step.inputs:
        use_web = step.inputs.get("use_web", decision.needs_web)
    # If the router under-called web, backstop with a heuristic so Tavily still runs for data questions.
    if not use_web and guess_needs_web(question):
        use_web = True
    search_depth = reasoning_to_search_depth(decision.reasoning_level, search_depth_mode, depth_profile)
    search_budget = depth_profile.get("tool_budget", {}).get("tavily_search", decision.max_results or 6)
    extract_budget = depth_profile.get("tool_budget", {}).get("tavily_extract", 6)
    gathered_sources = []
    tool_requests = parsed.get("tool_requests", [])
    if not isinstance(tool_requests, list):
        tool_requests = []
    if use_web:
        if not tavily.enabled:
            await bus.emit(run_id, "tavily_error", {"step": step.step_id, "message": "Tavily API key missing"})
        # Always ensure we have at least one usable query.
        if not queries:
            queries = [question]
        # Execute Tavily for each query
        per_query_max = max(3, min(search_budget, decision.max_results if decision else 6))
        for query in queries[:5]:
            await bus.emit(run_id, "tavily_search", {"step": step.step_id, "query": query})
            search_resp = await tavily.search(
                query=query,
                search_depth=search_depth,
                max_results=per_query_max,
                topic=decision.topic if decision else "general",
            )
            await db.add_search(run_id, f"Step{step.step_id}", query, search_depth, per_query_max, search_resp)
            if search_resp.get("error"):
                await bus.emit(run_id, "tavily_error", {"step": step.step_id, "message": search_resp.get("error")})
            for res in search_resp.get("results", [])[: per_query_max]:
                src = {
                    "url": res.get("url"),
                    "title": res.get("title"),
                    "publisher": res.get("source"),
                    "date_published": res.get("published_date"),
                    "snippet": res.get("content", "")[:400],
                    "extracted_text": res.get("content", ""),
                }
                gathered_sources.append(src)
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
        # Extract for a subset
        urls = [s["url"] for s in gathered_sources if s.get("url")]
        if urls:
            url_slice = urls[: max(3, min(extract_budget, len(urls)))]
            await bus.emit(run_id, "tavily_extract", {"step": step.step_id, "urls": url_slice})
            extract_resp = await tavily.extract(url_slice, extract_depth=decision.extract_depth if decision else "basic")
            await db.add_extract(run_id, f"Step{step.step_id}", ",".join(url_slice), decision.extract_depth, extract_resp)
            if extract_resp.get("results"):
                gathered_sources = []
                for res in extract_resp["results"]:
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
        # Fallback: if searches returned nothing, try one more broad query using the main question.
        if use_web and not gathered_sources and question:
            fallback_query = question.strip()
            await bus.emit(run_id, "tavily_search", {"step": step.step_id, "query": fallback_query, "mode": "fallback"})
            search_resp = await tavily.search(
                query=fallback_query,
                search_depth=search_depth,
                max_results=per_query_max,
                topic=decision.topic if decision else "general",
            )
            await db.add_search(run_id, f"Step{step.step_id}", fallback_query, search_depth, per_query_max, search_resp)
            for res in search_resp.get("results", [])[: per_query_max]:
                src = {
                    "url": res.get("url"),
                    "title": res.get("title"),
                    "publisher": res.get("source"),
                    "date_published": res.get("published_date"),
                    "snippet": res.get("content", "")[:400],
                    "extracted_text": res.get("content", ""),
                }
                gathered_sources.append(src)
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
    tool_results = resolve_tool_requests(tool_requests)
    evidence = {
        "lane": step.agent_profile,
        "queries": parsed.get("queries", []),
        "sources": gathered_sources,
        "claims": parsed.get("claims", []),
        "gaps": parsed.get("gaps", []),
        "conflicts_found": parsed.get("conflicts_found", False),
        "tool_requests": tool_requests,
        "tool_results": tool_results,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }
    for claim in evidence["claims"]:
        await db.add_claim(run_id, claim if isinstance(claim, str) else json.dumps(claim), [s.get("url", "") for s in gathered_sources], confidence="MED", notes=step.agent_profile)
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
) -> Tuple[Dict[str, Any], List[Artifact], str]:
    answer_hint = ""
    if step.type == "draft":
        answer_hint = response_guidance or response_guidance_text(question, decision.reasoning_level, progress_meta)
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
            run_id, question, step, prompt, decision, search_depth_mode, depth_profile, lm_client, tavily, db, bus, model_map
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
        draft_profile = step.agent_profile or "Orchestrator"
        draft_resp = await run_worker(lm_client, draft_profile, model_map, prompt, temperature=0.3, max_tokens=800)
        artifact = Artifact(
            step_id=step.step_id,
            key="draft_answer",
            artifact_type="draft",
            content_text=draft_resp,
            content_json={"draft": draft_resp},
        )
        return {"draft": draft_resp}, [artifact], prompt
    elif step.type == "verify":
        # use verifier worker (Qwen8) but with verifier system
        ledger = merge_evidence_artifacts(artifacts)
        draft = next((a.content_text for a in artifacts if a.artifact_type == "draft"), "")
        verifier_prompt = (
            f"Question: {question}\nDraft: {draft}\nClaims ledger: {json.dumps(ledger)[:3000]}\n"
            "Return JSON verdict: PASS/NEEDS_REVISION, issues[], revised_answer?, extra_steps[]."
        )
        verifier_profile = step.agent_profile or "Verifier"
        report = await run_worker(
            lm_client, verifier_profile, model_map, verifier_prompt, temperature=0.0, max_tokens=700
        )
        parsed = await safe_json_parse(report, lm_client, model_map["verifier"])
        if not parsed:
            parsed = {"issues": [], "verdict": "PASS", "extra_steps": []}
        artifact = Artifact(
            step_id=step.step_id,
            key="verifier_report",
            artifact_type="verifier",
            content_text=json.dumps(parsed),
            content_json=parsed,
        )
        return parsed, [artifact], prompt
    elif step.type == "analysis":
        analysis_profile = step.agent_profile or "Summarizer"
        summary = await run_worker(lm_client, analysis_profile, model_map, prompt, temperature=0.2, max_tokens=400)
        artifact = Artifact(
            step_id=step.step_id,
            key="success_criteria",
            artifact_type="criteria",
            content_text=summary,
            content_json={"criteria": summary},
        )
        return {"criteria": summary}, [artifact], prompt
    else:
        generic = await run_worker(lm_client, step.agent_profile, model_map, prompt, temperature=0.2, max_tokens=500)
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
) -> Tuple[List[Artifact], str]:
    """Analyze uploads with vision (8B) and secretary (4B) models."""
    if not upload_ids:
        return [], ""
    artifacts: List[Artifact] = []
    summaries: List[str] = []
    vision_endpoint = model_map.get("worker") or model_map.get("worker_a") or model_map.get("orch")
    secretary_endpoint = model_map.get("summarizer") or model_map.get("router") or model_map.get("worker")
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
                resp = await lm_client.chat_completion(
                    model=vision_endpoint["model"],
                    messages=[
                        {"role": "system", "content": agents.VISION_ANALYST_SYSTEM},
                        {"role": "user", "content": image_block},
                    ],
                    temperature=0.2,
                    max_tokens=600,
                    base_url=vision_endpoint["base_url"],
                )
                content = resp["choices"][0]["message"]["content"]
                vision_json = await safe_json_parse(content, lm_client, vision_endpoint["model"]) or {"caption": content}
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
            sec_resp = await lm_client.chat_completion(
                model=secretary_endpoint["model"],
                messages=[
                    {"role": "system", "content": agents.UPLOAD_SECRETARY_SYSTEM},
                    {"role": "user", "content": secretary_prompt},
                ],
                temperature=0.2,
                max_tokens=320,
                base_url=secretary_endpoint["base_url"],
            )
            sec_content = sec_resp["choices"][0]["message"]["content"]
            sec_json = await safe_json_parse(sec_content, lm_client, secretary_endpoint["model"]) or {"summary": sec_content}
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
        except Exception as exc:
            await db.update_upload_status(
                record["id"], "failed", summary_text=str(exc), summary_json={"error": str(exc)}
            )
            await bus.emit(
                run_id,
                "upload_failed",
                {"upload_id": record["id"], "name": record["original_name"], "error": str(exc)},
            )
    summary_line = "; ".join(summaries)
    return artifacts, summary_line


async def run_question(
    run_id: str,
    question: str,
    decision_mode: str,
    manual_level: str,
    model_tier: str,
    deep_mode: str,
    search_depth_mode: str,
    max_results_override: int,
    strict_mode: bool,
    auto_memory: bool,
    db: Database,
    bus: EventBus,
    lm_client: LMStudioClient,
    tavily: TavilyClient,
    settings_models: Dict[str, Dict[str, str]],
    upload_ids: Optional[List[int]] = None,
) -> None:
    """Main orchestration loop for a single run (now with parallel step execution)."""
    try:
        await db.insert_run(run_id, question=question, reasoning_mode=decision_mode)
        user_msg = await db.add_message(run_id, "user", question)
        await bus.emit(
            run_id,
            "message_added",
            {"id": user_msg.get("id"), "role": "user", "content": question, "run_id": run_id, "created_at": user_msg.get("created_at")},
        )
        await bus.emit(run_id, "run_started", {"question": question})

        base_router_endpoint = settings_models.get("router") or settings_models.get("summarizer") or settings_models["orch"]
        router_decision = await call_router(
            lm_client, base_router_endpoint, question, manual_level if decision_mode == "manual" else None, strict_mode=strict_mode
        )
        requested_tier = model_tier
        if requested_tier == "fast":
            router_decision.reasoning_level = "LOW"
            router_decision.expected_passes = 1
        if decision_mode == "manual":
            router_decision.reasoning_level = manual_level
        if max_results_override:
            router_decision.max_results = max_results_override
        if strict_mode:
            router_decision.reasoning_level = "HIGH" if router_decision.reasoning_level in ("LOW", "MED") else router_decision.reasoning_level
            router_decision.extract_depth = "advanced"
            router_decision.max_results = max(router_decision.max_results, 10)
        if strict_mode or router_decision.reasoning_level in ("HIGH", "ULTRA"):
            router_decision.expected_passes = max(router_decision.expected_passes or 1, 2)
        if guess_needs_web(question):
            router_decision.needs_web = True
        effective_tier = requested_tier
        if requested_tier == "auto":
            effective_tier = resolve_auto_tier(router_decision)
        model_tier = effective_tier
        depth_profile = REASONING_DEPTHS.get(router_decision.reasoning_level, REASONING_DEPTHS["MED"])
        if model_tier == "fast":
            depth_profile = REASONING_DEPTHS["LOW"]
        if depth_profile.get("tool_budget", {}).get("tavily_extract"):
            router_decision.max_results = max(router_decision.max_results, depth_profile["tool_budget"]["tavily_extract"] // 2)
        if search_depth_mode == "auto" and depth_profile.get("advanced"):
            search_depth_mode = "advanced"
        if not router_decision.tool_budget:
            router_decision.tool_budget = depth_profile.get("tool_budget", {})
        deep_route_used = deep_mode
        if model_tier == "deep":
            deep_route_used = await choose_deep_route(lm_client, base_router_endpoint, question, deep_mode, router_decision)
        active_models, planner_endpoint, control_endpoint, allow_parallel, execution_mode = select_model_suite(
            settings_models, model_tier, deep_route_used
        )
        # Copy so we can safely adjust per-run without mutating global settings
        active_models = {k: (v.copy() if isinstance(v, dict) else v) for k, v in active_models.items()}
        # Ensure research steps use the tool-savvy orchestrator model when web search is needed.
        if router_decision.needs_web and active_models.get("orch"):
            for role in ("worker", "worker_b", "worker_c"):
                if active_models.get(role) != active_models["orch"]:
                    active_models[role] = active_models["orch"]
        decision_payload = router_decision.model_dump()
        decision_payload.update(
            {
                "model_tier": model_tier,
                "requested_tier": requested_tier,
                "deep_route": deep_route_used,
                "execution_mode": execution_mode,
            }
        )
        await db.update_run_router(run_id, decision_payload)
        await bus.emit(run_id, "router_decision", decision_payload)
        if strict_mode:
            await bus.emit(run_id, "strict_mode", {"enabled": True})
        tier_note = model_tier.upper()
        if requested_tier != model_tier:
            tier_note = f"{requested_tier.upper()}->{model_tier.upper()}"
        await bus.emit(run_id, "client_note", {"note": f"{tier_note} mode: {execution_mode} (route {deep_route_used})"})

        # Memory retrieval
        mem_hits = await db.search_memory(question, limit=5)
        memory_context = "; ".join([f"{m['title']}: {m['content']}" for m in mem_hits])
        artifacts: List[Artifact] = []
        if mem_hits:
            mem_art = Artifact(step_id=0, key="memory_context", artifact_type="memory", content_text=memory_context, content_json={"items": mem_hits})
            artifacts.append(mem_art)
        await bus.emit(run_id, "memory_retrieved", {"count": len(mem_hits)})

        upload_id_list = upload_ids or [u["id"] for u in await db.list_uploads(run_id)]
        if upload_id_list:
            upload_artifacts, upload_summary = await process_uploads(
                run_id, question, upload_id_list, db, bus, lm_client, active_models
            )
            if upload_artifacts:
                artifacts.extend(upload_artifacts)
                if upload_summary:
                    memory_context = (memory_context + "; " if memory_context else "") + f"Uploads: {upload_summary}"

        if execution_mode in ("fast_linear", "oss_linear"):
            step_plan = build_linear_plan(question, router_decision, depth_profile, needs_verify=True)
        else:
            step_plan = await build_step_plan(
                lm_client, active_models["orch"], question, router_decision, depth_profile, memory_context, planner_endpoint=planner_endpoint
            )
        if len(step_plan.steps) > depth_profile.get("max_steps", len(step_plan.steps)):
            step_plan.steps = step_plan.steps[: depth_profile["max_steps"]]
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
        progress_meta["response_guidance"] = response_guidance
        await db.add_step_plan(run_id, step_plan.model_dump())
        await bus.emit(
            run_id,
            "plan_created",
            {
                "steps": len(step_plan.steps),
                "expected_total_steps": progress_meta["total_steps"],
                "expected_passes": progress_meta["counted_passes"],
            },
        )
        # Build lookup for dependency-aware scheduling
        step_lookup: Dict[int, PlanStep] = {s.step_id: s for s in step_plan.steps}
        completed_steps: Set[int] = set()
        running_tasks: Dict[int, asyncio.Task] = {}
        max_loops = max(step_plan.global_constraints.get("max_loops", 1), progress_meta["counted_passes"] - 1)
        loops = 0
        stop_requested = False
        fast_endpoint = active_models.get("summarizer") or active_models.get("router") or active_models["orch"]

        async def cancel_running_tasks() -> None:
            if not running_tasks:
                return
            for t in running_tasks.values():
                t.cancel()
            await asyncio.gather(*running_tasks.values(), return_exceptions=True)
            running_tasks.clear()

        async def start_step(step: PlanStep, snapshot: List[Artifact]) -> asyncio.Task:
            await bus.emit(run_id, "step_started", {"step_id": step.step_id, "name": step.name})
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
                        "control": await evaluate_control(lm_client, control_endpoint, step, output),
                    }
                except Exception as exc:
                    await db.update_step_run(step_run_id, status="error", error_text=str(exc))
                    await bus.emit(
                        run_id,
                        "step_error",
                        {"step": step.step_id, "name": step.name, "message": str(exc)},
                    )
                    return {"status": "error", "step": step, "error": str(exc)}

            return asyncio.create_task(runner())

        def deps_satisfied(step: PlanStep) -> bool:
            return all(dep in completed_steps for dep in step.depends_on)

        while len(completed_steps) < len(step_plan.steps) and not stop_requested:
            ready_steps = [
                s for s in step_plan.steps if s.step_id not in completed_steps and s.step_id not in running_tasks and deps_satisfied(s)
            ]
            if not allow_parallel and ready_steps:
                ready_steps = sorted(ready_steps, key=lambda s: s.step_id)[:1]
            if ready_steps:
                start_ids = await allocate_ready_steps(lm_client, fast_endpoint, ready_steps, artifacts, len(running_tasks)) if allow_parallel else [ready_steps[0].step_id]
                for step in ready_steps:
                    if step.step_id in start_ids and step.step_id not in running_tasks:
                        running_tasks[step.step_id] = await start_step(step, list(artifacts))

            if not running_tasks:
                # No runnable steps left; avoid deadlock
                break

            done, _ = await asyncio.wait(running_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = task.result()
                step_id = result["step"].step_id
                running_tasks.pop(step_id, None)
                if result["status"] != "completed":
                    completed_steps.add(step_id)  # prevent deadlock; move on if a step errors
                    continue
                completed_steps.add(step_id)
                artifacts.extend(result["artifacts"])
                await bus.emit(run_id, "step_completed", {"step_id": step_id, "name": result["step"].name})

                fast_control, escalate = await evaluate_control_fast(lm_client, fast_endpoint, result["step"], result["output"])
                control: ControlCommand = fast_control
                # Only pull in the OSS orchestrator for heavyweight checkpoints or when escalation is requested.
                heavy_types = {"merge", "draft", "verify", "analysis"}
                needs_oss = (
                    escalate
                    or result["step"].type in heavy_types
                    or (strict_mode and fast_control.control != "CONTINUE")
                )
                if needs_oss:
                    control = await evaluate_control(lm_client, control_endpoint, result["step"], result["output"])
                if control.control != "CONTINUE":
                    await db.add_control_action(run_id, control.model_dump())
                    await bus.emit(run_id, "control_action", control.model_dump())
                    # Handle control signals with minimal disruption; cancel in-flight work if we need to rerun/backtrack.
                    if control.control == "ADD_STEPS" and control.steps:
                        insertion = len(step_plan.steps)
                        for offset, new_step in enumerate(control.steps):
                            ps = PlanStep(**new_step)
                            step_plan.steps.insert(insertion + offset, ps)
                            step_lookup[ps.step_id] = ps
                        await db.add_step_plan(run_id, step_plan.model_dump())
                    elif control.control == "BACKTRACK" and control.to_step:
                        await cancel_running_tasks()
                        completed_steps = {sid for sid in completed_steps if sid < control.to_step}
                    elif control.control == "RERUN_STEP" and control.step_id:
                        await cancel_running_tasks()
                        if control.step_id in completed_steps:
                            completed_steps.remove(control.step_id)
                    elif control.control == "STOP":
                        stop_requested = True
                        await cancel_running_tasks()
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
                    await bus.emit(
                        run_id,
                        "loop_iteration",
                        {
                            "iteration": loops,
                            "expected_total_steps": progress_meta.get("total_steps"),
                            "completed_reset_to": completed_reset_to,
                            "counted_passes": progress_meta.get("counted_passes"),
                        },
                    )
                    completed_steps = {s.step_id for s in step_plan.steps if s.type == "analysis"}  # keep upfront steps
                    # keep artifacts but rerun research+draft+verify
                    await cancel_running_tasks()
                    continue

        # finalize
        draft_art = next((a for a in artifacts if a.artifact_type == "draft"), None)
        verifier_art = next((a for a in artifacts if a.artifact_type == "verifier"), None)
        ledger_art = next((a for a in artifacts if a.artifact_type == "ledger"), None)
        final_answer = (verifier_art.content_json.get("revised_answer") if verifier_art and verifier_art.content_json else None) or (
            draft_art.content_text if draft_art else ""
        )
        if not final_answer:
            # Fallback: force a concise answer from available context + model knowledge.
            ledger_json = ledger_art.content_json if ledger_art else merge_evidence_artifacts(artifacts)
            fallback_prompt = (
                f"Question: {question}\n"
                f"Evidence (may be partial): {json.dumps(ledger_json)[:2800]}\n"
                "Provide the best direct answer you can. If evidence is light, rely on your own knowledge but flag any uncertainty.\n"
                "Return a short, clear answer without chain-of-thought."
            )
            try:
                final_answer = await run_worker(
                    lm_client, "Orchestrator", active_models, fallback_prompt, temperature=0.25, max_tokens=700
                )
            except Exception:
                final_answer = final_answer or "Unable to produce an answer with the available context."
        confidence = "MED"
        if verifier_art and verifier_art.content_json:
            verdict = verifier_art.content_json.get("verdict", "PASS")
            confidence = "HIGH" if verdict == "PASS" else "LOW"
        assistant_msg = await db.add_message(run_id, "assistant", final_answer)
        await bus.emit(
            run_id,
            "message_added",
            {"id": assistant_msg.get("id"), "role": "assistant", "content": final_answer, "run_id": run_id, "created_at": assistant_msg.get("created_at")},
        )
        await db.finalize_run(run_id, final_answer, confidence)
        existing_run_memory = await db.get_run_memory(run_id)
        if auto_memory and final_answer and not existing_run_memory:
            mem_id = await db.add_memory_item(
                kind="answer",
                title=question[:80],
                content=final_answer[:800],
                tags=[router_decision.reasoning_level],
                pinned=False,
                relevance_score=1.0,
            )
            await db.link_memory_to_run(run_id, mem_id, "auto")
            await bus.emit(run_id, "memory_saved", {"count": 1})
            existing_run_memory.append({"id": mem_id})
        if final_answer and not existing_run_memory:
            try:
                summary_prompt = (
                    f"Conversation snippet to index for recall.\nQuestion: {question}\nAnswer: {final_answer}\n"
                    f"Memory hints: {memory_context or 'n/a'}\nSummarize key takeaways (<=120 words) and keep it scannable."
                )
                summary_text = await run_worker(
                    lm_client, "Summarizer", active_models, summary_prompt, temperature=0.2, max_tokens=180
                )
            except Exception:
                summary_text = (final_answer or question)[:400]
            mem_id = await db.add_memory_item(
                kind="summary",
                title=f"Chat summary: {question[:60]}",
                content=summary_text,
                tags=[router_decision.reasoning_level, "summary"],
                pinned=False,
                relevance_score=0.9,
            )
            await db.link_memory_to_run(run_id, mem_id, "auto_summary")
            await bus.emit(run_id, "memory_saved", {"count": 1})
        try:
            ui_note_prompt = (
                f"Question: {question}\nAnswer: {final_answer[:320]}\n"
                f"Tier: {model_tier}, Route: {deep_route_used}, Confidence: {confidence}\n"
                "Summarize in one short status line for the UI ticker."
            )
            ui_note = await run_worker(
                lm_client, "Summarizer", active_models, ui_note_prompt, temperature=0.1, max_tokens=120
            )
            await bus.emit(run_id, "client_note", {"note": ui_note, "tier": model_tier, "route": deep_route_used})
        except Exception:
            pass
        await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": confidence})
    except Exception as exc:
        await db.update_run_status(run_id, f"error: {exc}")
        await bus.emit(run_id, "error", {"message": str(exc), "fatal": True})


def new_run_id() -> str:
    return str(uuid.uuid4())
