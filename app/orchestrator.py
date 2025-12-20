import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

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


def reasoning_to_search_depth(level: str, preferred: str, depth_profile: Optional[dict] = None) -> str:
    if preferred != "auto":
        return preferred
    if depth_profile and depth_profile.get("advanced"):
        return "advanced"
    if level in ("HIGH", "ULTRA"):
        return "advanced"
    return "basic"


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


async def call_router(
    lm_client: LMStudioClient, endpoint: Dict[str, str], question: str, manual_level: Optional[str] = None
) -> RouterDecision:
    user_msg = f"User question: {question}\nReturn JSON only."
    parsed = None
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
        needs_web_guess = any(
            token in question.lower()
            for token in ["today", "current", "latest", "news", "recent", "update", "breaking"]
        )
        parsed = {
            "needs_web": needs_web_guess,
            "reasoning_level": manual_level or "MED",
            "topic": "general",
            "max_results": 6,
            "extract_depth": "basic",
            "stop_conditions": {},
        }
    decision = RouterDecision(**parsed)
    if manual_level:
        decision.reasoning_level = manual_level
    return decision


async def build_step_plan(
    lm_client: LMStudioClient,
    orch_endpoint: Dict[str, str],
    question: str,
    decision: RouterDecision,
    depth_profile: dict,
    memory_context: str = "",
) -> StepPlan:
    plan_prompt = (
        "Produce a JSON step plan for answering the question. "
        "Include step_id, name, type, depends_on (list of ids), agent_profile, acceptance_criteria. "
        "Keep 6-12 steps for typical questions."
    )
    user_content = (
        f"Question: {question}\nNeeds web: {decision.needs_web}\nReasoning level: {decision.reasoning_level}\n"
        f"Memory context: {memory_context}\n"
        "Return JSON only as {\"plan_id\": \"...\", \"goal\": \"...\", \"global_constraints\": {...}, \"steps\": [...]}"
    )
    parsed = None
    try:
        resp = await lm_client.chat_completion(
            model=orch_endpoint["model"],
            messages=[
                {"role": "system", "content": agents.MICROMANAGER_SYSTEM + plan_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.25,
            max_tokens=900,
            base_url=orch_endpoint["base_url"],
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, orch_endpoint["model"])
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
) -> str:
    context = "\n".join([f"{a.key}: {a.content_text or ''}" for a in artifacts[-5:]])
    prompt = (
        f"User question: {question}\n"
        f"Step: {step.step_id} - {step.name} ({step.type})\n"
        f"Acceptance: {step.acceptance_criteria}\n"
        f"Recent artifacts:\n{context}\n"
        f"Produce the needed output for this step."
    )
    # For most steps this generic prompt suffices; for research we include instruction.
    if step.type == "research":
        prompt += (
            "\nReturn JSON with queries, sources (url,title,snippet), claims, gaps. "
            "Do not provide final answer."
        )
    return prompt


def merge_evidence_artifacts(artifacts: List[Artifact]) -> Dict[str, Any]:
    sources_by_url: Dict[str, dict] = {}
    claims: List[dict] = []
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
    conflicts = [c for c in claims if c.get("conflict")]
    return {"sources": list(sources_by_url.values()), "claims": claims, "conflicts": conflicts}


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


async def execute_research_step(
    run_id: str,
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
        parsed = {"queries": [prompt], "sources": [], "claims": [], "gaps": []}
    artifacts: List[Artifact] = []
    search_depth = reasoning_to_search_depth(decision.reasoning_level, search_depth_mode, depth_profile)
    search_budget = depth_profile.get("tool_budget", {}).get("tavily_search", decision.max_results or 6)
    extract_budget = depth_profile.get("tool_budget", {}).get("tavily_extract", 6)
    gathered_sources = []
    # Execute Tavily for each query
    per_query_max = max(3, min(search_budget, decision.max_results if decision else 6))
    for query in parsed.get("queries", [])[:5]:
        await bus.emit(run_id, "tavily_search", {"step": step.step_id, "query": query})
        search_resp = await tavily.search(
            query=query,
            search_depth=search_depth,
            max_results=per_query_max,
            topic=decision.topic if decision else "general",
        )
        await db.add_search(run_id, f"Step{step.step_id}", query, search_depth, per_query_max, search_resp)
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
    evidence = {
        "lane": step.agent_profile,
        "queries": parsed.get("queries", []),
        "sources": gathered_sources,
        "claims": parsed.get("claims", []),
        "gaps": parsed.get("gaps", []),
        "conflicts_found": parsed.get("conflicts_found", False),
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
    lm_client: LMStudioClient,
    tavily: TavilyClient,
    db: Database,
    bus: EventBus,
    model_map: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, Any], List[Artifact], str]:
    prompt = await generate_step_prompt(lm_client, model_map["orch"], question, step, artifacts)
    if step.type == "research":
        return await execute_research_step(
            run_id, step, prompt, decision, search_depth_mode, depth_profile, lm_client, tavily, db, bus, model_map
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


async def run_question(
    run_id: str,
    question: str,
    decision_mode: str,
    manual_level: str,
    search_depth_mode: str,
    max_results_override: int,
    strict_mode: bool,
    auto_memory: bool,
    db: Database,
    bus: EventBus,
    lm_client: LMStudioClient,
    tavily: TavilyClient,
    settings_models: Dict[str, Dict[str, str]],
) -> None:
    """Main orchestration loop for a single run."""
    try:
        await db.insert_run(run_id, question=question, reasoning_mode=decision_mode)
        await db.add_message(run_id, "user", question)
        await bus.emit(run_id, "run_started", {"question": question})

        router_decision = await call_router(
            lm_client, settings_models["router"], question, manual_level if decision_mode == "manual" else None
        )
        if decision_mode == "manual":
            router_decision.reasoning_level = manual_level
        if max_results_override:
            router_decision.max_results = max_results_override
        if strict_mode:
            router_decision.reasoning_level = "HIGH" if router_decision.reasoning_level in ("LOW", "MED") else router_decision.reasoning_level
            router_decision.extract_depth = "advanced"
            router_decision.max_results = max(router_decision.max_results, 10)
        depth_profile = REASONING_DEPTHS.get(router_decision.reasoning_level, REASONING_DEPTHS["MED"])
        if depth_profile.get("tool_budget", {}).get("tavily_extract"):
            router_decision.max_results = max(router_decision.max_results, depth_profile["tool_budget"]["tavily_extract"] // 2)
        if search_depth_mode == "auto" and depth_profile.get("advanced"):
            search_depth_mode = "advanced"
        if not router_decision.tool_budget:
            router_decision.tool_budget = depth_profile.get("tool_budget", {})
        await db.update_run_router(run_id, router_decision.model_dump())
        await bus.emit(run_id, "router_decision", router_decision.model_dump())
        if strict_mode:
            await bus.emit(run_id, "strict_mode", {"enabled": True})

        # Memory retrieval
        mem_hits = await db.search_memory(question, limit=5)
        memory_context = "; ".join([f"{m['title']}: {m['content']}" for m in mem_hits])
        artifacts: List[Artifact] = []
        if mem_hits:
            mem_art = Artifact(step_id=0, key="memory_context", artifact_type="memory", content_text=memory_context, content_json={"items": mem_hits})
            artifacts.append(mem_art)
        await bus.emit(run_id, "memory_retrieved", {"count": len(mem_hits)})

        step_plan = await build_step_plan(lm_client, settings_models["orch"], question, router_decision, depth_profile, memory_context)
        if len(step_plan.steps) > depth_profile.get("max_steps", len(step_plan.steps)):
            step_plan.steps = step_plan.steps[: depth_profile["max_steps"]]
        await db.add_step_plan(run_id, step_plan.model_dump())
        await bus.emit(run_id, "plan_created", {"steps": len(step_plan.steps)})

        step_index = 0
        loops = 0
        while step_index < len(step_plan.steps):
            step = step_plan.steps[step_index]
            await bus.emit(run_id, "step_started", {"step_id": step.step_id, "name": step.name})
            step_run_id = await db.add_step_run(
                run_id,
                step.step_id,
                status="running",
                agent_profile=step.agent_profile,
                prompt_text="",
            )
            try:
                output, new_artifacts, prompt_used = await execute_step(
                    run_id,
                    question,
                    step,
                    router_decision,
                    search_depth_mode,
                    depth_profile,
                    artifacts,
                    lm_client,
                    tavily,
                    db,
                    bus,
                    settings_models,
                )
                await db.update_step_run(step_run_id, status="completed", output_json=output)
                # persist prompt text separately
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
                    artifacts.append(art)
                await bus.emit(run_id, "step_completed", {"step_id": step.step_id, "name": step.name})
                control = await evaluate_control(lm_client, settings_models["orch"], step, output)
                if control.control != "CONTINUE":
                    await db.add_control_action(run_id, control.model_dump())
                    await bus.emit(run_id, "control_action", control.model_dump())
                    if control.control == "BACKTRACK" and control.to_step:
                        # find index of target step_id
                        targets = [i for i, s in enumerate(step_plan.steps) if s.step_id == control.to_step]
                        if targets:
                            step_index = targets[0]
                            continue
                    elif control.control == "RERUN_STEP" and control.step_id:
                        targets = [i for i, s in enumerate(step_plan.steps) if s.step_id == control.step_id]
                        if targets:
                            step_index = targets[0]
                            continue
                    elif control.control == "ADD_STEPS" and control.steps:
                        # Append new steps after current
                        insertion = step_index + 1
                        for offset, new_step in enumerate(control.steps):
                            ps = PlanStep(**new_step)
                            step_plan.steps.insert(insertion + offset, ps)
                        await db.add_step_plan(run_id, step_plan.model_dump())
                    elif control.control == "STOP":
                        break
            except Exception as exc:
                await db.update_step_run(step_run_id, status="error", error_text=str(exc))
                await bus.emit(run_id, "error", {"step": step.step_id, "message": str(exc)})
            step_index += 1
            if step_index >= len(step_plan.steps) and loops < step_plan.global_constraints.get("max_loops", 1):
                # if verifier artifact says needs revision, loop
                verifier_art = next((a for a in artifacts if a.artifact_type == "verifier"), None)
                if verifier_art and verifier_art.content_json and verifier_art.content_json.get("verdict") == "NEEDS_REVISION":
                    loops += 1
                    await bus.emit(run_id, "loop_iteration", {"iteration": loops})
                    # Backtrack to merge/draft
                    targets = [i for i, s in enumerate(step_plan.steps) if s.type in ("merge", "research")]
                    step_index = targets[0] if targets else 0

        # finalize
        draft_art = next((a for a in artifacts if a.artifact_type == "draft"), None)
        verifier_art = next((a for a in artifacts if a.artifact_type == "verifier"), None)
        ledger_art = next((a for a in artifacts if a.artifact_type == "ledger"), None)
        final_answer = (verifier_art.content_json.get("revised_answer") if verifier_art and verifier_art.content_json else None) or (
            draft_art.content_text if draft_art else ""
        )
        confidence = "MED"
        if verifier_art and verifier_art.content_json:
            verdict = verifier_art.content_json.get("verdict", "PASS")
            confidence = "HIGH" if verdict == "PASS" else "LOW"
        await db.add_message(run_id, "assistant", final_answer)
        await db.finalize_run(run_id, final_answer, confidence)
        if auto_memory and final_answer:
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
        await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": confidence})
    except Exception as exc:
        await db.update_run_status(run_id, f"error: {exc}")
        await bus.emit(run_id, "error", {"message": str(exc)})


def new_run_id() -> str:
    return str(uuid.uuid4())
