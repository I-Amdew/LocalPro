import asyncio
import json
import uuid
from typing import Dict, List, Optional

from . import agents
from .db import Database
from .llm import LMStudioClient
from .schemas import EvidencePack, EvidenceSource, RouterDecision, VerifierReport
from .tavily import TavilyClient


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
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        prompt = f"Repair the following JSON. Only return JSON.\n{raw}"
        resp = await lm_client.chat_completion(
            model=fixer_model,
            messages=[{"role": "system", "content": agents.SUMMARIZER_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        fixed = resp["choices"][0]["message"]["content"]
        return json.loads(fixed)
    except Exception:
        return None


def merge_evidence(packs: List[EvidencePack]) -> dict:
    sources_by_url: Dict[str, dict] = {}
    claims: List[dict] = []
    conflicts = 0
    for pack in packs:
        for src in pack.sources:
            if src.url not in sources_by_url:
                sources_by_url[src.url] = src.model_dump()
        for claim in pack.claims:
            claims.append({"claim": claim, "lane": pack.lane})
        if pack.conflicts_found:
            conflicts += 1
    return {"sources": list(sources_by_url.values()), "claims": claims, "conflicts": conflicts}


async def call_router(
    lm_client: LMStudioClient, model: str, question: str, manual_level: Optional[str] = None
) -> RouterDecision:
    user_msg = f"User question: {question}\nReturn JSON only."
    try:
        resp = await lm_client.chat_completion(
            model=model,
            messages=[{"role": "system", "content": agents.ROUTER_SYSTEM}, {"role": "user", "content": user_msg}],
            temperature=0.1,
            max_tokens=300,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, model)
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


async def call_plan(lm_client: LMStudioClient, orch_model: str, question: str, decision: RouterDecision) -> dict:
    plan_prompt = (
        "Produce a short JSON task list to answer the user. Include research queries, "
        "extraction hints, computation steps, and verification focus areas."
    )
    user_content = (
        f"Question: {question}\nNeeds web: {decision.needs_web}\nReasoning level: {decision.reasoning_level}\n"
        "Return JSON only as {\"tasks\": [\"...\"]}"
    )
    try:
        resp = await lm_client.chat_completion(
            model=orch_model,
            messages=[
                {"role": "system", "content": agents.ORCHESTRATOR_SYSTEM + plan_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, orch_model)
    except Exception:
        parsed = None
    return parsed or {"tasks": ["analyze question", "research", "synthesize", "verify"]}


async def call_lane_plan(
    lm_client: LMStudioClient,
    model: str,
    lane_system: str,
    question: str,
    decision: RouterDecision,
    lane_name: str,
) -> EvidencePack:
    user_content = (
        f"User question: {question}\nReasoning level: {decision.reasoning_level}\n"
        "Return only the JSON evidence pack with 3-5 focused queries."
    )
    try:
        resp = await lm_client.chat_completion(
            model=model,
            messages=[{"role": "system", "content": lane_system}, {"role": "user", "content": user_content}],
            temperature=0.4,
            max_tokens=500,
        )
        raw = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(raw, lm_client, model)
    except Exception:
        parsed = None
    if not parsed:
        parsed = {"lane": lane_name, "queries": [question], "sources": [], "claims": [], "gaps": []}
    pack = EvidencePack(**parsed)
    pack.lane = lane_name
    return pack


async def call_draft(
    lm_client: LMStudioClient, orch_model: str, question: str, ledger: dict, decision: RouterDecision
) -> str:
    summary_sources = [s.get("url") for s in ledger.get("sources", [])][:6]
    summary_claims = [c.get("claim") for c in ledger.get("claims", [])][:6]
    prompt = (
        "Draft a concise answer with inline sources (URLs). Include a confidence label HIGH/MED/LOW and a short justification. "
        "No hidden chain-of-thought."
    )
    user_msg = (
        f"Question: {question}\nSources: {summary_sources}\nClaims: {summary_claims}\n"
        f"Reasoning level: {decision.reasoning_level}\nProvide a helpful answer with citations."
    )
    try:
        resp = await lm_client.chat_completion(
            model=orch_model,
            messages=[{"role": "system", "content": agents.ORCHESTRATOR_SYSTEM + prompt}, {"role": "user", "content": user_msg}],
            temperature=0.35,
            max_tokens=800,
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"(draft unavailable due to error: {exc})"


async def call_verifier(
    lm_client: LMStudioClient, model: str, question: str, draft: str, ledger: dict
) -> VerifierReport:
    user_msg = (
        f"Question: {question}\nDraft: {draft}\nEvidence ledger: {json.dumps(ledger)[:4000]}"
        "\nReturn JSON verdict only."
    )
    try:
        resp = await lm_client.chat_completion(
            model=model,
            messages=[{"role": "system", "content": agents.VERIFIER_SYSTEM}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=600,
        )
        raw = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(raw, lm_client, model)
    except Exception:
        parsed = None
    if not parsed:
        parsed = {"verdict": "PASS", "issues": [], "revised_answer": "", "extra_queries": []}
    return VerifierReport(**parsed)


def reasoning_to_search_depth(level: str, preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if level in ("HIGH", "ULTRA"):
        return "advanced"
    return "basic"


async def run_question(
    run_id: str,
    question: str,
    decision_mode: str,
    manual_level: str,
    search_depth_mode: str,
    max_results_override: int,
    strict_mode: bool,
    db: Database,
    bus: EventBus,
    lm_client: LMStudioClient,
    tavily: TavilyClient,
    settings_models: Dict[str, str],
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
        await db.update_run_router(run_id, router_decision.model_dump())
        await bus.emit(run_id, "router_decision", router_decision.model_dump())
        if strict_mode:
            await bus.emit(run_id, "strict_mode", {"enabled": True})

        plan = await call_plan(lm_client, settings_models["orch"], question, router_decision)
        for task in plan.get("tasks", []):
            await db.add_task(run_id, "orchestrator_plan", {"task": task}, status="planned")
        await bus.emit(run_id, "plan", {"tasks": plan.get("tasks", [])})

        packs: List[EvidencePack] = []
        if router_decision.needs_web:
            lane_prompts = {
                "A": agents.RESEARCH_LANE_A_SYSTEM,
                "B": agents.RESEARCH_LANE_B_SYSTEM,
                "C": agents.RESEARCH_LANE_C_SYSTEM,
            }
            search_depth = reasoning_to_search_depth(router_decision.reasoning_level, search_depth_mode)
            async def run_lane(lane_name: str, lane_system: str) -> EvidencePack:
                await bus.emit(run_id, "lane_started", {"lane": lane_name})
                pack = await call_lane_plan(
                    lm_client, settings_models["worker"], lane_system, question, router_decision, lane_name
                )
                await bus.emit(run_id, "lane_queries", {"lane": lane_name, "queries": pack.queries})
                # Execute Tavily calls for each query
                gathered_urls: List[str] = []
                for query in pack.queries:
                    search_resp = await tavily.search(
                        query=query,
                        search_depth=search_depth,
                        max_results=router_decision.max_results,
                        topic=router_decision.topic,
                    )
                    await db.add_search(run_id, lane_name, query, search_depth, router_decision.max_results, search_resp)
                    if search_resp.get("results"):
                        for res in search_resp["results"][: router_decision.max_results]:
                            url = res.get("url") or ""
                            if url:
                                gathered_urls.append(url)
                                await db.add_source(
                                    run_id,
                                    lane_name,
                                    url,
                                    res.get("title") or "",
                                    res.get("source") or "",
                                    res.get("published_date") or "",
                                    res.get("content") or "",
                                    res.get("content") or "",
                                )
                # Extract details on a subset of URLs
                if gathered_urls:
                    extract_resp = await tavily.extract(gathered_urls[:6], extract_depth=router_decision.extract_depth)
                    await bus.emit(run_id, "lane_extract", {"lane": lane_name, "urls": gathered_urls[:6]})
                    if extract_resp.get("results"):
                        pack.sources = []
                        for res in extract_resp["results"]:
                            await db.add_extract(
                                run_id,
                                lane_name,
                                res.get("url", ""),
                                router_decision.extract_depth,
                                res,
                            )
                            source_obj = EvidenceSource(
                                url=res.get("url", ""),
                                title=res.get("title", ""),
                                publisher=res.get("source", ""),
                                date_published=res.get("published_date", ""),
                                snippet=res.get("content", "")[:400],
                                extracted_text=res.get("content", ""),
                            )
                            pack.sources.append(source_obj)
                            await db.add_source(
                                run_id,
                                lane_name,
                                source_obj.url,
                                source_obj.title or "",
                                source_obj.publisher or "",
                                source_obj.date_published or "",
                                source_obj.snippet or "",
                                source_obj.extracted_text or "",
                            )
                # Persist claims
                for claim in pack.claims:
                    await db.add_claim(run_id, claim, [s.url for s in pack.sources], confidence="MED", notes=lane_name)
                await bus.emit(run_id, "lane_finished", {"lane": lane_name, "sources": len(pack.sources)})
                return pack

            packs = await asyncio.gather(*(run_lane(name, prompt) for name, prompt in lane_prompts.items()))
        else:
            await bus.emit(run_id, "router_skip_web", {"needs_web": False})

        ledger = merge_evidence(packs) if packs else {"sources": [], "claims": [], "conflicts": 0}
        await bus.emit(
            run_id,
            "merge_summary",
            {"sources": len(ledger.get("sources", [])), "claims": len(ledger.get("claims", [])), "conflicts": ledger.get("conflicts", 0)},
        )

        draft = await call_draft(lm_client, settings_models["orch"], question, ledger, router_decision)
        await db.add_draft(run_id, draft)
        await bus.emit(run_id, "draft_ready", {"chars": len(draft)})

        verifier = await call_verifier(lm_client, settings_models["verifier"], question, draft, ledger)
        await db.add_verifier_report(run_id, verifier.verdict, verifier.issues, verifier.revised_answer)
        await bus.emit(
            run_id,
            "verifier_verdict",
            {"verdict": verifier.verdict, "issues": len(verifier.issues), "extra_queries": len(verifier.extra_queries)},
        )

        final_answer = verifier.revised_answer or draft
        confidence = "HIGH" if verifier.verdict == "PASS" else "MED"

        loops = 0
        while verifier.verdict == "NEEDS_REVISION" and verifier.extra_queries and loops < 2:
            loops += 1
            await bus.emit(run_id, "loop_iteration", {"iteration": loops, "extra_queries": verifier.extra_queries})
            # Run targeted searches
            for query in verifier.extra_queries:
                search_resp = await tavily.search(
                    query=query,
                    search_depth=reasoning_to_search_depth(router_decision.reasoning_level, search_depth_mode),
                    max_results=router_decision.max_results,
                    topic=router_decision.topic,
                )
                await db.add_search(run_id, "Loop", query, search_depth_mode, router_decision.max_results, search_resp)
                urls = []
                if search_resp.get("results"):
                    urls = [r.get("url") for r in search_resp["results"] if r.get("url")]
                if urls:
                    extract_resp = await tavily.extract(urls[:4], extract_depth=router_decision.extract_depth)
                    await db.add_extract(run_id, "Loop", ",".join(urls[:4]), router_decision.extract_depth, extract_resp)
                    if extract_resp.get("results"):
                        for res in extract_resp["results"]:
                            ledger.setdefault("sources", []).append(
                                {
                                    "url": res.get("url", ""),
                                    "title": res.get("title", ""),
                                    "publisher": res.get("source", ""),
                                    "date_published": res.get("published_date", ""),
                                    "snippet": res.get("content", "")[:400],
                                    "extracted_text": res.get("content", ""),
                                }
                            )
            # Re-draft and re-verify
            draft = await call_draft(lm_client, settings_models["orch"], question, ledger, router_decision)
            await db.add_draft(run_id, draft)
            verifier = await call_verifier(lm_client, settings_models["verifier"], question, draft, ledger)
            await db.add_verifier_report(run_id, verifier.verdict, verifier.issues, verifier.revised_answer)
            final_answer = verifier.revised_answer or draft
            confidence = "MED" if verifier.verdict == "PASS" else "LOW"
            await bus.emit(
                run_id,
                "verifier_verdict",
                {"verdict": verifier.verdict, "issues": len(verifier.issues), "extra_queries": len(verifier.extra_queries)},
            )

        await db.finalize_run(run_id, final_answer, confidence)
        await db.add_message(run_id, "assistant", final_answer)
        await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": confidence})
    except Exception as exc:
        await db.update_run_status(run_id, f"error: {exc}")
        await bus.emit(run_id, "error", {"message": str(exc)})


def new_run_id() -> str:
    return str(uuid.uuid4())
