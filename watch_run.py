import argparse
import json
import os
import sys
import time
from urllib.parse import urlparse
from typing import Any, Dict, Iterable, List, Optional

import httpx


DEFAULT_PROMPT = (
    "Compare SQLite vs Postgres for storing a local multi-agent app's run logs. "
    "Provide a pros/cons table, a 5-step rollout plan, and a short recommendation with risks."
)


def load_env_port(env_path: str) -> Optional[int]:
    if not os.path.exists(env_path):
        return None
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip().upper() != "PORT":
                    continue
                cleaned = value.strip().strip('"').strip("'")
                if cleaned.isdigit():
                    return int(cleaned)
    except OSError:
        return None
    return None


def resolve_base_url(base_url: Optional[str], port: Optional[int]) -> str:
    if base_url:
        return base_url.rstrip("/")
    env_port = None
    if os.getenv("PORT", "").strip().isdigit():
        env_port = int(os.getenv("PORT", "").strip())
    env_port = env_port or load_env_port(".env")
    port_value = port or env_port or 8000
    return f"http://127.0.0.1:{port_value}"


def condense_text(value: str, limit: int = 160) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def condense_list(items: Iterable[str], limit: int = 3) -> str:
    cleaned = [i for i in items if i]
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return ", ".join(cleaned)
    tail = cleaned[-limit:]
    return f"{', '.join(tail)} (+{len(cleaned) - limit} more)"


def condense_urls(urls: Iterable[str], limit: int = 2, show_full: bool = False) -> str:
    cleaned = [str(u).strip() for u in urls if u and str(u).strip()]
    if not cleaned:
        return ""
    display: List[str] = []
    for url in cleaned[:limit]:
        if show_full:
            display.append(condense_text(url, 80))
        else:
            parsed = urlparse(url)
            host = parsed.netloc or url
            host = host.replace("www.", "")
            display.append(condense_text(host, 40))
    if len(cleaned) > limit:
        display.append(f"+{len(cleaned) - limit} more")
    return ", ".join(display)


def record_event_counts(event_type: str, payload: Dict[str, Any], counts: Dict[str, Any]) -> None:
    if event_type in ("step_started", "step_completed", "step_error"):
        counts[event_type] = counts.get(event_type, 0) + 1
        step_id = payload.get("step_id") or payload.get("step")
        active_steps = counts.setdefault("active_steps", set())
        if not isinstance(active_steps, set):
            active_steps = set(active_steps)
            counts["active_steps"] = active_steps
        if event_type == "step_started" and step_id is not None:
            active_steps.add(step_id)
        elif event_type in ("step_completed", "step_error") and step_id is not None:
            active_steps.discard(step_id)
        counts["max_parallel"] = max(counts.get("max_parallel", 0), len(active_steps))
        return
    if event_type == "model_selected":
        model = payload.get("model") or ""
        if model:
            models = counts.setdefault("models", set())
            if not isinstance(models, set):
                models = set(models)
                counts["models"] = models
            models.add(model)
        return
    if event_type == "tool_request":
        requests = payload.get("requests") or payload.get("tool_requests") or []
        counts["tool_request"] = counts.get("tool_request", 0) + len(requests)
        return
    if event_type == "tool_result":
        results = payload.get("results") or payload.get("tool_results") or []
        counts["tool_result"] = counts.get("tool_result", 0) + len(results)
        return
    if event_type == "tavily_search":
        counts["search_queries"] = counts.get("search_queries", 0) + 1
        for key, metric in (
            ("result_count", "search_results"),
            ("new_sources", "search_new"),
            ("duplicate_sources", "search_duplicate"),
        ):
            try:
                counts[metric] = counts.get(metric, 0) + int(payload.get(key) or 0)
            except Exception:
                counts[metric] = counts.get(metric, 0)
        return
    if event_type == "search_skipped":
        counts["search_skipped"] = counts.get("search_skipped", 0) + 1
        return
    if event_type == "source_found":
        counts["source_found"] = counts.get("source_found", 0) + 1
        return
    if event_type == "claim_found":
        counts["claim_found"] = counts.get("claim_found", 0) + 1
        return
    if event_type == "archived":
        counts["archived"] = counts.get("archived", 0) + 1
        return
    if event_type == "error":
        counts["error"] = counts.get("error", 0) + 1

def summarize_tools(items: Optional[List[Dict[str, Any]]]) -> str:
    tools: List[str] = []
    for item in items or []:
        tool = str(item.get("tool") or item.get("type") or item.get("name") or "").strip()
        if not tool:
            continue
        if tool in ("model_call", "agent_call"):
            profile = item.get("profile") or item.get("agent_profile") or item.get("agent")
            if profile:
                tool = f"{tool}:{profile}"
        tools.append(tool)
    return condense_list(tools, limit=4) or "none"


def format_event(event: Dict[str, Any], counts: Dict[str, Any], view: str = "human") -> Optional[str]:
    event_type = event.get("event_type") or ""
    payload = event.get("payload") or {}
    seq = event.get("seq")
    prefix = f"[{seq:04d}] " if isinstance(seq, int) else ""
    record_event_counts(event_type, payload, counts)

    if view == "human":
        if event_type == "run_started":
            question = payload.get("question") or ""
            if question:
                return f"{prefix}run_started: {condense_text(question, 140)}"
            return f"{prefix}run_started"
        if event_type == "plan_created":
            steps = payload.get("expected_total_steps") or payload.get("steps")
            if isinstance(steps, list):
                steps = len(steps)
            return f"{prefix}plan_ready: steps={steps}" if steps else f"{prefix}plan_ready"
        if event_type == "plan_updated":
            steps = payload.get("expected_total_steps") or payload.get("steps")
            if isinstance(steps, list):
                steps = len(steps)
            return f"{prefix}plan_updated: steps={steps}" if steps else f"{prefix}plan_updated"
        if event_type == "work_log":
            text = payload.get("text") or payload.get("message") or ""
            urls = payload.get("urls") or []
            note = condense_text(str(text), 160) if text else ""
            if urls:
                sources = condense_urls(urls, limit=2, show_full=False)
                if sources:
                    note = f"{note} (sources: {sources})" if note else f"sources: {sources}"
            return f"{prefix}update: {note}" if note else f"{prefix}update"
        if event_type == "narration":
            text = payload.get("text") or ""
            urls = payload.get("urls") or []
            note = condense_text(str(text), 160) if text else ""
            if urls:
                sources = condense_urls(urls, limit=2, show_full=False)
                if sources:
                    note = f"{note} (sources: {sources})" if note else f"sources: {sources}"
            return f"{prefix}note: {note}" if note else f"{prefix}note"
        if event_type == "source_found":
            title = payload.get("title") or ""
            publisher = payload.get("publisher") or ""
            url = payload.get("url") or ""
            bits = []
            if title:
                bits.append(condense_text(str(title), 80))
            if publisher:
                bits.append(condense_text(str(publisher), 40))
            if url:
                bits.append(condense_text(str(url), 80))
            return f"{prefix}source: " + " | ".join(bits) if bits else f"{prefix}source"
        if event_type == "claim_found":
            claim = payload.get("claim") or ""
            urls = payload.get("urls") or []
            bits = []
            if claim:
                bits.append(condense_text(str(claim), 120))
            if urls:
                sources = condense_urls(urls, limit=2, show_full=False)
                if sources:
                    bits.append(f"sources={sources}")
            return f"{prefix}claim: " + ", ".join(bits) if bits else f"{prefix}claim"
        if event_type == "client_note":
            note = payload.get("note") or ""
            return f"{prefix}note: {condense_text(str(note), 160)}" if note else f"{prefix}note"
        if event_type == "tavily_error":
            message = payload.get("message") or ""
            return f"{prefix}warning: {condense_text(str(message), 160)}" if message else f"{prefix}warning"
        if event_type == "tavily_search":
            query = payload.get("query") or ""
            result_count = payload.get("result_count") or 0
            new_sources = payload.get("new_sources") or 0
            duplicate_sources = payload.get("duplicate_sources") or 0
            bits = []
            if query:
                bits.append(condense_text(str(query), 120))
            if result_count:
                counts = []
                if new_sources:
                    counts.append(f"{new_sources} new")
                if duplicate_sources:
                    counts.append(f"{duplicate_sources} seen")
                if not counts:
                    counts.append(f"{result_count} results")
                bits.append(" | ".join(counts))
            return f"{prefix}search: " + " - ".join(bits) if bits else f"{prefix}search"
        if event_type == "search_skipped":
            query = payload.get("query") or ""
            if query:
                return f"{prefix}search_skip: {condense_text(str(query), 120)}"
            return f"{prefix}search_skip"
        if event_type == "step_error":
            name = payload.get("name") or payload.get("type") or "step"
            message = payload.get("message") or payload.get("error") or ""
            if message:
                return f"{prefix}issue: {name} - {condense_text(str(message), 140)}"
            return f"{prefix}issue: {name}"
        if event_type == "error":
            msg = payload.get("message") or payload.get("error") or ""
            return f"{prefix}error: {condense_text(str(msg), 160)}" if msg else f"{prefix}error"
        if event_type == "control_action":
            control = payload.get("control") or ""
            steps = payload.get("steps") or []
            reason = payload.get("reason") or ""
            constraints = payload.get("new_constraints") or {}
            origin = payload.get("origin") or ""
            bits = [control.lower()] if control else []
            if control == "ADD_STEPS" and steps:
                bits.append(f"add_steps={len(steps)}")
            if control == "BACKTRACK" and payload.get("to_step"):
                bits.append(f"to_step={payload.get('to_step')}")
            if control == "RERUN_STEP" and payload.get("step_id"):
                bits.append(f"step_id={payload.get('step_id')}")
            if constraints:
                keys = condense_list([str(k) for k in constraints.keys()], limit=3)
                bits.append(f"constraints={keys}")
            if reason:
                bits.append(f"reason={condense_text(str(reason), 80)}")
            if origin:
                bits.append(f"origin={origin}")
            return f"{prefix}plan_update: " + ", ".join(bits) if bits else f"{prefix}plan_update"
        if event_type == "archived":
            confidence = payload.get("confidence") or ""
            stopped = payload.get("stopped")
            errored = payload.get("error")
            bits = []
            if confidence:
                bits.append(f"confidence={confidence}")
            if stopped:
                bits.append("stopped=true")
            if errored:
                bits.append("error=true")
            return f"{prefix}done: " + ", ".join(bits) if bits else f"{prefix}done"
        return None

    if event_type == "run_started":
        question = payload.get("question") or ""
        if question:
            return f"{prefix}run_started: {condense_text(question, 140)}"
        return f"{prefix}run_started"
    if event_type == "router_decision":
        level = payload.get("reasoning_level") or ""
        needs_web = payload.get("needs_web")
        passes = payload.get("expected_passes")
        route = payload.get("deep_route") or payload.get("deep_route_used") or ""
        mode = payload.get("execution_mode") or ""
        bits = [
            b
            for b in [
                f"level={level}" if level else "",
                f"web={needs_web}" if needs_web is not None else "",
                f"passes={passes}" if passes else "",
                f"route={route}" if route else "",
                f"mode={mode}" if mode else "",
            ]
            if b
        ]
        return f"{prefix}router_decision: " + ", ".join(bits) if bits else f"{prefix}router_decision"
    if event_type == "team_roster":
        planner = payload.get("planner") or "-"
        executor = payload.get("executor") or "-"
        workers = condense_list([str(w) for w in (payload.get("workers") or []) if w], limit=3) or "-"
        verifier = payload.get("verifier") or "-"
        return f"{prefix}team_roster: planner={planner} executor={executor} workers={workers} verifier={verifier}"
    if event_type == "resource_budget":
        budget = payload.get("budget") or payload
        desired = budget.get("desired_parallel") or payload.get("desired_parallel")
        max_slots = budget.get("max_parallel") or budget.get("max")
        return f"{prefix}resource_budget: desired={desired} max={max_slots}"
    if event_type in ("plan_created", "plan_updated", "loop_iteration"):
        steps = payload.get("expected_total_steps") or payload.get("steps")
        passes = payload.get("expected_passes") or payload.get("passes")
        extra = []
        if steps:
            extra.append(f"steps={steps}")
        if passes:
            extra.append(f"passes={passes}")
        return f"{prefix}{event_type}: " + ", ".join(extra) if extra else f"{prefix}{event_type}"
    if event_type in ("step_started", "step_completed", "step_error"):
        step_id = payload.get("step_id") or payload.get("step")
        name = payload.get("name") or payload.get("type") or "step"
        profile = payload.get("agent_profile") or ""
        message = payload.get("message") or ""
        bits = [f"{name}", f"id={step_id}" if step_id else "", f"profile={profile}" if profile else ""]
        if event_type == "step_error" and message:
            bits.append(f"error={condense_text(str(message), 100)}")
        return f"{prefix}{event_type}: " + ", ".join([b for b in bits if b])
    if event_type == "model_selected":
        profile = payload.get("profile") or ""
        model = payload.get("model") or ""
        base_url = payload.get("base_url") or ""
        context = payload.get("context") or ""
        step_id = payload.get("step_id") or payload.get("step")
        fallback = payload.get("fallback")
        bits = [f"profile={profile}" if profile else "", f"model={model}" if model else ""]
        if step_id is not None:
            bits.append(f"step={step_id}")
        if context:
            bits.append(f"context={context}")
        if base_url:
            bits.append(f"base={base_url}")
        if fallback:
            bits.append("fallback=true")
        return f"{prefix}model_selected: " + ", ".join([b for b in bits if b])
    if event_type == "executor_brief":
        note = payload.get("note") or ""
        slots = payload.get("parallel_slots")
        bits = [f"slots={slots}" if slots is not None else ""]
        if note:
            bits.append(f"note={condense_text(str(note), 120)}")
        return f"{prefix}executor_brief: " + ", ".join([b for b in bits if b])
    if event_type == "role_map":
        roles = payload.get("roles") or {}
        if isinstance(roles, dict) and roles:
            pairs = []
            for role, cfg in roles.items():
                if not isinstance(cfg, dict):
                    continue
                model = cfg.get("model")
                status = cfg.get("status")
                preferred = cfg.get("preferred")
                if not model:
                    continue
                label = f"{role}={model}"
                if status and status != "ok":
                    label = f"{label}({status})"
                if preferred and preferred != model:
                    label = f"{label}<-" + str(preferred)
                pairs.append(label)
            return f"{prefix}role_map: " + condense_list([p for p in pairs if p], limit=4)
        return f"{prefix}role_map"
    if event_type == "tool_request":
        requests = payload.get("requests") or payload.get("tool_requests") or []
        step_id = payload.get("step")
        context = payload.get("context") or ""
        tools = summarize_tools(requests)
        bits = [f"step={step_id}" if step_id else "", f"tools={tools}"]
        if context:
            bits.append(f"context={context}")
        return f"{prefix}tool_request: " + ", ".join([b for b in bits if b])
    if event_type == "tool_result":
        results = payload.get("results") or payload.get("tool_results") or []
        step_id = payload.get("step")
        tools = summarize_tools(results)
        bits = [f"step={step_id}" if step_id else "", f"tools={tools}"]
        return f"{prefix}tool_result: " + ", ".join([b for b in bits if b])
    if event_type == "work_log":
        text = payload.get("text") or payload.get("message") or ""
        detail = payload.get("detail") or payload.get("error")
        urls = payload.get("urls") or []
        if detail:
            text = f"{text} ({condense_text(str(detail), 80)})" if text else str(detail)
        if urls:
            sources = condense_urls(urls, limit=2, show_full=False)
            if sources:
                text = f"{text} (sources: {sources})" if text else f"sources: {sources}"
        if text:
            return f"{prefix}work_log: {condense_text(str(text), 160)}"
        return f"{prefix}work_log"
    if event_type == "dev_trace":
        text = payload.get("text") or payload.get("message") or ""
        detail = payload.get("detail") or payload.get("error")
        if detail:
            text = f"{text} ({condense_text(str(detail), 80)})" if text else str(detail)
        if text:
            return f"{prefix}dev_trace: {condense_text(str(text), 160)}"
        return f"{prefix}dev_trace"
    if event_type == "narration":
        text = payload.get("text") or ""
        urls = payload.get("urls") or []
        note = condense_text(str(text), 160) if text else ""
        if urls:
            sources = condense_urls(urls, limit=2, show_full=False)
            if sources:
                note = f"{note} (sources: {sources})" if note else f"sources: {sources}"
        return f"{prefix}note: {note}" if note else f"{prefix}note"
    if event_type == "source_found":
        title = payload.get("title") or ""
        publisher = payload.get("publisher") or ""
        url = payload.get("url") or ""
        bits = []
        if title:
            bits.append(condense_text(str(title), 80))
        if publisher:
            bits.append(condense_text(str(publisher), 40))
        if url:
            bits.append(condense_text(str(url), 80))
        return f"{prefix}source_found: " + " | ".join(bits) if bits else f"{prefix}source_found"
    if event_type == "claim_found":
        claim = payload.get("claim") or ""
        urls = payload.get("urls") or []
        bits = []
        if claim:
            bits.append(condense_text(str(claim), 120))
        if urls:
            sources = condense_urls(urls, limit=2, show_full=False)
            if sources:
                bits.append(f"sources={sources}")
        return f"{prefix}claim_found: " + ", ".join(bits) if bits else f"{prefix}claim_found"
    if event_type == "client_note":
        note = payload.get("note") or ""
        return f"{prefix}client_note: {condense_text(str(note), 160)}" if note else f"{prefix}client_note"
    if event_type in ("tavily_search", "tavily_extract", "tavily_error"):
        query = payload.get("query") or ""
        urls = payload.get("urls") or []
        message = payload.get("message") or ""
        bits = []
        if query:
            bits.append(f"query={condense_text(str(query), 100)}")
        if urls:
            sources = condense_urls(urls, limit=2, show_full=False)
            bits.append(f"sources={sources}" if sources else f"urls={len(urls)}")
        if message:
            bits.append(f"message={condense_text(str(message), 100)}")
        if event_type == "tavily_search":
            result_count = payload.get("result_count")
            new_sources = payload.get("new_sources")
            duplicate_sources = payload.get("duplicate_sources")
            if result_count:
                bits.append(f"results={result_count}")
            if new_sources:
                bits.append(f"new={new_sources}")
            if duplicate_sources:
                bits.append(f"dupes={duplicate_sources}")
        return f"{prefix}{event_type}: " + ", ".join(bits) if bits else f"{prefix}{event_type}"
    if event_type == "archived":
        confidence = payload.get("confidence") or ""
        stopped = payload.get("stopped")
        errored = payload.get("error")
        bits = []
        if confidence:
            bits.append(f"confidence={confidence}")
        if stopped:
            bits.append("stopped=true")
        if errored:
            bits.append("error=true")
        return f"{prefix}archived: " + ", ".join(bits) if bits else f"{prefix}archived"
    if event_type == "error":
        msg = payload.get("message") or payload.get("error") or ""
        return f"{prefix}error: {condense_text(str(msg), 160)}"

    try:
        fallback = condense_text(json.dumps(payload, ensure_ascii=True), 160)
    except TypeError:
        fallback = condense_text(str(payload), 160)
    return f"{prefix}{event_type}: {fallback}" if event_type else f"{prefix}{fallback}"


def iter_sse_events(response: httpx.Response) -> Iterable[Dict[str, Any]]:
    data_lines: List[str] = []
    for raw_line in response.iter_lines():
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            if data_lines:
                joined = "\n".join(data_lines)
                data_lines.clear()
                try:
                    yield json.loads(joined)
                except json.JSONDecodeError:
                    continue
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        joined = "\n".join(data_lines)
        try:
            yield json.loads(joined)
        except json.JSONDecodeError:
            return


def safe_print(text: str) -> None:
    try:
        print(text)
    except OSError:
        cleaned = text.encode("ascii", "backslashreplace").decode("ascii")
        print(cleaned)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Start a LocalPro run and stream SSE events.")
    parser.add_argument("prompt", nargs="*", help="Prompt text. Defaults to a built-in diagnostic prompt.")
    parser.add_argument("--prompt-file", help="Path to a prompt text file.")
    parser.add_argument("--run-id", help="Attach to an existing run instead of starting a new one.")
    parser.add_argument("--base-url", help="Base URL for the app (ex: http://127.0.0.1:8000).")
    parser.add_argument("--port", type=int, help="Port override if base URL is not set.")
    parser.add_argument("--model-tier", default="pro", choices=["fast", "pro", "deep", "auto"])
    parser.add_argument("--reasoning-mode", default="manual", choices=["auto", "manual"])
    parser.add_argument("--manual-level", default="HIGH", choices=["LOW", "MED", "HIGH", "ULTRA"])
    parser.add_argument("--deep-mode", default="auto", choices=["auto", "oss", "cluster"])
    parser.add_argument("--strict", action="store_true", help="Enable strict verification mode.")
    parser.add_argument("--evidence-dump", action="store_true", help="Enable evidence dump artifacts.")
    parser.add_argument(
        "--view",
        default="human",
        choices=["human", "debug"],
        help="Output view: human for plain-language updates, debug for full event details.",
    )
    args = parser.parse_args()

    base_url = resolve_base_url(args.base_url, args.port)

    prompt_text = ""
    if args.prompt_file:
        try:
            prompt_text = open(args.prompt_file, "r", encoding="utf-8").read().strip()
        except OSError as exc:
            print(f"Failed to read prompt file: {exc}", file=sys.stderr)
            return 1
    if not prompt_text and args.prompt:
        prompt_text = " ".join(args.prompt).strip()
    if not prompt_text and not args.run_id:
        prompt_text = DEFAULT_PROMPT

    timeout = httpx.Timeout(10.0, read=None)
    counts: Dict[str, Any] = {
        "start_ts": None,
        "active_steps": set(),
        "max_parallel": 0,
        "models": set(),
    }
    start_ts = time.time()
    try:
        with httpx.Client(timeout=timeout) as client:
            run_id = args.run_id
            if not run_id:
                payload: Dict[str, Any] = {
                    "question": prompt_text,
                    "model_tier": args.model_tier,
                    "reasoning_mode": args.reasoning_mode,
                    "strict_mode": args.strict,
                    "deep_mode": args.deep_mode,
                    "evidence_dump": args.evidence_dump,
                }
                if args.reasoning_mode == "manual":
                    payload["manual_level"] = args.manual_level
                resp = client.post(f"{base_url}/api/run", json=payload)
                resp.raise_for_status()
                run_id = resp.json().get("run_id")
                if not run_id:
                    print("Run did not return a run_id.", file=sys.stderr)
                    return 1
                safe_print(f"Run started: {run_id}")
                if prompt_text:
                    safe_print(f"Prompt: {condense_text(prompt_text, 180)}")
            else:
                safe_print(f"Attaching to run: {run_id}")

            events_url = f"{base_url}/runs/{run_id}/events"
            with client.stream("GET", events_url) as response:
                response.raise_for_status()
                for event in iter_sse_events(response):
                    if counts["start_ts"] is None:
                        counts["start_ts"] = time.time()
                    line = format_event(event, counts, view=args.view)
                    if line:
                        safe_print(line)
                    if event.get("event_type") == "archived":
                        break
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    except httpx.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - start_ts
    steps_started = counts.get("step_started", 0)
    steps_completed = counts.get("step_completed", 0)
    step_errors = counts.get("step_error", 0)
    tool_requests = counts.get("tool_request", 0)
    tool_results = counts.get("tool_result", 0)
    errors = counts.get("error", 0)
    max_parallel = counts.get("max_parallel", 0)
    search_queries = counts.get("search_queries", 0)
    search_new = counts.get("search_new", 0)
    search_duplicate = counts.get("search_duplicate", 0)
    search_skipped = counts.get("search_skipped", 0)
    sources_found = counts.get("source_found", 0)
    claims_found = counts.get("claim_found", 0)
    models = counts.get("models", set()) or set()
    unique_models = len(models)
    model_list = condense_list(sorted(models), limit=4) if models else "none"
    safe_print(
        "Summary: steps_started={}; steps_completed={}; step_errors={}; tool_requests={}; tool_results={}; searches={}; search_new={}; search_dupes={}; search_skipped={}; sources={}; claims={}; errors={}; max_parallel={}; models_used={}; elapsed={:.1f}s".format(
            steps_started,
            steps_completed,
            step_errors,
            tool_requests,
            tool_results,
            search_queries,
            search_new,
            search_duplicate,
            search_skipped,
            sources_found,
            claims_found,
            errors,
            max_parallel,
            f"{unique_models} ({model_list})",
            elapsed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
