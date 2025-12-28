import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import AppSettings, CONFIG_PATH, load_settings, save_settings
from .db import Database
from .llm import LMStudioClient
from .orchestrator import EventBus, new_run_id, run_question
from .schemas import ControlCommand, StartRunRequest
from .tavily import TavilyClient
from .system_info import compute_worker_slots, get_resource_snapshot


_MODEL_SIZE_RE = re.compile(r"(\\d+(?:\\.\\d+)?b)", re.IGNORECASE)
REASONING_LEVELS = {"LOW", "MED", "HIGH", "ULTRA"}


def _normalize_model_id(value: str) -> str:
    base = value.split(":")[0].strip()
    if "/" in base:
        base = base.rsplit("/", 1)[-1]
    return base.lower()


def _resolve_model_id(configured_id: Optional[str], available: List[str]) -> Optional[str]:
    if not configured_id or not available:
        return None
    if configured_id in available:
        return configured_id
    base = configured_id.split(":")[0]
    if base in available:
        return base
    target = _normalize_model_id(configured_id)
    for mid in available:
        if _normalize_model_id(mid) == target:
            return mid
    size_match = _MODEL_SIZE_RE.search(configured_id)
    if size_match:
        size_hint = size_match.group(1).lower()
        for mid in available:
            if size_hint in mid.lower():
                return mid
    return None


def _match_model_id(configured_id: Optional[str], available: List[str]) -> Optional[str]:
    """Resolve model IDs without falling back to an arbitrary available model."""
    if not configured_id or not available:
        return None
    if configured_id in available:
        return configured_id
    base = configured_id.split(":")[0]
    if base in available:
        return base
    target = _normalize_model_id(configured_id)
    for mid in available:
        if _normalize_model_id(mid) == target:
            return mid
    return None


def build_model_map(settings_obj: AppSettings, availability: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, str]]:
    model_map = {
        "orch": {"base_url": settings_obj.orch_endpoint.base_url, "model": settings_obj.orch_endpoint.model_id},
        "worker": {"base_url": settings_obj.worker_a_endpoint.base_url, "model": settings_obj.worker_a_endpoint.model_id},
        "worker_b": {"base_url": settings_obj.worker_b_endpoint.base_url, "model": settings_obj.worker_b_endpoint.model_id},
        "worker_c": {"base_url": settings_obj.worker_c_endpoint.base_url, "model": settings_obj.worker_c_endpoint.model_id},
        "fast": {"base_url": settings_obj.fast_endpoint.base_url, "model": settings_obj.fast_endpoint.model_id},
        "deep_planner": {"base_url": settings_obj.deep_planner_endpoint.base_url, "model": settings_obj.deep_planner_endpoint.model_id},
        "deep_orch": {"base_url": settings_obj.deep_orchestrator_endpoint.base_url, "model": settings_obj.deep_orchestrator_endpoint.model_id},
        "router": {"base_url": settings_obj.router_endpoint.base_url, "model": settings_obj.router_endpoint.model_id},
        "summarizer": {"base_url": settings_obj.summarizer_endpoint.base_url, "model": settings_obj.summarizer_endpoint.model_id},
        "verifier": {"base_url": settings_obj.verifier_endpoint.base_url, "model": settings_obj.verifier_endpoint.model_id},
    }
    if availability:
        for role, cfg in model_map.items():
            info = availability.get(role) if isinstance(availability, dict) else None
            if not isinstance(info, dict):
                continue
            available = info.get("available") or []
            resolved = _resolve_model_id(cfg.get("model"), available)
            if resolved and resolved != cfg.get("model"):
                model_map[role] = {**cfg, "model": resolved}

        def fallback(role: str, target: str, allow_unloaded: bool = False) -> None:
            info = availability.get(role)
            if not isinstance(info, dict):
                return
            cfg = model_map.get(role)
            target_cfg = model_map.get(target)
            if not cfg or not target_cfg:
                return
            configured_id = cfg.get("model")
            configured_url = cfg.get("base_url")
            if not configured_id or not configured_url:
                model_map[role] = target_cfg
                return
            available = info.get("available") or []
            missing = info.get("ok") is False or (configured_id and configured_id not in available)
            if missing:
                if allow_unloaded and not info.get("error"):
                    return
                model_map[role] = target_cfg

        fallback("worker_b", "worker")
        fallback("worker_c", "worker")
        fallback("fast", "worker")
        fallback("deep_planner", "worker")
        fallback("deep_orch", "orch")
        fallback("router", "orch")
        fallback("summarizer", "router")
        fallback("verifier", "worker")
    return model_map


async def refresh_model_check(settings_obj: AppSettings, lm_client: LMStudioClient) -> Dict[str, Any]:
    """Fetch available models per role and cache the results for routing fallbacks."""
    checks: Dict[str, Any] = {}
    raw_map = build_model_map(settings_obj)
    base_cache: Dict[str, Dict[str, Any]] = {}
    for role, cfg in raw_map.items():
        base_url = cfg.get("base_url") or ""
        if not base_url:
            checks[role] = {"ok": False, "error": "missing_base_url"}
            continue
        if base_url not in base_cache:
            try:
                resp = await lm_client.list_models(base_url)
                ids = [m.get("id") for m in resp.get("data", []) if m.get("id")]
                base_cache[base_url] = {"ok": True, "available": ids}
            except Exception as exc:
                base_cache[base_url] = {"ok": False, "error": str(exc), "available": []}
        base_info = base_cache[base_url]
        if not base_info.get("ok"):
            checks[role] = {"ok": False, "error": base_info.get("error") or "unreachable"}
            continue
        ids = base_info.get("available") or []
        resolved = _match_model_id(cfg.get("model"), ids)
        ok = resolved is not None
        entry = {"ok": ok, "missing": [] if ok else [cfg.get("model")], "available": ids}
        if ok and resolved and resolved != cfg.get("model"):
            entry["resolved"] = resolved
        checks[role] = entry
    resources = get_resource_snapshot()
    checks["resources"] = resources
    try:
        active_map = build_model_map(settings_obj, checks)
        checks["worker_slots"] = compute_worker_slots(active_map, "pro", checks, resources)
    except Exception:
        pass
    return checks


def normalize_reasoning_level(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip().upper()
    if cleaned in REASONING_LEVELS:
        return cleaned
    return None


def normalize_control_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"control": "CONTINUE"}
    control_raw = payload.get("control") or payload.get("action") or payload.get("command")
    steps = payload.get("steps") or payload.get("plan_steps") or []
    new_constraints = payload.get("new_constraints") or payload.get("constraints") or {}
    if not control_raw:
        if steps:
            control_raw = "ADD_STEPS"
        else:
            control_raw = "CONTINUE"
    control_map = {
        "ADD_STEP": "ADD_STEPS",
        "ADD_STEPS": "ADD_STEPS",
        "BACKTRACK": "BACKTRACK",
        "RERUN": "RERUN_STEP",
        "RETRY": "RERUN_STEP",
        "RERUN_STEP": "RERUN_STEP",
        "STOP": "STOP",
        "CANCEL": "STOP",
        "CONTINUE": "CONTINUE",
        "UPDATE": "CONTINUE",
        "UPDATE_PLAN": "CONTINUE",
    }
    control = control_map.get(str(control_raw).strip().upper(), str(control_raw).strip().upper())
    normalized = dict(payload)
    normalized["control"] = control
    normalized["steps"] = steps if isinstance(steps, list) else []
    normalized["new_constraints"] = new_constraints if isinstance(new_constraints, dict) else {}
    return normalized


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_event_bus(request: Request) -> EventBus:
    return request.app.state.bus


def get_lm_client(request: Request) -> LMStudioClient:
    return request.app.state.lm_client


def get_tavily_client(request: Request) -> TavilyClient:
    return request.app.state.tavily_client


def get_model_check(request: Request) -> Dict[str, Any]:
    return request.app.state.model_check


def get_upload_dir(request: Request) -> Path:
    return request.app.state.upload_dir


def get_max_upload_bytes(request: Request) -> int:
    return request.app.state.max_upload_bytes


def get_run_tasks(request: Request) -> Dict[str, asyncio.Task]:
    return request.app.state.run_tasks


def get_run_stop_events(request: Request) -> Dict[str, asyncio.Event]:
    return request.app.state.run_stop_events


def get_run_control_queues(request: Request) -> Dict[str, asyncio.Queue]:
    return request.app.state.run_control_queues


def get_config_path(request: Request) -> Path:
    return request.app.state.config_path


def sse_format(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def validate_upload(file: UploadFile) -> None:
    allowed = {"image/png", "image/jpeg", "image/webp", "image/gif", "application/pdf"}
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required.")
    raw_name = file.filename
    safe_name = Path(raw_name).name
    if safe_name != raw_name or safe_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only images or PDFs are allowed.")


async def stop_run_internal(
    run_id: str,
    db: Database,
    bus: EventBus,
    run_stop_events: Dict[str, asyncio.Event],
) -> Dict[str, Any]:
    run = await db.get_run_summary(run_id)
    stop_event = run_stop_events.get(run_id)
    if not run:
        if stop_event:
            if not stop_event.is_set():
                stop_event.set()
            return {"ok": True, "status": "stopping"}
        raise HTTPException(status_code=404, detail="Run not found")
    if stop_event and not stop_event.is_set():
        stop_event.set()
        return {"ok": True, "status": "stopping"}
    if run.get("status") and run["status"].startswith("error"):
        return {"ok": True, "status": run["status"]}
    if run.get("status") in ("completed", "stopped"):
        return {"ok": True, "status": run["status"]}
    await db.update_run_status(run_id, "stopped")
    await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "LOW", "stopped": True})
    return {"ok": True, "status": "stopped"}


router = APIRouter()


@router.get("/")
async def index(request: Request):
    static_dir = request.app.state.static_dir
    return FileResponse(static_dir / "index.html")


@router.get("/settings")
async def get_settings_route(
    request: Request,
    settings: AppSettings = Depends(get_settings),
    lm_client: LMStudioClient = Depends(get_lm_client),
    model_check: Dict[str, Any] = Depends(get_model_check),
):
    if not model_check:
        request.app.state.model_check = await refresh_model_check(settings, lm_client)
    return {"settings": settings.to_safe_dict(), "model_check": request.app.state.model_check}


@router.post("/settings")
async def update_settings_route(
    request: Request,
    settings: AppSettings = Depends(get_settings),
    db: Database = Depends(get_db),
    lm_client: LMStudioClient = Depends(get_lm_client),
    tavily_client: TavilyClient = Depends(get_tavily_client),
    config_path: Path = Depends(get_config_path),
):
    body = await request.json()
    new_settings = AppSettings(**{**settings.model_dump(), **body})
    save_settings(new_settings, config_path=config_path)
    await db.save_config(new_settings.model_dump())
    request.app.state.settings = new_settings
    lm_client.base_url = new_settings.lm_studio_base_url
    lm_client.max_output_tokens = new_settings.oss_max_tokens
    tavily_client.api_key = new_settings.tavily_api_key
    upload_dir = Path(new_settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    request.app.state.upload_dir = upload_dir
    request.app.state.max_upload_bytes = new_settings.upload_max_mb * 1024 * 1024
    request.app.state.model_check = await refresh_model_check(new_settings, lm_client)
    return {"ok": True, "model_check": request.app.state.model_check}


@router.post("/api/uploads")
async def upload_file(
    file: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
    settings: AppSettings = Depends(get_settings),
    db: Database = Depends(get_db),
    bus: EventBus = Depends(get_event_bus),
    upload_dir: Path = Depends(get_upload_dir),
    max_upload_bytes: int = Depends(get_max_upload_bytes),
):
    validate_upload(file)
    data = await file.read()
    if len(data) > max_upload_bytes:
        raise HTTPException(status_code=400, detail=f"File too large (>{settings.upload_max_mb} MB).")
    safe_name = Path(file.filename).name
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    upload_path = upload_dir / stored_name
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(data)
    upload_id = await db.add_upload(
        run_id,
        stored_name,
        safe_name,
        file.content_type or "application/octet-stream",
        len(data),
        str(upload_path),
    )
    if run_id:
        await bus.emit(
            run_id,
            "upload_received",
            {"upload_id": upload_id, "name": safe_name, "mime": file.content_type, "size": len(data)},
        )
    return {"id": upload_id, "filename": safe_name, "mime": file.content_type, "size": len(data)}


@router.get("/api/uploads/{upload_id}")
async def get_upload(upload_id: int, db: Database = Depends(get_db)):
    record = await db.get_upload(upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")
    path = Path(record["storage_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(path, media_type=record["mime"], filename=record["original_name"])


@router.post("/api/discover")
async def discover_models(
    payload: Dict[str, Any],
    settings: AppSettings = Depends(get_settings),
    lm_client: LMStudioClient = Depends(get_lm_client),
):
    base_urls = payload.get("base_urls") or settings.discovery_base_urls
    results = {}
    for url in base_urls:
        try:
            resp = await lm_client.list_models(url)
            ids = [m.get("id") for m in resp.get("data", [])]
            results[url] = {"ok": True, "models": ids}
        except Exception as exc:
            results[url] = {"ok": False, "error": str(exc)}
    return {"results": results}


@router.post("/api/run")
async def start_run(
    payload: StartRunRequest,
    request: Request,
    settings: AppSettings = Depends(get_settings),
    db: Database = Depends(get_db),
    bus: EventBus = Depends(get_event_bus),
    lm_client: LMStudioClient = Depends(get_lm_client),
    tavily_client: TavilyClient = Depends(get_tavily_client),
    model_check: Dict[str, Any] = Depends(get_model_check),
    upload_dir: Path = Depends(get_upload_dir),
    run_tasks: Dict[str, asyncio.Task] = Depends(get_run_tasks),
    run_stop_events: Dict[str, asyncio.Event] = Depends(get_run_stop_events),
    run_control_queues: Dict[str, asyncio.Queue] = Depends(get_run_control_queues),
):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    fields_set = payload.model_fields_set or set()
    reasoning_auto_set = "reasoning_auto" in fields_set
    reasoning_mode = payload.reasoning_mode
    if reasoning_auto_set and payload.reasoning_auto:
        reasoning_mode = "auto"
    elif reasoning_auto_set and not payload.reasoning_auto:
        reasoning_mode = payload.reasoning_mode

    default_level = normalize_reasoning_level(settings.reasoning_depth_default)
    manual_level_set = "manual_level" in fields_set
    manual_level = payload.manual_level if manual_level_set else None
    effective_manual_level = manual_level or default_level or payload.manual_level
    if reasoning_mode == "manual" and manual_level is None:
        manual_level = effective_manual_level

    conversation_id = payload.conversation_id or await db.get_default_conversation_id()
    conversation = None
    if conversation_id:
        conversation = await db.get_conversation(conversation_id)
    if payload.conversation_id and not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if not conversation:
        title_seed = question if len(question) <= 80 else question[:77].rstrip() + "..."
        conversation = await db.create_conversation(
            title=title_seed,
            model_tier=payload.model_tier,
            reasoning_mode=reasoning_mode,
            manual_level=effective_manual_level,
            deep_mode=payload.deep_mode,
        )
        conversation_id = conversation["id"]
    else:
        await db.update_conversation(
            conversation_id,
            model_tier=payload.model_tier,
            reasoning_mode=reasoning_mode,
            manual_level=effective_manual_level,
            deep_mode=payload.deep_mode,
        )
        title_seed = question if len(question) <= 80 else question[:77].rstrip() + "..."
        await db.ensure_conversation_title(conversation_id, title_seed)
    run_id = new_run_id()
    bus.register_run(run_id, conversation_id)
    prompt_state = await db.set_prompt_state(question, run_id)
    await bus.emit("conversation", "prompt_updated", prompt_state)
    models = build_model_map(settings)
    upload_ids = payload.upload_ids or []
    for uid in upload_ids:
        await db.assign_upload_to_run(uid, run_id)
        await db.update_upload_status(uid, "queued")
    stop_event = asyncio.Event()
    run_stop_events[run_id] = stop_event
    control_queue: asyncio.Queue = asyncio.Queue()
    run_control_queues[run_id] = control_queue

    async def run_and_cleanup() -> None:
        try:
            await run_question(
                run_id=run_id,
                conversation_id=conversation_id,
                question=payload.question,
                decision_mode=reasoning_mode,
                manual_level=manual_level,
                default_reasoning_level=default_level,
                model_tier=payload.model_tier,
                deep_mode=payload.deep_mode,
                search_depth_mode=payload.search_depth_mode,
                max_results_override=payload.max_results or 0,
                strict_mode=payload.strict_mode,
                auto_memory=payload.auto_memory,
                evidence_dump=payload.evidence_dump,
                db=db,
                bus=bus,
                lm_client=lm_client,
                tavily=tavily_client,
                settings_models=models,
                model_availability=model_check,
                upload_ids=upload_ids,
                upload_dir=upload_dir,
                stop_event=stop_event,
                control_queue=control_queue,
            )
        finally:
            run_tasks.pop(run_id, None)
            run_stop_events.pop(run_id, None)
            run_control_queues.pop(run_id, None)
            # Ensure the prompt unlocks and a terminal event is emitted even if the run ended early.
            try:
                prompt = await db.get_prompt_state()
                if prompt and prompt.get("run_id") == run_id:
                    updated_at = await db.clear_prompt_state()
                    await bus.emit("conversation", "prompt_cleared", {"updated_at": updated_at})
            except Exception:
                pass
            try:
                archived_row = await db.fetchone(
                    "SELECT 1 FROM events WHERE run_id=? AND event_type='archived' LIMIT 1",
                    (run_id,),
                )
                if not archived_row:
                    run_summary = await db.get_run_summary(run_id)
                    status = (run_summary or {}).get("status") or ""
                    confidence = (run_summary or {}).get("confidence") or "LOW"
                    archive_payload = {"run_id": run_id, "confidence": confidence}
                    if status == "stopped":
                        archive_payload["stopped"] = True
                    if status.startswith("error"):
                        archive_payload["error"] = True
                    await bus.emit(run_id, "archived", archive_payload)
            except Exception:
                pass

    task = asyncio.create_task(run_and_cleanup())
    run_tasks[run_id] = task
    return {"run_id": run_id, "conversation_id": conversation_id}


@router.get("/api/run/latest")
async def get_latest_run(
    conversation_id: Optional[str] = None,
    db: Database = Depends(get_db),
):
    reset_at = await db.get_conversation_reset()
    convo_id = conversation_id or await db.get_default_conversation_id()
    run = await db.get_latest_run(after=reset_at, conversation_id=convo_id)
    return {"run": run, "reset_at": reset_at, "conversation_id": convo_id}


@router.get("/api/run/{run_id}")
async def get_run(run_id: str, db: Database = Depends(get_db)):
    run = await db.get_run_summary(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/api/run/{run_id}/events")
async def list_run_events(run_id: str, after_seq: int = 0, db: Database = Depends(get_db)):
    run = await db.get_run_summary(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    events = await db.list_events(run_id, after_seq=after_seq)
    last_seq = events[-1]["seq"] if events else after_seq
    return {"events": events, "last_seq": last_seq}


@router.post("/api/run/{run_id}/stop")
async def stop_run(
    run_id: str,
    db: Database = Depends(get_db),
    bus: EventBus = Depends(get_event_bus),
    run_stop_events: Dict[str, asyncio.Event] = Depends(get_run_stop_events),
):
    return await stop_run_internal(run_id, db, bus, run_stop_events)


@router.post("/api/run/{run_id}/control")
async def control_run(
    run_id: str,
    payload: Dict[str, Any] = Body(default={}),
    db: Database = Depends(get_db),
    run_control_queues: Dict[str, asyncio.Queue] = Depends(get_run_control_queues),
):
    queue = run_control_queues.get(run_id)
    if not queue:
        run = await db.get_run_summary(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        raise HTTPException(status_code=409, detail="Run is not active")
    normalized = normalize_control_payload(payload)
    try:
        control = ControlCommand(**normalized)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid control payload")
    await queue.put(control)
    return {"ok": True}


@router.get("/api/run/{run_id}/artifacts")
async def get_artifacts(run_id: str, db: Database = Depends(get_db)):
    run = await db.get_run_summary(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    sources = await db.get_sources(run_id)
    claims = await db.get_claims(run_id)
    draft = await db.get_latest_draft(run_id)
    verifier = await db.get_verifier_report(run_id)
    artifacts = await db.get_artifacts(run_id)
    uploads = await db.list_uploads(run_id)
    return {
        "run": run,
        "sources": sources,
        "claims": claims,
        "draft": draft,
        "verifier": verifier,
        "artifacts": artifacts,
        "uploads": uploads,
    }


@router.get("/api/memory")
async def list_memory(q: Optional[str] = None, db: Database = Depends(get_db)):
    if q:
        items = await db.search_memory(q, limit=50)
    else:
        items = await db.list_memory(limit=50)
    return {"items": items}


@router.post("/api/memory")
async def create_memory(item: Dict[str, Any], db: Database = Depends(get_db)):
    mem_id = await db.add_memory_item(
        item.get("kind", "note"),
        item.get("title", ""),
        item.get("content", ""),
        item.get("tags", []),
        pinned=item.get("pinned", False),
        relevance_score=item.get("relevance_score", 0.0),
    )
    return {"id": mem_id}


@router.patch("/api/memory/{item_id}")
async def update_memory(item_id: int, item: Dict[str, Any], db: Database = Depends(get_db)):
    await db.update_memory_item(
        item_id, title=item.get("title"), content=item.get("content"), pinned=item.get("pinned")
    )
    return {"ok": True}


@router.delete("/api/memory/{item_id}")
async def delete_memory(item_id: int, db: Database = Depends(get_db)):
    await db.delete_memory_item(item_id)
    return {"ok": True}


@router.get("/api/conversations")
async def list_conversations(db: Database = Depends(get_db)):
    conversations = await db.list_conversations()
    return {"conversations": conversations}


@router.post("/api/conversations")
async def create_conversation(
    payload: Dict[str, Any] = Body(default={}),
    db: Database = Depends(get_db),
    bus: EventBus = Depends(get_event_bus),
):
    convo = await db.create_conversation(
        title=payload.get("title"),
        model_tier=payload.get("model_tier", "pro"),
        reasoning_mode=payload.get("reasoning_mode", "auto"),
        manual_level=payload.get("manual_level", "MED"),
        deep_mode=payload.get("deep_mode", "auto"),
    )
    await bus.emit("conversation", "conversation_created", {"conversation_id": convo["id"], "conversation": convo})
    return {"conversation": convo}


@router.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, db: Database = Depends(get_db)):
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": convo}


@router.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, limit: int = 200, db: Database = Depends(get_db)):
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await db.list_messages(conversation_id=conversation_id, limit=limit)
    return {"messages": messages}


@router.patch("/api/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    payload: Dict[str, Any] = Body(default={}),
    db: Database = Depends(get_db),
    bus: EventBus = Depends(get_event_bus),
):
    convo = await db.update_conversation(
        conversation_id,
        title=payload.get("title"),
        model_tier=payload.get("model_tier"),
        reasoning_mode=payload.get("reasoning_mode"),
        manual_level=payload.get("manual_level"),
        deep_mode=payload.get("deep_mode"),
    )
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await bus.emit("conversation", "conversation_updated", {"conversation_id": convo["id"], "conversation": convo})
    return {"conversation": convo}


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: Database = Depends(get_db),
    bus: EventBus = Depends(get_event_bus),
):
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete_conversation(conversation_id)
    await db.execute("UPDATE conversation_state SET default_conversation_id=NULL WHERE id=1")
    await db.get_default_conversation_id()
    await bus.emit("conversation", "conversation_deleted", {"conversation_id": conversation_id})
    return {"ok": True}


@router.get("/api/conversation")
async def conversation_history(
    limit: int = 200,
    conversation_id: Optional[str] = None,
    db: Database = Depends(get_db),
):
    convo_id = conversation_id or await db.get_default_conversation_id()
    messages = await db.list_messages(conversation_id=convo_id, limit=limit)
    reset_at = await db.get_conversation_reset()
    return {"messages": messages, "reset_at": reset_at, "conversation_id": convo_id}


@router.delete("/api/conversation")
async def reset_conversation(
    conversation_id: Optional[str] = None,
    db: Database = Depends(get_db),
    bus: EventBus = Depends(get_event_bus),
):
    reset_at = await db.reset_conversation(conversation_id=conversation_id)
    convo_id = conversation_id or await db.get_default_conversation_id()
    await bus.emit("conversation", "conversation_reset", {"reset_at": reset_at, "conversation_id": convo_id})
    return {"ok": True, "reset_at": reset_at, "conversation_id": convo_id}


@router.get("/api/prompt")
async def get_prompt(db: Database = Depends(get_db)):
    prompt = await db.get_prompt_state()
    return {"prompt": prompt}


@router.delete("/api/prompt")
async def clear_prompt(
    db: Database = Depends(get_db),
    bus: EventBus = Depends(get_event_bus),
    run_stop_events: Dict[str, asyncio.Event] = Depends(get_run_stop_events),
):
    prompt = await db.get_prompt_state()
    run_id = prompt.get("run_id") if prompt else None
    if run_id:
        try:
            await stop_run_internal(run_id, db, bus, run_stop_events)
        except HTTPException:
            pass
        except Exception:
            pass
    updated_at = await db.clear_prompt_state()
    await bus.emit("conversation", "prompt_cleared", {"updated_at": updated_at})
    return {"ok": True, "updated_at": updated_at}


@router.get("/events")
async def stream_global_events(bus: EventBus = Depends(get_event_bus)):
    async def event_generator():
        queue = await bus.subscribe_global()
        try:
            while True:
                ev = await queue.get()
                yield sse_format(ev)
        except asyncio.CancelledError:
            pass
        finally:
            await bus.unsubscribe_global(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/runs/{run_id}/events")
async def stream_events(
    run_id: str,
    db: Database = Depends(get_db),
    bus: EventBus = Depends(get_event_bus),
):
    # Preload past events then stream new ones
    async def event_generator():
        queue = await bus.subscribe(run_id)
        try:
            past = await db.list_events(run_id)
            for ev in past:
                yield sse_format(ev)
            while True:
                ev = await queue.get()
                yield sse_format(ev)
        except asyncio.CancelledError:
            pass
        finally:
            await bus.unsubscribe(run_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def create_app(
    settings: AppSettings,
    *,
    db: Optional[Database] = None,
    lm_client: Optional[LMStudioClient] = None,
    tavily_client: Optional[TavilyClient] = None,
    config_path: Optional[Path] = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.db.init()
        await app.state.db.save_config(app.state.settings.model_dump())
        app.state.upload_dir.mkdir(parents=True, exist_ok=True)
        app.state.model_check = await refresh_model_check(app.state.settings, app.state.lm_client)
        try:
            yield
        finally:
            await app.state.lm_client.close()
            await app.state.tavily_client.close()

    app = FastAPI(title="LocalPro Chat Orchestrator", lifespan=lifespan)
    app.state.settings = settings
    app.state.db = db or Database(settings.database_path)
    app.state.lm_client = lm_client or LMStudioClient(
        settings.lm_studio_base_url, max_output_tokens=settings.oss_max_tokens
    )
    app.state.tavily_client = tavily_client or TavilyClient(settings.tavily_api_key)
    app.state.bus = EventBus(app.state.db)
    app.state.run_tasks = {}
    app.state.run_stop_events = {}
    app.state.run_control_queues = {}
    app.state.model_check = {}
    app.state.static_dir = Path(__file__).parent / "web" / "static"
    app.state.upload_dir = Path(settings.upload_dir).resolve()
    app.state.max_upload_bytes = settings.upload_max_mb * 1024 * 1024
    app.state.config_path = config_path or CONFIG_PATH

    app.mount("/static", StaticFiles(directory=app.state.static_dir), name="static")
    app.include_router(router)
    return app


app = create_app(load_settings())


if __name__ == "__main__":
    import os
    import uvicorn

    settings = app.state.settings
    reload_enabled = os.getenv("LOCALPRO_RELOAD", "").lower() in ("1", "true", "yes", "on")
    try:
        uvicorn.run(
            "app.main:app",
            host=getattr(settings, "host", "0.0.0.0"),
            port=settings.port,
            reload=reload_enabled,
        )
    except KeyboardInterrupt:
        pass
