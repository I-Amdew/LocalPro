import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import AppSettings, CONFIG_PATH, load_settings, save_settings
from .db import Database
from .llm import LMStudioClient
from .orchestrator import EventBus, new_run_id, run_question
from .schemas import StartRunRequest
from .tavily import TavilyClient
from .system_info import compute_worker_slots, get_resource_snapshot


settings = load_settings()
db = Database(settings.database_path)
lm_client = LMStudioClient(settings.lm_studio_base_url, max_output_tokens=settings.oss_max_tokens)
tavily_client = TavilyClient(settings.tavily_api_key)
bus = EventBus(db)
RUN_TASKS: Dict[str, asyncio.Task] = {}
RUN_STOP_EVENTS: Dict[str, asyncio.Event] = {}

app = FastAPI(title="LocalPro Chat Orchestrator")
static_dir = Path(__file__).parent / "web" / "static"
upload_dir = Path(settings.upload_dir).resolve()
upload_dir.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = settings.upload_max_mb * 1024 * 1024
app.mount("/static", StaticFiles(directory=static_dir), name="static")

model_check: Dict[str, Any] = {}

_MODEL_SIZE_RE = re.compile(r"(\\d+(?:\\.\\d+)?b)", re.IGNORECASE)


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
    return available[0] if available else None


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


async def refresh_model_check(settings_obj: AppSettings) -> Dict[str, Any]:
    """Fetch available models per role and cache the results for routing fallbacks."""
    global model_check
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
        ok = cfg.get("model") in ids
        checks[role] = {"ok": ok, "missing": [] if ok else [cfg.get("model")], "available": ids}
    resources = get_resource_snapshot()
    checks["resources"] = resources
    try:
        active_map = build_model_map(settings_obj, checks)
        checks["worker_slots"] = compute_worker_slots(active_map, "pro", checks, resources)
    except Exception:
        pass
    model_check = checks
    return checks


@app.on_event("startup")
async def startup_event():
    await db.init()
    await db.save_config(settings.model_dump())
    await refresh_model_check(settings)


@app.on_event("shutdown")
async def shutdown_event():
    await lm_client.close()
    await tavily_client.close()


def sse_format(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.get("/settings")
async def get_settings():
    if not model_check:
        await refresh_model_check(settings)
    return {"settings": settings.to_safe_dict(), "model_check": model_check}


@app.post("/settings")
async def update_settings(request: Request):
    global settings, upload_dir, MAX_UPLOAD_BYTES
    body = await request.json()
    new_settings = AppSettings(**{**settings.model_dump(), **body})
    save_settings(new_settings)
    await db.save_config(new_settings.model_dump())
    settings = new_settings
    lm_client.base_url = settings.lm_studio_base_url
    lm_client.max_output_tokens = settings.oss_max_tokens
    tavily_client.api_key = settings.tavily_api_key
    upload_dir = Path(settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    MAX_UPLOAD_BYTES = settings.upload_max_mb * 1024 * 1024
    await refresh_model_check(settings)
    return {"ok": True, "model_check": model_check}


def validate_upload(file: UploadFile) -> None:
    allowed = {"image/png", "image/jpeg", "image/webp", "image/gif", "application/pdf"}
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required.")
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only images or PDFs are allowed.")


@app.post("/api/uploads")
async def upload_file(file: UploadFile = File(...), run_id: Optional[str] = Form(None)):
    validate_upload(file)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large (>{settings.upload_max_mb} MB).")
    safe_name = Path(file.filename).name
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    upload_path = upload_dir / stored_name
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(data)
    upload_id = await db.add_upload(run_id, stored_name, safe_name, file.content_type or "application/octet-stream", len(data), str(upload_path))
    if run_id:
        await bus.emit(run_id, "upload_received", {"upload_id": upload_id, "name": safe_name, "mime": file.content_type, "size": len(data)})
    return {"id": upload_id, "filename": safe_name, "mime": file.content_type, "size": len(data)}


@app.get("/api/uploads/{upload_id}")
async def get_upload(upload_id: int):
    record = await db.get_upload(upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")
    path = Path(record["storage_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(path, media_type=record["mime"], filename=record["original_name"])


@app.post("/api/discover")
async def discover_models(payload: Dict[str, Any]):
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


@app.post("/api/run")
async def start_run(payload: StartRunRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
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
            reasoning_mode=payload.reasoning_mode,
            manual_level=payload.manual_level,
            deep_mode=payload.deep_mode,
        )
        conversation_id = conversation["id"]
    else:
        await db.update_conversation(
            conversation_id,
            model_tier=payload.model_tier,
            reasoning_mode=payload.reasoning_mode,
            manual_level=payload.manual_level,
            deep_mode=payload.deep_mode,
        )
        title_seed = question if len(question) <= 80 else question[:77].rstrip() + "..."
        await db.ensure_conversation_title(conversation_id, title_seed)
    run_id = new_run_id()
    bus.register_run(run_id, conversation_id)
    prompt_state = await db.set_prompt_state(question, run_id)
    await bus.emit("conversation", "prompt_updated", prompt_state)
    models = build_model_map(settings, model_check)
    upload_ids = payload.upload_ids or []
    for uid in upload_ids:
        await db.assign_upload_to_run(uid, run_id)
        await db.update_upload_status(uid, "queued")
    stop_event = asyncio.Event()
    RUN_STOP_EVENTS[run_id] = stop_event

    async def run_and_cleanup() -> None:
        try:
            await run_question(
                run_id=run_id,
                conversation_id=conversation_id,
                question=payload.question,
                decision_mode=payload.reasoning_mode,
                manual_level=payload.manual_level,
                model_tier=payload.model_tier,
                deep_mode=payload.deep_mode,
                search_depth_mode=payload.search_depth_mode,
                max_results_override=payload.max_results or 0,
                strict_mode=payload.strict_mode,
                auto_memory=payload.auto_memory,
                db=db,
                bus=bus,
                lm_client=lm_client,
                tavily=tavily_client,
                settings_models=models,
                model_availability=model_check,
                upload_ids=upload_ids,
                upload_dir=upload_dir,
                stop_event=stop_event,
            )
        finally:
            RUN_TASKS.pop(run_id, None)
            RUN_STOP_EVENTS.pop(run_id, None)

    task = asyncio.create_task(run_and_cleanup())
    RUN_TASKS[run_id] = task
    return {"run_id": run_id, "conversation_id": conversation_id}


@app.get("/api/run/{run_id}")
async def get_run(run_id: str):
    run = await db.get_run_summary(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/run/latest")
async def get_latest_run(conversation_id: Optional[str] = None):
    reset_at = await db.get_conversation_reset()
    convo_id = conversation_id or await db.get_default_conversation_id()
    run = await db.get_latest_run(after=reset_at, conversation_id=convo_id)
    return {"run": run, "reset_at": reset_at, "conversation_id": convo_id}


@app.post("/api/run/{run_id}/stop")
async def stop_run(run_id: str):
    run = await db.get_run_summary(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    stop_event = RUN_STOP_EVENTS.get(run_id)
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


@app.get("/api/run/{run_id}/artifacts")
async def get_artifacts(run_id: str):
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


@app.get("/api/memory")
async def list_memory(q: Optional[str] = None):
    if q:
        items = await db.search_memory(q, limit=50)
    else:
        items = await db.list_memory(limit=50)
    return {"items": items}


@app.post("/api/memory")
async def create_memory(item: Dict[str, Any]):
    mem_id = await db.add_memory_item(
        item.get("kind", "note"),
        item.get("title", ""),
        item.get("content", ""),
        item.get("tags", []),
        pinned=item.get("pinned", False),
        relevance_score=item.get("relevance_score", 0.0),
    )
    return {"id": mem_id}


@app.patch("/api/memory/{item_id}")
async def update_memory(item_id: int, item: Dict[str, Any]):
    await db.update_memory_item(
        item_id, title=item.get("title"), content=item.get("content"), pinned=item.get("pinned")
    )
    return {"ok": True}


@app.delete("/api/memory/{item_id}")
async def delete_memory(item_id: int):
    await db.delete_memory_item(item_id)
    return {"ok": True}

@app.get("/api/conversations")
async def list_conversations():
    conversations = await db.list_conversations()
    return {"conversations": conversations}


@app.post("/api/conversations")
async def create_conversation(payload: Dict[str, Any] = Body(default={})):
    convo = await db.create_conversation(
        title=payload.get("title"),
        model_tier=payload.get("model_tier", "pro"),
        reasoning_mode=payload.get("reasoning_mode", "auto"),
        manual_level=payload.get("manual_level", "MED"),
        deep_mode=payload.get("deep_mode", "auto"),
    )
    await bus.emit("conversation", "conversation_created", {"conversation_id": convo["id"], "conversation": convo})
    return {"conversation": convo}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": convo}


@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, limit: int = 200):
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await db.list_messages(conversation_id=conversation_id, limit=limit)
    return {"messages": messages}


@app.patch("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, payload: Dict[str, Any] = Body(default={})):
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


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete_conversation(conversation_id)
    await db.execute("UPDATE conversation_state SET default_conversation_id=NULL WHERE id=1")
    await db.get_default_conversation_id()
    await bus.emit("conversation", "conversation_deleted", {"conversation_id": conversation_id})
    return {"ok": True}


@app.get("/api/conversation")
async def conversation_history(limit: int = 200, conversation_id: Optional[str] = None):
    convo_id = conversation_id or await db.get_default_conversation_id()
    messages = await db.list_messages(conversation_id=convo_id, limit=limit)
    reset_at = await db.get_conversation_reset()
    return {"messages": messages, "reset_at": reset_at, "conversation_id": convo_id}


@app.delete("/api/conversation")
async def reset_conversation(conversation_id: Optional[str] = None):
    reset_at = await db.reset_conversation(conversation_id=conversation_id)
    convo_id = conversation_id or await db.get_default_conversation_id()
    await bus.emit("conversation", "conversation_reset", {"reset_at": reset_at, "conversation_id": convo_id})
    return {"ok": True, "reset_at": reset_at, "conversation_id": convo_id}


@app.get("/api/prompt")
async def get_prompt():
    prompt = await db.get_prompt_state()
    return {"prompt": prompt}


@app.delete("/api/prompt")
async def clear_prompt():
    updated_at = await db.clear_prompt_state()
    await bus.emit("conversation", "prompt_cleared", {"updated_at": updated_at})
    return {"ok": True, "updated_at": updated_at}


@app.get("/events")
async def stream_global_events():
    async def event_generator():
        queue = await bus.subscribe_global()
        try:
            while True:
                ev = await queue.get()
                yield sse_format(ev)
        finally:
            await bus.unsubscribe_global(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str):
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
        finally:
            await bus.unsubscribe(run_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import os
    import uvicorn

    reload_enabled = os.getenv("LOCALPRO_RELOAD", "").lower() in ("1", "true", "yes", "on")
    uvicorn.run(
        "app.main:app",
        host=getattr(settings, "host", "0.0.0.0"),
        port=settings.port,
        reload=reload_enabled,
    )
