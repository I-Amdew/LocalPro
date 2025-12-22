import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import AppSettings, CONFIG_PATH, load_settings, save_settings
from .db import Database
from .llm import LMStudioClient
from .orchestrator import EventBus, new_run_id, run_question
from .schemas import StartRunRequest
from .tavily import TavilyClient


settings = load_settings()
db = Database(settings.database_path)
lm_client = LMStudioClient(settings.lm_studio_base_url, max_output_tokens=settings.oss_max_tokens)
tavily_client = TavilyClient(settings.tavily_api_key)
bus = EventBus(db)

app = FastAPI(title="LocalPro Chat Orchestrator")
static_dir = Path(__file__).parent / "web" / "static"
upload_dir = Path(settings.upload_dir).resolve()
upload_dir.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = settings.upload_max_mb * 1024 * 1024
app.mount("/static", StaticFiles(directory=static_dir), name="static")

model_check: Dict[str, Any] = {}


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
        available_ids = set()
        for info in availability.values():
            for mid in info.get("available", []):
                if mid:
                    available_ids.add(mid)

        def fallback(role: str, target: str) -> None:
            info = availability.get(role)
            if info is None:
                return
            cfg = model_map.get(role)
            target_cfg = model_map.get(target)
            if not cfg or not target_cfg:
                return
            configured_id = cfg.get("model")
            missing = info.get("ok") is False or (configured_id and configured_id not in available_ids)
            if missing:
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
    for role, cfg in raw_map.items():
        try:
            resp = await lm_client.list_models(cfg["base_url"])
            ids = [m.get("id") for m in resp.get("data", [])]
            ok = cfg["model"] in ids
            checks[role] = {"ok": ok, "missing": [] if ok else [cfg["model"]], "available": ids}
        except Exception as exc:
            checks[role] = {"ok": False, "error": str(exc)}
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
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    run_id = new_run_id()
    models = build_model_map(settings, model_check)
    upload_ids = payload.upload_ids or []
    for uid in upload_ids:
        await db.assign_upload_to_run(uid, run_id)
        await db.update_upload_status(uid, "queued")
    asyncio.create_task(
        run_question(
            run_id=run_id,
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
            upload_ids=upload_ids,
        )
    )
    return {"run_id": run_id}


@app.get("/api/run/{run_id}")
async def get_run(run_id: str):
    run = await db.get_run_summary(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/run/latest")
async def get_latest_run():
    reset_at = await db.get_conversation_reset()
    run = await db.get_latest_run(after=reset_at)
    return {"run": run, "reset_at": reset_at}


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


@app.get("/api/conversation")
async def conversation_history(limit: int = 200):
    messages = await db.list_messages(limit=limit)
    reset_at = await db.get_conversation_reset()
    return {"messages": messages, "reset_at": reset_at}


@app.delete("/api/conversation")
async def reset_conversation():
    reset_at = await db.reset_conversation()
    return {"ok": True, "reset_at": reset_at}


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
    import uvicorn

    uvicorn.run("app.main:app", host=getattr(settings, "host", "0.0.0.0"), port=settings.port, reload=True)
