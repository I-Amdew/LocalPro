import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
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
lm_client = LMStudioClient(settings.lm_studio_base_url)
tavily_client = TavilyClient(settings.tavily_api_key)
bus = EventBus(db)

app = FastAPI(title="LocalPro Chat Orchestrator")
static_dir = Path(__file__).parent / "web" / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

model_check: Dict[str, Any] = {}


def build_model_map(settings_obj: AppSettings) -> Dict[str, Dict[str, str]]:
    return {
        "orch": {"base_url": settings_obj.orch_endpoint.base_url, "model": settings_obj.orch_endpoint.model_id},
        "worker": {"base_url": settings_obj.worker_a_endpoint.base_url, "model": settings_obj.worker_a_endpoint.model_id},
        "worker_b": {"base_url": settings_obj.worker_b_endpoint.base_url, "model": settings_obj.worker_b_endpoint.model_id},
        "worker_c": {"base_url": settings_obj.worker_c_endpoint.base_url, "model": settings_obj.worker_c_endpoint.model_id},
        "router": {"base_url": settings_obj.router_endpoint.base_url, "model": settings_obj.router_endpoint.model_id},
        "summarizer": {"base_url": settings_obj.summarizer_endpoint.base_url, "model": settings_obj.summarizer_endpoint.model_id},
        "verifier": {"base_url": settings_obj.verifier_endpoint.base_url, "model": settings_obj.verifier_endpoint.model_id},
    }


@app.on_event("startup")
async def startup_event():
    await db.init()
    await db.save_config(settings.model_dump())
    global model_check
    checks = {}
    model_map = build_model_map(settings)
    for role, cfg in model_map.items():
        try:
            resp = await lm_client.list_models(cfg["base_url"])
            ids = [m.get("id") for m in resp.get("data", [])]
            ok = cfg["model"] in ids
            checks[role] = {"ok": ok, "missing": [] if ok else [cfg["model"]], "available": ids}
        except Exception as exc:
            checks[role] = {"ok": False, "error": str(exc)}
    model_check = checks


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
    return {"settings": settings.to_safe_dict(), "model_check": model_check}


@app.post("/settings")
async def update_settings(request: Request):
    body = await request.json()
    new_settings = AppSettings(**{**settings.model_dump(), **body})
    save_settings(new_settings)
    await db.save_config(new_settings.model_dump())
    global settings
    settings = new_settings
    lm_client.base_url = settings.lm_studio_base_url
    tavily_client.api_key = settings.tavily_api_key
    return {"ok": True}


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
    models = build_model_map(settings)
    asyncio.create_task(
        run_question(
            run_id=run_id,
            question=payload.question,
            decision_mode=payload.reasoning_mode,
            manual_level=payload.manual_level,
            search_depth_mode=payload.search_depth_mode,
            max_results_override=payload.max_results or 0,
            strict_mode=payload.strict_mode,
            auto_memory=payload.auto_memory,
            db=db,
            bus=bus,
            lm_client=lm_client,
            tavily=tavily_client,
            settings_models=models,
        )
    )
    return {"run_id": run_id}


@app.get("/api/run/{run_id}")
async def get_run(run_id: str):
    run = await db.get_run_summary(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


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
    return {"run": run, "sources": sources, "claims": claims, "draft": draft, "verifier": verifier, "artifacts": artifacts}


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

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
