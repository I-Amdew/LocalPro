import asyncio
import ast
import base64
import io
import json
import math
import operator
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import httpx
from pypdf import PdfReader

try:
    from PIL import Image
except Exception:
    Image = None

from . import agents
from .db import Database
from .llm import LMStudioClient, resolve_model_id
from .schemas import (
    Artifact,
    ControlCommand,
    PlanStep,
    RouterDecision,
    StepPlan,
    VerifierReport,
)
from .tavily import TavilyClient
from .system_info import compute_worker_slots, get_resource_snapshot


# Reasoning depth mapping
REASONING_DEPTHS = {
    "LOW": {"max_steps": 6, "research_rounds": 1, "tool_budget": {"tavily_search": 4, "tavily_extract": 6}},
    "MED": {"max_steps": 10, "research_rounds": 2, "tool_budget": {"tavily_search": 8, "tavily_extract": 10}},
    "HIGH": {"max_steps": 14, "research_rounds": 3, "tool_budget": {"tavily_search": 12, "tavily_extract": 16}, "advanced": True},
    "ULTRA": {"max_steps": 20, "research_rounds": 3, "tool_budget": {"tavily_search": 18, "tavily_extract": 24}, "advanced": True, "strict_verify": True},
}

# Cache models that LM Studio reports as unloaded or missing.
UNAVAILABLE_MODELS: Set[Tuple[str, str]] = set()


@dataclass
class RunState:
    can_chat: bool = True
    can_web: bool = False
    chat_error: Optional[str] = None
    web_error: Optional[str] = None
    freshness_required: bool = False
    work_log_flags: Set[str] = field(default_factory=set)
    dev_trace_cb: Optional[Callable[[str, Optional[dict]], None]] = None

    def mark_chat_unavailable(self, reason: str) -> None:
        self.can_chat = False
        self.chat_error = reason

    def add_dev_trace(self, message: str, detail: Optional[dict] = None) -> None:
        if self.dev_trace_cb:
            self.dev_trace_cb(message, detail)


async def emit_work_log(bus: "EventBus", run_id: str, text: str, tone: str = "info") -> None:
    await bus.emit(run_id, "work_log", {"text": text, "tone": tone})


async def maybe_emit_work_log(
    run_state: RunState,
    bus: "EventBus",
    run_id: str,
    key: str,
    text: str,
    tone: str = "info",
) -> None:
    if key in run_state.work_log_flags:
        return
    run_state.work_log_flags.add(key)
    await emit_work_log(bus, run_id, text, tone=tone)


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
) -> Optional[dict]:
    """Try to parse JSON, and fallback to the JSONRepair profile to fix."""
    try:
        return json.loads(raw)
    except Exception:
        pass
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


def safe_eval_expr(expr: str, names: Optional[Dict[str, Any]] = None) -> Any:
    """Evaluate a basic expression safely (no attribute access/imports)."""
    tree = ast.parse(expr, mode="eval")
    allowed_names = dict(SAFE_NAMES)
    if names:
        allowed_names.update(names)

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool, str)) or node.value is None:
                return node.value
            raise ValueError("Unsupported literal")
        if isinstance(node, ast.Tuple):
            return tuple(_eval(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [_eval(elt) for elt in node.elts]
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise ValueError("Dict unpacking not allowed")
            return {_eval(k): _eval(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BIN_OPS:
            return SAFE_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPS:
            return SAFE_UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in allowed_names:
                raise ValueError("Function not allowed")
            args = [_eval(arg) for arg in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise ValueError("Keyword splat not allowed")
                kwargs[kw.arg] = _eval(kw.value)
            return allowed_names[func_name](*args, **kwargs)
        if isinstance(node, ast.Name) and node.id in allowed_names:
            return allowed_names[node.id]
        raise ValueError("Disallowed expression")

    return _eval(tree)


TOOL_TEXT_MAX_CHARS = 4000
TOOL_BYTES_MAX = 200000
TOOL_LIST_MAX = 60
TOOL_IMAGE_MAX_SIZE = 1024
TOOL_IMAGE_MAX_LIMIT = 2048


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


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


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

    return {
        "read_text": read_text,
        "read_bytes": read_bytes,
        "list_files": list_files,
        "image_info": image_info,
        "image_load": image_load,
        "image_zoom": image_zoom,
        "image_crop": image_zoom,
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


def resolve_tool_requests(tool_requests: List[dict], upload_dir: Optional[Path] = None) -> List[dict]:
    """Resolve lightweight tool requests locally (date, calculator, code_eval, execute_code, image/pdf hints)."""
    resolved: List[dict] = []
    now_iso = datetime.utcnow().isoformat()
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
            elif tool in ("execute_code", "exec_code", "code_exec", "execute", "python_exec"):
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
    if any(ch.isdigit() for ch in q):
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


async def check_web_access(tavily: TavilyClient) -> Tuple[bool, Optional[str]]:
    """Return (can_web, error). Non-auth errors are reported but do not disable web."""
    if not tavily.enabled:
        return False, "missing_api_key"
    resp = await tavily.search(query="ping", search_depth="basic", max_results=1)
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
        style = "Compact but complete (<=350 words) with short sections and sourced bullets."
    if passes > 1:
        style += f" Note progress and clarify if another pass ({passes}) is in flight or expected."
    if reasoning_level in ("ULTRA", "HIGH"):
        style += " Keep sources prominent and state any remaining risks."
    return style


def desired_parallelism(
    decision: RouterDecision,
    worker_budget: Dict[str, Any],
    strict_mode: bool = False,
) -> int:
    """Choose how many worker slots to actively fill based on task depth."""
    max_parallel = int(worker_budget.get("max_parallel") or 1)
    if max_parallel <= 1:
        return 1
    if strict_mode or decision.expected_passes > 1:
        return max_parallel
    if decision.reasoning_level in ("HIGH", "ULTRA"):
        return max_parallel
    if decision.needs_web or decision.extract_depth == "advanced":
        return max_parallel
    if decision.reasoning_level == "MED":
        return min(max_parallel, 2)
    return 1


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
        "Executor": agents.EXECUTOR_SYSTEM,
        "JSONRepair": agents.JSON_REPAIR_SYSTEM,
        "Verifier": agents.VERIFIER_SYSTEM,
    }.get(profile, agents.RESEARCH_PRIMARY_SYSTEM)


def profile_model(profile: str, model_map: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    """Return (base_url, model_id) for a given profile."""
    if profile == "Orchestrator":
        cfg = model_map.get("orch")
    elif profile == "Executor":
        cfg = model_map.get("executor") or model_map.get("summarizer") or model_map.get("router")
    elif profile == "Writer":
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
    planner = base_map.get("orch") or base_map.get("worker") or {}
    router = base_map.get("router") or {}
    summarizer = base_map.get("summarizer") or {}
    executor = base_map.get("executor") or summarizer or router or base_map.get("deep_orch") or planner
    worker = base_map.get("worker") or planner
    worker_b = base_map.get("worker_b") or worker
    worker_c = base_map.get("worker_c") or worker
    verifier = base_map.get("verifier") or worker
    suite = {
        "orch": planner,
        "worker": worker,
        "worker_b": worker_b,
        "worker_c": worker_c,
        "router": router or executor,
        "summarizer": summarizer or executor,
        "verifier": verifier,
        "deep_planner": base_map.get("deep_planner") or planner,
        "deep_orch": base_map.get("deep_orch") or executor,
        "fast": base_map.get("fast") or worker,
    }
    allow_parallel = True
    if tier == "fast":
        execution_mode = "fast_team"
    elif tier == "deep":
        execution_mode = "oss_team" if deep_route == "oss" else "deep_cluster"
    else:
        execution_mode = "pro_full"
    return suite, planner, executor, allow_parallel, execution_mode


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
    research_steps: List[dict] = []
    parallel_research = prefer_parallel and worker_slots > 1
    if parallel_research:
        research_defs = [
            ("ResearchPrimary", "Research primary", "lane_primary"),
            ("ResearchRecency", "Research recency", "lane_recency"),
            ("ResearchAdversarial", "Research adversarial", "lane_adversarial"),
        ]
        max_research = min(worker_slots, len(research_defs))
        for profile, name, key in research_defs[:max_research]:
            research_steps.append(
                {
                    "step_id": next_step_id,
                    "name": name,
                    "type": "research",
                    "depends_on": [1],
                    "agent_profile": profile,
                    "inputs": {"use_web": decision.needs_web},
                    "outputs": [{"artifact_type": "evidence", "key": key}],
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
    run_state: Optional[RunState] = None,
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
            run_state=run_state,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, router_endpoint["model"], run_state=run_state)
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
    strict_mode: bool = False,
    run_state: Optional[RunState] = None,
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
            run_state=run_state,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, endpoint["model"], run_state=run_state)
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
    desired_parallel: int = 1,
    run_state: Optional[RunState] = None,
) -> StepPlan:
    plan_prompt = (
        "Produce a JSON step plan for answering the question. "
        "Include step_id, name, type, depends_on (list of ids), agent_profile, acceptance_criteria. "
        "Keep 6-12 steps for typical questions and use agent_profile 'Writer' for the draft step. "
        "Add global_constraints.expected_passes (1-3) if a verifier rerun is likely, and response_guidance describing how long the final answer should be based on task complexity. "
        "Include parallel research lanes when worker slots allow."
    )
    user_content = (
        f"Question: {question}\nNeeds web: {decision.needs_web}\nReasoning level: {decision.reasoning_level}\nExpected passes: {decision.expected_passes}\n"
        f"Available worker slots: {max(desired_parallel, 1)} (use them in parallel when useful)\n"
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
            run_state=run_state,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, plan_ep["model"], run_state=run_state)
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
                    "agent_profile": "Writer",
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
    run_id: Optional[str] = None,
    bus: Optional["EventBus"] = None,
    step_id: Optional[int] = None,
    context: str = "",
    run_state: Optional[RunState] = None,
) -> str:
    if run_state and not run_state.can_chat:
        raise RuntimeError("Local model unavailable.")
    system_prompt = profile_system(profile)
    last_error: Optional[Exception] = None
    for base_url, model in candidate_endpoints(profile, model_map):
        if run_state and not run_state.can_chat:
            break
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
                await bus.emit(
                    run_id,
                    "model_selected",
                    {
                        "profile": profile,
                        "model": model,
                        "base_url": base_url,
                        "step_id": step_id,
                        "context": context,
                    },
                )
            return resp["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if run_state and not run_state.can_chat:
                break
            detail = ""
            try:
                detail = exc.response.text or ""
            except Exception:
                detail = ""
            detail_lower = detail.lower()
            # Retry with fallback models if the current model is unavailable or unloaded.
            if exc.response is not None and exc.response.status_code in (400, 404):
                if (
                    "model unloaded" in detail_lower
                    or "model not found" in detail_lower
                    or "invalid model identifier" in detail_lower
                    or "valid downloaded model" in detail_lower
                ):
                    UNAVAILABLE_MODELS.add((base_url, model))
                    if run_id and bus:
                        await bus.emit(
                            run_id,
                            "model_unavailable",
                            {
                                "profile": profile,
                                "model": model,
                                "base_url": base_url,
                                "step_id": step_id,
                                "context": context,
                            },
                    )
                    continue
                # Avoid repeating the same rejected payload against other endpoints.
                break
            # For other status errors, try fallbacks but keep the last error.
            if run_id and bus:
                await bus.emit(
                    run_id,
                    "model_error",
                    {
                        "profile": profile,
                        "model": model,
                        "base_url": base_url,
                        "step_id": step_id,
                        "context": context,
                        "error": str(exc),
                    },
                )
            continue
        except httpx.RequestError as exc:
            last_error = exc
            if run_state and not run_state.can_chat:
                break
            if run_id and bus:
                await bus.emit(
                    run_id,
                    "model_error",
                    {
                        "profile": profile,
                        "model": model,
                        "base_url": base_url,
                        "step_id": step_id,
                        "context": context,
                        "error": str(exc),
                    },
                )
            continue
    if last_error:
        raise last_error
    raise RuntimeError("No available model endpoint for profile.")


async def generate_step_prompt(
    lm_client: LMStudioClient,
    orch_model: str,
    question: str,
    step: PlanStep,
    artifacts: List[Artifact],
    answer_guidance: str = "",
    toolbox_hint: str = "",
) -> str:
    def _strip_data_urls(item: Any) -> Any:
        if isinstance(item, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in item.items():
                if key == "data_url":
                    cleaned[key] = "<omitted>"
                elif isinstance(value, (dict, list)):
                    cleaned[key] = _strip_data_urls(value)
                else:
                    cleaned[key] = value
            return cleaned
        if isinstance(item, list):
            return [_strip_data_urls(v) for v in item]
        return item

    context_parts: List[str] = []
    for art in artifacts[-5:]:
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
                    slim["tool_results"] = _strip_data_urls(tool_results)
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
        prompt += "\nIf sources are present in artifacts, cite them in a Sources section with URLs."
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
    run_state: Optional[RunState] = None,
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
            run_state=run_state,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, orch_endpoint["model"], run_state=run_state)
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
    try:
        resp = await lm_client.chat_completion(
            model=fast_endpoint["model"],
            messages=[{"role": "system", "content": agents.EXECUTOR_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            base_url=fast_endpoint["base_url"],
            run_state=run_state,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, fast_endpoint["model"], run_state=run_state)
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
    target_slots: int,
    run_state: Optional[RunState] = None,
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
        "You are the executor allocator. Choose which ready steps to start now to keep worker slots busy."
        " Return JSON {\"start_ids\":[step_ids...]}. Prefer research steps in parallel; keep drafts/verify after research."
        f" Ready: {ready_desc}. Running now: {running_count}. Target slots: {target_slots}. Recent artifacts: {recent}."
    )
    try:
        resp = await lm_client.chat_completion(
            model=fast_endpoint["model"],
            messages=[{"role": "system", "content": agents.EXECUTOR_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            base_url=fast_endpoint["base_url"],
            run_state=run_state,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, fast_endpoint["model"], run_state=run_state)
        if parsed and isinstance(parsed.get("start_ids"), list):
            allowed = {s.step_id for s in ready_steps}
            filtered = [sid for sid in parsed["start_ids"] if sid in allowed]
            if filtered:
                return filtered
    except Exception:
        pass
    return [s.step_id for s in ready_steps]


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
        f"\nQuestion: {question}\nParallel slots: {target_slots}\nSteps: {json.dumps(steps_summary)[:1500]}"
    )
    try:
        resp = await lm_client.chat_completion(
            model=executor_endpoint["model"],
            messages=[{"role": "system", "content": agents.EXECUTOR_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=220,
            base_url=executor_endpoint["base_url"],
            run_state=run_state,
        )
        content = resp["choices"][0]["message"]["content"]
        parsed = await safe_json_parse(content, lm_client, executor_endpoint["model"], run_state=run_state)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def format_tavily_error(resp: Dict[str, Any]) -> str:
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
    parsed = await safe_json_parse(raw, lm_client, fixer_model, run_state=run_state)
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
) -> Tuple[List[dict], List[str]]:
    gathered_sources: List[dict] = []
    tavily_errors: List[str] = []
    executed_set = executed if executed is not None else set()
    for query in queries:
        q = str(query or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in executed_set:
            continue
        executed_set.add(key)
        if executed_order is not None:
            executed_order.append(q)
        payload = {"step": step.step_id, "query": q}
        if mode:
            payload["mode"] = mode
        await bus.emit(run_id, "tavily_search", payload)
        search_resp = await tavily.search(
            query=q,
            search_depth=search_depth,
            max_results=per_query_max,
            topic=topic,
            time_range=time_range,
        )
        await db.add_search(run_id, f"Step{step.step_id}", q, search_depth, per_query_max, search_resp)
        if search_resp.get("error"):
            error_msg = format_tavily_error(search_resp)
            await bus.emit(
                run_id,
                "tavily_error",
                {"step": step.step_id, "message": error_msg},
            )
            tavily_errors.append(error_msg)
            continue
        for res in search_resp.get("results", [])[:per_query_max]:
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
    return gathered_sources, tavily_errors


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
    upload_dir: Optional[Path] = None,
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
            "tool_requests": [],
            "tool_results": [],
            "timestamp_utc": datetime.utcnow().isoformat(),
        }
        artifact = Artifact(
            step_id=step.step_id,
            key=f"evidence_step_{step.step_id}",
            artifact_type="evidence",
            content_text="",
            content_json=evidence,
        )
        return evidence, [artifact], prompt
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
    parsed_raw = await safe_json_parse(raw, lm_client, fixer_model, run_state=run_state) if raw else None
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
            tavily_errors.append("Tavily API key missing")
        else:
            if not queries and question:
                queries = [question]
            primary_queries = queries[:5] if queries else []
            if primary_queries:
                sources, errors = await run_tavily_queries(
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
                )
                gathered_sources.extend(sources)
                tavily_errors.extend(errors)
            if not gathered_sources and question:
                fallback_query = question.strip()
                sources, errors = await run_tavily_queries(
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
                )
                gathered_sources.extend(sources)
                tavily_errors.extend(errors)
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
                    sources, errors = await run_tavily_queries(
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
                    )
                    gathered_sources.extend(sources)
                    tavily_errors.extend(errors)
            urls = [s["url"] for s in gathered_sources if s.get("url")]
            if urls:
                url_slice = urls[: max(3, min(extract_budget, len(urls)))]
                await bus.emit(run_id, "tavily_extract", {"step": step.step_id, "urls": url_slice})
                extract_resp = await tavily.extract(url_slice, extract_depth=extract_depth)
                await db.add_extract(run_id, f"Step{step.step_id}", ",".join(url_slice), extract_depth, extract_resp)
                if extract_resp.get("error"):
                    error_msg = format_tavily_error(extract_resp)
                    await bus.emit(
                        run_id,
                        "tavily_error",
                        {"step": step.step_id, "message": error_msg},
                    )
                    tavily_errors.append(error_msg)
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
            if gathered_sources and run_state:
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "read_sources",
                    "Reading sources to pull out key details.",
                )
    if tavily_errors and run_state:
        await maybe_emit_work_log(
            run_state,
            bus,
            run_id,
            "web_tool_error",
            "I couldn't reach the web search tool, so I'll proceed with what I have.",
            tone="warn",
        )
    if tool_requests:
        await bus.emit(run_id, "tool_request", {"step": step.step_id, "requests": tool_requests})
    tool_results = resolve_tool_requests(tool_requests, upload_dir=upload_dir)
    if tool_results:
        def _strip_data_urls(item: Any) -> Any:
            if isinstance(item, dict):
                cleaned: Dict[str, Any] = {}
                for key, value in item.items():
                    if key == "data_url":
                        cleaned[key] = "<omitted>"
                    elif isinstance(value, (dict, list)):
                        cleaned[key] = _strip_data_urls(value)
                    else:
                        cleaned[key] = value
                return cleaned
            if isinstance(item, list):
                return [_strip_data_urls(v) for v in item]
            return item

        await bus.emit(
            run_id,
            "tool_result",
            {"step": step.step_id, "results": _strip_data_urls(tool_results)},
        )
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
        "timestamp_utc": datetime.utcnow().isoformat(),
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
            "timestamp_utc": datetime.utcnow().isoformat(),
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
    if not tavily.enabled:
        await bus.emit(run_id, "tavily_error", {"step": step.step_id, "message": "Tavily API key missing"})
        tavily_errors.append("Tavily API key missing")
    else:
        primary_queries = queries[:5] if queries else []
        if primary_queries:
            sources, errors = await run_tavily_queries(
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
            )
            gathered_sources.extend(sources)
            tavily_errors.extend(errors)
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
                sources, errors = await run_tavily_queries(
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
                )
                gathered_sources.extend(sources)
                tavily_errors.extend(errors)
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
        "timestamp_utc": datetime.utcnow().isoformat(),
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
            "timestamp_utc": datetime.utcnow().isoformat(),
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
        tavily_errors.append("Tavily API key missing")
    if urls:
        await bus.emit(run_id, "tavily_extract", {"step": step.step_id, "urls": urls})
        extract_resp = await tavily.extract(urls, extract_depth=extract_depth)
        await db.add_extract(run_id, f"Step{step.step_id}", ",".join(urls), extract_depth, extract_resp)
        if extract_resp.get("error"):
            error_msg = format_tavily_error(extract_resp)
            await bus.emit(run_id, "tavily_error", {"step": step.step_id, "message": error_msg})
            tavily_errors.append(error_msg)
        for res in extract_resp.get("results", []) or []:
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
        "timestamp_utc": datetime.utcnow().isoformat(),
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
        if not draft_profile or draft_lower == "orchestrator":
            draft_profile = "Writer"
        elif draft_lower == "writer":
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
        verifier_profile = step.agent_profile or "Verifier"
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
        )
        verifier_model = (
            (model_map.get("verifier") or {}).get("model")
            or (model_map.get("worker") or {}).get("model")
            or (model_map.get("orch") or {}).get("model")
            or ""
        )
        parsed = await safe_json_parse(report, lm_client, verifier_model, run_state=run_state)
        if not parsed:
            parsed = {"issues": [], "verdict": "PASS", "extra_steps": []}
        if decision.reasoning_level in ("HIGH", "ULTRA") or decision.expected_passes > 1 or decision.needs_web:
            planner_model = (model_map.get("orch") or {}).get("model")
            planner_url = (model_map.get("orch") or {}).get("base_url")
            if planner_model and planner_url and planner_model != verifier_model:
                try:
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
                    planner_parsed = await safe_json_parse(planner_content, lm_client, planner_model, run_state=run_state)
                    if planner_parsed:
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
        )
        summary_text = summary_raw
        tool_requests = []
        tool_results = []
        fixer_model = (
            (model_map.get("summarizer") or {}).get("model")
            or (model_map.get("orch") or {}).get("model")
            or ""
        )
        parsed = await safe_json_parse(summary_raw, lm_client, fixer_model, run_state=run_state)
        if isinstance(parsed, dict):
            lines = parsed.get("activity_lines") or parsed.get("memory_notes") or parsed.get("criteria")
            if isinstance(lines, list):
                summary_text = " ".join(str(x) for x in lines if str(x).strip()).strip() or summary_raw
            elif isinstance(lines, str) and lines.strip():
                summary_text = lines.strip()
            tool_requests = parsed.get("tool_requests") or []
            if not isinstance(tool_requests, list):
                tool_requests = []
        if tool_requests:
            await bus.emit(run_id, "tool_request", {"step": step.step_id, "requests": tool_requests})
            tool_results = resolve_tool_requests(tool_requests, upload_dir=upload_dir)
            if tool_results:
                def _strip_data_urls(item: Any) -> Any:
                    if isinstance(item, dict):
                        cleaned: Dict[str, Any] = {}
                        for key, value in item.items():
                            if key == "data_url":
                                cleaned[key] = "<omitted>"
                            elif isinstance(value, (dict, list)):
                                cleaned[key] = _strip_data_urls(value)
                            else:
                                cleaned[key] = value
                        return cleaned
                    if isinstance(item, list):
                        return [_strip_data_urls(v) for v in item]
                    return item

                await bus.emit(
                    run_id,
                    "tool_result",
                    {"step": step.step_id, "results": _strip_data_urls(tool_results)},
                )
        artifact = Artifact(
            step_id=step.step_id,
            key="success_criteria",
            artifact_type="criteria",
            content_text=summary_text,
            content_json={"criteria": summary_text, "tool_requests": tool_requests, "tool_results": tool_results},
        )
        return {"criteria": summary_text}, [artifact], prompt
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
                    run_state=run_state,
                )
                content = resp["choices"][0]["message"]["content"]
                vision_json = (
                    await safe_json_parse(content, lm_client, vision_endpoint["model"], run_state=run_state)
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
                await safe_json_parse(sec_content, lm_client, secretary_endpoint["model"], run_state=run_state)
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
    conversation_id: str,
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
    model_availability: Optional[Dict[str, Any]] = None,
    upload_ids: Optional[List[int]] = None,
    upload_dir: Optional[Path] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Main orchestration loop for a single run (now with parallel step execution)."""
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
        run_state.freshness_required = needs_freshness(question)
        run_state.dev_trace_cb = make_dev_trace_cb(bus, run_id)
        await maybe_emit_work_log(run_state, bus, run_id, "goal", f"Goal: {question}")
        await maybe_emit_work_log(run_state, bus, run_id, "access_check", "Checking what I can access...")

        run_state.can_web, run_state.web_error = await check_web_access(tavily)
        if run_state.web_error and run_state.can_web:
            run_state.add_dev_trace("Web probe failed; continuing.", {"error": run_state.web_error})
            await maybe_emit_work_log(
                run_state,
                bus,
                run_id,
                "web_warning",
                f"Web search looks flaky ({run_state.web_error}); I'll still try to fetch sources.",
                tone="warn",
            )
        if not run_state.can_web:
            await maybe_emit_work_log(
                run_state,
                bus,
                run_id,
                "no_web",
                "I can't browse the web in this mode, so I'll answer from what's provided and flag assumptions.",
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

        base_check = settings_models.get("orch") or settings_models.get("router") or settings_models.get("summarizer") or {}
        check_url = base_check.get("base_url") or lm_client.base_url
        check_model = base_check.get("model") or ""
        can_chat, chat_detail = await lm_client.check_chat(base_url=check_url, model=check_model, run_state=run_state)
        if not can_chat:
            run_state.mark_chat_unavailable(chat_detail or "Local model unavailable")
            await maybe_emit_work_log(
                run_state,
                bus,
                run_id,
                "no_chat",
                "I can't reach the local model right now, so I'll stop and explain how to fix it.",
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

        base_router_endpoint = settings_models.get("router") or settings_models.get("summarizer") or settings_models["orch"]
        router_decision = await call_router(
            lm_client,
            base_router_endpoint,
            question,
            manual_level if decision_mode == "manual" else None,
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
        if run_state.can_web and (guess_needs_web(question) or run_state.freshness_required):
            router_decision.needs_web = True
        if not run_state.can_web:
            router_decision.needs_web = False
            router_decision.tool_budget = {**(router_decision.tool_budget or {}), "tavily_search": 0, "tavily_extract": 0}
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
        decision_payload = router_decision.model_dump()
        decision_payload.update(
            {
                "model_tier": model_tier,
                "requested_tier": requested_tier,
                "deep_route": deep_route_used,
                "execution_mode": execution_mode,
                "web_available": run_state.can_web,
                "freshness_required": run_state.freshness_required,
            }
        )
        try:
            resource_snapshot = get_resource_snapshot()
            worker_budget = compute_worker_slots(active_models, model_tier, model_availability, resource_snapshot)
            max_parallel_slots = worker_budget.get("max_parallel", 1)
        except Exception:
            resource_snapshot = {}
            worker_budget = {"max_parallel": 1, "configured": 1, "variants": 1, "ram_slots": 1, "vram_slots": 1}
            max_parallel_slots = 1
        desired_slots = desired_parallelism(router_decision, worker_budget, strict_mode=strict_mode)
        desired_slots = max(1, min(max_parallel_slots, desired_slots))
        worker_budget["desired_parallel"] = desired_slots
        if max_parallel_slots > 1 and not allow_parallel:
            allow_parallel = True
        if not allow_parallel:
            max_parallel_slots = 1
            desired_slots = 1
        target_parallel_slots = max(1, min(max_parallel_slots, desired_slots))
        decision_payload["resource_budget"] = worker_budget
        decision_payload["desired_parallel"] = target_parallel_slots
        team_roster = {
            "planner": (planner_endpoint or {}).get("model"),
            "executor": (executor_endpoint or {}).get("model"),
            "workers": [
                (active_models.get("worker") or {}).get("model"),
                (active_models.get("worker_b") or {}).get("model"),
                (active_models.get("worker_c") or {}).get("model"),
            ],
            "verifier": (active_models.get("verifier") or {}).get("model"),
        }
        decision_payload["team"] = team_roster
        await db.update_run_router(run_id, decision_payload)
        await bus.emit(run_id, "router_decision", decision_payload)
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
        await bus.emit(run_id, "team_roster", team_roster)
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
                        "I reviewed the uploaded material and noted key details.",
                    )

        if execution_mode in ("fast_team", "oss_team"):
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
        if run_state.can_web:
            step_plan = ensure_parallel_research(step_plan, target_parallel_slots, router_decision)
        else:
            step_plan = strip_research_steps(step_plan)
        for step in step_plan.steps:
            if step.type != "draft":
                continue
            profile = (step.agent_profile or "").strip()
            profile_lower = profile.lower()
            if not profile or profile_lower == "orchestrator":
                step.agent_profile = "Writer"
            elif profile_lower == "writer":
                step.agent_profile = "Writer"
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
        if not run_state.can_web:
            response_guidance += (
                " Web browsing is unavailable. Clearly label what is verified from provided materials vs assumptions, "
                "and invite the user to share links for verification."
            )
            if run_state.freshness_required:
                response_guidance += " The user asked for up-to-date verification; request sources or suggest a browsing-enabled lane."
            step_plan.global_constraints["response_guidance"] = response_guidance
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
        if run_state.can_web:
            plan_note = "Plan: Gather sources where needed, compare notes, then write a clear answer."
        else:
            plan_note = "Plan: Use what you provided and any local context, then write a best-effort answer and flag uncertainties."
        if upload_id_list:
            plan_note = "Plan: Review your uploads, then " + plan_note.split("Plan: ", 1)[-1]
        await maybe_emit_work_log(run_state, bus, run_id, "plan", plan_note)
        executor_brief = await build_executor_brief(
            lm_client, executor_endpoint, question, step_plan, target_parallel_slots, run_state=run_state
        )
        if executor_brief:
            await bus.emit(run_id, "executor_brief", executor_brief)
        # Build lookup for dependency-aware scheduling
        step_lookup: Dict[int, PlanStep] = {s.step_id: s for s in step_plan.steps}
        completed_steps: Set[int] = set()
        running_tasks: Dict[int, asyncio.Task] = {}
        max_loops = max(step_plan.global_constraints.get("max_loops", 1), progress_meta["counted_passes"] - 1)
        loops = 0
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

        async def start_step(step: PlanStep, snapshot: List[Artifact]) -> asyncio.Task:
            if step.type in ("research", "tavily_search", "search") and run_state.can_web:
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "search",
                    "Looking for sources to ground the answer.",
                )
            elif step.type in ("tavily_extract", "extract") and run_state.can_web:
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "read_sources",
                    "Reading sources to pull out key details.",
                )
            elif step.type == "merge":
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "compare",
                    "Comparing notes across sources to check for conflicts.",
                )
            elif step.type == "verify":
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "verify",
                    "Checking the draft for issues and consistency.",
                )
            elif step.type == "draft":
                await maybe_emit_work_log(
                    run_state,
                    bus,
                    run_id,
                    "draft",
                    "Writing the answer with clear caveats.",
                )
            await bus.emit(
                run_id,
                "step_started",
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "type": step.type,
                    "agent_profile": step.agent_profile,
                },
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
                        "control": await evaluate_control(lm_client, planner_endpoint, step, output, run_state=run_state),
                    }
                except Exception as exc:
                    await db.update_step_run(step_run_id, status="error", error_text=str(exc))
                    await bus.emit(
                        run_id,
                        "step_error",
                        {
                            "step": step.step_id,
                            "name": step.name,
                            "type": step.type,
                            "agent_profile": step.agent_profile,
                            "message": str(exc),
                        },
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
            ready_steps = [
                s for s in step_plan.steps if s.step_id not in completed_steps and s.step_id not in running_tasks and deps_satisfied(s)
            ]
            capacity = target_parallel_slots - len(running_tasks)
            if not allow_parallel:
                # Force one-at-a-time execution in linear modes.
                capacity = 1 - len(running_tasks)
            if ready_steps and capacity > 0:
                start_ids = (
                    await allocate_ready_steps(
                        lm_client,
                        fast_endpoint,
                        ready_steps,
                        artifacts,
                        len(running_tasks),
                        target_parallel_slots,
                        run_state=run_state,
                    )
                    if allow_parallel
                    else [ready_steps[0].step_id]
                )
                start_ids = start_ids[:capacity]
                if allow_parallel:
                    await bus.emit(
                        run_id,
                        "allocator_decision",
                        {
                            "start_ids": start_ids,
                            "ready_ids": [s.step_id for s in ready_steps],
                            "target_slots": target_parallel_slots,
                        },
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
                await bus.emit(
                    run_id,
                    "step_completed",
                    {
                        "step_id": step_id,
                        "name": result["step"].name,
                        "type": result["step"].type,
                        "agent_profile": result["step"].agent_profile,
                    },
                )

                fast_control, escalate = await evaluate_control_fast(
                    lm_client, fast_endpoint, result["step"], result["output"], run_state=run_state
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
                        lm_client, planner_endpoint, result["step"], result["output"], run_state=run_state
                    )
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
                )
            except Exception:
                final_answer = final_answer or "Unable to produce an answer with the available context."
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
                    lm_client,
                    "Summarizer",
                    active_models,
                    summary_prompt,
                    temperature=0.2,
                    max_tokens=180,
                    run_id=run_id,
                    bus=bus,
                    context="memory_summary",
                    run_state=run_state,
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
            )
            await bus.emit(run_id, "client_note", {"note": ui_note, "tier": model_tier, "route": deep_route_used})
        except Exception:
            pass
        await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": confidence})
    except asyncio.CancelledError:
        await db.update_run_status(run_id, "stopped")
        await bus.emit(run_id, "archived", {"run_id": run_id, "confidence": "LOW", "stopped": True})
        raise
    except Exception as exc:
        await db.update_run_status(run_id, f"error: {exc}")
        await bus.emit(run_id, "error", {"message": str(exc), "fatal": True})


def new_run_id() -> str:
    return str(uuid.uuid4())
