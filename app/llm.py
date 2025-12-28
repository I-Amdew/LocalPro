import json
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
DISALLOWED_FIELDS = {
    "tools",
    "tool_choice",
    "response_format",
    "reasoning",
    "seed",
    "logprobs",
    "top_logprobs",
    "parallel_tool_calls",
    "json_schema",
    "modalities",
    "audio",
}
_MODEL_SIZE_RE = re.compile(r"(\\d+(?:\\.\\d+)?b)", re.IGNORECASE)


def _normalize_model_id(value: str) -> str:
    base = value.split(":")[0].strip()
    if "/" in base:
        base = base.rsplit("/", 1)[-1]
    return base.lower()


def _model_family(value: str) -> str:
    base = value.split(":")[0].strip()
    if "/" in base:
        base = base.rsplit("/", 1)[-1]
    base = _MODEL_SIZE_RE.sub("", base).strip("-_ ")
    return base.lower()


def resolve_model_id(preferred: Optional[str], available: List[str]) -> Optional[str]:
    if not preferred or not available:
        return None
    if preferred in available:
        return preferred
    base = preferred.split(":")[0]
    if base in available:
        return base
    target = _normalize_model_id(preferred)
    for mid in available:
        if _normalize_model_id(mid) == target:
            return mid
    size_match = _MODEL_SIZE_RE.search(preferred)
    if size_match:
        size_hint = size_match.group(1).lower()
        for mid in available:
            if size_hint in mid.lower():
                return mid
    return None


def _run_state_can_chat(run_state: Optional[Any]) -> bool:
    if run_state is None:
        return True
    return bool(getattr(run_state, "can_chat", True))


def _run_state_mark_unavailable(run_state: Optional[Any], reason: str) -> None:
    if run_state is None:
        return
    marker = getattr(run_state, "mark_chat_unavailable", None)
    if callable(marker):
        marker(reason)
        return
    setattr(run_state, "can_chat", False)
    setattr(run_state, "chat_error", reason)


def _run_state_add_trace(run_state: Optional[Any], message: str, detail: Optional[dict] = None) -> None:
    if run_state is None:
        return
    tracer = getattr(run_state, "add_dev_trace", None)
    if callable(tracer):
        tracer(message, detail)


class LMStudioClient:
    def __init__(self, base_url: str, max_output_tokens: Optional[int] = None):
        self.base_url = base_url.rstrip("/")
        self.max_output_tokens = max_output_tokens
        self.client = httpx.AsyncClient(timeout=60)
        self.model_cache: Dict[str, Dict[str, Any]] = {}
        self.model_cache_ttl = 60.0
        self.unavailable_models: Dict[Tuple[str, str], float] = {}
        self.unavailable_ttl = self.model_cache_ttl

    async def list_models(self, base_url: Optional[str] = None) -> Dict[str, Any]:
        url = f"{(base_url or self.base_url).rstrip('/')}/models"
        resp = await self.client.get(url)
        resp.raise_for_status()
        return resp.json()

    async def list_models_cached(self, base_url: Optional[str] = None, force: bool = False) -> List[str]:
        url = (base_url or self.base_url).rstrip("/")
        now = time.monotonic()
        cached = self.model_cache.get(url)
        if cached and not force and now - cached["ts"] < self.model_cache_ttl:
            return cached["ids"]
        resp = await self.list_models(url)
        ids = [m.get("id") for m in resp.get("data", []) if m.get("id")]
        self.model_cache[url] = {"ts": now, "ids": ids}
        return ids

    def mark_model_unavailable(self, base_url: str, model: str) -> None:
        if not base_url or not model:
            return
        key = (base_url.rstrip("/"), model)
        self.unavailable_models[key] = time.monotonic()

    def _prune_unavailable(self) -> None:
        if not self.unavailable_models:
            return
        now = time.monotonic()
        ttl = self.unavailable_ttl
        if ttl <= 0:
            self.unavailable_models.clear()
            return
        expired = [key for key, ts in self.unavailable_models.items() if now - ts > ttl]
        for key in expired:
            self.unavailable_models.pop(key, None)

    def _is_model_unavailable(self, base_url: str, model: str) -> bool:
        if not base_url or not model:
            return False
        self._prune_unavailable()
        return (base_url.rstrip("/"), model) in self.unavailable_models

    def _normalize_error_text(self, detail: str) -> str:
        text = detail or ""
        for _ in range(2):
            try:
                parsed = json.loads(text)
            except Exception:
                break
            if isinstance(parsed, dict):
                found = False
                for key in ("error", "detail", "message"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        text = val
                        found = True
                        break
                if not found:
                    break
            elif isinstance(parsed, str):
                text = parsed
            else:
                break
        return text

    def _is_model_unavailable_error(self, detail: str) -> bool:
        text = self._normalize_error_text(detail).lower()
        return any(
            token in text
            for token in (
                "failed to load model",
                "operation canceled",
                "out of memory",
                "insufficient memory",
                "model is unloaded",
                "model is not loaded",
                "model unloaded",
                "model not found",
                "invalid model identifier",
                "valid downloaded model",
                "no such model",
            )
        )

    def _select_fallback_model(self, base_url: str, preferred: str, available: List[str]) -> Optional[str]:
        base = base_url.rstrip("/")
        self._prune_unavailable()
        candidates = [
            mid
            for mid in available
            if mid and "embed" not in mid.lower() and (base, mid) not in self.unavailable_models
        ]
        if not candidates:
            return None
        if preferred and preferred in candidates:
            return preferred
        family = _model_family(preferred) if preferred else ""
        if family:
            for mid in candidates:
                if family and family in _model_family(mid):
                    return mid
        return candidates[0]

    def _resolve_available_model(
        self,
        base_url: str,
        preferred: str,
        available: List[str],
    ) -> str:
        if not self._is_model_unavailable(base_url, preferred):
            return preferred
        fallback = self._select_fallback_model(base_url, preferred, available)
        if fallback and fallback != preferred:
            return fallback
        return preferred

    async def resolve_model_id(self, preferred: str, base_url: Optional[str] = None) -> Optional[str]:
        available = await self.list_models_cached(base_url)
        return resolve_model_id(preferred, available)

    def _sanitize_messages(self, messages: Any) -> List[Dict[str, Any]]:
        if not isinstance(messages, list):
            return []
        sanitized: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ALLOWED_ROLES:
                continue
            content = msg.get("content")
            if content is None:
                continue
            cleaned_content: Any
            if isinstance(content, str):
                if not content.strip():
                    continue
                cleaned_content = content
            elif isinstance(content, list):
                cleaned_items = [
                    item
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type")
                    and (item.get("text") or item.get("image_url"))
                ]
                if not cleaned_items:
                    continue
                cleaned_content = cleaned_items
            else:
                cleaned_text = json.dumps(content, ensure_ascii=True)
                if not cleaned_text.strip():
                    continue
                cleaned_content = cleaned_text
            sanitized.append({"role": role, "content": cleaned_content})
        return sanitized

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {k: v for k, v in payload.items() if k not in DISALLOWED_FIELDS}
        return cleaned

    def _extract_error_detail(self, response: httpx.Response) -> str:
        try:
            data = response.json()
            if isinstance(data, dict):
                return json.dumps(data, ensure_ascii=True)
        except Exception:
            pass
        try:
            return response.text
        except Exception:
            return ""

    async def _prepare_payload(
        self,
        payload: Dict[str, Any],
        base_url: str,
        run_state: Optional[Any],
    ) -> Dict[str, Any]:
        cleaned = self._sanitize_payload(payload)
        cleaned["messages"] = self._sanitize_messages(cleaned.get("messages"))
        if not cleaned.get("messages"):
            _run_state_add_trace(run_state, "Invalid payload: empty messages.")
            raise ValueError("messages must include at least one non-empty entry")
        model = str(cleaned.get("model") or "").strip()
        if not model:
            _run_state_add_trace(run_state, "Invalid payload: missing model.")
            raise ValueError("model is required")
        try:
            available = await self.list_models_cached(base_url)
        except Exception as exc:
            _run_state_add_trace(run_state, "Model list lookup failed", {"error": str(exc)})
            raise
        available = [m for m in available if m and "embed" not in m.lower()]
        resolved = resolve_model_id(model, available)
        if not resolved:
            _run_state_add_trace(run_state, "Model not found in /v1/models.", {"model": model})
            raise ValueError("model not found in /v1/models")
        cleaned["model"] = self._resolve_available_model(base_url, resolved, available)
        return cleaned

    async def _minimal_chat_check(
        self,
        base_url: str,
        preferred_model: str,
        run_state: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        try:
            available = await self.list_models_cached(base_url)
        except Exception as exc:
            return False, str(exc)
        available = [m for m in available if m and "embed" not in m.lower()]
        ping_model = resolve_model_id(preferred_model, available) or (available[0] if available else "")
        if not ping_model:
            return False, "no models available"
        return True, ping_model

    async def check_chat(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        run_state: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        target_base = (base_url or self.base_url).rstrip("/")
        preferred_model = model or ""
        ok, ping_detail = await self._minimal_chat_check(target_base, preferred_model, run_state=run_state)
        if not ok:
            return False, ping_detail
        ping_model = ping_detail
        url = f"{target_base}/chat/completions"
        payload = {
            "model": ping_model,
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0.2,
            "max_tokens": 1,
            "stream": False,
        }
        payload = self._sanitize_payload(payload)
        try:
            resp = await self.client.post(url, json=payload, timeout=10.0)
            resp.raise_for_status()
            return True, ""
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            _run_state_add_trace(run_state, "Minimal payload failed", {"detail": detail})
            if self._is_model_unavailable_error(detail):
                self.mark_model_unavailable(target_base, ping_model)
            return False, detail
        except httpx.RequestError as exc:
            return False, str(exc)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
        response_format: Optional[dict] = None,
        base_url: Optional[str] = None,
        run_state: Optional[Any] = None,
        allow_minimal_retry: bool = True,
    ) -> Any:
        if not _run_state_can_chat(run_state):
            raise RuntimeError("Local model unavailable.")
        final_max_tokens = max_tokens
        if self.max_output_tokens:
            final_max_tokens = min(max_tokens, self.max_output_tokens)
        target_base = (base_url or self.base_url).rstrip("/")
        url = f"{target_base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": final_max_tokens,
            "stream": stream,
        }
        if response_format:
            payload["response_format"] = response_format
        try:
            payload = await self._prepare_payload(payload, target_base, run_state)
        except Exception:
            raise
        attempts = 0
        max_fallbacks = 10
        fallback_note: Optional[Dict[str, str]] = None
        while True:
            try:
                resp = await self.client.post(url, json=payload)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                detail = self._extract_error_detail(exc.response) if exc.response is not None else ""
                if status in (400, 404) and self._is_model_unavailable_error(detail):
                    self.mark_model_unavailable(target_base, str(payload.get("model") or "").strip())
                    if allow_minimal_retry and attempts < max_fallbacks:
                        retry_payload = await self._prepare_payload(payload, target_base, run_state)
                        if retry_payload.get("model") and retry_payload.get("model") != payload.get("model"):
                            fallback_note = {"from": str(payload.get("model")), "to": str(retry_payload.get("model"))}
                            payload = retry_payload
                            attempts += 1
                            continue
                if status == 400 and allow_minimal_retry:
                    _run_state_add_trace(run_state, "LM Studio rejected request (400).", {"detail": detail})
                    ok, min_detail = await self._minimal_chat_check(
                        target_base, payload.get("model", ""), run_state=run_state
                    )
                    if not ok:
                        _run_state_mark_unavailable(run_state, min_detail or "minimal payload failed")
                raise
        if fallback_note:
            _run_state_add_trace(run_state, "Model unavailable; using fallback.", fallback_note)
        if stream:
            return resp
        data = resp.json()
        if isinstance(data, dict) and payload.get("model"):
            data["_model_used"] = payload.get("model")
        try:
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if content is None or content == "":
                    fallback = message.get("reasoning") or message.get("reasoning_content")
                    if fallback:
                        message["content"] = fallback
                        choices[0]["message"] = message
        except Exception:
            pass
        return data

    async def stream_text(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
        run_state: Optional[Any] = None,
    ) -> AsyncGenerator[str, None]:
        response = await self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            base_url=base_url,
            run_state=run_state,
        )
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            chunk = line.replace("data:", "").strip()
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
                delta_obj = data.get("choices", [{}])[0].get("delta", {})
                delta = delta_obj.get("content") or delta_obj.get("reasoning") or delta_obj.get("reasoning_content")
                if delta:
                    yield delta
            except Exception:
                continue

    async def close(self) -> None:
        await self.client.aclose()
