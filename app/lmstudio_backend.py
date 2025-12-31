import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from .model_manager import ModelCandidate, ModelInstanceInfo, ModelBackend, ResourceEstimate


_DEFAULT_TIMEOUT = 60.0
_CLI_TIMEOUT_S = 8.0
_CLI_ESTIMATE_TIMEOUT_S = 15.0
_CLI_LOAD_TIMEOUT_S = 120.0
_CLI_UNLOAD_TIMEOUT_S = 30.0
_CLI_SERVER_TIMEOUT_S = 30.0
_VRAM_RE = re.compile(r"(\\d+(?:\\.\\d+)?)\\s*mb", re.IGNORECASE)
_INSTANCE_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def _to_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _parse_json_blob(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _looks_like_instance_id(model_key: str) -> bool:
    if not model_key:
        return False
    if model_key.startswith("profile-"):
        return True
    return bool(_INSTANCE_SUFFIX_RE.search(model_key))


class LMStudioBackend(ModelBackend):
    id = "lmstudio"

    def __init__(
        self,
        *,
        base_url: str,
        host: str = "127.0.0.1",
        port: int = 1234,
        use_cli: bool = True,
        cli_path: Optional[str] = None,
        default_ttl_s: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.host = host
        self.port = port
        self.use_cli = use_cli
        self.cli_path = cli_path or "lms"
        self.default_ttl_s = default_ttl_s
        self.client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
        self._responses_supported = True
        self._server_check_at = 0.0
        self._server_check_ok = False

    def _cli_exe(self) -> Optional[str]:
        if not self.use_cli:
            return None
        if self.cli_path:
            return self.cli_path
        return "lms"

    async def _run_cli(self, args: List[str], timeout_s: Optional[float] = None) -> str:
        exe = self._cli_exe()
        if not exe:
            raise RuntimeError("LM Studio CLI disabled")

        def _runner() -> str:
            return subprocess.check_output(
                [exe, *args],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=timeout_s or _CLI_TIMEOUT_S,
                encoding="utf-8",
                errors="replace",
            )

        return await asyncio.to_thread(_runner)

    async def ensure_server_running(self) -> None:
        url = f"{self.base_url}/models"
        try:
            resp = await self.client.get(url, timeout=3.0)
            if resp.status_code < 400:
                self._server_check_ok = True
                self._server_check_at = time.monotonic()
                return
        except Exception:
            pass
        if not self._cli_exe():
            return
        await self._run_cli(["server", "start", "--port", str(self.port)], timeout_s=_CLI_SERVER_TIMEOUT_S)

    async def is_server_reachable(self) -> bool:
        now = time.monotonic()
        if (now - self._server_check_at) < 2.0:
            return self._server_check_ok
        try:
            resp = await self.client.get(f"{self.base_url}/models", timeout=2.0)
            ok = resp.status_code < 400
        except Exception:
            ok = False
        self._server_check_ok = ok
        self._server_check_at = now
        return ok

    async def discover(self) -> List[ModelCandidate]:
        if not await self.is_server_reachable():
            return []
        candidates: Dict[str, ModelCandidate] = {}
        if self._cli_exe():
            try:
                raw = await self._run_cli(["ls", "--json"])
                payload = _parse_json_blob(raw)
                items = []
                if isinstance(payload, dict):
                    items = payload.get("models") or payload.get("data") or []
                elif isinstance(payload, list):
                    items = payload
                for item in items or []:
                    if not isinstance(item, dict):
                        continue
                    model_key = item.get("id") or item.get("modelKey") or item.get("model_key")
                    if not model_key:
                        continue
                    display = item.get("name") or item.get("displayName") or model_key
                    candidates[model_key] = ModelCandidate(
                        backend_id=self.id,
                        model_key=model_key,
                        display_name=str(display),
                        metadata=item,
                    )
            except Exception:
                candidates = {}
        try:
            resp = await self.client.get(f"{self.base_url}/models", timeout=5.0)
            if resp.status_code < 400:
                data = resp.json()
                allow_add = not candidates
                for item in data.get("data", []):
                    if not isinstance(item, dict):
                        continue
                    model_key = item.get("id")
                    if not model_key or _looks_like_instance_id(model_key):
                        continue
                    display = item.get("name") or model_key
                    if model_key in candidates:
                        candidates[model_key].metadata.setdefault("loaded", True)
                    elif allow_add:
                        candidates[model_key] = ModelCandidate(
                            backend_id=self.id,
                            model_key=model_key,
                            display_name=str(display),
                            metadata={"loaded": True},
                        )
        except Exception:
            pass
        return list(candidates.values())

    async def list_loaded(self) -> List[ModelInstanceInfo]:
        if not await self.is_server_reachable():
            return []
        instances: List[ModelInstanceInfo] = []
        cli_instances: List[ModelInstanceInfo] = []
        if self._cli_exe():
            try:
                raw = await self._run_cli(["ps", "--json"])
                payload = _parse_json_blob(raw)
                items = payload if isinstance(payload, list) else payload.get("data") if isinstance(payload, dict) else []
                for item in items or []:
                    if not isinstance(item, dict):
                        continue
                    identifier = item.get("identifier") or item.get("id") or item.get("model")
                    model_key = item.get("modelKey") or item.get("model_key") or item.get("model") or identifier
                    if not identifier:
                        continue
                    cli_instances.append(
                        ModelInstanceInfo(
                            backend_id=self.id,
                            instance_id=str(identifier),
                            model_key=str(model_key),
                            api_identifier=str(identifier),
                            endpoint=self.base_url,
                            status="ready",
                            ttl_seconds=item.get("ttl") or item.get("ttl_seconds"),
                        )
                    )
            except Exception:
                cli_instances = []
        if cli_instances:
            instances.extend(cli_instances)
        seen_ids = {inst.api_identifier or inst.instance_id for inst in instances if inst}
        try:
            resp = await self.client.get(f"{self.base_url}/models", timeout=5.0)
            if resp.status_code < 400:
                data = resp.json()
                for item in data.get("data", []):
                    if not isinstance(item, dict):
                        continue
                    model_id = item.get("id")
                    if not model_id:
                        continue
                    if model_id in seen_ids:
                        continue
                    instances.append(
                        ModelInstanceInfo(
                            backend_id=self.id,
                            instance_id=str(model_id),
                            model_key=str(model_id),
                            api_identifier=str(model_id),
                            endpoint=self.base_url,
                            status="ready",
                        )
                    )
                    seen_ids.add(model_id)
        except Exception:
            pass
        return instances

    async def load_instance(self, model_key: str, opts: Dict[str, Any]) -> ModelInstanceInfo:
        identifier = opts.get("identifier")
        if not identifier:
            identifier = f"{model_key}-{uuid.uuid4().hex[:8]}"
        ttl = int(opts.get("ttl_seconds") or self.default_ttl_s)
        args = ["load", model_key, "--identifier", str(identifier)]
        if ttl:
            args.extend(["--ttl", str(ttl)])
        if opts.get("context_length"):
            args.extend(["--context-length", str(opts["context_length"])])
        if opts.get("gpu"):
            args.extend(["--gpu", str(opts["gpu"])])
        if self._cli_exe():
            await self._run_cli(args, timeout_s=_CLI_LOAD_TIMEOUT_S)
        return ModelInstanceInfo(
            backend_id=self.id,
            instance_id=str(identifier),
            model_key=model_key,
            api_identifier=str(identifier),
            endpoint=self.base_url,
            status="ready",
            ttl_seconds=ttl,
        )

    async def unload_instance(self, instance_id_or_identifier: str) -> None:
        if not self._cli_exe():
            return
        await self._run_cli(["unload", str(instance_id_or_identifier)], timeout_s=_CLI_UNLOAD_TIMEOUT_S)

    async def estimate_resources(self, model_key: str, opts: Dict[str, Any]) -> Optional[ResourceEstimate]:
        if not self._cli_exe():
            return None
        args = ["load", "--estimate-only", model_key]
        if opts.get("context_length"):
            args.extend(["--context-length", str(opts["context_length"])])
        if opts.get("gpu"):
            args.extend(["--gpu", str(opts["gpu"])])
        try:
            raw = await self._run_cli(args, timeout_s=_CLI_ESTIMATE_TIMEOUT_S)
        except Exception:
            return None
        payload = _parse_json_blob(raw)
        if isinstance(payload, dict):
            return ResourceEstimate(
                vram_mb=_to_float(payload.get("vram_mb") or payload.get("vram")),
                ram_mb=_to_float(payload.get("ram_mb") or payload.get("ram")),
                cpu_pct=_to_float(payload.get("cpu_pct") or payload.get("cpu")),
                gpu_id=_to_float(payload.get("gpu_id")),
            )
        match = _VRAM_RE.search(raw or "")
        if match:
            return ResourceEstimate(vram_mb=_to_float(match.group(1)))
        return None

    async def _post_with_model_fallback(
        self,
        url: str,
        payload: Dict[str, Any],
        instance: ModelInstanceInfo,
        *,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        resp = await self.client.post(url, json=payload, timeout=timeout)
        if resp.status_code == 400 and instance.model_key and instance.model_key != payload.get("model"):
            should_retry = False
            message = ""
            code = ""
            try:
                data = resp.json()
                message = str((data.get("error") or {}).get("message") or "")
                code = str((data.get("error") or {}).get("code") or "")
            except Exception:
                data = None
            if "Invalid model identifier" in message or code == "model_not_found":
                should_retry = True
            if not should_retry:
                text = ""
                try:
                    text = resp.text or ""
                except Exception:
                    text = ""
                lower = text.lower()
                if not message and (not text.strip() or "model" in lower or "identifier" in lower or "not found" in lower):
                    should_retry = True
            if should_retry:
                retry_payload = dict(payload)
                retry_payload["model"] = instance.model_key
                resp = await self.client.post(url, json=retry_payload, timeout=timeout)
        resp.raise_for_status()
        return resp

    async def call_chat_completion(self, instance: ModelInstanceInfo, request: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(request)
        payload["model"] = instance.api_identifier
        url = f"{self.base_url}/chat/completions"
        try_responses = payload.pop("use_responses", True)
        if try_responses and self._responses_supported:
            try:
                resp = await self._post_with_model_fallback(
                    f"{self.base_url}/responses",
                    payload,
                    instance,
                    timeout=5.0,
                )
                data = resp.json()
                return self._normalize_response(data)
            except Exception:
                self._responses_supported = False
                pass
        resp = await self._post_with_model_fallback(url, payload, instance)
        data = resp.json()
        return data

    async def call_responses(self, instance: ModelInstanceInfo, request: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(request)
        payload["model"] = instance.api_identifier
        resp = await self._post_with_model_fallback(f"{self.base_url}/responses", payload, instance)
        data = resp.json()
        return self._normalize_response(data)

    def _normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        if data.get("choices"):
            return data
        output = data.get("output") or []
        if not output:
            return data
        content = ""
        tool_calls = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                raw_content = item.get("content", "")
                if isinstance(raw_content, list):
                    parts = []
                    for part in raw_content:
                        if isinstance(part, dict) and part.get("text"):
                            parts.append(str(part.get("text")))
                    content = "".join(parts)
                else:
                    content = str(raw_content or "")
            if item.get("type") == "tool_call":
                func = item.get("function") or {}
                name = func.get("name") or item.get("name") or "tool"
                args = func.get("arguments") or item.get("arguments") or ""
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=True)
                tool_calls.append(
                    {
                        "type": "function",
                        "function": {"name": str(name), "arguments": str(args)},
                    }
                )
        if tool_calls:
            message = {"content": content, "tool_calls": tool_calls}
        else:
            message = {"content": content}
        return {"choices": [{"message": message}]}

    def supports_tools(self) -> bool:
        return True

    async def close(self) -> None:
        await self.client.aclose()
