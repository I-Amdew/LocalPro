import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from app.model_manager import ModelCandidate, ModelInstanceInfo, ResourceEstimate


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True)


class FakeLMStudioClient:
    def __init__(
        self,
        model_ids: Optional[List[str]] = None,
        router_response: Optional[Any] = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.base_url = "http://lm.test/v1"
        self.max_output_tokens = None
        self.model_ids = model_ids or ["test-model"]
        self.router_response = router_response
        self.delay_seconds = delay_seconds
        self.calls: List[Dict[str, Any]] = []
        self.router_calls = 0

    async def list_models(self, base_url: Optional[str] = None) -> Dict[str, Any]:
        return {"data": [{"id": mid} for mid in self.model_ids]}

    async def list_models_cached(self, base_url: Optional[str] = None, force: bool = False) -> List[str]:
        return list(self.model_ids)

    async def check_chat(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        run_state: Optional[Any] = None,
    ) -> tuple[bool, str]:
        return True, ""

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
    ) -> Dict[str, Any]:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        system_text = _message_text(messages[0]) if messages else ""
        user_text = _message_text(messages[-1]) if messages else ""
        self.calls.append({"model": model, "system": system_text, "user": user_text})

        if "You are the Router" in system_text:
            self.router_calls += 1
            if self.router_response is None:
                response = {
                    "needs_web": False,
                    "reasoning_level": "MED",
                    "topic": "general",
                    "max_results": 3,
                    "extract_depth": "basic",
                    "tool_budget": {"tavily_search": 0, "tavily_extract": 0},
                    "expected_passes": 1,
                    "stop_conditions": {},
                }
                content = json.dumps(response)
            elif isinstance(self.router_response, str):
                content = self.router_response
            else:
                content = json.dumps(self.router_response)
        elif "JSONRepair" in system_text or "repaired JSON" in system_text:
            content = "{}"
        elif "SYSTEM (WORKER: Finalizer)" in system_text:
            content = json.dumps({"tool_requests": [{"tool": "finalize_answer"}]})
        elif "SYSTEM (WORKER: Verifier)" in system_text:
            content = json.dumps({"verdict": "PASS", "issues": [], "revised_answer": "", "extra_steps": []})
        elif "SYSTEM (WORKER: Writer)" in system_text:
            content = "Test answer."
        elif "SYSTEM (WORKER: Research" in system_text:
            content = json.dumps({"queries": ["test query"], "tool_requests": []})
        elif "SYSTEM (WORKER: UploadSecretary)" in system_text:
            content = json.dumps(
                {
                    "summary": "Upload summary",
                    "bullet_points": [],
                    "suggested_queries": [],
                    "tags": [],
                }
            )
        elif "SYSTEM (WORKER: VisionAnalyst)" in system_text:
            content = json.dumps(
                {
                    "caption": "Test image",
                    "objects": [],
                    "text": "",
                    "details": "",
                    "safety_notes": "",
                }
            )
        elif "SYSTEM (EXECUTOR" in system_text:
            if "prompt_hint" in user_text:
                content = json.dumps({"prompts": [{"step_id": 1, "prompt_hint": "Focus on basics."}]})
            else:
                content = json.dumps({"start_ids": [], "queue_ids": [], "target_slots": 1, "note": "ok"})
        elif "SYSTEM (ORCHESTRATOR" in system_text:
            if "Evaluate the step output" in user_text:
                content = json.dumps({"control": "CONTINUE"})
            else:
                content = "{}"
        elif "execution lane" in user_text and "route" in user_text:
            content = json.dumps({"route": "oss"})
        else:
            content = "{}"

        return {"choices": [{"message": {"content": content}}]}

    async def close(self) -> None:
        return None


class FakeTavilyClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_response: Optional[Dict[str, Any]] = None,
        extract_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.api_key = api_key
        self.search_response = search_response
        self.extract_response = extract_response
        self.search_calls: List[Dict[str, Any]] = []
        self.extract_calls: List[Dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        topic: Optional[str] = None,
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.search_calls.append(
            {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "topic": topic,
                "time_range": time_range,
            }
        )
        if self.search_response is not None:
            return self.search_response
        if not self.enabled:
            return {"error": "missing_api_key"}
        return {"results": []}

    async def extract(self, urls: List[str], extract_depth: str = "basic") -> Dict[str, Any]:
        self.extract_calls.append({"urls": urls, "extract_depth": extract_depth})
        if self.extract_response is not None:
            return self.extract_response
        if not self.enabled:
            return {"error": "missing_api_key"}
        return {"results": []}

    async def close(self) -> None:
        return None


class FakeModelBackend:
    id = "fake"

    def __init__(
        self,
        model_keys: Optional[List[str]] = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.model_keys = model_keys or ["test-model"]
        self.delay_seconds = delay_seconds
        self.fake_client = FakeLMStudioClient(model_ids=self.model_keys)
        self._loaded: Dict[str, ModelInstanceInfo] = {}
        self.load_calls: List[Dict[str, Any]] = []
        self.unload_calls: List[str] = []

    async def discover(self) -> List[ModelCandidate]:
        return [
            ModelCandidate(
                backend_id=self.id,
                model_key=key,
                display_name=key,
                metadata={},
                capabilities={"tool_use": True, "structured_output": True},
            )
            for key in self.model_keys
        ]

    async def list_loaded(self) -> List[ModelInstanceInfo]:
        return list(self._loaded.values())

    async def ensure_server_running(self) -> None:
        return None

    async def load_instance(self, model_key: str, opts: Dict[str, Any]) -> ModelInstanceInfo:
        identifier = opts.get("identifier") or f"{model_key}-{uuid.uuid4().hex[:8]}"
        ttl = int(opts.get("ttl_seconds") or 0) or None
        info = ModelInstanceInfo(
            backend_id=self.id,
            instance_id=str(identifier),
            model_key=model_key,
            api_identifier=str(identifier),
            endpoint="http://fake.local/v1",
            status="ready",
            ttl_seconds=ttl,
        )
        self._loaded[info.instance_id] = info
        self.load_calls.append({"model_key": model_key, "identifier": identifier})
        return info

    async def unload_instance(self, instance_id_or_identifier: str) -> None:
        self.unload_calls.append(str(instance_id_or_identifier))
        self._loaded.pop(str(instance_id_or_identifier), None)

    async def estimate_resources(self, model_key: str, opts: Dict[str, Any]) -> Optional[ResourceEstimate]:
        return ResourceEstimate(vram_mb=1200.0, ram_mb=2400.0, cpu_pct=10.0, gpu_id=0)

    async def call_chat_completion(self, instance: ModelInstanceInfo, request: Dict[str, Any]) -> Dict[str, Any]:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if request.get("tools") or request.get("tool_choice"):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"function": {"arguments": "{\"value\": 7}"}, "type": "function"}
                            ],
                        }
                    }
                ]
            }
        if request.get("response_format"):
            return {"choices": [{"message": {"content": "{\"ok\": true}"}}]}
        messages = request.get("messages") or []
        return await self.fake_client.chat_completion(
            model=instance.api_identifier,
            messages=messages,
            temperature=request.get("temperature", 0.2),
            max_tokens=request.get("max_tokens", 512),
        )

    async def call_responses(self, instance: ModelInstanceInfo, request: Dict[str, Any]) -> Dict[str, Any]:
        return await self.call_chat_completion(instance, request)

    def supports_tools(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class FakeTelemetry:
    def __init__(self, snapshots: Optional[List[Dict[str, Any]]] = None) -> None:
        self.snapshots = snapshots or []
        self.calls = 0

    def snapshot(self) -> Dict[str, Any]:
        if not self.snapshots:
            return {"ram": {}, "gpus": [], "captured_at": "1970-01-01T00:00:00Z"}
        idx = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        return self.snapshots[idx]

    def monitor_start(self, sample_interval_ms: int = 250) -> str:
        return "monitor"

    def monitor_stop(self, monitor_id: str) -> Dict[str, Any]:
        snap = self.snapshot()
        return {"peak_snapshot": snap, "samples_summary": {"count": 1, "duration_ms": 0}}
