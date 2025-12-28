import asyncio
import json
from typing import Any, Dict, List, Optional


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
