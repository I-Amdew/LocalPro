import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx


class LMStudioClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60)

    async def list_models(self, base_url: Optional[str] = None) -> Dict[str, Any]:
        url = f"{(base_url or self.base_url).rstrip('/')}/models"
        resp = await self.client.get(url)
        resp.raise_for_status()
        return resp.json()

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
        response_format: Optional[dict] = None,
        base_url: Optional[str] = None,
    ) -> Any:
        url = f"{(base_url or self.base_url).rstrip('/')}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if response_format:
            payload["response_format"] = response_format
        resp = await self.client.post(url, json=payload)
        resp.raise_for_status()
        if stream:
            return resp
        return resp.json()

    async def stream_text(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        response = await self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            base_url=base_url,
        )
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            chunk = line.replace("data:", "").strip()
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta
            except Exception:
                continue

    async def close(self) -> None:
        await self.client.aclose()
