from typing import Any, Dict, List, Optional

import httpx


class TavilyClient:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        # Tune connection limits so concurrent model calls share a pool instead of opening
        # a new TCP connection per request.
        self.client = httpx.AsyncClient(
            timeout=60,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

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
        if not self.enabled:
            return {"error": "missing_api_key"}
        allowed_topics = {"general", "news", "finance"}
        if topic:
            cleaned = str(topic).strip().lower()
            topic = cleaned if cleaned in allowed_topics else None
        payload: Dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        }
        if topic:
            payload["topic"] = topic
        if time_range:
            payload["time_range"] = time_range
        return await self._post("https://api.tavily.com/search", payload)

    async def extract(self, urls: List[str], extract_depth: str = "basic") -> Dict[str, Any]:
        if not self.enabled:
            return {"error": "missing_api_key"}
        payload = {"urls": urls, "extract_depth": extract_depth}
        return await self._post("https://api.tavily.com/extract", payload)

    async def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Shared POST helper with minimal retry/response handling."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # Tavily's dev keys expect the key in the JSON payload; include it there and keep the header for compatibility.
            payload = {**payload, "api_key": self.api_key}
            headers["X-API-Key"] = self.api_key
        try:
            resp = await self.client.post(
                url,
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            detail: Any
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            return {"error": "http_status", "status_code": e.response.status_code, "detail": detail}
        except httpx.RequestError as e:
            return {"error": "request_failed", "detail": str(e)}

    async def close(self) -> None:
        # Safe to call multiple times
        if not self.client.is_closed:
            await self.client.aclose()
