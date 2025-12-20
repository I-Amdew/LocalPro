from typing import Any, Dict, List, Optional

import httpx


class TavilyClient:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=60)

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
        payload: Dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        }
        if topic:
            payload["topic"] = topic
        if time_range:
            payload["time_range"] = time_range
        resp = await self.client.post(
            "https://api.tavily.com/search",
            json=payload,
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
        )
        resp.raise_for_status()
        return resp.json()

    async def extract(self, urls: List[str], extract_depth: str = "basic") -> Dict[str, Any]:
        if not self.enabled:
            return {"error": "missing_api_key"}
        payload = {"urls": urls, "extract_depth": extract_depth}
        resp = await self.client.post(
            "https://api.tavily.com/extract",
            json=payload,
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self.client.aclose()

