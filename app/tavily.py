from typing import Any, Dict, List, Optional

import base64
import html as html_lib
import re
from urllib.parse import parse_qs, unquote, urlparse

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

    def _strip_html(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
        cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
        cleaned = html_lib.unescape(cleaned)
        return re.sub(r"\\s+", " ", cleaned).strip()

    def _unwrap_bing_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/a"):
                qs = parse_qs(parsed.query)
                token = qs.get("u", [None])[0]
                if token:
                    if token.startswith("a1"):
                        token = token[2:]
                    padding = "=" * (-len(token) % 4)
                    decoded = base64.urlsafe_b64decode(token + padding).decode("utf-8", "ignore")
                    return decoded or url
        except Exception:
            return url
        return url

    async def _fallback_search(self, query: str, max_results: int) -> Dict[str, Any]:
        if not query:
            return {"results": [], "fallback": True, "provider": "bing"}
        try:
            resp = await self.client.get(
                "https://www.bing.com/search",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            html_text = resp.text
        except Exception as exc:
            return {
                "results": [],
                "fallback": True,
                "provider": "bing",
                "fallback_error": f"request_failed:{exc}",
            }
        results: List[Dict[str, Any]] = []
        blocks = re.findall(r'<li class="b_algo".*?</li>', html_text, re.DOTALL)
        for block in blocks:
            match = re.search(r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            if not match:
                continue
            raw_url = html_lib.unescape(match.group(1))
            url = self._unwrap_bing_url(raw_url)
            title = self._strip_html(match.group(2))
            snippet_match = re.search(r"<p>(.*?)</p>", block, re.DOTALL)
            snippet = self._strip_html(snippet_match.group(1)) if snippet_match else ""
            if not url or not title:
                continue
            results.append(
                {
                    "url": url,
                    "title": title,
                    "content": snippet or title,
                    "source": "bing",
                }
            )
            if len(results) >= max_results:
                break
        return {"results": results, "fallback": True, "provider": "bing"}

    async def _fallback_extract(self, urls: List[str], extract_depth: str) -> Dict[str, Any]:
        max_chars = 2000 if extract_depth == "basic" else 8000
        results: List[Dict[str, Any]] = []
        for url in urls:
            if not url:
                continue
            try:
                resp = await self.client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    follow_redirects=True,
                )
                resp.raise_for_status()
                title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", resp.text or "")
                title = self._strip_html(title_match.group(1)) if title_match else ""
                body = self._strip_html(resp.text or "")
                if max_chars and len(body) > max_chars:
                    body = body[:max_chars]
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": body,
                        "raw_content": body,
                        "source": urlparse(url).hostname or "",
                    }
                )
            except Exception:
                continue
        return {"results": results, "fallback": True, "provider": "http_fetch"}

    async def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        topic: Optional[str] = None,
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return await self._fallback_search(query, max_results)
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
        resp = await self._post("https://api.tavily.com/search", payload)
        if resp.get("error"):
            fallback = await self._fallback_search(query, max_results)
            if fallback.get("results"):
                fallback["fallback_error"] = resp
                return fallback
        return resp

    async def extract(self, urls: List[str], extract_depth: str = "basic") -> Dict[str, Any]:
        if not self.enabled:
            return await self._fallback_extract(urls, extract_depth)
        payload = {"urls": urls, "extract_depth": extract_depth}
        resp = await self._post("https://api.tavily.com/extract", payload)
        if resp.get("error"):
            fallback = await self._fallback_extract(urls, extract_depth)
            if fallback.get("results"):
                fallback["fallback_error"] = resp
                return fallback
        return resp

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
