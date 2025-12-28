import json

import pytest
import respx
from httpx import Response

from app.tavily import TavilyClient


@pytest.mark.asyncio
async def test_tavily_search_payload_and_headers():
    client = TavilyClient("test-key")
    captured = {}
    try:
        with respx.mock(assert_all_called=True) as respx_mock:
            def handler(request):
                captured["json"] = json.loads(request.content.decode("utf-8"))
                captured["headers"] = request.headers
                return Response(200, json={"results": []})

            respx_mock.post("https://api.tavily.com/search").mock(side_effect=handler)
            resp = await client.search("hello", search_depth="basic", max_results=3, topic="news")
            assert resp == {"results": []}
            assert captured["json"]["api_key"] == "test-key"
            assert captured["headers"]["X-API-Key"] == "test-key"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tavily_extract_handles_http_error():
    client = TavilyClient("test-key")
    try:
        with respx.mock(assert_all_called=True) as respx_mock:
            respx_mock.post("https://api.tavily.com/extract").mock(
                return_value=Response(500, json={"error": "boom"})
            )
            resp = await client.extract(["http://example.com"], extract_depth="basic")
            assert resp["error"] == "http_status"
            assert resp["status_code"] == 500
    finally:
        await client.close()
