import httpx
import pytest

from app.lmstudio_backend import LMStudioBackend


class StubResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=httpx.Request("POST", "http://test"), response=httpx.Response(self.status_code)
            )


class StubClient:
    def __init__(self):
        self.post_calls = []

    async def post(self, url, json=None):
        self.post_calls.append({"url": url, "json": json})
        if url.endswith("/responses"):
            return StubResponse({"error": "not found"}, status_code=404)
        return StubResponse({"choices": [{"message": {"content": "ok"}}]}, status_code=200)

    async def get(self, url, timeout=None):
        return StubResponse({"data": []}, status_code=200)

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_lmstudio_backend_load_call_unload():
    backend = LMStudioBackend(
        base_url="http://127.0.0.1:1234/v1",
        host="127.0.0.1",
        port=1234,
        use_cli=True,
        cli_path="lms",
        default_ttl_s=60,
    )
    calls = []

    async def fake_run_cli(args):
        calls.append(args)
        return ""

    backend._run_cli = fake_run_cli  # type: ignore[assignment]
    backend.client = StubClient()
    instance = await backend.load_instance("qwen/qwen3-vl-8b", {"identifier": "qwen-test", "ttl_seconds": 33})
    assert instance.api_identifier == "qwen-test"
    assert calls and calls[0][:4] == ["load", "qwen/qwen3-vl-8b", "--identifier", "qwen-test"]

    response = await backend.call_chat_completion(
        instance,
        {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 5, "use_responses": True},
    )
    assert response["choices"][0]["message"]["content"] == "ok"
    assert backend.client.post_calls[-1]["json"]["model"] == "qwen-test"

    await backend.unload_instance(instance.api_identifier)
    assert ["unload", "qwen-test"] in calls
