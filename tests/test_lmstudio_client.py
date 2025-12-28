import json

import pytest
import respx
from httpx import Response

from app.llm import LMStudioClient


@pytest.mark.asyncio
async def test_list_models_hits_models_endpoint():
    client = LMStudioClient("http://lm.test/v1")
    try:
        with respx.mock(assert_all_called=True) as respx_mock:
            respx_mock.get("http://lm.test/v1/models").mock(
                return_value=Response(200, json={"data": [{"id": "test-model"}]})
            )
            data = await client.list_models()
            assert data["data"][0]["id"] == "test-model"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_completion_payload_shape_and_caps_tokens():
    client = LMStudioClient("http://lm.test/v1", max_output_tokens=5)
    captured = {}
    try:
        with respx.mock(assert_all_called=True) as respx_mock:
            respx_mock.get("http://lm.test/v1/models").mock(
                return_value=Response(200, json={"data": [{"id": "test-model"}]})
            )

            def handler(request):
                captured["json"] = json.loads(request.content.decode("utf-8"))
                return Response(200, json={"choices": [{"message": {"content": "ok"}}]})

            respx_mock.post("http://lm.test/v1/chat/completions").mock(side_effect=handler)
            resp = await client.chat_completion(
                model="test-model",
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                temperature=0.5,
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            assert resp["choices"][0]["message"]["content"] == "ok"
            payload = captured["json"]
            assert payload["model"] == "test-model"
            assert payload["max_tokens"] == 5
            assert payload["temperature"] == 0.5
            assert payload["stream"] is False
            assert "response_format" not in payload
            assert payload["messages"][0]["role"] == "system"
    finally:
        await client.close()
