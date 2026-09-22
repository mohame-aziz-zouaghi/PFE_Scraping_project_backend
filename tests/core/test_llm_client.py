import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from app.core.llm_client import LLMClient, LLMError


class Echo(BaseModel):
    echo: str
    ok: bool


@pytest.mark.asyncio
async def test_chat_json_parses_valid_response():
    client = LLMClient(endpoint="test")
    with patch.object(
        client,
        "_call_provider",
        new=AsyncMock(return_value=(
            json.dumps({"echo": "hi", "ok": True}),
            {"prompt_tokens": 10, "completion_tokens": 5},
        )),
    ):
        result = await client.chat_json(
            system="test", user={"echo": "hi"}, schema=Echo,
        )
    assert result.echo == "hi"
    assert result.ok is True


@pytest.mark.asyncio
async def test_chat_json_extracts_from_markdown_fence():
    client = LLMClient(endpoint="test")
    raw = '```json\n{"echo": "x", "ok": true}\n```'
    with patch.object(
        client,
        "_call_provider",
        new=AsyncMock(return_value=(
            raw, {"prompt_tokens": 10, "completion_tokens": 5},
        )),
    ):
        result = await client.chat_json(system="t", user="t", schema=Echo)
    assert result.echo == "x"


@pytest.mark.asyncio
async def test_chat_json_raises_after_all_retries():
    client = LLMClient(endpoint="test")
    with patch.object(
        client,
        "_call_provider",
        new=AsyncMock(return_value=("not json at all", {})),
    ):
        with pytest.raises(LLMError):
            await client.chat_json(system="t", user="t", schema=Echo)


def test_extract_json_handles_prose():
    raw = 'Sure! Here is the answer: {"a": 1} Done.'
    assert LLMClient._extract_json(raw) == {"a": 1}
