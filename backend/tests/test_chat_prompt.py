import pytest

from app.routers.chat import DEFAULT_SYSTEM_PROMPT, prepare_messages
from schemas.schemas import ChatRequest, Message, MessageRole


def request_with(messages, **kwargs):
    return ChatRequest(messages=[Message(**m) for m in messages], **kwargs)


@pytest.mark.asyncio
async def test_a_caller_system_prompt_is_kept_when_rag_is_off():
    req = request_with([
        {"role": MessageRole.system, "content": "Answer only in French."},
        {"role": MessageRole.user, "content": "Hello"},
    ])

    messages, _ = await prepare_messages(req)

    assert messages[0]["content"] == "Answer only in French."


@pytest.mark.asyncio
async def test_the_default_prompt_is_added_when_the_caller_sends_none():
    req = request_with([{"role": MessageRole.user, "content": "Hello"}])

    messages, _ = await prepare_messages(req)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_only_one_system_message_is_ever_sent():
    req = request_with([
        {"role": MessageRole.system, "content": "Be terse."},
        {"role": MessageRole.user, "content": "Hello"},
    ])

    messages, _ = await prepare_messages(req)

    assert [m["role"] for m in messages].count("system") == 1


@pytest.mark.asyncio
async def test_rag_context_replaces_the_caller_prompt(monkeypatch):
    from app.routers import chat

    async def build_context(query_text, collection_name, top_k):
        return "[Source 1: a.txt]\nthe answer is 42", []

    monkeypatch.setattr(chat.rag_service, "build_context", build_context)

    req = request_with(
        [
            {"role": MessageRole.system, "content": "Answer only in French."},
            {"role": MessageRole.user, "content": "What is the answer?"},
        ],
        use_rag=True,
        collection_name="docs",
    )

    messages, _ = await prepare_messages(req)

    assert "the answer is 42" in messages[0]["content"]
