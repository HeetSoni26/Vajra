from __future__ import annotations

from fastapi import APIRouter

from api.schemas import ChatRequest

router = APIRouter()


@router.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    last = request.messages[-1].content if request.messages else ""
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": "vajra-lm",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": last}, "finish_reason": "stub"}],
    }
