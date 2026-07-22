from __future__ import annotations

from fastapi import APIRouter

from api.schemas import CompletionRequest

router = APIRouter()


@router.post("/v1/completions")
def completions(request: CompletionRequest):
    return {
        "id": "cmpl-local",
        "object": "text_completion",
        "model": "vajra-lm",
        "choices": [{"index": 0, "text": request.prompt, "finish_reason": "stub"}],
    }
