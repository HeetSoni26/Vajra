from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from api.middleware import setup_middleware
from api.schemas import (
    CompletionRequest,
    ChatRequest,
    TokenizeRequest,
    DetokenizeRequest,
)

_ENGINE = None


def _get_engine():
    """Return the global inference engine (initialised at startup)."""
    return _ENGINE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the InferenceEngine once on startup and tear down on shutdown."""
    global _ENGINE
    config_path = os.environ.get("FLM_CONFIG", "configs/training/pretrain_tiny.yaml")
    checkpoint = os.environ.get("FLM_CHECKPOINT", "")

    from inference.engine import InferenceEngine

    _ENGINE = InferenceEngine.from_config(config_path, checkpoint or None)
    app.state.engine = _ENGINE
    yield
    _ENGINE = None


app = FastAPI(
    title="Foundation LM API",
    version="0.6.0",
    description="Production API for FoundationLM text generation",
    lifespan=lifespan,
)

setup_middleware(app)


# --- Health & Model Info ---


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    engine = _get_engine()
    if engine:
        info = engine.model_info()
        return {"data": [{"id": info["model_name"], "object": "model", **info}]}
    return {"data": [{"id": "vajra-lm", "object": "model"}]}


@app.get("/model")
async def model_info():
    engine = _get_engine()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialised",
        )
    return engine.model_info()


# --- Generation ---


@app.post("/v1/completions")
@app.post("/generate")
async def completions(request: CompletionRequest):
    engine = _get_engine()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialised",
        )

    from inference.engine import GenerationConfig

    gen_cfg = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        do_sample=(request.temperature > 0),
        seed=request.seed,
    )

    if request.stream:

        async def _async_stream():
            for token_text in engine.generate_stream(request.prompt, gen_cfg):
                yield f"data: {token_text}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_async_stream(), media_type="text/event-stream")

    try:
        results = engine.generate(request.prompt, gen_cfg)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )

    return {
        "id": "cmpl-local",
        "object": "text_completion",
        "model": engine.model_info()["model_name"],
        "choices": [{"index": 0, "text": results[0], "finish_reason": "stop"}],
    }


# --- Chat ---


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    engine = _get_engine()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialised",
        )

    # Flatten messages into a single prompt
    prompt_parts = []
    for msg in request.messages:
        prompt_parts.append(f"<|{msg.role}|>{msg.content}")
    prompt = "".join(prompt_parts) + "<|assistant|>"

    from inference.engine import GenerationConfig

    gen_cfg = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        do_sample=(request.temperature > 0),
        seed=request.seed,
    )

    try:
        results = engine.generate(prompt, gen_cfg)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(e)}",
        )

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": engine.model_info()["model_name"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": results[0]},
                "finish_reason": "stop",
            }
        ],
    }


# --- Tokenize / Detokenize ---


@app.post("/tokenize")
async def tokenize_text(request: TokenizeRequest):
    engine = _get_engine()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialised",
        )
    return engine.tokenize(request.text)


@app.post("/detokenize")
async def detokenize_ids(request: DetokenizeRequest):
    engine = _get_engine()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialised",
        )
    return {"text": engine.detokenize(request.ids)}
