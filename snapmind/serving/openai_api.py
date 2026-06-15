# ─── SECTION: OpenAI API ─────────────────────────────────
import json
import time
import uuid
from collections.abc import AsyncGenerator

import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

import snapmind.sampling.greedy  # noqa: F401
import snapmind.sampling.temperature  # noqa: F401
import snapmind.sampling.top_k  # noqa: F401
import snapmind.sampling.top_p  # noqa: F401
import snapmind.tokenizer.hf  # noqa: F401
from snapmind.core.config import ModelConfig
from snapmind.core.registry import SAMPLER, TOKENIZER
from snapmind.engine.generate import GenerateEngine
from snapmind.sampling.mirostat import MirostatSampler


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "gpt2"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    mirostat_tau: float | None = None
    mirostat_learning_rate: float | None = None


def resolve_sampler(req: ChatRequest) -> tuple:
    sampler_kwargs: dict = {}

    if req.mirostat_tau is not None:
        tau = req.mirostat_tau
        lr = req.mirostat_learning_rate if req.mirostat_learning_rate is not None else 0.1
        return MirostatSampler(tau=tau, learning_rate=lr), sampler_kwargs

    if req.top_k is not None:
        sampler_kwargs["top_k"] = req.top_k
        return SAMPLER.create("top_k"), sampler_kwargs

    if req.top_p is not None:
        sampler_kwargs["top_p"] = req.top_p
        return SAMPLER.create("top_p"), sampler_kwargs

    if req.temperature != 1.0:
        return SAMPLER.create("temperature"), sampler_kwargs

    return SAMPLER.create("greedy"), sampler_kwargs


# ANCHOR: create_app
def create_app(model: nn.Module, config: ModelConfig, model_name: str) -> FastAPI:
    app = FastAPI(title="snapmind", version="0.1.0")
    tok = TOKENIZER.create("hf", model_name=model_name)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "user"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        if not req.messages:
            raise HTTPException(400, "messages is required")

        prompt = req.messages[-1].content
        sampler, sampler_kwargs = resolve_sampler(req)
        engine = GenerateEngine(model, tok, sampler)

        if req.stream:
            return StreamingResponse(
                stream_chat(engine, prompt, req.model, req.max_tokens, req.temperature, sampler_kwargs),
                media_type="text/event-stream",
            )

        return await complete_chat(engine, prompt, req.model, req.max_tokens, req.temperature, sampler_kwargs)

    return app


# ENDANCHOR: create_app


# ANCHOR: complete_chat
async def complete_chat(
    engine: GenerateEngine, prompt: str, model_name: str, max_tokens: int, temperature: float, sampler_kwargs: dict
) -> dict:
    tokens = []
    async for token in engine.generate(prompt, max_tokens=max_tokens, temperature=temperature, **sampler_kwargs):
        tokens.append(token)

    content = "".join(tokens)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(tokens), "total_tokens": len(tokens)},
    }


# ENDANCHOR: complete_chat


# ANCHOR: stream_chat
async def stream_chat(
    engine: GenerateEngine, prompt: str, model_name: str, max_tokens: int, temperature: float, sampler_kwargs: dict
) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    yield (
        "data: "
        + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        + "\n\n"
    )

    token_count = 0
    async for token in engine.generate(prompt, max_tokens=max_tokens, temperature=temperature, **sampler_kwargs):
        token_count += 1
        yield (
            "data: "
            + json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
            )
            + "\n\n"
        )

    yield (
        "data: "
        + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + "\n\n"
    )
    yield "data: [DONE]\n\n"


# ENDANCHOR: stream_chat
# ─── ENDSECTION: OpenAI API ──────────────────────────────
