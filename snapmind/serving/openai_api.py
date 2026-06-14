# ─── SECTION: OpenAI API ─────────────────────────────────
import json
import time
import uuid
from collections.abc import AsyncGenerator

import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from snapmind.core.config import ModelConfig
from snapmind.engine.generate import GenerateEngine
from snapmind.sampling.greedy import GreedySampler
from snapmind.sampling.top_p import TopPSampler
from snapmind.tokenizer.hf import HFTokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gpt2"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float | None = None


# ANCHOR: create_app
def create_app(model: nn.Module, config: ModelConfig, model_name: str) -> FastAPI:
    app = FastAPI(title="snapmind", version="0.1.0")
    tok = HFTokenizer(model_name=model_name)

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

        sampler_cls = TopPSampler if req.top_p is not None else GreedySampler
        sampler_kwargs = {}
        if req.top_p is not None:
            sampler_kwargs["top_p"] = req.top_p

        sampler = sampler_cls()
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
