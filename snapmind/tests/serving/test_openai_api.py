import json

import pytest


class TestChatRequest:
    def test_extra_fields_ignored(self):
        from snapmind.serving.openai_api import ChatRequest

        req = ChatRequest(messages=[{"role": "user", "content": "hi"}], unknown_field="hello", temperature=1.0)
        assert req.model == "gpt2"
        assert req.temperature == 1.0
        assert not hasattr(req, "unknown_field")

    def test_accepts_extended_params(self):
        from snapmind.serving.openai_api import ChatRequest

        req = ChatRequest(messages=[], top_k=50, mirostat_tau=3.0, mirostat_learning_rate=0.2)
        assert req.top_k == 50
        assert req.mirostat_tau == 3.0
        assert req.mirostat_learning_rate == 0.2


class TestResolveSampler:
    def test_greedy_when_no_params(self):
        from snapmind.serving.openai_api import ChatRequest, resolve_sampler

        s, kw = resolve_sampler(ChatRequest(messages=[], temperature=1.0))
        assert type(s).__name__ == "GreedySampler"
        assert kw == {}

    def test_top_p_when_set(self):
        from snapmind.serving.openai_api import ChatRequest, resolve_sampler

        s, kw = resolve_sampler(ChatRequest(messages=[], top_p=0.9))
        assert type(s).__name__ == "TopPSampler"
        assert kw == {"top_p": 0.9}

    def test_top_k_when_set(self):
        from snapmind.serving.openai_api import ChatRequest, resolve_sampler

        s, kw = resolve_sampler(ChatRequest(messages=[], top_k=50))
        assert type(s).__name__ == "TopKSampler"
        assert kw == {"top_k": 50}

    def test_temperature_when_not_one(self):
        from snapmind.serving.openai_api import ChatRequest, resolve_sampler

        s, kw = resolve_sampler(ChatRequest(messages=[], temperature=0.8))
        assert type(s).__name__ == "TemperatureSampler"
        assert kw == {}

    def test_mirostat_when_tau_set(self):
        from snapmind.serving.openai_api import ChatRequest, resolve_sampler

        s, kw = resolve_sampler(ChatRequest(messages=[], mirostat_tau=3.0))
        assert type(s).__name__ == "MirostatSampler"
        assert s.tau == 3.0
        assert s.rate == 0.1

    def test_mirostat_with_learning_rate(self):
        from snapmind.serving.openai_api import ChatRequest, resolve_sampler

        s, kw = resolve_sampler(ChatRequest(messages=[], mirostat_tau=2.5, mirostat_learning_rate=0.3))
        assert type(s).__name__ == "MirostatSampler"
        assert s.tau == 2.5
        assert s.rate == 0.3

    def test_top_k_overrides_top_p(self):
        from snapmind.serving.openai_api import ChatRequest, resolve_sampler

        s, kw = resolve_sampler(ChatRequest(messages=[], top_k=50, top_p=0.9))
        assert type(s).__name__ == "TopKSampler"
        assert kw == {"top_k": 50}

    def test_mirostat_overrides_all(self):
        from snapmind.serving.openai_api import ChatRequest, resolve_sampler

        s, kw = resolve_sampler(ChatRequest(messages=[], mirostat_tau=3.0, top_k=50, top_p=0.9, temperature=0.8))
        assert type(s).__name__ == "MirostatSampler"


class TestCreateApp:
    @pytest.fixture
    def model(self):
        import torch.nn as nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = nn.Module()
                self.config.n_layers = 2

            def forward(self, *args, **kwargs):
                import torch

                return torch.randn(1, 1, 100)

        return FakeModel()

    @pytest.fixture
    def config(self):
        from snapmind.core.config import ModelConfig

        return ModelConfig()

    def test_routes_registered(self, model, config):
        from snapmind.serving.openai_api import create_app

        app = create_app(model, config, "gpt2")
        routes = {r.path for r in app.routes}
        assert "/v1/models" in routes
        assert "/v1/chat/completions" in routes


class TestCompleteChat:
    @pytest.mark.asyncio
    async def test_response_format(self):
        from snapmind.serving.openai_api import complete_chat

        calls = 0

        class FakeEngine:
            async def generate(self, prompt, max_tokens=100, temperature=1.0, **kw):
                nonlocal calls
                calls += 1
                for t in ["hello", " ", "world"]:
                    yield t

        result = await complete_chat(FakeEngine(), "hi", "gpt2", 10, 1.0, {})
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt2"
        assert result["choices"][0]["message"]["content"] == "hello world"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["completion_tokens"] == 3


class TestStreamChat:
    @pytest.mark.asyncio
    async def test_streaming_events(self):
        from snapmind.serving.openai_api import stream_chat

        class FakeEngine:
            async def generate(self, prompt, max_tokens=100, temperature=1.0, **kw):
                for t in ["hello", " ", "world"]:
                    yield t

        events = []
        async for chunk in stream_chat(FakeEngine(), "hi", "gpt2", 10, 1.0, {}):
            events.append(chunk)

        assert len(events) >= 5
        assert "role" in events[0]
        assert "content" in events[1] or "delta" in json.loads(events[1].replace("data: ", ""))
        assert events[-1] == "data: [DONE]\n\n"
