# snapmind — Framework Agent Memory

## Current Focus
Completed: Phase 0-2 + Phase 3 Llama model + Phase 4 serving (CLI + API, in progress).
206 tests passing.

## Key Recent Decisions
- RoPE uses `dim` (head_dim) not `d_model` for the cos/sin table
- `apply_to_qk()` added to PositionalEncodingABC with default no-op
- LlamaTransformerBlock is separate from TransformerBlock (not a config flag)
- GatedFeedForward is a separate layer class
- SiLU is registered as `"silu"` in ACTIVATION registry
- GQA accepts optional `pe` parameter for RoPE injection
- MODEL registry added for multi-model dispatch via string key
- Serving CLI uses argparse (no extra deps) + FastAPI optional as `[serve]` extra
- Streaming uses `json.dumps` (not f-string brace escaping)
- RMSNorm test uses `atol=0.15` to avoid flakiness from random variance

## Open Questions
- Integration test for Llama weights (needs download)?
- Benchmark corpus for KV cache experiments?
- Should GPT-2 also register in MODEL registry?
- Test `pytorch.bin` loader?

## Known Issues
- `feed_forward.py` accepts `activation_type` param but hardcodes GELU (minor)
- `test_tokens_small_vocab` fixture unused (leftover from generate test fix)
- Weight tying warning on generate (missing `lm_head.weight` with random weights)
- NumPy warning (torch misconfiguration, harmless)

## Working Context
- Project: /Users/petehaughie/Projects/python-inference-framework
- Package: snapmind
- 206 tests, all passing
- 2 models: GPT-2 (124M, 12 layers, tied weights) and Llama (8B, 32 layers, separate lm_head)
- Serving CLI works: `snapmind list`, `snapmind generate`, `snapmind serve`
