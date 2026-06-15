# Benchmarks

Standardised KV cache performance measurements.

## Methodology

All benchmarks use random model weights (no download). Results are collected on a single CPU/MPS device with no batch parallelism.

### Metrics

| Metric | Definition |
|---|---|
| **TTFT** | Time to first token = milliseconds from prefill start to logits returned |
| **Decode throughput** | Tokens per second during autoregressive decode (average over N steps after prefill) |
| **Cache memory** | Total bytes across all KV cache layers divided by 1 MiB |

### Corpus

The prompt corpus in [`corpus.json`](corpus.json) contains passages of varying length:

| Label | Target tokens | Content |
|---|---|---|
| `t16` | 16 | Short sentence |
| `t32` | 32 | Two-sentence definition |
| `t64` | 64 | Paragraph description |
| `t128` | 128 | Multi-paragraph summary |
| `t256` | 256 | Technical explanation (KV cache) |
| `t512` | 512 | Survey of cache strategies |

Actual token counts depend on the tokenizer and are recorded in each run's TSV.

### Runner

Use `scripts/benchmark.py` from the project root:

```bash
# GPT-2 with default settings
python scripts/benchmark.py

# Custom model and lengths
python scripts/benchmark.py --model tinyllama --seq-lens 16 128 512

# Save to results file
python scripts/benchmark.py --output docs/benchmarks/results/run.tsv

# Precise timing
python scripts/benchmark.py --warmup 5 --samples 10 --decode-steps 20
```

## Results

### GPT-2 (124M, random bf16 weights, CPU)

| Prompt | Seq Len | Prefill TTFT | Decode tok/s | Cache MB |
|---|---|---|---|---|
| t16 | 13 | 31.05ms | 78.1 | 0.81 |
| t32 | 30 | 60.08ms | 77.5 | 1.41 |
| t64 | 69 | 127.37ms | 72.0 | 2.78 |
| t128 | 141 | 270.67ms | 70.5 | 5.31 |
| t256 | 208 | 427.26ms | 69.5 | 7.66 |
| t512 | 359 | 1254.79ms | 66.2 | 12.97 |

Full TSV: [`results/gpt2.tsv`](results/gpt2.tsv)

### Interpreting

- **Prefill** scales roughly linearly with sequence length for a fixed model (O(n) per layer × n layers = O(n²) total, but in practice mostly memory-bound).
- **Decode throughput** drops slightly at longer contexts due to larger attention matrices in the QK^T product.
- **Cache memory** grows linearly with sequence length: `2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_elem`.
