<div align="center">

# YouAI

**Train your own language model from scratch — in a few lines of Python.**

A small, professional, batteries-included toolkit for building, training, serving
and exporting transformer language models. Modern architecture, correct code,
a real test suite, and an API you can learn in five minutes.

</div>

---

## Why YouAI?

Most people who want to *understand and own* a language model face a choice
between 500-line research repos and heavyweight frameworks. YouAI gives you a
clean, modern implementation you can actually read, with the ergonomics of a
high-level library.

| | YouAI | Typical from-scratch repo | Heavy framework |
|---|:---:|:---:|:---:|
| Lines to train a model | **~5** | 200+ | 50–100 |
| Modern architecture (RoPE, RMSNorm, SwiGLU, GQA) | ✅ built-in | sometimes | ✅ |
| Fused / flash attention | ✅ automatic | rare | ✅ |
| KV-cache generation | ✅ | rare | ✅ |
| Mixed precision (fp16/bf16) | ✅ one flag | manual | ✅ |
| Streaming generation | ✅ built-in | ✗ | manual |
| One-command CLI | ✅ | ✗ | partial |
| Export (ONNX / TorchScript / quantized) | ✅ | ✗ | external |
| Readable, documented, **tested** | ✅ 45 tests | ✗ | large surface |

## Installation

```bash
pip install -e .          # core (torch + transformers)
pip install -e ".[all]"   # + datasets, onnx, wandb, web, dev tools
```

## Quick start

```python
import youai

youai.set_seed(42)

# 1. Create a modern transformer (RoPE + RMSNorm + SwiGLU, weights tied)
model = youai.create_model("125m")

# 2. Get some data (synthetic sample, your own files, or a HuggingFace dataset)
train_file, val_file = youai.create_sample_data(2000)
# train_file, val_file = youai.download_dataset("tinystories", num_examples=50_000)

# 3. Train — mixed precision, packing, warmup+cosine schedule handled for you
youai.train(model, train_file, val_file, epochs=1, mixed_precision="bf16")

# 4. Generate
print(youai.generate("The future of AI", checkpoint_path="./checkpoints/final")[0])

# 5. Chat / stream
bot = youai.load_model("./checkpoints/final")
print(bot.chat("Hello!"))
for piece in youai.stream_generate("./checkpoints/final", prompt="Once upon a time"):
    print(piece, end="", flush=True)
```

## Command line

```bash
youai info --preset 125m                      # architecture + parameter count
youai datasets                                # list dataset presets
youai train --preset 125m --dataset tinystories --epochs 3 --mixed-precision bf16
youai generate --checkpoint ./checkpoints/final --prompt "Hello"
youai chat --checkpoint ./checkpoints/final
youai export --checkpoint ./checkpoints/final --format onnx
youai benchmark --checkpoint ./checkpoints/final
```

## Model presets

| Preset | Parameters | Notes |
|--------|-----------:|-------|
| `nano`  | ~7.5M   | Tiny — unit tests & laptop experiments |
| `micro` | ~19M    | Quick smoke training |
| `125m`  | ~152M   | Great starting point |
| `350m`  | ~454M   | |
| `750m`  | ~983M   | |
| `1.3b`  | ~1.7B   | |
| `2b`    | ~3.5B   | Requires a serious GPU |

Every preset defaults to a modern architecture. Pass `modern=False` for a
classic GPT-2 style model, or override any field:

```python
model = youai.create_model("125m", num_key_value_heads=4, activation="gelu")
```

## Feature highlights

- **Modern architecture** — rotary/learned/ALiBi positions, RMSNorm/LayerNorm,
  SwiGLU/GEGLU/GELU MLPs, grouped-query attention, weight tying.
- **Fast & memory efficient** — fused scaled-dot-product ("flash") attention,
  KV-cache decoding, gradient checkpointing, mixed precision.
- **Correct training** — warmup + cosine/linear schedules, gradient
  accumulation & clipping, padding masked out of the loss, token packing,
  best-checkpoint tracking, early stopping, seamless resume.
- **Great inference** — batch-correct top-k / nucleus sampling, repetition
  penalty, greedy or sampled decoding, token streaming, chat sessions,
  perplexity scoring.
- **Deployment** — ONNX & TorchScript export, int8/fp16 quantization, GGUF
  metadata, benchmarking.
- **Quality** — typed, documented, and covered by a 45-test pytest suite.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite runs in seconds on CPU and needs no GPU.

## Documentation

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full API reference and
[`CHANGELOG.md`](CHANGELOG.md) for what changed in 1.0.

## License

MIT — see [`LICENSE`](LICENSE).
