# YouAI Documentation

Complete reference for the YouAI library (v1.0).

- [Overview](#overview)
- [Installation](#installation)
- [Architecture](#architecture)
- [High-level API](#high-level-api)
- [Configuration](#configuration)
- [Data pipeline](#data-pipeline)
- [Training](#training)
- [Generation & inference](#generation--inference)
- [Streaming & chat](#streaming--chat)
- [Export & quantization](#export--quantization)
- [Command-line interface](#command-line-interface)
- [Utilities](#utilities)
- [FAQ](#faq)

---

## Overview

YouAI is a compact PyTorch library for training decoder-only transformer
language models from scratch. It aims to be:

- **Correct** — the math is right and the behaviour is covered by tests.
- **Modern** — the same architectural building blocks as contemporary open LLMs.
- **Ergonomic** — five-line training, one-command CLI, sensible defaults.
- **Readable** — every module is small and documented.

## Installation

```bash
pip install -e .          # core
pip install -e ".[all]"   # everything (datasets, onnx, wandb, web, dev)
```

Requirements: Python ≥ 3.8, PyTorch ≥ 2.0, transformers ≥ 4.30.

## Architecture

A YouAI model is a pre-norm, decoder-only transformer. Each block is:

```
x = x + Attention(Norm(x))
x = x + FeedForward(Norm(x))
```

Configurable components:

| Component | Options | Default (modern) |
|-----------|---------|------------------|
| Positional info | `learned`, `rotary`, `alibi` | `rotary` |
| Normalisation | `layernorm`, `rmsnorm` | `rmsnorm` |
| MLP activation | `gelu`, `relu`, `silu`, `swiglu`, `geglu` | `swiglu` |
| Attention | multi-head or grouped-query (`num_key_value_heads`) | MHA |
| Attention kernel | fused SDPA ("flash") or manual | fused when available |
| Embedding tying | on/off (`tie_word_embeddings`) | on |

Generation uses a **key/value cache** — verified to produce token-for-token
identical results to the cacheless path under greedy decoding, at a fraction of
the cost.

## High-level API

All of the following are attributes of the top-level `youai` module.

### `create_model(preset="125m", modern=True, **overrides) -> YouAIModel`
Build a model from a preset. `overrides` may set any `YouAIConfig` field.

```python
model = youai.create_model("350m")
custom = youai.create_model("125m", num_key_value_heads=4, activation="gelu")
```

### `train(model, train_file, val_file=None, ...) -> dict`
Train a model end-to-end and return metrics. Key arguments:

| Argument | Default | Meaning |
|----------|---------|---------|
| `epochs` | 3 | Number of passes over the data |
| `batch_size` | 8 | Examples per step |
| `learning_rate` | 3e-4 | Peak LR (after warmup) |
| `max_length` | 512 | Sequence / block length |
| `mixed_precision` | `None` | `"fp16"` or `"bf16"` |
| `gradient_checkpointing` | `False` | Trade compute for memory |
| `gradient_accumulation_steps` | 1 | Simulate larger batches |
| `packing` | `True` | Pack tokens into full blocks (no padding waste) |
| `resume_from` | `None` | Checkpoint dir to resume from |
| `device` | `"auto"` | `"cuda"`, `"cpu"`, `"mps"` or `"auto"` |

Extra keyword arguments flow through to [`TrainingConfig`](#training).

### `generate(prompt, checkpoint_path, ...) -> list[str]`
Load a checkpoint and produce completions. Supports `max_new_tokens`,
`temperature`, `top_k`, `top_p`, `repetition_penalty`, `do_sample`,
`num_return_sequences`.

### `load_model(checkpoint_path, device="auto") -> YouAIInference`
Load a checkpoint once and reuse it for `generate` / `chat` / `perplexity`.

### `stream_generate(model_or_checkpoint, tokenizer=None, prompt="", ...)`
Generator yielding text fragments as they are produced.

### `download_dataset(name, output_dir, num_examples=None) -> (train, val)`
Download a curated HuggingFace dataset. Presets: `tinystories`, `wikipedia`,
`openwebtext`, `code`. Large corpora are streamed when `num_examples` is set.

### Other helpers
`create_sample_data`, `prepare_data`, `list_datasets`, `list_presets`,
`estimate_training`, `find_learning_rate`, `export_onnx`, `export_quantized`,
`benchmark_model`, `set_seed`, `set_log_level`.

## Configuration

```python
from youai import YouAIConfig

config = YouAIConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,        # grouped-query attention
    intermediate_size=3072,
    max_position_embeddings=2048,
    activation="swiglu",
    norm_type="rmsnorm",
    position_embedding_type="rotary",
    tie_word_embeddings=True,
)
print(config.total_params_formatted)   # e.g. "151.9M"
```

- `config.validate()` runs automatically and raises `ValueError` on bad values
  (e.g. `hidden_size` not divisible by `num_attention_heads`).
- `config.to_dict()` / `YouAIConfig.from_dict()` round-trip via JSON, and
  `from_dict` ignores unknown keys for forward compatibility.
- `get_preset_config(name, modern=True, **overrides)` returns a preset.

## Data pipeline

Two dataset strategies:

- **`LineTextDataset`** — one example per line, padded to `max_length`. Padding
  positions are set to `-100` in the labels so they are ignored by the loss.
- **`PackedTextDataset`** — concatenates the corpus into one token stream and
  slices it into full `block_size` blocks. No padding, no wasted compute. This
  is the default (`packing=True`).

```python
from youai.data import create_dataloaders
train_loader, val_loader = create_dataloaders(
    "train.txt", "val.txt", batch_size=8, max_length=512, packing=True,
)
```

## Training

`youai.Trainer` is the engine behind `youai.train`. Configure it with
`TrainingConfig`:

```python
from youai import Trainer, TrainingConfig

config = TrainingConfig(
    learning_rate=3e-4,
    num_epochs=3,
    warmup_ratio=0.03,
    lr_scheduler="cosine",          # cosine | linear | constant
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    eval_steps=200,
    save_steps=500,
    early_stopping_patience=5,
    save_total_limit=3,
    output_dir="./checkpoints",
)
trainer = Trainer(model, train_loader, val_loader, config, device="auto")
metrics = trainer.train()
```

Features: automatic mixed precision (correct fp16 gradient scaling), gradient
accumulation and clipping, warmup + decay LR schedule stepped every optimizer
step, periodic evaluation with best-checkpoint tracking, early stopping,
`save_total_limit` cleanup, optional Weights & Biases logging, and user
callbacks (`callbacks=[fn]`, called as `fn(trainer, metrics)`).

**Resuming** restores model, optimizer, scheduler, AMP scaler and step count:

```python
trainer.load_checkpoint("./checkpoints/step-500")
trainer.train()
```

## Generation & inference

```python
bot = youai.load_model("./checkpoints/final")

bot.generate("Hello", max_new_tokens=50, temperature=0.8, top_p=0.9,
             repetition_penalty=1.1, num_return_sequences=1, skip_prompt=True)
bot.chat("What is machine learning?", history=[])
bot.perplexity("The quick brown fox")   # lower is better
```

`GenerationConfig` controls decoding: `max_new_tokens`, `temperature`, `top_k`,
`top_p`, `repetition_penalty`, `do_sample`, `min_new_tokens`, `eos_token_id`.
Sampling is batch-correct (top-p uses `scatter`, not naive indexing).

## Streaming & chat

```python
from youai import StreamingGenerator, ChatSession

gen = StreamingGenerator(model, tokenizer)
for fragment in gen.stream("Once upon a time", max_new_tokens=100):
    print(fragment, end="", flush=True)

session = ChatSession(model, tokenizer, system_prompt="You are helpful.")
print(session.chat("Hi!"))
for fragment in session.chat_stream("Tell me a joke"):
    print(fragment, end="", flush=True)
```

Streaming decodes incrementally so multi-token characters render correctly, and
supports `stop_sequences`. An `async_stream` coroutine is provided for web apps.

## Export & quantization

```python
from youai.export import ModelExporter, ModelQuantizer, benchmark_model

ModelExporter(model).export_onnx("model.onnx")          # cross-platform
ModelExporter(model).export_torchscript("model.pt")     # production PyTorch

q = ModelQuantizer(model)
q.save(q.dynamic_quantize("qint8"), "model-int8")        # ~4x smaller on CPU

benchmark_model(model, device="cpu", num_runs=50)        # latency / throughput
```

Export wraps the model so it returns a plain logits tensor — the reason the
previous dict-returning `forward` could not be traced.

## Command-line interface

```
youai info [--preset NAME] [--classic]
youai datasets
youai train --preset 125m [--dataset tinystories | --data FILE]
            [--epochs N] [--batch-size N] [--grad-accum N]
            [--mixed-precision fp16|bf16] [--gradient-checkpointing]
            [--no-packing] [--resume DIR] [--estimate] [--seed N]
youai generate --checkpoint DIR --prompt TEXT [--max-new-tokens N]
               [--temperature F] [--top-k N] [--top-p F]
               [--repetition-penalty F] [--num-sequences N]
youai chat --checkpoint DIR
youai export --checkpoint DIR --format onnx|torchscript|quantized [--output PATH]
youai benchmark --checkpoint DIR [--device cpu|cuda] [--runs N]
```

## Utilities

```python
youai.set_seed(42)                 # Python + NumPy + torch RNGs
youai.set_log_level("DEBUG")       # or set env YOUAI_LOG_LEVEL
from youai.utils import resolve_device, count_parameters, human_bytes
```

## FAQ

**Do I need a GPU?** No — everything runs on CPU (tiny presets train in
seconds). A GPU is strongly recommended for the 125M+ presets.

**Which tokenizer is used?** The GPT-2 byte-pair tokenizer (`vocab_size=50257`),
loaded and cached via `youai.tokenizer.get_tokenizer`.

**Can I resume an interrupted run?** Yes — pass `resume_from=` to `youai.train`
or call `trainer.load_checkpoint(dir)`.

**How do I train on my own text?** Put your text in files and call
`youai.prepare_data([...])`, or point `youai.train` at a plain text file (one
example per line).

**Is it tested?** Yes — `pytest` runs 45 tests covering config, model,
generation, data, trainer and inference, all on CPU with no network.
