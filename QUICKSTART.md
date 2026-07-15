# YouAI Quick Start Guide

Get from zero to a trained model, then scale up. Every step below uses the
`youai` library or CLI — no editing source files.

## Install

```bash
pip install -e .            # core
pip install -e ".[all]"     # + datasets / onnx / wandb / web
```

---

## Path 1 — Just testing (free, minutes, CPU)

Understand the whole pipeline on a laptop.

```python
import youai

youai.set_seed(42)
model = youai.create_model("nano")                     # ~7.5M params
train_file, val_file = youai.create_sample_data(2000)  # synthetic corpus
youai.train(model, train_file, val_file, epochs=1, device="cpu")
print(youai.generate("Science is", checkpoint_path="./checkpoints/final")[0])
```

Or entirely from the command line:

```bash
youai train --preset nano --num-examples 2000 --epochs 1 --device cpu
youai generate --checkpoint ./checkpoints/final --prompt "Science is"
```

---

## Path 2 — A real small model ($10–50, hours–days, one GPU)

Google Colab / a single T4 or better.

```python
import youai

youai.set_seed(42)
model = youai.create_model("125m")                     # ~152M params

# Download real data (streamed, capped for a quick run)
train_file, val_file = youai.download_dataset("tinystories", num_examples=200_000)

youai.train(
    model, train_file, val_file,
    epochs=1, batch_size=16, mixed_precision="bf16",
    gradient_accumulation_steps=4,
)

bot = youai.load_model("./checkpoints/final")
print(bot.chat("Tell me a short story."))
```

Estimate cost/time before committing:

```python
print(youai.estimate_training(model, dataset_size=200_000, batch_size=16))
```

---

## Path 3 — A capable 350M–750M model ($200–1000, one A100)

```bash
youai train --preset 350m --dataset openwebtext \
    --epochs 3 --batch-size 16 --grad-accum 4 \
    --mixed-precision bf16 --gradient-checkpointing \
    --output-dir ./ckpt-350m
```

- `--gradient-checkpointing` roughly halves activation memory.
- Resume any interrupted run: add `--resume ./ckpt-350m/step-XXXX`.
- Deploy: `pip install -e ".[web]" && python web_interface.py --checkpoint ./ckpt-350m/final`.

---

## Path 4 — Large models ($1000+, multi-GPU)

```python
model = youai.create_model("1.3b")   # or "2b"
```

Use a big streamed corpus, `mixed_precision="bf16"`,
`gradient_checkpointing=True`, and a large effective batch via
`gradient_accumulation_steps`. Save often (`save_steps`) and rely on `--resume`.

---

## Model size reference

| Preset | Params | Typical GPU memory | Good for |
|--------|-------:|-------------------:|----------|
| nano   | 7.5M   | CPU / <1GB | Learning, tests |
| micro  | 19M    | <2GB | Quick experiments |
| 125m   | 152M   | ~8GB | First real model |
| 350m   | 454M   | ~16GB | Real applications |
| 750m   | 983M   | ~24GB | Strong performance |
| 1.3b   | 1.7B   | ~32GB | Serious projects |
| 2b     | 3.5B   | 40GB+ | Production quality |

---

## Healthy training signals

- ✅ Loss decreases steadily; perplexity falls from 1000+ toward <100.
- ✅ Generated text becomes word-like, then sentence-like.
- ❌ Loss = NaN or rising → lower `learning_rate`.
- ❌ Loss flat after thousands of steps → check data quality / raise LR.

## Common fixes

| Problem | Fix |
|---------|-----|
| Out of memory | Lower `batch_size`, enable `gradient_checkpointing`, reduce `max_length` |
| Too slow | Raise `batch_size`, use `mixed_precision="bf16"`, keep `packing=True` |
| Loss not dropping | Lower `learning_rate` (e.g. 1e-4), increase `warmup_ratio`, verify data |

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full API.
