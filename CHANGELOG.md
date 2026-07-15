# Changelog

## 1.0.0

A ground-up rewrite of the core library: modern architecture, correctness fixes
and a real test suite.

### Added
- **Modern architecture** — rotary (RoPE), learned and ALiBi positional
  embeddings; RMSNorm and LayerNorm; SwiGLU / GEGLU / GELU / SiLU / ReLU MLPs;
  grouped-query attention (`num_key_value_heads`).
- **Fused ("flash") attention** via `torch.nn.functional.scaled_dot_product_attention`,
  with a correct manual fallback that produces identical results.
- **KV-cache generation** — verified token-for-token identical to the cacheless
  path under greedy decoding.
- **Weight tying**, scaled residual initialisation, working gradient checkpointing.
- **`save_pretrained` / `from_pretrained`** on the model.
- **Shared, batch-correct sampling** (`youai.generation`) with repetition
  penalty, greedy/sampled decoding, `min_new_tokens` and EOS handling.
- **Token packing** dataset (`PackedTextDataset`) — trains on full blocks with
  no wasted padding.
- **Consolidated trainer** with mixed precision (new `torch.amp` API), gradient
  accumulation, warmup + cosine/linear/constant schedules, best-checkpoint
  tracking, early stopping, `save_total_limit`, callbacks and full resume.
- **Streaming** with incremental decoding and stop sequences; `ChatSession`.
- **Export** to ONNX / TorchScript / quantized, plus GGUF metadata and
  benchmarking.
- **Utilities** — `set_seed`, device resolution, structured logging.
- **New presets** `nano`, `micro`, `1.3b`; accurate parameter counting.
- **`pytest` suite** — 45 tests covering config, model, generation, data,
  trainer and inference; runs on CPU with no network.
- `pyproject.toml`, richer `setup.py`, `CHANGELOG.md`.

### Fixed
- **Context overflow** — generating past `max_position_embeddings` crashed; the
  context is now truncated (and `forward` raises a clear error).
- **Batched top-p sampling bug** — nucleus filtering removed the wrong tokens
  for batch sizes > 1; now implemented with `scatter`.
- **Padding leaked into the loss** — pad positions are masked to `-100`.
- **CLI `train` crash** — the trainer was constructed with a `None` dataloader.
- **ONNX / TorchScript export** — tracing failed on the dict-returning `forward`;
  a logits wrapper fixes it.
- **Hard-coded EOS id (50256)** — the model now uses `config.eos_token_id`.
- Removed deprecated `torch.cuda.amp` usage.

### Changed
- `training_advanced.MixedPrecisionTrainer` is now an alias of the consolidated
  `youai.Trainer` (kept for backward compatibility).
- `generate` defaults to `max_new_tokens` semantics (with `max_length` alias).
