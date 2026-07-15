"""YouAI — train your own language model from scratch.

A small, professional, batteries-included library for building, training,
serving and exporting transformer language models with a friendly API.

Quick start::

    import youai

    # From scratch
    model = youai.create_model("125m")
    train_file, val_file = youai.create_sample_data(1000)
    youai.train(model, train_file=train_file, epochs=1)

    # Load any pretrained model (15+ families)
    model = youai.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = youai.from_pretrained("Qwen/Qwen2-7B")
    model = youai.from_pretrained("gpt2")

    # LoRA fine-tune in 5 lines
    model = youai.from_pretrained("gpt2")
    youai.train(model, train_file=train_file, lora=True, epochs=1)

    # Serve with streaming
    youai.serve(pretrained="gpt2", port=8000)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .config import (
    YouAIConfig, get_preset_config, list_presets,
    list_architecture_families, list_family_presets, create_from_family,
    ARCHITECTURE_FAMILIES,
)
from .model import YouAIModel
from .generation import GenerationConfig
from .trainer import Trainer, TrainingConfig, estimate_training_time
from .inference import YouAIInference
from .streaming import StreamingGenerator, ChatSession
from .tokenizer import get_tokenizer
from .pretrained import (
    from_pretrained, from_pretrained_gpt2,
    GPT2_VARIANTS, SUPPORTED_FAMILIES, POPULAR_MODELS,
)
from .lora import (
    apply_lora, apply_qlora, merge_lora, save_lora, load_lora,
    trainable_parameter_summary, LoRALinear,
)
from .utils import set_seed, resolve_device, get_logger, set_log_level, format_count
from .data import (
    TextDataset,
    LineTextDataset,
    PackedTextDataset,
    create_sample_dataset,
    prepare_custom_text,
    create_dataloaders,
    download_dataset,
    list_datasets,
    list_datasets_by_category,
    DATASET_PRESETS,
)
from .export import (
    export_model, list_export_formats,
    ModelExporter, ModelQuantizer, EXPORT_FORMATS,
    benchmark_model as _benchmark_model_fn,
)

__version__ = "2.0.0"

__all__ = [
    # High-level functions
    "create_model", "from_pretrained", "from_pretrained_gpt2",
    "train", "generate", "load_model", "chat",
    "create_sample_data", "prepare_data", "download_dataset", "list_datasets",
    "list_datasets_by_category",
    "list_presets", "list_family_presets", "list_architecture_families",
    "create_from_family",
    "estimate_training", "find_learning_rate",
    "stream_generate", "export_model", "list_export_formats",
    "benchmark_model",
    "set_seed", "set_log_level",
    # Pretrained + LoRA
    "from_pretrained", "from_pretrained_gpt2",
    "GPT2_VARIANTS", "SUPPORTED_FAMILIES", "POPULAR_MODELS",
    "apply_lora", "apply_qlora", "merge_lora", "save_lora", "load_lora",
    "trainable_parameter_summary", "LoRALinear", "serve",
    # Export
    "export_model", "list_export_formats", "EXPORT_FORMATS",
    "ModelExporter", "ModelQuantizer",
    # Classes / config
    "YouAIConfig", "YouAIModel", "YouAIInference", "Trainer", "TrainingConfig",
    "GenerationConfig", "StreamingGenerator", "ChatSession",
    "TextDataset", "LineTextDataset", "PackedTextDataset",
    "get_tokenizer", "get_preset_config", "create_from_family",
    "ARCHITECTURE_FAMILIES",
    "__version__",
]


# ----------------------------------------------------------------------
# Model creation
# ----------------------------------------------------------------------
def create_model(preset: str = "125m", modern: bool = True, **kwargs) -> YouAIModel:
    """Create a model from a preset, with optional config overrides.

    Args:
        preset: A size preset (``nano``, ``micro``, ``125m``, ``350m``,
            ``750m``, ``1.3b``, ``2b``, ``3b``, ``7b``, ``13b``, ``34b``,
            ``70b``) or a family preset (``llama2-7b``, ``qwen2-7b``,
            ``mistral-7b``, ``phi-2``, etc.).
        modern: Use modern architecture defaults (RoPE + RMSNorm + SwiGLU).
            Ignored for family presets.
        **kwargs: Any :class:`YouAIConfig` field to override.
    """
    config = get_preset_config(preset, modern=modern, **kwargs)
    return YouAIModel(config)


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def create_sample_data(num_examples: int = 10000, output_dir: str = "./data") -> Tuple[str, str]:
    """Create a small synthetic dataset for testing the pipeline."""
    return create_sample_dataset(output_dir=output_dir, num_examples=num_examples)


def prepare_data(text_files: List[str], output_dir: str = "./data",
                 train_split: float = 0.9) -> Tuple[str, str]:
    """Prepare train/val splits from your own text files."""
    return prepare_custom_text(text_files, output_dir=output_dir, train_split=train_split)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(
    model: YouAIModel,
    train_file: str,
    val_file: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    max_length: int = 512,
    output_dir: str = "./checkpoints",
    device: str = "auto",
    mixed_precision: Optional[str] = None,
    gradient_checkpointing: bool = False,
    gradient_accumulation_steps: int = 1,
    packing: bool = True,
    resume_from: Optional[str] = None,
    lora: bool = False,
    qlora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
    **kwargs,
) -> dict:
    """Train (or fine-tune) a model end-to-end and return final metrics.

    Set ``lora=True`` (or ``qlora=True``) for parameter-efficient fine-tuning of
    a pretrained model — e.g. ``youai.from_pretrained("gpt2")``. When LoRA
    is used, the adapter is saved to ``<output_dir>/adapter`` and a merged,
    ready-to-deploy checkpoint to ``<output_dir>/merged``.

    Any extra keyword arguments are passed through to :class:`TrainingConfig`.
    """
    if lora or qlora:
        from .lora import apply_lora, apply_qlora, merge_lora, save_lora, trainable_parameter_summary

        target = tuple(lora_target_modules) if lora_target_modules else None
        kw = dict(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        if target:
            kw["target_modules"] = target
        (apply_qlora if qlora else apply_lora)(model, **kw)

    train_loader, val_loader = create_dataloaders(
        train_file=train_file, val_file=val_file, batch_size=batch_size,
        max_length=max_length, device=device, packing=packing,
    )
    config = TrainingConfig(
        num_epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        output_dir=output_dir, mixed_precision=mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps, **kwargs,
    )
    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    if resume_from:
        trainer.load_checkpoint(resume_from)
    metrics = trainer.train()

    if lora or qlora:
        from .lora import merge_lora, save_lora, trainable_parameter_summary
        import os

        metrics["trainable"] = trainable_parameter_summary(model)
        save_lora(model, os.path.join(output_dir, "adapter"),
                  meta={"r": lora_r, "alpha": lora_alpha})
        merge_lora(model)
        model.save_pretrained(os.path.join(output_dir, "merged"))
        metrics["adapter_dir"] = os.path.join(output_dir, "adapter")
        metrics["merged_dir"] = os.path.join(output_dir, "merged")
    return metrics


def estimate_training(model: YouAIModel, dataset_size: int, batch_size: int = 8,
                      num_epochs: int = 3, device: str = "cuda") -> dict:
    """Estimate training time, memory and cost."""
    return estimate_training_time(model, dataset_size, batch_size, num_epochs, device)


def find_learning_rate(model: YouAIModel, train_dataloader, **kwargs) -> float:
    """Suggest a learning rate using the LR-range test."""
    from .training_advanced import LearningRateFinder

    return LearningRateFinder(model, train_dataloader, **kwargs).find()


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
def load_model(checkpoint_path: str, device: str = "auto") -> YouAIInference:
    """Load a trained checkpoint for generation / chat."""
    return YouAIInference(checkpoint_path, device=device)


def generate(prompt: str, checkpoint_path: str = "./checkpoints/final",
             max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50,
             top_p: float = 0.9, repetition_penalty: float = 1.1,
             num_return_sequences: int = 1, device: str = "auto", **kwargs) -> List[str]:
    """Load a checkpoint and generate completion(s) for a prompt."""
    inferencer = YouAIInference(checkpoint_path, device=device)
    return inferencer.generate(
        prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, **kwargs,
    )


def chat(message: str, checkpoint_path: str = "./checkpoints/final",
         device: str = "auto", **kwargs) -> str:
    """One-shot chat helper."""
    return YouAIInference(checkpoint_path, device=device).chat(message, **kwargs)


def stream_generate(model_or_checkpoint, tokenizer=None, prompt: str = "",
                    device: str = "auto", **kwargs):
    """Stream tokens from a model instance or a checkpoint path."""
    if isinstance(model_or_checkpoint, str):
        inferencer = YouAIInference(model_or_checkpoint, device=device)
        model, tokenizer = inferencer.model, inferencer.tokenizer
    else:
        model = model_or_checkpoint
    generator = StreamingGenerator(model, tokenizer, device=device)
    yield from generator.stream(prompt, **kwargs)


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
def export_model(model: YouAIModel, format: str, output_path: str,
                 tokenizer=None, **kwargs) -> str:
    """Export a model to the specified format.

    Args:
        model: The model to export.
        format: One of: ``onnx``, ``torchscript``, ``safetensors``, ``gguf``,
            ``huggingface``, ``vllm``, ``int8``, ``fp16``, ``coreml``, ``openvino``.
        output_path: Output path or directory.
        tokenizer: Optional tokenizer (needed for some formats).

    Run ``youai.list_export_formats()`` to see all options.
    """
    from .export import export_model as _export
    return _export(model, format, output_path, tokenizer=tokenizer, **kwargs)


def benchmark_model(model: YouAIModel, device: str = "cpu", **kwargs) -> dict:
    """Benchmark model inference speed."""
    return _benchmark_model_fn(model, device=device, **kwargs)


def serve(checkpoint: str = None, host: str = "0.0.0.0", port: int = 8000,
          device: str = "auto", pretrained: str = None, **kwargs):
    """Launch the FastAPI inference server (blocking).

    Example::

        youai.serve(checkpoint="./checkpoints/final", port=8000)
        youai.serve(pretrained="gpt2")   # serve real GPT-2 out of the box
    """
    from .server import serve as _serve

    return _serve(checkpoint=checkpoint, host=host, port=port,
                  device=device, pretrained=pretrained, **kwargs)
