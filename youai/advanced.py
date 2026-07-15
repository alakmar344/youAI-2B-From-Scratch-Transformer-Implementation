"""Performance and deployment utilities.

* **torch.compile** — free speedup via graph compilation.
* **Model card generation** — HuggingFace-compatible model cards.
* **Training visualization** — ASCII training curves.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .utils import get_logger, format_count

logger = get_logger()


# ======================================================================
# torch.compile integration
# ======================================================================
def compile_model(
    model: nn.Module,
    mode: str = "default",
    backend: str = "inductor",
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
) -> nn.Module:
    """Compile a model with torch.compile for faster inference/training.

    Args:
        model: The model to compile.
        mode: Compilation mode:
            - ``"default"``: balanced compile time and runtime performance
            - ``"reduce-overhead"``: minimize overhead (good for small batches)
            - ``"max-autotune"``: maximize performance (long compile time)
        backend: Compilation backend (``"inductor"``, ``"cudagraphs"``).
        fullgraph: Capture the full graph (fails on dynamic shapes).
        dynamic: Use dynamic shapes (None = auto-detect).

    Returns:
        The compiled model (same object, compiled in-place).

    Example::

        model = youai.create_model("125m")
        model = youai.compile_model(model, mode="reduce-overhead")
        # Now inference is faster!
    """
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile requires PyTorch >= 2.0. Skipping compilation.")
        return model

    try:
        compiled = torch.compile(
            model,
            mode=mode,
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        logger.info("Model compiled with mode=%s, backend=%s", mode, backend)
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed: %s. Returning uncompiled model.", e)
        return model


# ======================================================================
# Model card generation
# ======================================================================
def generate_model_card(
    model: nn.Module,
    output_path: str,
    model_name: str = "youai-model",
    base_model: Optional[str] = None,
    dataset: Optional[str] = None,
    license: str = "apache-2.0",
    language: List[str] = None,
    tags: List[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> str:
    """Generate a HuggingFace-compatible model card (README.md).

    Args:
        model: The model.
        output_path: Path to write the model card.
        model_name: Name of the model.
        base_model: Base model name (if fine-tuned).
        dataset: Dataset used for training.
        license: License identifier.
        language: List of supported languages.
        tags: List of tags.
        metrics: Evaluation metrics.
        training_config: Training configuration.
        description: Model description.

    Returns:
        The generated model card text.
    """
    config = model.config if hasattr(model, "config") else None
    num_params = sum(p.numel() for p in model.parameters())

    language = language or ["en"]
    tags = tags or ["youai", "language-model", "text-generation"]
    if config and hasattr(config, "architecture_family"):
        tags.append(config.architecture_family)

    lines = [
        "---",
        f"language: {json.dumps(language)}",
        f"license: {license}",
        f"tags: {json.dumps(tags)}",
        "---",
        "",
        f"# {model_name}",
        "",
    ]

    if description:
        lines.append(description)
        lines.append("")

    lines.extend([
        "## Model Details",
        "",
        f"- **Parameters**: {format_count(num_params)} ({num_params:,})",
    ])

    if config:
        lines.extend([
            f"- **Architecture**: {config.architecture_family if hasattr(config, 'architecture_family') else 'transformer'}",
            f"- **Hidden size**: {config.hidden_size}",
            f"- **Layers**: {config.num_hidden_layers}",
            f"- **Attention heads**: {config.num_attention_heads}",
            f"- **Vocab size**: {config.vocab_size}",
            f"- **Max sequence length**: {config.max_position_embeddings}",
        ])

    if base_model:
        lines.append(f"- **Base model**: {base_model}")
    if dataset:
        lines.append(f"- **Training data**: {dataset}")

    lines.extend(["", "## Usage", "", "```python", "import youai", ""])

    if base_model:
        lines.append(f"# Load the model")
        lines.append(f"model = youai.from_pretrained('{model_name}')")
    else:
        lines.append(f"# Create the model")
        lines.append(f"model = youai.create_model('{model_name}')")

    lines.extend([
        "",
        "# Generate text",
        "from youai.inference import YouAIInference",
        "inferencer = YouAIInference.from_model(model)",
        "print(inferencer.generate('Hello, I am')[0])",
        "",
        "# Chat",
        "print(inferencer.chat('What is machine learning?'))",
        "```",
        "",
    ])

    if metrics:
        lines.extend(["## Evaluation Metrics", ""])
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append(f"| {key}/{sub_key} | {sub_value} |")
            else:
                lines.append(f"| {key} | {value} |")
        lines.append("")

    if training_config:
        lines.extend(["## Training Configuration", "", "```json"])
        lines.append(json.dumps(training_config, indent=2))
        lines.extend(["```", ""])

    lines.extend([
        "## License",
        "",
        f"This model is released under the {license} license.",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@software{youai,",
        "  author = {YouAI Contributors},",
        f"  title = {{{model_name}}},",
        f"  year = {{{datetime.now().year}}},",
        "  url = {https://github.com/youai/youai}",
        "}",
        "```",
        "",
    ])

    card = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card)

    logger.info("Generated model card: %s", output_path)
    return card


# ======================================================================
# Training visualization
# ======================================================================
def plot_training_curve(
    metrics: List[Dict[str, float]],
    metric_key: str = "train/loss",
    width: int = 60,
    height: int = 15,
) -> str:
    """Generate an ASCII training curve plot.

    Args:
        metrics: List of metric dicts (from training callbacks).
        metric_key: Which metric to plot.
        width: Plot width in characters.
        height: Plot height in characters.

    Returns:
        ASCII art string of the training curve.
    """
    values = [m.get(metric_key) for m in metrics if metric_key in m]
    if not values:
        return f"No data for metric '{metric_key}'"

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val > min_val else 1.0

    # Create the plot grid.
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Map values to grid positions.
    n = len(values)
    for i, val in enumerate(values):
        x = int(i * (width - 1) / max(1, n - 1))
        y = int((val - min_val) / val_range * (height - 1))
        y = height - 1 - y  # Invert y-axis.
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = "█"

    # Build the output.
    lines = []
    lines.append(f"  {metric_key}")
    lines.append(f"  {max_val:.4f} ┤{''.join(grid[0])}")
    for row in grid[1:-1]:
        mid_val = max_val - (max_val - min_val) * (grid.index(row) / (height - 1))
        lines.append(f"  {'':>8} │{''.join(row)}")
    lines.append(f"  {min_val:.4f} ┤{''.join(grid[-1])}")
    lines.append(f"  {'':>8} └{'─' * width}")
    lines.append(f"  {'':>8}  step 0{' ' * (width - 10)}step {n}")

    return "\n".join(lines)
