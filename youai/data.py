"""Data preparation utilities.

Two dataset strategies are provided:

* :class:`LineTextDataset` — one training example per line, padded/truncated to a
  fixed length.  Padding positions are masked out of the loss (``-100``), fixing
  a bug in the previous version where the model was trained to predict pad tokens.
* :class:`PackedTextDataset` — concatenates the whole corpus into one token
  stream and slices it into contiguous, fully-utilised blocks.  This is how
  modern LMs are trained and wastes no compute on padding.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import get_tokenizer
from .utils import get_logger

logger = get_logger()


# Curated HuggingFace datasets that work well for from-scratch training.
DATASET_PRESETS = {
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "description": "Simple short stories — ideal for fast experiments and testing.",
        "size": "~2GB",
        "split": "train",
        "text_column": "text",
    },
    "wikipedia": {
        "name": "wikitext",
        "subconfig": "wikitext-103-raw-v1",
        "description": "Wikipedia articles (wikitext-103) — encyclopedic knowledge.",
        "size": "~500MB",
        "split": "train",
        "text_column": "text",
    },
    "openwebtext": {
        "name": "Skylion007/openwebtext",
        "description": "Web text from Reddit-linked pages — broad general knowledge.",
        "size": "~40GB",
        "split": "train",
        "text_column": "text",
    },
    "code": {
        "name": "codeparrot/codeparrot-clean-valid",
        "description": "Clean Python source code — for training a code model.",
        "size": "~2GB",
        "split": "train",
        "text_column": "content",
    },
}


class LineTextDataset(Dataset):
    """One example per line, padded to ``max_length``. Pads are ignored in loss."""

    def __init__(self, file_path: str, tokenizer=None, max_length: int = 512):
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_length = max_length
        with open(file_path, "r", encoding="utf-8") as f:
            self.examples = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.examples[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # do not learn to predict padding
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class PackedTextDataset(Dataset):
    """Concatenates the corpus and slices it into contiguous ``block_size`` chunks."""

    def __init__(self, file_path: str, tokenizer=None, block_size: int = 512):
        self.tokenizer = tokenizer or get_tokenizer()
        self.block_size = block_size

        eos = self.tokenizer.eos_token_id
        token_stream: List[int] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_stream.extend(self.tokenizer.encode(line))
                token_stream.append(eos)

        n_blocks = max(0, (len(token_stream) - 1) // block_size)
        usable = n_blocks * block_size + 1
        self.tokens = torch.tensor(token_stream[:usable], dtype=torch.long)
        self.n_blocks = n_blocks
        if n_blocks == 0:
            logger.warning(
                "Corpus is smaller than one block (%d tokens); consider LineTextDataset.",
                len(token_stream),
            )

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.block_size
        input_ids = self.tokens[start: start + self.block_size].contiguous()
        # The model shifts logits/labels internally, so aligned labels give the
        # standard next-token objective with no wasted padding.
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


# Backwards-compatible alias.
TextDataset = LineTextDataset


def create_dataloaders(
    train_file: str,
    val_file: Optional[str] = None,
    tokenizer=None,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0,
    device: str = "cuda",
    packing: bool = False,
    shuffle: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training / validation ``DataLoader``s.

    Args:
        train_file: Path to the training text file (one example per line).
        val_file: Optional validation text file.
        tokenizer: Tokenizer to use (defaults to GPT-2).
        batch_size: Batch size.
        max_length: Sequence / block length.
        num_workers: DataLoader workers.
        device: Used to decide whether to pin memory.
        packing: Use :class:`PackedTextDataset` (no wasted padding) instead of
            :class:`LineTextDataset`.
        shuffle: Shuffle the training data.
    """
    tokenizer = tokenizer or get_tokenizer()
    ds_cls = PackedTextDataset if packing else LineTextDataset
    kw = {"block_size": max_length} if packing else {"max_length": max_length}

    pin = str(device).startswith("cuda") and torch.cuda.is_available()
    train_ds = ds_cls(train_file, tokenizer, **kw)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )

    val_loader = None
    if val_file and os.path.exists(val_file):
        val_ds = ds_cls(val_file, tokenizer, **kw)
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin,
            )
    return train_loader, val_loader


# ----------------------------------------------------------------------
# Sample / custom / HuggingFace data preparation
# ----------------------------------------------------------------------
def _write_split(texts: List[str], output_dir: str, prefix: str, train_split: float) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(texts)
    split = int(len(texts) * train_split)
    train_file = os.path.join(output_dir, f"{prefix}_train.txt")
    val_file = os.path.join(output_dir, f"{prefix}_val.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(texts[:split]) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join(texts[split:]) + "\n")
    return train_file, val_file


def create_sample_dataset(output_dir: str = "./data", num_examples: int = 10000) -> Tuple[str, str]:
    """Generate a small synthetic corpus for smoke-testing the pipeline."""
    templates = [
        "The importance of {topic} cannot be overstated in modern {field}.",
        "Recent advances in {topic} have transformed how we approach {concept}.",
        "Researchers found that {topic} plays a crucial role in {process}.",
        "Understanding {topic} is essential for anyone working in {field}.",
        "Experts agree that {topic} will keep shaping the future of {field}.",
        "{topic} has emerged as a key factor in understanding {concept}.",
        "New studies suggest {topic} matters more than we previously thought.",
        "The interplay between {topic} and {concept} rewards careful {process}.",
    ]
    topics = ["artificial intelligence", "climate change", "quantum computing", "biotechnology",
              "renewable energy", "space exploration", "nanotechnology", "genetics", "robotics",
              "neuroscience", "cryptography", "materials science", "ecology", "psychology"]
    fields = ["science", "technology", "medicine", "education", "industry", "research",
              "engineering", "business", "society", "the environment"]
    concepts = ["innovation", "sustainability", "efficiency", "progress", "discovery",
                "understanding", "resilience", "evolution", "transformation"]
    processes = ["learning", "adaptation", "analysis", "communication", "experimentation",
                 "collaboration", "optimization", "reasoning"]

    def gen(n):
        out = []
        for _ in range(n):
            out.append(random.choice(templates).format(
                topic=random.choice(topics), field=random.choice(fields),
                concept=random.choice(concepts), process=random.choice(processes)))
        return out

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "sample_train.txt")
    val_file = os.path.join(output_dir, "sample_val.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(gen(num_examples)) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join(gen(max(1, num_examples // 10))) + "\n")
    logger.info("Wrote sample data: %s / %s", train_file, val_file)
    return train_file, val_file


def prepare_custom_text(text_files: List[str], output_dir: str = "./data",
                        train_split: float = 0.9) -> Tuple[str, str]:
    """Prepare train/val splits from one or more raw text files."""
    all_lines: List[str] = []
    for path in text_files:
        with open(path, "r", encoding="utf-8") as f:
            all_lines.extend(line.strip() for line in f if line.strip())
    if not all_lines:
        raise ValueError("No non-empty lines found in the provided text files.")
    return _write_split(all_lines, output_dir, "custom", train_split)


def list_datasets() -> None:
    """Print the available HuggingFace dataset presets."""
    print("\nAvailable HuggingFace datasets")
    print("=" * 60)
    for key, info in DATASET_PRESETS.items():
        print(f"  {key:<12} {info['size']:>7}  {info['description']}")
    print("=" * 60)
    print("Usage: train_file, val_file = youai.download_dataset('tinystories')\n")


def download_dataset(dataset: str, output_dir: str = "./data",
                     num_examples: Optional[int] = None,
                     train_split: float = 0.9,
                     min_chars: int = 16) -> Tuple[str, str]:
    """Download a HuggingFace dataset preset and materialise train/val text files."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the datasets library: pip install datasets") from exc

    if dataset not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_PRESETS)}")

    preset = DATASET_PRESETS[dataset]
    column = preset.get("text_column", "text")
    logger.info("Downloading %s (%s): %s", dataset, preset["size"], preset["description"])

    load_kwargs = {"split": preset["split"]}
    if num_examples:
        # Stream to avoid downloading a 40GB corpus just to keep 50k rows.
        load_kwargs["streaming"] = True

    if "subconfig" in preset:
        hf = load_dataset(preset["name"], preset["subconfig"], **load_kwargs)
    else:
        hf = load_dataset(preset["name"], **load_kwargs)

    texts: List[str] = []
    if num_examples:
        for i, example in enumerate(hf):
            if i >= num_examples:
                break
            text = str(example.get(column, "")).strip().replace("\n", " ")
            if len(text) >= min_chars:
                texts.append(text)
    else:
        for example in hf:
            text = str(example.get(column, "")).strip().replace("\n", " ")
            if len(text) >= min_chars:
                texts.append(text)

    if not texts:
        raise RuntimeError(f"No usable text extracted from dataset '{dataset}'.")

    train_file, val_file = _write_split(texts, output_dir, dataset, train_split)
    logger.info("Dataset ready: %d train / %d val examples",
                int(len(texts) * train_split), len(texts) - int(len(texts) * train_split))
    return train_file, val_file
