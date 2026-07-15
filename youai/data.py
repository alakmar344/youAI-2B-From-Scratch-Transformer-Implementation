"""Data preparation utilities.

Two dataset strategies are provided:

* :class:`LineTextDataset` — one training example per line, padded/truncated to a
  fixed length.  Padding positions are masked out of the loss (``-100``), fixing
  a bug in the previous version where the model was trained to predict pad tokens.
* :class:`PackedTextDataset` — concatenates the whole corpus into one token
  stream and slices it into contiguous, fully-utilised blocks.  This is how
  modern LMs are trained and wastes no compute on padding.

Includes 20 curated HuggingFace dataset presets for from-scratch training,
fine-tuning, and evaluation.
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


# ======================================================================
# Curated HuggingFace datasets
# ======================================================================
# Each entry has: name, subconfig (optional), description, size, split,
# text_column, and category (pretrain, finetune, code, science, chat).
DATASET_PRESETS = {
    # ---- General pre-training ----
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "description": "Simple short stories — ideal for fast experiments and testing.",
        "size": "~2GB",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },
    "wikipedia": {
        "name": "wikitext",
        "subconfig": "wikitext-103-raw-v1",
        "description": "Wikipedia articles (wikitext-103) — encyclopedic knowledge.",
        "size": "~500MB",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },
    "openwebtext": {
        "name": "Skylion007/openwebtext",
        "description": "Web text from Reddit-linked pages — broad general knowledge.",
        "size": "~40GB",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },
    "c4": {
        "name": "allenai/c4",
        "subconfig": "en",
        "description": "Colossal Clean Crawled Corpus — cleaned web text used to train T5.",
        "size": "~750GB (stream)",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },
    "slimpajama": {
        "name": "cerebras/SlimPajama-627B",
        "description": "Cleaned, deduplicated RedPajama — high-quality web-scale corpus.",
        "size": "~627GB (stream)",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },
    "redpajama": {
        "name": "togethercomputer/RedPajama-Data-1T",
        "description": "1T token open reproduction of LLaMA training data.",
        "size": "~1TB (stream)",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },
    "the_pile": {
        "name": "EleutherAI/the_pile",
        "description": "800GB diverse text corpus used to train GPT-NeoX, Pythia.",
        "size": "~800GB (stream)",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },
    "oscar": {
        "name": "oscar-corpus/OSCAR-2301",
        "subconfig": "en",
        "description": "Open Super-large Crawled ALMAnaCH corpus — multilingual web text.",
        "size": "~100GB (stream)",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },
    "bookcorpus": {
        "name": "bookcorpus/bookcorpus",
        "description": "Over 11,000 books — narrative prose for language understanding.",
        "size": "~5GB",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },

    # ---- Code ----
    "code": {
        "name": "codeparrot/codeparrot-clean-valid",
        "description": "Clean Python source code — for training a code model.",
        "size": "~2GB",
        "split": "train",
        "text_column": "content",
        "category": "code",
    },
    "codesearchnet": {
        "name": "code_search_net",
        "subconfig": "python",
        "description": "6M Python docstrings + code from GitHub.",
        "size": "~3GB",
        "split": "train",
        "text_column": "func_code_string",
        "category": "code",
    },
    "stackexchange": {
        "name": "HuggingFaceFW/fineweb-edu",
        "subconfig": "default",
        "description": "FineWeb-Edu — high-quality educational web pages.",
        "size": "~1.3TB (stream)",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },

    # ---- Instruction / fine-tuning ----
    "alpaca": {
        "name": "tatsu-lab/alpaca",
        "description": "52K instruction-following examples (Stanford Alpaca format).",
        "size": "~25MB",
        "split": "train",
        "text_column": "text",
        "category": "finetune",
    },
    "dolly": {
        "name": "databricks/databricks-dolly-15k",
        "description": "15K human-written instruction/response pairs from Databricks.",
        "size": "~12MB",
        "split": "train",
        "text_column": "instruction",
        "category": "finetune",
    },
    "sharegpt": {
        "name": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "description": "Real user conversations with ChatGPT — dialogue training.",
        "size": "~7GB (stream)",
        "split": "train",
        "text_column": "conversations",
        "category": "chat",
    },

    # ---- Science / medical ----
    "pubmed": {
        "name": "ccdv/pubmed-summarization",
        "description": "PubMed abstracts — biomedical text for domain-specific LMs.",
        "size": "~2GB",
        "split": "train",
        "text_column": "article",
        "category": "science",
    },
    "arxiv": {
        "name": "ccdv/arxiv-summarization",
        "description": "arXiv paper abstracts — scientific/technical text.",
        "size": "~3GB",
        "split": "train",
        "text_column": "article",
        "category": "science",
    },

    # ---- Multilingual ----
    "mc4": {
        "name": "allenai/c4",
        "subconfig": "en",
        "description": "Multilingual C4 — web text in 101 languages.",
        "size": "~750GB (stream)",
        "split": "train",
        "text_column": "text",
        "category": "pretrain",
    },
}


def list_datasets(category: Optional[str] = None) -> None:
    """Print the available HuggingFace dataset presets.

    Args:
        category: Filter by category (``pretrain``, ``code``, ``finetune``,
            ``chat``, ``science``). ``None`` shows all.
    """
    print("\nAvailable HuggingFace datasets")
    print("=" * 70)
    for key, info in DATASET_PRESETS.items():
        if category and info.get("category") != category:
            continue
        cat = info.get("category", "general")
        print(f"  {key:<18} [{cat:>8}]  {info['size']:>12}  {info['description']}")
    print("=" * 70)
    print("Usage: train_file, val_file = youai.download_dataset('tinystories')\n")


def list_datasets_by_category() -> dict:
    """Return datasets grouped by category."""
    result = {}
    for key, info in DATASET_PRESETS.items():
        cat = info.get("category", "general")
        result.setdefault(cat, []).append(key)
    return result


# ======================================================================
# Dataset classes
# ======================================================================
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


def download_dataset(dataset: str, output_dir: str = "./data",
                     num_examples: Optional[int] = None,
                     train_split: float = 0.9,
                     min_chars: int = 16) -> Tuple[str, str]:
    """Download a HuggingFace dataset preset and materialise train/val text files.

    Args:
        dataset: One of the dataset preset names (run ``youai.list_datasets()``).
        output_dir: Directory to write the text files.
        num_examples: Limit the number of examples (useful for large datasets).
        train_split: Fraction of data for training (rest is validation).
        min_chars: Minimum character length for an example to be included.

    Returns:
        Tuple of (train_file, val_file) paths.
    """
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
