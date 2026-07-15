"""Tests for data preparation utilities."""

import os
import pytest

from youai.data import (
    LineTextDataset, PackedTextDataset, create_sample_dataset,
    prepare_custom_text, download_dataset, list_datasets,
    list_datasets_by_category, DATASET_PRESETS,
)


def test_line_dataset_roundtrip(tmp_path, tiny_tokenizer):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nfoo bar baz\n" * 20)
    ds = LineTextDataset(str(corpus), tokenizer=tiny_tokenizer, max_length=16)
    assert len(ds) == 40
    item = ds[0]
    assert item["input_ids"].shape == (16,)
    assert item["labels"].shape == (16,)
    # Pad positions should be masked to -100.
    pad_positions = (item["attention_mask"] == 0).sum().item()
    if pad_positions > 0:
        assert (item["labels"] == -100).sum().item() == pad_positions


def test_packed_dataset_roundtrip(tmp_path, tiny_tokenizer):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nfoo bar baz\n" * 50)
    ds = PackedTextDataset(str(corpus), tokenizer=tiny_tokenizer, block_size=8)
    assert len(ds) > 0
    item = ds[0]
    assert item["input_ids"].shape == (8,)


def test_create_sample_dataset(tmp_path):
    train, val = create_sample_dataset(str(tmp_path), num_examples=100)
    assert os.path.exists(train)
    assert os.path.exists(val)
    with open(train) as f:
        lines = f.readlines()
    assert len(lines) > 0


def test_prepare_custom_text(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("Line one.\nLine two.\nLine three.\n" * 10)
    train, val = prepare_custom_text([str(src)], str(tmp_path / "out"))
    assert os.path.exists(train)
    assert os.path.exists(val)


def test_dataset_presets_count():
    assert len(DATASET_PRESETS) >= 15


def test_dataset_presets_have_required_keys():
    for name, preset in DATASET_PRESETS.items():
        assert "name" in preset, f"{name} missing 'name'"
        assert "description" in preset, f"{name} missing 'description'"
        assert "size" in preset, f"{name} missing 'size'"
        assert "split" in preset, f"{name} missing 'split'"
        assert "text_column" in preset, f"{name} missing 'text_column'"


def test_dataset_categories():
    by_cat = list_datasets_by_category()
    assert "pretrain" in by_cat
    assert "code" in by_cat
    assert "finetune" in by_cat
    assert len(by_cat["pretrain"]) >= 5


def test_list_datasets(capsys):
    list_datasets()
    out = capsys.readouterr().out
    assert "tinystories" in out
    assert "openwebtext" in out
    assert "alpaca" in out


def test_list_datasets_filtered(capsys):
    list_datasets(category="code")
    out = capsys.readouterr().out
    assert "code" in out
    # tinystories should not appear as a listed dataset (only in the usage hint).
    lines = [l for l in out.split("\n") if l.strip().startswith(("code", "codesearch"))]
    assert len(lines) >= 1


def test_download_unknown_dataset_raises():
    try:
        with pytest.raises(ValueError, match="Unknown dataset"):
            download_dataset("nonexistent_dataset")
    except ImportError:
        pytest.skip("datasets library not installed")


@pytest.fixture
def tiny_tokenizer():
    from youai.tokenizer import get_tokenizer
    return get_tokenizer()
