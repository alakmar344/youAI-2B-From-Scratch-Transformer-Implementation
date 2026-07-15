import torch

from youai.data import LineTextDataset, PackedTextDataset, create_sample_dataset


class FakeTokenizer:
    """A deterministic offline tokenizer so data tests need no network."""

    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, **kw):
        return [ord(c) % 97 + 1 for c in text][:64]

    def __call__(self, text, max_length=32, truncation=True, padding="max_length", return_tensors="pt"):
        ids = self.encode(text)[:max_length]
        attn = [1] * len(ids)
        if padding == "max_length":
            pad = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad
            attn = attn + [0] * pad
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([attn])}


def _corpus(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("\n".join(f"line {i}" for i in range(50)))
    return str(path)


def test_line_dataset_masks_padding(tmp_path):
    ds = LineTextDataset(_corpus(tmp_path), FakeTokenizer(), max_length=32)
    item = ds[0]
    assert item["input_ids"].shape == (32,)
    # Padding positions must be ignored in the loss.
    assert (item["labels"] == -100).any()
    assert torch.equal(item["labels"] == -100, item["attention_mask"] == 0)


def test_packed_dataset_has_no_padding(tmp_path):
    ds = PackedTextDataset(_corpus(tmp_path), FakeTokenizer(), block_size=16)
    assert len(ds) > 0
    item = ds[0]
    assert item["input_ids"].shape == (16,)
    assert (item["labels"] == -100).sum() == 0


def test_create_sample_dataset(tmp_path):
    train, val = create_sample_dataset(str(tmp_path), num_examples=40)
    assert open(train).read().strip()
    assert open(val).read().strip()
