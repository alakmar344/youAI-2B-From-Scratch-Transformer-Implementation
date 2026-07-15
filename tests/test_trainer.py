import torch
from torch.utils.data import DataLoader, TensorDataset

from youai.config import YouAIConfig
from youai.model import YouAIModel
from youai.trainer import Trainer, TrainingConfig, estimate_training_time


class DictLoaderDataset(torch.utils.data.Dataset):
    """A small, *learnable* dataset: a handful of fixed sequences the model can
    memorise, so training genuinely reduces the loss."""

    def __init__(self, n=32, seq=8, vocab=64):
        torch.manual_seed(1234)
        patterns = torch.randint(0, vocab, (4, seq))
        self.data = patterns[torch.arange(n) % 4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = self.data[i]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _model():
    return YouAIModel(YouAIConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32,
    ))


def _loader():
    return DataLoader(DictLoaderDataset(), batch_size=4)


def test_training_reduces_loss():
    model = _model()
    cfg = TrainingConfig(num_epochs=3, learning_rate=5e-3, log_steps=1000,
                         eval_steps=10_000, save_steps=10_000, output_dir="/tmp/youai_test_ck")
    trainer = Trainer(model, _loader(), device="cpu", config=cfg)

    loader = _loader()
    batch = next(iter(loader))
    with torch.no_grad():
        before = model(batch["input_ids"], labels=batch["labels"])["loss"].item()
    trainer.train()
    with torch.no_grad():
        after = model(batch["input_ids"], labels=batch["labels"])["loss"].item()
    assert after < before


def test_checkpoint_save_and_resume(tmp_path):
    model = _model()
    cfg = TrainingConfig(num_epochs=1, output_dir=str(tmp_path), log_steps=1000,
                         eval_steps=10_000, save_steps=10_000)
    trainer = Trainer(model, _loader(), device="cpu", config=cfg)
    trainer.train()
    assert (tmp_path / "final" / "pytorch_model.bin").exists()

    model2 = _model()
    trainer2 = Trainer(model2, _loader(), device="cpu", config=cfg)
    trainer2.load_checkpoint(str(tmp_path / "final"))
    assert trainer2.global_step == trainer.global_step


def test_grad_accumulation_runs(tmp_path):
    model = _model()
    cfg = TrainingConfig(num_epochs=1, gradient_accumulation_steps=2, output_dir=str(tmp_path),
                         log_steps=1000, eval_steps=10_000, save_steps=10_000)
    trainer = Trainer(model, _loader(), device="cpu", config=cfg)
    metrics = trainer.train()
    assert metrics["global_step"] > 0


def test_estimate_training_time():
    est = estimate_training_time(_model(), dataset_size=10_000, batch_size=8, num_epochs=3)
    assert est["total_steps"] > 0 and est["estimated_time_hours"] >= 0
