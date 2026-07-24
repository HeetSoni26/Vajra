import torch
import torch.nn as nn
from training.trainer import Trainer

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, input_ids, labels=None):
        out = self.linear(input_ids.float())
        loss = out.mean()
        return {"loss": loss}

def test_exploding_gradient_no_longer_aborts():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        max_grad_norm_threshold=1.0,
        grad_clip=1.0,
    )
    
    dummy_input = {
        "input_ids": torch.randn(2, 10) * 1000,
        "labels": torch.randn(2, 10) * 1000
    }
    
    trainer.accumulated_loss = 1.0 # mock
    
    # We will let train_step compute forward on its own, wait, train_step calls model(**batch)
    # So model needs to accept input_ids
    
    # Force a large gradient manually
    loss = model(dummy_input["input_ids"])["loss"]
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= 1000.0
            
    # train_step performs forward and backward itself, so we actually want to give it input that causes a large gradient
    # Or we can just run train_step which will do forward/backward. 
    # With dummy_input * 1000, gradient will be large.
    metrics = trainer.train_step(dummy_input, step=1, is_accum_step=True)
    
    assert metrics is not None
    assert "grad_norm" in metrics

def test_tokens_per_sec_always_exists():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
    )
    
    dummy_input = {
        "input_ids": torch.randn(4, 10),
        "labels": torch.randn(4, 10)
    } # batch_size=4, seq_len=10
    
    metrics = trainer.train_step(dummy_input, step=1, is_accum_step=True)
    
    assert metrics is not None
    assert "tokens_per_sec" in metrics
    assert metrics["tokens_per_sec"] >= 0.0
    assert "step_time" in metrics
    assert "forward_time" in metrics
    assert "backward_time" in metrics
    assert "optimizer_time" in metrics
    assert "gpu_mem_allocated_mb" in metrics
    assert "gpu_mem_reserved_mb" in metrics
