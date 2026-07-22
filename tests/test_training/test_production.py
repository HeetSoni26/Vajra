import torch
from training.production.config import ProductionConfig
from training.production.watchdog import NumericalStabilityWatchdog
from training.production.profiler import MemoryProfiler, PerformanceProfiler
from training.production.optimisation import optimise_model_for_production
from training.production.attention import apply_flash_attention
from model.modeling import VajraForCausalLM
from model.config import VajraConfig

def test_production_config():
    config = ProductionConfig(dataset_dir="dummy", output_dir="dummy")
    assert not config.optimisation.gradient_checkpointing
    assert config.fault_tolerance.enable_watchdog
    assert not config.profiling.enable_memory_profiling

def test_watchdog_loss():
    watchdog = NumericalStabilityWatchdog()
    assert watchdog.check_loss(1.0)
    assert not watchdog.check_loss(float('nan'))
    assert not watchdog.check_loss(float('inf'))

def test_watchdog_gradients():
    watchdog = NumericalStabilityWatchdog()
    model = torch.nn.Linear(10, 10)
    loss = model(torch.randn(1, 10)).sum()
    loss.backward()
    
    assert watchdog.check_gradients(model)
    
    # Corrupt gradients
    model.weight.grad[0, 0] = float('nan')
    assert not watchdog.check_gradients(model)

def test_profiler_init():
    mem = MemoryProfiler(torch.device("cpu"))
    assert not mem.enabled
    
    perf = PerformanceProfiler()
    perf.start_step()
    perf.start_forward()
    perf.end_forward()
    perf.start_backward()
    perf.end_backward()
    stats = perf.end_step()
    
    assert "forward_time_ms" in stats
    assert "step_time_ms" in stats

def test_optimisation_fallback():
    config = VajraConfig(vocab_size=10, hidden_size=16, num_layers=1, num_attention_heads=1, num_key_value_heads=1)
    model = VajraForCausalLM(config)
    
    prod_config = ProductionConfig(dataset_dir="dummy", output_dir="dummy")
    prod_config.optimisation.gradient_checkpointing = True
    prod_config.optimisation.compile_model = True
    
    # Should apply cleanly or fallback safely without crashing
    opt_model = optimise_model_for_production(model, prod_config.optimisation)
    assert opt_model is not None

def test_flash_attention_fallback():
    q = torch.randn(1, 2, 10, 16)
    k = torch.randn(1, 2, 10, 16)
    v = torch.randn(1, 2, 10, 16)
    
    out = apply_flash_attention(q, k, v, is_causal=True)
    assert out.shape == (1, 2, 10, 16)
