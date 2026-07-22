# Vajra Model Architecture

The Vajra Model Architecture defines the foundational parameters spanning across the entirety of the Vajra Language Model family. Structurally designed to cleanly scale from `370M` out to sizes exceeding `7B` parameters, the engine leverages standardized PyTorch topologies strictly focusing on modular abstraction.

## Components

- **`config.py` (`VajraConfig`)**: Outlines the bounding topologies spanning dimensions across variables such as `hidden_size`, `num_layers`, `vocab_size`, and dynamic preset retrieval mappings explicitly targeting variants (e.g. `Vajra-370M`).
- **`layers/rmsnorm.py` (`RMSNorm`)**: Stabilized Root Mean Square Layer Normalization providing faster scaling limits natively mapped locally inside `float32` prior to output translation formatting.
- **`layers/rope.py` (`RotaryEmbedding`)**: Defines RoPE positional mappings injecting coordinate spaces seamlessly tracking sequence scaling mathematically cleanly preserving zero-copy arrays locally.
- **`layers/attention.py` (`VajraAttention`)**: GQA (Grouped Query Attention) mechanisms actively spanning dimensional groups wrapping dynamically back across past_key_values mapping memory efficiency. 
- **`layers/mlp.py` (`VajraMLP`)**: SwiGLU implementations actively projecting mathematically isolated sequences mapping dynamically matching boundaries over activation functions implicitly.
- **`blocks.py` (`VajraBlock`)**: Wraps internal attention models logically between `RMSNorm` structures projecting residuals dynamically through forward evaluations preserving non-linear limits natively.
- **`modeling.py` (`VajraModel`, `VajraForCausalLM`)**: Integrates structural layouts wrapping LM heads internally supporting output evaluation tracking loss functions across labels.
- **`generation/engine.py` (`GenerationEngine`)**: Dynamically samples tensors tracking iterative loops across Top K, Top P, Temperature, and greedy structures without requiring Hugging Face boundaries natively tracking cache values.
- **`checkpoints.py` (`CheckpointManager`)**: Marshals PyTorch dictionaries mapping physically to disk matching configuration layouts natively spanning Safetensors configurations natively avoiding malicious executions globally.
- **`utils.py`**: Exports reporting boundaries calculating model definitions sizing architectures visually across configurations.

## CLI Utility

Operate physically bounding states straight out of `manage_model.py`:

```bash
# Export 370M physical layouts onto disk directly scaling boundaries tracking sizes.
python model/scripts/manage_model.py create --preset Vajra-370M --output-dir models/vajra-370m

# Inspect structural sizes manually querying bounds physically visually mapped.
python model/scripts/manage_model.py summary models/vajra-370m
```
