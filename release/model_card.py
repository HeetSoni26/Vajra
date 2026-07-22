from pathlib import Path
from typing import Dict, Any

class ModelCardGenerator:
    """Generates the README.md / Model Card for the release."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(self, model_name: str, config: Dict[str, Any], eval_metrics: Dict[str, Any] = None) -> Path:
        content = f"""# {model_name}

## Model Description
{model_name} is a foundation language model built with the Vajra Framework.

## Architecture
- **Model Type**: vajra
- **Hidden Size**: {config.get('hidden_size', 'N/A')}
- **Layers**: {config.get('num_layers', 'N/A')}
- **Attention Heads**: {config.get('num_attention_heads', 'N/A')}
- **Context Length**: {config.get('max_position_embeddings', 'N/A')}
- **Vocab Size**: {config.get('vocab_size', 'N/A')}

## Training
- **Framework**: Vajra Framework
- **Dataset**: Custom Sharded Corpus
- **Precision**: bf16 / fp32

## Evaluation
"""
        if eval_metrics:
            for bench, metrics in eval_metrics.items():
                content += f"- **{bench}**: {metrics}\n"
        else:
            content += "Evaluation metrics not available for this release.\n"
            
        content += """
## Limitations
This model is a base foundation model and has not undergone safety alignment or instruction tuning. It may produce inaccurate, biased, or inappropriate content.

## License
MIT License (or equivalent Open Source License).

## Citation
```bibtex
@misc{{vajra2026,
  author = {{Vajra Dev Team}},
  title = {{{model_name}} Foundation Model},
  year = {{2026}}
}}
```
"""
        
        path = self.output_dir / "README.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
