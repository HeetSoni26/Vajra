import json
from pathlib import Path

RECIPES = {
    "quick_smoke_test": {
        "batch_size": 2,
        "learning_rate": 1e-3,
        "warmup_steps": 2,
        "weight_decay": 0.01,
        "gradient_clipping": 1.0,
        "scheduler": "cosine",
        "precision": "none",
        "optimizer": "AdamW",
        "expected_runtime": "1 minute",
        "estimated_vram": "2GB"
    },
    "10M_token_run": {
        "batch_size": 8,
        "learning_rate": 5e-4,
        "warmup_steps": 100,
        "weight_decay": 0.1,
        "gradient_clipping": 1.0,
        "scheduler": "cosine",
        "precision": "bf16",
        "optimizer": "AdamW_Fused",
        "expected_runtime": "30 minutes",
        "estimated_vram": "8GB"
    },
    "100M_token_run": {
        "batch_size": 16,
        "learning_rate": 3e-4,
        "warmup_steps": 500,
        "weight_decay": 0.1,
        "gradient_clipping": 1.0,
        "scheduler": "cosine",
        "precision": "bf16",
        "optimizer": "AdamW_Fused",
        "expected_runtime": "4 hours",
        "estimated_vram": "16GB"
    },
    "1B_token_run": {
        "batch_size": 32,
        "learning_rate": 3e-4,
        "warmup_steps": 2000,
        "weight_decay": 0.1,
        "gradient_clipping": 1.0,
        "scheduler": "cosine",
        "precision": "bf16",
        "optimizer": "AdamW_Fused",
        "expected_runtime": "24 hours",
        "estimated_vram": "24GB"
    },
    "full_Vajra_370M": {
        "batch_size": 128,
        "learning_rate": 3e-4,
        "warmup_steps": 2000,
        "weight_decay": 0.1,
        "gradient_clipping": 1.0,
        "scheduler": "cosine",
        "precision": "bf16",
        "optimizer": "AdamW_Fused",
        "expected_runtime": "14 days",
        "estimated_vram": "80GB"
    }
}

HARDWARE_PROFILES = {
    "1_GPU": {"gradient_accumulation": 8, "batch_size_per_gpu": 16},
    "2_GPU": {"gradient_accumulation": 4, "batch_size_per_gpu": 16},
    "4_GPU": {"gradient_accumulation": 2, "batch_size_per_gpu": 16},
    "8_GPU": {"gradient_accumulation": 1, "batch_size_per_gpu": 16},
    "Consumer_RTX_3090": {"batch_size_per_gpu": 8, "precision": "bf16", "gradient_checkpointing": True},
    "Consumer_RTX_4090": {"batch_size_per_gpu": 16, "precision": "bf16", "gradient_checkpointing": True},
    "Consumer_RTX_5090": {"batch_size_per_gpu": 32, "precision": "fp8", "gradient_checkpointing": False},
    "A100_80GB": {"batch_size_per_gpu": 64, "precision": "bf16", "gradient_checkpointing": False},
    "H100_80GB": {"batch_size_per_gpu": 128, "precision": "fp8", "gradient_checkpointing": False}
}

def generate_configs(output_dir: str = "configs"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    for name, recipe in RECIPES.items():
        (out / f"recipe_{name}.json").write_text(json.dumps(recipe, indent=2))
        
    (out / "hardware_profiles.json").write_text(json.dumps(HARDWARE_PROFILES, indent=2))

if __name__ == "__main__":
    generate_configs()
