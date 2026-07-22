from __future__ import annotations

import argparse
from model import FoundationLM, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/model_1b.yaml")
    args = parser.parse_args()
    cfg = ModelConfig.from_yaml(args.config)
    model = FoundationLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({"model": cfg.model_name, "parameters": total, "trainable": trainable})


if __name__ == "__main__":
    main()
