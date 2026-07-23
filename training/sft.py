from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning entry point.")
    parser.add_argument("--config", default="configs/training/sft.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install alignment dependencies: pip install transformers datasets trl accelerate") from exc

    dataset = load_dataset("json", data_files={"train": cfg["train_file"], "validation": cfg["validation_file"]})
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype="auto", device_map="auto")

    # Use eval_strategy for transformers >= 4.41 compatibility
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=float(cfg["weight_decay"]),
        bf16=bool(cfg["bf16"]),
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        report_to=cfg.get("report_to", "none"),
    )

    # Pass processing_class for modern TRL compatibility
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "args": training_args,
        "max_seq_length": int(cfg["max_seq_length"]),
        "dataset_text_field": "text",
    }
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
