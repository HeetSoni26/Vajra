import csv
import json
from pathlib import Path

from utils.logging import setup_logger

logger = setup_logger("compare_checkpoints")


def generate_leaderboard(eval_dir: str | Path = "evaluations") -> None:
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        return

    results = []
    for metrics_file in eval_dir.glob("*/metrics.json"):
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            data["checkpoint_dir"] = metrics_file.parent.name
            results.append(data)
        except Exception as e:
            logger.warning(f"Failed to read {metrics_file}: {e}")

    if not results:
        logger.error("No evaluation results found.")
        return

    # Sort by validation loss (ascending)
    results.sort(key=lambda x: x.get("validation_loss", float("inf")))

    # 1. Output to terminal
    print("\n" + "="*80)
    print("VAJRA CHECKPOINT LEADERBOARD")
    print("="*80)
    print(f"{'Rank':<5} | {'Checkpoint Dir':<30} | {'Step':<6} | {'Loss':<8} | {'Perplexity':<10}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        loss = f"{r.get('validation_loss', 0):.4f}"
        ppl = f"{r.get('perplexity', 0):.4f}"
        print(f"{i:<5} | {r['checkpoint_dir']:<30} | {r.get('global_step', 0):<6} | {loss:<8} | {ppl:<10}")
    print("="*80 + "\n")

    # 2. Save leaderboard.json
    json_path = eval_dir / "leaderboard.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 3. Save leaderboard.csv
    csv_path = eval_dir / "leaderboard.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        headers = ["Rank", "Checkpoint Dir", "Global Step", "Tokens Seen", "Validation Loss", "Perplexity", "Timestamp", "Dataset"]
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, r in enumerate(results, 1):
            writer.writerow([
                i,
                r.get("checkpoint_dir"),
                r.get("global_step"),
                r.get("tokens_seen"),
                r.get("validation_loss"),
                r.get("perplexity"),
                r.get("evaluation_timestamp"),
                r.get("dataset_name"),
            ])

    # 4. Save leaderboard.md
    md_path = eval_dir / "leaderboard.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Vajra Checkpoint Leaderboard\n\n")
        f.write("| Rank | Checkpoint | Step | Validation Loss | Perplexity |\n")
        f.write("|------|------------|------|-----------------|------------|\n")
        for i, r in enumerate(results, 1):
            loss = f"{r.get('validation_loss', 0):.4f}"
            ppl = f"{r.get('perplexity', 0):.4f}"
            f.write(f"| {i} | `{r.get('checkpoint_dir')}` | {r.get('global_step')} | {loss} | {ppl} |\n")

    logger.info("Leaderboard generated successfully (JSON, CSV, MD).")

if __name__ == "__main__":
    import sys
    eval_dir_arg = sys.argv[1] if len(sys.argv) > 1 else "evaluations"
    generate_leaderboard(eval_dir_arg)
