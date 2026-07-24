import argparse
import subprocess
import sys
from pathlib import Path

from evaluation.compare_checkpoints import generate_leaderboard
from utils.logging import setup_logger

logger = setup_logger("evaluate_all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in an experiment directory.")
    parser.add_argument("--experiment-dir", required=True, help="Path to experiment directory (e.g. checkpoints/run_1)")
    parser.add_argument("--config", required=True, help="Path to training config.yaml")
    parser.add_argument("--data-dir", default=None, help="Evaluation data directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt", default="The future of AI is", help="Prompt for sample generation")
    parser.add_argument("--run-benchmarks", action="store_true", help="Run benchmark suite after evaluation")
    
    args = parser.parse_args()
    
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        logger.error(f"Experiment directory not found: {exp_dir}")
        return

    checkpoints = sorted(exp_dir.glob("checkpoint_step_*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
    
    if not checkpoints:
        logger.warning(f"No checkpoint_step_*.pt found in {exp_dir}.")
        return

    logger.info(f"Found {len(checkpoints)} checkpoints to evaluate.")

    for ckpt_path in checkpoints:
        step = ckpt_path.stem.split("_")[-1]
        ckpt_name = f"checkpoint_{step}"
        logger.info(f"--- Evaluating {ckpt_name} ---")

        # 1. Evaluate
        cmd_eval = [
            sys.executable, "-m", "evaluation.evaluate",
            "--checkpoint", str(ckpt_path),
            "--config", args.config,
            "--batch-size", str(args.batch_size),
            "--sequence-length", "256"
        ]
        if args.data_dir:
            cmd_eval.extend(["--data-dir", args.data_dir])
            
        logger.info(f"Running evaluate.py on {ckpt_name}")
        subprocess.run(cmd_eval, check=True)
        
        # The evaluate.py script saves metrics to evaluations/checkpoint_step_XYZ/metrics.json
        # 2. Generate Sample
        eval_out_dir = Path("evaluations") / ckpt_name
        cmd_gen = [
            sys.executable, "-m", "evaluation.generate",
            "--checkpoint", str(ckpt_path),
            "--config", args.config,
            "--prompt", args.prompt,
            "--output-dir", str(eval_out_dir),
            "--max-new-tokens", "64"
        ]
        logger.info(f"Running generate.py on {ckpt_name}")
        subprocess.run(cmd_gen, check=True)

        # 3. Benchmark (Optional)
        if args.run_benchmarks:
            cmd_bench = [
                sys.executable, "-m", "benchmarks.benchmark",
                "--checkpoint", str(ckpt_path),
                "--config", args.config,
                "--eval-dir", "evaluations"
            ]
            logger.info(f"Running benchmark.py on {ckpt_name}")
            subprocess.run(cmd_bench, check=True)

    logger.info("All checkpoints evaluated. Updating leaderboard...")
    generate_leaderboard("evaluations")
    
    if args.run_benchmarks:
        from benchmarks.compare_benchmarks import generate_comparison_report
        generate_comparison_report(Path("benchmarks/reports"))

if __name__ == "__main__":
    main()
