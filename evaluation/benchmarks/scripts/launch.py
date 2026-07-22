import argparse
from evaluation.benchmarks.registry import BenchmarkRegistry
# Ensure adapters are registered

def main():
    parser = argparse.ArgumentParser(description="Vajra Benchmark Suite CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    bench_p = subparsers.add_parser("benchmark")
    bench_p.add_argument("--name", required=True)
    bench_p.add_argument("--output-dir", required=True)
    
    all_p = subparsers.add_parser("benchmark-all")
    all_p.add_argument("--output-dir", required=True)
    
    ckpt_p = subparsers.add_parser("benchmark-checkpoint")
    ckpt_p.add_argument("--checkpoint", required=True)
    ckpt_p.add_argument("--output-dir", required=True)
    
    lead_p = subparsers.add_parser("leaderboard")
    lead_p.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    if args.command == "benchmark":
        print(f"Running benchmark {args.name}")
    elif args.command == "benchmark-all":
        benchmarks = BenchmarkRegistry.list_benchmarks()
        print(f"Running all benchmarks: {benchmarks}")
    elif args.command == "benchmark-checkpoint":
        print(f"Evaluating checkpoint: {args.checkpoint}")
    elif args.command == "leaderboard":
        print("Generating leaderboard...")

if __name__ == "__main__":
    main()
