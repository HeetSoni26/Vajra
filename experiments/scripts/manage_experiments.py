import argparse
import sys
import json
from pathlib import Path

from experiments.config import ExperimentConfig
from experiments.manager import RunManager
from experiments.search import SearchEngine
from experiments.comparison import ComparisonEngine
from experiments.export import ExportManager

def cmd_create(args):
    config = ExperimentConfig()
    rm = RunManager(project_id=args.project, config=config, run_name=args.name, tags=args.tags)
    print(f"Created run {rm.run_name} (ID: {rm.run_id})")

def cmd_list(args):
    engine = SearchEngine(Path("output/experiments"))
    runs = engine.filter_runs(project=args.project, status=args.status, tags=args.tags)
    for r in runs:
        print(f"[{r.get('status')}] {r.get('run_name')} - Project: {r.get('project_id')} - Path: {r.get('_path')}")

def cmd_inspect(args):
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print("Run directory not found")
        sys.exit(1)
    
    with open(run_dir / "run_metadata.json", "r") as f:
        meta = json.load(f)
        print(json.dumps(meta, indent=2))

def cmd_compare(args):
    path_a = Path(args.run_a)
    path_b = Path(args.run_b)
    diff = ComparisonEngine.compare_runs(path_a, path_b)
    print(json.dumps(diff, indent=2))

def cmd_export(args):
    run_dir = Path(args.run_dir)
    with open(run_dir / "run_metadata.json", "r") as f:
        meta = json.load(f)
    
    ExportManager.export_summary(meta, Path(args.output), format=args.format)
    print(f"Exported to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Vajra Experiment Manager CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    create_p = subparsers.add_parser("create")
    create_p.add_argument("--project", required=True)
    create_p.add_argument("--name")
    create_p.add_argument("--tags", nargs="*")
    
    list_p = subparsers.add_parser("list")
    list_p.add_argument("--project")
    list_p.add_argument("--status")
    list_p.add_argument("--tags", nargs="*")
    
    inspect_p = subparsers.add_parser("inspect")
    inspect_p.add_argument("run_dir")
    
    compare_p = subparsers.add_parser("compare")
    compare_p.add_argument("run_a")
    compare_p.add_argument("run_b")
    
    export_p = subparsers.add_parser("export")
    export_p.add_argument("run_dir")
    export_p.add_argument("--output", required=True)
    export_p.add_argument("--format", choices=["json", "csv", "md"], default="md")
    
    args = parser.parse_args()
    
    if args.command == "create":
        cmd_create(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "export":
        cmd_export(args)

if __name__ == "__main__":
    main()
