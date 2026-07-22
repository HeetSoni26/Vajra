import argparse
from release.package import ReleasePackager
from release.inference import InferenceExamplesGenerator
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Vajra Release CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    pack_p = subparsers.add_parser("package")
    pack_p.add_argument("--output-dir", required=True)
    
    ver_p = subparsers.add_parser("verify-release")
    ver_p.add_argument("--dir", required=True)
    
    args = parser.parse_args()
    
    if args.command == "package":
        print(f"Packaging release to {args.output_dir}")
        # Mocking model and config for CLI execution
        model = nn.Linear(10, 10)
        config = {"vocab_size": 100}
        packager = ReleasePackager(args.output_dir)
        packager.create_package(model, config)
        
        gen = InferenceExamplesGenerator(args.output_dir)
        gen.generate_all()
        print("Done.")
        
    elif args.command == "verify-release":
        packager = ReleasePackager(args.dir)
        if packager.verify_package():
            print("Release package is valid.")
        else:
            print("Release package is missing required files.")

if __name__ == "__main__":
    main()
