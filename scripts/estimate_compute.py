from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=float, default=1.05e9)
    parser.add_argument("--tokens", type=float, default=1.0e11)
    args = parser.parse_args()
    print({"flops": 6 * args.params * args.tokens})


if __name__ == "__main__":
    main()
