from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from training.builder import auto_discover_cases, split_case_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_ids = auto_discover_cases(Path(args.train_dir))
    _, val_ids = split_case_ids(all_ids, args.val_split, args.seed)

    print(f"Val cases ({len(val_ids)}):")
    for c in val_ids:
        print(c)


if __name__ == "__main__":
    main()
