"""Reproducible feature-table benchmark CLI."""

import argparse
import csv
from pathlib import Path

import numpy as np

from regenbench.benchmark import evaluate, save_results
from regenbench.data import load_table


def write_demo(path, seed=0):
    """A synthetic signal with repeated technical measurements per donor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_id", "label", "donor_id", "batch_id", "f_signal", "f_nuisance"])
        for donor in range(30):
            offset = rng.normal(0, 0.4)
            for label in range(2):
                for replicate in range(3):
                    writer.writerow([f"d{donor}-c{label}-r{replicate}", str(label), f"d{donor}",
                                     f"b{donor // 3}", 1.5 * label + offset + rng.normal(0, 0.6),
                                     rng.normal()])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Group-aware research benchmarks (no clinical interpretation)")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="write an explicitly synthetic CSV")
    demo.add_argument("--out", required=True)
    demo.add_argument("--seed", type=int, default=0)
    run = sub.add_parser("run")
    run.add_argument("csv")
    run.add_argument("--group-by", required=True, help="comma-separated metadata columns; joint blocking uses connected groups")
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--bootstrap-draws", type=int, default=2000)
    run.add_argument("--out", required=True, help="new output directory (never overwrites results)")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            write_demo(args.out, args.seed)
        else:
            data = load_table(args.csv, [c.strip() for c in args.group_by.split(",")])
            report, predictions = evaluate(data, folds=args.folds, seed=args.seed,
                                           bootstrap_draws=args.bootstrap_draws)
            save_results(report, predictions, args.out)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
