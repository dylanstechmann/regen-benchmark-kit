"""Reproducible feature-table benchmark CLI."""

import argparse
import csv
from pathlib import Path

import numpy as np

from regenbench.benchmark import evaluate, save_results
from regenbench.data import load_table
from regenbench.regression import evaluate_regression, save_regression_results


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
    regress = sub.add_parser("regress", help="whole-group regression; leave one group out by default")
    regress.add_argument("csv")
    regress.add_argument("--group-by", required=True)
    regress.add_argument("--target", default="target")
    regress.add_argument("--folds", type=int, help="use GroupKFold instead of leaving each whole group out")
    regress.add_argument("--seed", type=int, default=0)
    regress.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            write_demo(args.out, args.seed)
        elif args.command == "regress":
            data = load_table(args.csv, [c.strip() for c in args.group_by.split(",")],
                              task="regression", target_column=args.target)
            report, predictions = evaluate_regression(data, folds=args.folds, seed=args.seed)
            save_regression_results(report, predictions, args.out)
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
