"""Run budget-sensitivity experiments for selected synthetic datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import GAMMA_RANGE, add_common_args, method_rows, setup_logger, summarize, tests, set_all_seeds
from run_synthetic_benchmark import DATASETS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--budgets", nargs="*", type=int, default=[7, 10, 15])
    parser.add_argument("--datasets", nargs="*", default=["H2-Medium"])
    args = parser.parse_args()
    if args.smoke:
        args.budgets = [3]; args.n_runs = 1
    n_runs = args.n_runs if args.n_runs is not None else 30
    logger = setup_logger("run_budget_sensitivity", ROOT / "logs" / "run_budget_sensitivity.log")
    set_all_seeds(args.seed)
    selected = set(args.datasets)
    rows = []
    for budget in args.budgets:
        config = {"seed": args.seed, "budget": budget, "n_init": args.n_init, "max_iter": 5 if args.smoke else args.max_iter, "gamma_range": GAMMA_RANGE, "n_initial": min(5, budget), "alpha": 0.3}
        for name, generator, kwargs, n_clusters in DATASETS:
            if name not in selected:
                continue
            local_kwargs = dict(kwargs)
            if args.smoke and "n_per_cluster" in local_kwargs:
                local_kwargs["n_per_cluster"] = 12
            x, y, cat = generator(random_state=args.seed, **local_kwargs)
            logger.info("Running %s at budget=%d", name, budget)
            for run_id in range(n_runs):
                for row in method_rows(x, y, cat, n_clusters, run_id, config, include_optuna=not args.no_optuna, use_ari_objective=True):
                    rows.append({"dataset": name, "budget": budget, "run_id": run_id, **row})
    df = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "budget_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "budget_all_runs.csv", index=False)
    summarize(df, ["dataset", "budget"]).to_csv(out_dir / "budget_summary.csv", index=False)
    tests(df, ["dataset", "budget"]).to_csv(out_dir / "budget_tests.csv", index=False)
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
    import logging
    import os
    logging.shutdown()
    os._exit(0)
