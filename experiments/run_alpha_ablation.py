"""Run alpha-ablation experiments for the DMKPO-Log scoring rule."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from common import BudgetedObjective, GAMMA_RANGE, add_common_args, dmkpo_search, setup_logger, set_all_seeds, summarize
from run_synthetic_benchmark import DATASETS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.3, 0.5, 1.0])
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    if args.smoke:
        args.alphas = [0.0, 0.3]; args.n_runs = 1
    n_runs = args.n_runs if args.n_runs is not None else 30
    budget = 3 if args.smoke else args.budget
    logger = setup_logger("run_alpha_ablation", ROOT / "logs" / "run_alpha_ablation.log")
    set_all_seeds(args.seed)
    selected = set(args.datasets) if args.datasets else None
    rows = []
    for name, generator, kwargs, n_clusters in DATASETS:
        if selected and name not in selected:
            continue
        local_kwargs = dict(kwargs)
        if args.smoke and "n_per_cluster" in local_kwargs:
            local_kwargs["n_per_cluster"] = 12
        x, y, cat = generator(random_state=args.seed, **local_kwargs)
        for alpha in args.alphas:
            logger.info("Running %s at alpha=%.2f", name, alpha)
            for run_id in range(n_runs):
                seed = args.seed + run_id * 997
                obj = BudgetedObjective(x, y, cat, n_clusters, args.n_init, 5 if args.smoke else args.max_iter, seed, budget, True)
                dmkpo_search(obj, GAMMA_RANGE, budget, True, min(5, budget), alpha=alpha)
                labels = obj.best_labels
                ari = adjusted_rand_score(y, labels) if labels is not None and len(np.unique(labels)) > 1 else np.nan
                rows.append({"dataset": name, "alpha": alpha, "run_id": run_id, "method": "DMKPO-Log", "ari": ari, "nmi": np.nan, "gamma": obj.best_gamma, "n_eval": obj.n_eval, "time_s": np.nan})
    df = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "alpha_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "alpha_ablation_all_runs.csv", index=False)
    summarize(df, ["dataset", "alpha"]).to_csv(out_dir / "alpha_ablation_results.csv", index=False)
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
    import logging
    import os
    logging.shutdown()
    os._exit(0)
