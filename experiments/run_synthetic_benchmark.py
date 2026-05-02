"""Run the synthetic benchmark for seven mixed-type datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import GAMMA_RANGE, add_common_args, method_rows, setup_logger, summarize, tests, set_all_seeds

ROOT = Path(__file__).resolve().parents[1]


def generate_h1_dataset(n_per_cluster=200, n_continuous=2, n_categorical=8, n_clusters=3, cont_sep=1.0, cont_std=2.0, cat_n_values=4, cat_noise=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    total = n_per_cluster * n_clusters
    centers = rng.normal(size=(n_clusters, n_continuous)) * cont_sep
    x_cont = np.vstack([rng.normal(size=(n_per_cluster, n_continuous)) * cont_std + centers[k] for k in range(n_clusters)])
    blocks = []
    for k in range(n_clusters):
        block = np.zeros((n_per_cluster, n_categorical), dtype=object)
        for j in range(n_categorical):
            values = np.full(n_per_cluster, str((k + j) % cat_n_values), dtype=object)
            mask = rng.random(n_per_cluster) < cat_noise
            values[mask] = rng.integers(0, cat_n_values, size=int(mask.sum())).astype(str)
            block[:, j] = values
        blocks.append(block)
    x = np.hstack([x_cont, np.vstack(blocks)])
    y = np.repeat(np.arange(n_clusters), n_per_cluster)
    order = rng.permutation(total)
    return x[order], y[order], list(range(n_continuous, n_continuous + n_categorical))


def generate_control_dataset(random_state=42):
    return generate_h1_dataset(n_per_cluster=300, n_continuous=5, n_categorical=5, n_clusters=3, cont_sep=2.5, cont_std=1.2, cat_n_values=4, cat_noise=0.1, random_state=random_state)


DATASETS = [
    ("H1-Easy", generate_h1_dataset, dict(n_per_cluster=200, n_continuous=4, n_categorical=2, n_clusters=3, cont_sep=2.0, cont_std=1.0, cat_n_values=3, cat_noise=0.30), 3),
    ("H1-Medium", generate_h1_dataset, dict(n_per_cluster=200, n_continuous=4, n_categorical=2, n_clusters=3, cont_sep=3.0, cont_std=1.0, cat_n_values=3, cat_noise=0.45), 3),
    ("H1-Hard", generate_h1_dataset, dict(n_per_cluster=200, n_continuous=4, n_categorical=2, n_clusters=3, cont_sep=4.0, cont_std=1.0, cat_n_values=3, cat_noise=0.45), 3),
    ("H2-Easy", generate_h1_dataset, dict(n_per_cluster=200, n_continuous=1, n_categorical=3, n_clusters=3, cont_sep=2.0, cont_std=1.0, cat_n_values=3, cat_noise=0.30), 3),
    ("H2-Medium", generate_h1_dataset, dict(n_per_cluster=200, n_continuous=2, n_categorical=3, n_clusters=3, cont_sep=3.0, cont_std=1.0, cat_n_values=3, cat_noise=0.40), 3),
    ("H2-Hard", generate_h1_dataset, dict(n_per_cluster=200, n_continuous=2, n_categorical=3, n_clusters=3, cont_sep=4.0, cont_std=1.0, cat_n_values=3, cat_noise=0.45), 3),
    ("Control-Balanced", generate_control_dataset, dict(), 3),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    n_runs = args.n_runs if args.n_runs is not None else (1 if args.smoke else 30)
    budget = 3 if args.smoke else args.budget
    n_per_cluster_override = 12 if args.smoke else None
    config = {"seed": args.seed, "budget": budget, "n_init": args.n_init, "max_iter": 5 if args.smoke else args.max_iter, "gamma_range": GAMMA_RANGE, "n_initial": 3 if args.smoke else 5, "alpha": 0.3}
    out_dir = ROOT / "results" / "synthetic"
    logger = setup_logger("synthetic", ROOT / "logs" / "run_synthetic_benchmark.log")
    set_all_seeds(args.seed)
    rows = []
    selected = set(args.datasets) if args.datasets else None
    for name, generator, kwargs, n_clusters in DATASETS:
        if selected and name not in selected:
            continue
        local_kwargs = dict(kwargs)
        if n_per_cluster_override and "n_per_cluster" in local_kwargs:
            local_kwargs["n_per_cluster"] = n_per_cluster_override
        x, y, cat = generator(random_state=args.seed, **local_kwargs)
        logger.info("Running %s with n=%d, runs=%d, budget=%d", name, len(x), n_runs, budget)
        for run_id in range(n_runs):
            for row in method_rows(x, y, cat, n_clusters, run_id, config, include_optuna=not args.no_optuna, use_ari_objective=False):
                rows.append({"dataset": name, "run_id": run_id, **row})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_runs.csv", index=False)
    try:
        df.to_parquet(out_dir / "all_runs.parquet", index=False)
    except Exception as exc:
        logger.warning("Parquet output failed: %s", exc)
    summary = summarize(df, ["dataset"])
    stat = tests(df, ["dataset"])
    summary.to_csv(out_dir / "summary_all.csv", index=False)
    stat.to_csv(out_dir / "statistical_tests.csv", index=False)
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
    import logging
    import os
    logging.shutdown()
    os._exit(0)
