"""Run the primary real-data benchmark using ucimlrepo datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from common import GAMMA_RANGE, PRIMARY_REAL_DATASETS, add_common_args, method_rows, preprocess_mixed_data, setup_logger, summarize, tests, set_all_seeds

ROOT = Path(__file__).resolve().parents[1]
UCI_IDS = {"Adult": 2, "Credit Approval": 27, "Australian Credit": 143, "Heart Disease": 45, "Bank Marketing": 222, "Abalone": 1, "Automobile": 10, "Census KDD": 117, "German Credit": 144, "Horse Colic": 47}
MANUAL_CATEGORICAL_COLUMNS = {
    "Heart Disease": ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    "Bank Marketing": ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"],
}
N_CLUSTERS = {"Abalone": 3, "Automobile": 6}


def load_dataset(name: str, max_samples: int | None, seed: int):
    from ucimlrepo import fetch_ucirepo
    repo = fetch_ucirepo(id=UCI_IDS[name])
    x_df = repo.data.features.copy()
    y_raw = repo.data.targets.iloc[:, 0].values
    if name == "Heart Disease":
        y_values = (pd.to_numeric(pd.Series(y_raw), errors="coerce").fillna(0).values > 0).astype(int)
    elif name == "Adult":
        y_values = np.array([str(v).rstrip(".").strip() for v in y_raw])
    elif name == "Abalone":
        y_values = pd.qcut(pd.to_numeric(pd.Series(y_raw), errors="coerce"), q=3, labels=False, duplicates="drop").astype(int).values
    else:
        y_values = y_raw
    valid = pd.Series(y_values).notna().values
    x_df = x_df.loc[valid].copy()
    y_values = np.asarray(y_values)[valid]
    categorical_cols = [col for col in MANUAL_CATEGORICAL_COLUMNS.get(name, []) if col in x_df.columns]
    if not categorical_cols:
        categorical_cols = list(x_df.select_dtypes(include=["object", "category", "bool"]).columns)
    if not categorical_cols:
        for col in x_df.columns:
            unique_count = x_df[col].nunique(dropna=True)
            if unique_count <= 10:
                categorical_cols.append(col)
    for col in x_df.columns:
        if col in categorical_cols:
            x_df[col] = x_df[col].astype(str).replace({"nan": np.nan, "?": np.nan}).fillna("missing")
        else:
            values = pd.to_numeric(x_df[col], errors="coerce")
            x_df[col] = values.fillna(values.median())
    cat = [x_df.columns.get_loc(c) for c in categorical_cols]
    y = LabelEncoder().fit_transform(y_values.astype(str))
    if max_samples and len(x_df) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_df), size=max_samples, replace=False)
        x = x_df.values[idx]
        y = y[idx]
    else:
        x = x_df.values
    x = preprocess_mixed_data(x, cat)
    info = {"dataset": name, "uci_id": UCI_IDS[name], "n_samples": len(x), "n_features": x.shape[1], "n_categorical": len(cat), "n_continuous": x.shape[1] - len(cat), "n_classes": len(np.unique(y))}
    return x, y, cat, info


def run_benchmark(dataset_names, args, output_dir: Path, log_name: str):
    n_runs = args.n_runs if args.n_runs is not None else (1 if args.smoke else 30)
    config = {"seed": args.seed, "budget": 3 if args.smoke else args.budget, "n_init": args.n_init, "max_iter": 5 if args.smoke else args.max_iter, "gamma_range": GAMMA_RANGE, "n_initial": 3 if args.smoke else 5, "alpha": 0.3}
    logger = setup_logger(log_name, ROOT / "logs" / f"{log_name}.log")
    set_all_seeds(args.seed)
    rows = []
    metadata = []
    for name in dataset_names:
        logger.info("Loading %s", name)
        x, y, cat, info = load_dataset(name, args.max_samples if not args.smoke else min(args.max_samples, 300), args.seed)
        metadata.append(info)
        n_clusters = N_CLUSTERS.get(name, len(np.unique(y)))
        logger.info("Running %s with n=%d, k=%d, runs=%d", name, len(x), n_clusters, n_runs)
        for run_id in range(n_runs):
            for row in method_rows(x, y, cat, n_clusters, run_id, config, include_optuna=not args.no_optuna, use_ari_objective=False):
                rows.append({"dataset": name, "run_id": run_id, **row})
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "all_runs.csv", index=False)
    try:
        df.to_parquet(output_dir / "all_runs.parquet", index=False)
    except Exception as exc:
        logger.warning("Parquet output failed: %s", exc)
    summarize(df, ["dataset"]).to_csv(output_dir / "all_stats.csv", index=False)
    tests(df, ["dataset"]).to_csv(output_dir / "statistical_tests.csv", index=False)
    pd.DataFrame(metadata).to_csv(output_dir / "dataset_metadata.csv", index=False)
    logger.info("Wrote %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--datasets", nargs="*", default=PRIMARY_REAL_DATASETS)
    args = parser.parse_args()
    run_benchmark(args.datasets, args, ROOT / "results" / "real", "run_real_benchmark")


if __name__ == "__main__":
    main()
    import logging
    import os
    logging.shutdown()
    os._exit(0)
