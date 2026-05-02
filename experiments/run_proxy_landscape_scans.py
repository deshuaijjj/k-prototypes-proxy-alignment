"""Run extended proxy-alignment scans over real mixed-type datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from common import EXTENDED_REAL_DATASETS, GAMMA_RANGE, add_common_args, safe_name, setup_logger, set_all_seeds
from run_real_benchmark import N_CLUSTERS, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def scan_dataset(name, x, y, cat, n_clusters, n_points, n_repeats, seed, n_init, max_iter, logger):
    rows = []
    gammas = np.logspace(np.log10(GAMMA_RANGE[0]), np.log10(GAMMA_RANGE[1]), n_points)
    for i, gamma in enumerate(gammas):
        costs, aris = [], []
        for rep in range(n_repeats):
            model = KPrototypes(n_clusters=n_clusters, gamma=float(gamma), n_init=n_init, max_iter=max_iter, random_state=seed + i * 101 + rep, verbose=0)
            labels = model.fit_predict(x, categorical=cat)
            costs.append(float(model.cost_))
            aris.append(adjusted_rand_score(y, labels) if len(np.unique(labels)) > 1 else 0.0)
        rows.append({"dataset": name, "gamma": gamma, "log10_gamma": np.log10(gamma), "cost_mean": np.mean(costs), "cost_std": np.std(costs), "ari_mean": np.mean(aris), "ari_std": np.std(aris)})
        if (i + 1) % max(1, n_points // 5) == 0:
            logger.info("%s scan progress: %d/%d", name, i + 1, n_points)
    return pd.DataFrame(rows)


def alignment(df):
    rho, p = stats.spearmanr(df.cost_mean.values, df.ari_mean.values)
    j_idx = df.cost_mean.idxmin(); ari_idx = df.ari_mean.idxmax()
    return {"spearman_rho": rho, "spearman_p": p, "gamma_opt_J": df.loc[j_idx, "gamma"], "ari_at_opt_J": df.loc[j_idx, "ari_mean"], "gamma_opt_ARI": df.loc[ari_idx, "gamma"], "cost_at_opt_ARI": df.loc[ari_idx, "cost_mean"], "log10_distance": abs(np.log10(df.loc[j_idx, "gamma"]) - np.log10(df.loc[ari_idx, "gamma"])), "aligned": bool(rho < -0.3)}


def plot_scan(df, name, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = df.log10_gamma.values
    cost = (df.cost_mean.values - df.cost_mean.min()) / (df.cost_mean.max() - df.cost_mean.min() + 1e-12)
    ax1.plot(x, cost, color="#2166ac", label="J(gamma), normalized")
    ax1.set_xlabel("log10(gamma)"); ax1.set_ylabel("Normalized K-Prototypes objective", color="#2166ac")
    ax2 = ax1.twinx(); ax2.plot(x, df.ari_mean.values, color="#d6604d", linestyle="--", label="ARI")
    ax2.set_ylabel("ARI", color="#d6604d")
    fig.suptitle(f"Proxy alignment scan: {name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"proxy_alignment_{safe_name(name)}.png", dpi=300)
    fig.savefig(out_dir / f"proxy_alignment_{safe_name(name)}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--datasets", nargs="*", default=EXTENDED_REAL_DATASETS)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.smoke:
        args.n_points = min(args.n_points, 5); args.n_repeats = 1; args.max_samples = min(args.max_samples, 200); args.max_iter = min(args.max_iter, 5)
    logger = setup_logger("run_proxy_landscape_scans", ROOT / "logs" / "run_proxy_landscape_scans.log")
    set_all_seeds(args.seed)
    out_dir = ROOT / "results" / "proxy_scans"; fig_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in args.datasets:
        try:
            x, y, cat, _ = load_dataset(name, args.max_samples, args.seed)
            n_clusters = N_CLUSTERS.get(name, len(np.unique(y)))
            df = scan_dataset(name, x, y, cat, n_clusters, args.n_points, args.n_repeats, args.seed, args.n_init, args.max_iter, logger)
            df.to_csv(out_dir / f"{safe_name(name)}_scan.csv", index=False)
            info = alignment(df); summary.append({"dataset": name, **info})
            plot_scan(df, name, fig_dir)
        except Exception as exc:
            logger.exception("Failed dataset %s: %s", name, exc)
            summary.append({"dataset": name, "error": str(exc)})
    pd.DataFrame(summary).to_csv(out_dir / "proxy_alignment_summary.csv", index=False)
    logger.info("Wrote %s and %s", out_dir, fig_dir)


if __name__ == "__main__":
    main()
    import logging
    import os
    logging.shutdown()
    os._exit(0)
