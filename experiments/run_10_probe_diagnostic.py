"""Run the lightweight 10-probe proxy diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import EXTENDED_REAL_DATASETS, GAMMA_RANGE, PRIMARY_REAL_DATASETS, add_common_args, safe_name, setup_logger, set_all_seeds
from run_proxy_landscape_scans import alignment, scan_dataset
from run_real_benchmark import N_CLUSTERS, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--datasets", nargs="*", default=PRIMARY_REAL_DATASETS)
    parser.add_argument("--all-extended", action="store_true")
    parser.add_argument("--n-repeats", type=int, default=3)
    args = parser.parse_args()
    datasets = EXTENDED_REAL_DATASETS if args.all_extended else args.datasets
    if args.smoke:
        args.n_repeats = 1; args.max_samples = min(args.max_samples, 200); args.max_iter = min(args.max_iter, 5)
    logger = setup_logger("run_10_probe_diagnostic", ROOT / "logs" / "run_10_probe_diagnostic.log")
    set_all_seeds(args.seed)
    out_dir = ROOT / "results" / "diagnostic_10probe"
    full_dir = ROOT / "results" / "proxy_scans"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in datasets:
        try:
            x, y, cat, _ = load_dataset(name, args.max_samples, args.seed)
            n_clusters = N_CLUSTERS.get(name, len(np.unique(y)))
            df = scan_dataset(name, x, y, cat, n_clusters, 10, args.n_repeats, args.seed, args.n_init, args.max_iter, logger)
            df.to_csv(out_dir / f"{safe_name(name)}_10probe.csv", index=False)
            diag = alignment(df)
            full_path = full_dir / f"{safe_name(name)}_scan.csv"
            sign_agreement = np.nan
            if full_path.exists():
                full = alignment(pd.read_csv(full_path))
                sign_agreement = bool(np.sign(diag["spearman_rho"]) == np.sign(full["spearman_rho"]))
            rows.append({"dataset": name, **diag, "sign_agreement_with_full_scan": sign_agreement})
        except Exception as exc:
            logger.exception("Failed dataset %s: %s", name, exc)
            rows.append({"dataset": name, "error": str(exc)})
    pd.DataFrame(rows).to_csv(out_dir / "diagnostic_summary.csv", index=False)
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
    import logging
    import os
    logging.shutdown()
    os._exit(0)
