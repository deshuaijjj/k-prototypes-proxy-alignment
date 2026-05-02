"""Download and preprocess the four primary UCI real datasets.

This helper uses the same loader as the benchmark script. It writes processed
mixed-type CSV files that include the external label column. Labels are provided
for external evaluation only and are not used by gamma search.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from common import PRIMARY_REAL_DATASETS
from run_real_benchmark import load_dataset


def main() -> None:
    output_dir = ROOT / "data" / "real_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in PRIMARY_REAL_DATASETS:
        X, y, categorical_indices, info = load_dataset(name, max_samples=None, seed=42)
        feature_columns = [f"feature_{i + 1}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_columns)
        df["external_label"] = y
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        df.to_csv(output_dir / f"{safe_name}.csv", index=False)
        print(f"Wrote {name}: rows={len(df)}, categorical_indices={categorical_indices}, info={info}")


if __name__ == "__main__":
    main()
