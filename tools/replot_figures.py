"""Regenerate proxy-alignment figures from scan CSV files.

The script reads `*_scan.csv` files and writes PNG/PDF figures with consistent
font settings. By default it uses the final reproducibility package paths, but
input and output directories can be overridden from the command line.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "axes.unicode_minus": False,
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
})


def replot_from_csv(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    dataset_name = str(df["dataset"].iloc[0]) if "dataset" in df.columns else csv_path.stem.replace("_scan", "")
    safe_name = dataset_name.replace(" ", "_").replace("-", "_").lower()

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color_cost = "#2166ac"
    color_ari = "#d6604d"

    x = df["log10_gamma"].values
    cost_values = df["cost_mean"].values
    cost_norm = (cost_values - cost_values.min()) / (cost_values.max() - cost_values.min() + 1e-12)

    ax1.plot(x, cost_norm, color=color_cost, lw=2, label="J(gamma)")
    ax1.set_xlabel("log10(gamma)", fontsize=12)
    ax1.set_ylabel("Normalized internal objective", color=color_cost, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_cost)

    ax2 = ax1.twinx()
    ax2.plot(x, df["ari_mean"].values, color=color_ari, lw=2, linestyle="--", label="ARI(gamma)")
    ax2.set_ylabel("ARI (external quality)", color=color_ari, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_ari)

    cost_best_idx = df["cost_mean"].idxmin()
    ari_best_idx = df["ari_mean"].idxmax()
    ax1.axvline(df.loc[cost_best_idx, "log10_gamma"], color=color_cost, linestyle=":", alpha=0.5)
    ax2.axvline(df.loc[ari_best_idx, "log10_gamma"], color=color_ari, linestyle=":", alpha=0.5)

    plt.title(f"Dataset: {dataset_name}", fontsize=12)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"proxy_alignment_{safe_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"proxy_alignment_{safe_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered {dataset_name}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "final_peerj_ai_application_package" / "results" / "exp_proxy")
    parser.add_argument("--output-dir", type=Path, default=root / "final_peerj_ai_application_package" / "figures")
    args = parser.parse_args()

    csv_files = sorted(args.input_dir.glob("*_scan.csv"))
    print(f"Rendering {len(csv_files)} proxy-alignment scan files from {args.input_dir}")
    for csv_file in csv_files:
        replot_from_csv(csv_file, args.output_dir)


if __name__ == "__main__":
    main()
