# Results Directory

This directory contains the authoritative complete result artifacts used to support the manuscript tables and selected figures.

## Included result groups

- `exp_h1_h2/`: Complete seven-dataset synthetic benchmark results, including summary statistics, statistical tests, and per-dataset run tables.
- `exp_real/`: Complete four-dataset primary real benchmark results for Heart Disease, Credit Approval, Adult, and Bank Marketing.
- `exp_proxy/`: Proxy-alignment landscape scans for Heart Disease, Credit Approval, and Bank Marketing. The manuscript reports proxy-scan evidence only for datasets whose CSV artifacts are included here.
- `exp_alpha/`: Alpha-ablation result summary.
- `exp_bo/`: Bayesian-optimization comparison runs.

## Important note

Smoke-test outputs from development runs are intentionally excluded from this final package. The files in this directory are copied from the complete historical result artifacts and are the evidence base for the submitted manuscript.

## File formats

CSV files provide human-readable summary statistics and statistical-test outputs. Parquet files provide run-level result tables used to compute those summaries and tests. They are included as exact machine-readable evidence for the statistical analyses, while the CSV files remain the primary human-readable files for review.

## Label use

External labels are used to compute retrospective metrics such as ARI and NMI. Gamma search itself optimizes the internal K-Prototypes objective and does not use external labels.
