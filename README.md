# Reproducibility Package for K-Prototypes Gamma Search Reliability Study

## Title

On the Reliability of Gamma Search in K-Prototypes Clustering: A Proxy-Alignment Diagnostic and Reproducible Benchmark

## Description

This package supports the PeerJ Computer Science AI Application submission on gamma (`gamma`) search reliability in K-Prototypes clustering for mixed-type data. The study evaluates when internal-objective-driven gamma search can improve or harm external clustering quality, with emphasis on log-space parameterization, fixed-budget search, and proxy-alignment diagnostics.

The package is intended for editors, reviewers, and readers. It contains source code, synthetic benchmark datasets, third-party real-dataset source information, preprocessing documentation, result files used in the manuscript, selected proxy-alignment figures, and file manifests.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── setup.py
├── experiments/
│   ├── common.py
│   ├── run_synthetic_benchmark.py
│   ├── run_real_benchmark.py
│   ├── run_proxy_landscape_scans.py
│   ├── run_10_probe_diagnostic.py
│   ├── run_budget_sensitivity.py
│   └── run_alpha_ablation.py
├── data/
│   ├── synthetic/
│   ├── real_raw_or_external/
│   ├── real_processed/
│   └── metadata/
├── results/
│   ├── exp_h1_h2/
│   ├── exp_real/
│   ├── exp_proxy/
│   ├── exp_alpha/
│   └── exp_bo/
├── figures/
├── metadata/
└── submission_texts/
```

## Dataset Information

### Synthetic datasets

The study uses seven original synthetic mixed-type datasets:

- `H1-Easy`
- `H1-Medium`
- `H1-Hard`
- `H2-Easy`
- `H2-Medium`
- `H2-Hard`
- `Control-Balanced`

The generated CSV files are included in `data/synthetic/`. Their generation parameters, random seed, feature names, categorical indices, and label-use notes are provided in `data/synthetic/synthetic_generation_config.json` and `data/synthetic/README_synthetic.md`.

These datasets are controlled stress-test scenarios. They are not intended to represent natural populations. The `true_label` column is used only for external evaluation metrics such as ARI and NMI, not for unsupervised K-Prototypes fitting or gamma search.

### Primary real datasets

The four primary real datasets are public third-party datasets from the UCI Machine Learning Repository:

| Dataset | Role in study | Source |
|---|---|---|
| Heart Disease | Primary real benchmark; balanced case | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Credit Approval | Primary real benchmark; categorical-dominant case | https://archive.ics.uci.edu/dataset/27/credit+approval |
| Adult | Primary real benchmark; scale-heterogeneous case | https://archive.ics.uci.edu/dataset/2/adult |
| Bank Marketing | Primary real benchmark; noisy/high-sensitivity case | https://archive.ics.uci.edu/dataset/222/bank+marketing |

The source inventory is provided in `data/metadata/dataset_sources.csv`. The preprocessing summary is provided in `data/metadata/preprocessing_summary.md`. Processed CSV files for the four primary real datasets are included in `data/real_processed/`, and the regeneration helper is provided at `data/real_raw_or_external/download_real_datasets.py`.

External labels from real datasets are used only for retrospective evaluation and diagnostics. They are not used by gamma search.

## Code Information

The code implements:

- K-Prototypes objective evaluation;
- default gamma heuristic baseline;
- logarithmic grid search;
- random log-space search;
- Optuna-TPE baseline;
- linear-space adaptive search variant;
- logarithmic-space adaptive search variant;
- ARI, NMI, and statistical evaluation utilities;
- proxy-alignment scans and diagnostic scripts.

The final package intentionally excludes archival recovered source code with non-English comments. The main release scripts are in `experiments/`.

## Requirements

The main dependencies are listed in `requirements.txt`. They include:

- numpy
- scipy
- scikit-learn
- pandas
- kmodes
- ucimlrepo
- optuna
- matplotlib
- seaborn
- statsmodels
- pingouin
- PyYAML
- tqdm

## Installation

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If needed, install the package in editable mode:

```bash
python -m pip install -e .
```

## Usage Instructions

Smoke-test commands:

```bash
python -c "import numpy, pandas, sklearn, kmodes; print('import smoke ok')"
python experiments/run_synthetic_benchmark.py --smoke --datasets H1-Easy --no-optuna
python experiments/run_real_benchmark.py --smoke --datasets "Heart Disease" --no-optuna
```

Full experiment commands:

```bash
python experiments/run_synthetic_benchmark.py --n-runs 30 --budget 15
python experiments/run_real_benchmark.py --n-runs 30 --budget 15 --max-samples 5000
python experiments/run_proxy_landscape_scans.py --n-points 100 --n-repeats 5 --max-samples 5000
python experiments/run_10_probe_diagnostic.py --n-repeats 3 --max-samples 5000
python experiments/run_budget_sensitivity.py --datasets H2-Medium --budgets 7 10 15 --n-runs 30
python experiments/run_alpha_ablation.py --n-runs 30
```

Network access may be required for first-time real-dataset retrieval through `ucimlrepo`.

## Reproducing the Manuscript Results

The manuscript tables are supported by the complete result artifacts in `results/`:

- Synthetic benchmark: `results/exp_h1_h2/summary_all.csv` and `results/exp_h1_h2/statistical_tests.csv`
- Primary real-data benchmark: `results/exp_real/all_stats.csv`
- Proxy-alignment scans: `results/exp_proxy/proxy_alignment_summary.csv` and the corresponding `*_scan.csv` files
- Alpha ablation: `results/exp_alpha/alpha_ablation_results.csv`
- Bayesian-optimization comparison: `results/exp_bo/synthetic_bo_runs.csv`

Development smoke-test outputs are not included as final manuscript evidence.

## Methodology Overview

K-Prototypes combines squared Euclidean dissimilarity for continuous features and mismatch-based dissimilarity for categorical features, weighted by gamma. Because useful gamma values may span several orders of magnitude, the study compares linear-space and log-space search strategies under fixed evaluation budgets.

Search methods optimize the internal K-Prototypes objective only. External labels are used after search to compute ARI, NMI, and diagnostic summaries. The proxy-alignment diagnostic estimates whether lower internal objective values tend to correspond to higher external clustering quality over probe gamma values.

## Citations

If this package is used, cite the associated PeerJ Computer Science article once available. Third-party datasets should be cited through their official UCI Machine Learning Repository pages listed in `data/metadata/dataset_sources.csv`.

## License and Contribution Guidelines

This package is provided for peer review and reproducibility of the associated manuscript. Reuse of third-party datasets is governed by the terms of the original data providers. Contributions are not expected during peer review; questions or reproduction issues should be reported to the corresponding author.
