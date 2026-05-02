# Synthetic Datasets

These seven datasets are controlled stress-test scenarios for evaluating gamma search in K-Prototypes clustering. They are not intended to represent natural populations. Their purpose is to isolate feature-scale, feature-type balance, and categorical-noise conditions under which gamma selection can be reliable or unreliable.

## Files

- `H1-Easy.csv`, `H1-Medium.csv`, `H1-Hard.csv`: H1 scenarios with four continuous and two categorical features.
- `H2-Easy.csv`, `H2-Medium.csv`, `H2-Hard.csv`: H2 scenarios with stronger categorical-feature influence and fewer continuous features.
- `Control-Balanced.csv`: a sanity-check scenario where the default gamma heuristic is comparatively well calibrated.
- `synthetic_generation_config.json`: generation parameters, random seed, feature names, categorical indices, and label-use notes.

## Label use

The `true_label` column is provided to allow reviewers to verify external clustering metrics such as ARI and NMI. It is not used by the unsupervised gamma-search procedures. Gamma search optimizes the internal K-Prototypes objective only.

## Reproducibility

The datasets were generated with random seed 42 using the generator functions in `experiments/run_synthetic_benchmark.py`. The generated CSV files are included directly so that reviewers can inspect and reuse the exact benchmark inputs.
