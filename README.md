# K-Prototypes Proxy Alignment

Research artifacts for studying proxy-alignment diagnostics in log-space hyperparameter search for k-prototypes clustering.

## Overview

This repository packages the manuscript, figure assets, experimental result files, and utility scripts associated with the project on proxy alignment for k-prototypes model selection.

The current public release is organized as a research-artifact repository rather than a fully packaged software library. Its main purpose is to support paper submission, result inspection, and figure reproduction.

## Repository Structure

- `paper_material/` — submission manuscript source, compiled PDF, class files, and figure assets
- `results/` — experiment outputs used in the paper analyses
- `src/` — project module layout reserved for source components
- `experiments/` — experiment entry directory reserved for executable workflows
- `docs/` — supplementary project documentation
- `replot_figures.py` — utility script for regenerating figures from saved results
- `verify_plots.py` — utility script for validating generated plots

## Included Materials

This release currently includes:

- final submission manuscript files
- proxy-alignment figures
- benchmark result tables and serialized run artifacts
- lightweight plotting and verification utilities

## Suggested Citation

If you use these materials, please cite the associated manuscript on proxy-alignment diagnostics for log-space hyperparameter search in k-prototypes clustering.

## Notes

- Temporary logs, caches, and local-only artifacts were removed for a cleaner public presentation.
- Some directories are retained as project structure placeholders for future code release expansion.
