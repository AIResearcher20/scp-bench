# scp-bench

## A Reproducible Benchmarking Framework for Single-Cell Proteomics Classification

**Project Year:** 2025
**Repository:** Consolidated in 2026 from earlier development
**License:** MIT
**Author:** Sepideh Moafi

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

scp-bench is an open-source benchmarking framework for evaluating machine-learning models on real single-cell proteomics (SCP) data. It provides a leakage-safe, reproducible pipeline with cell-aware cross-validation, standardized evaluation, and computational profiling.

Benchmarked on the SCoPE2 dataset (Specht et al., Genome Biology 2021) from the Bioconductor scpdata package for Macrophage vs. Monocyte classification.

---

## Key Features

- Leakage-safe preprocessing: KNN imputation and CLR normalization fitted only on training data
- Cell-aware stratified cross-validation (StratifiedKFold)
- Five machine-learning models: Logistic Regression, Random Forest, XGBoost, LightGBM, PyTorch MLP
- Eight evaluation metrics: accuracy, balanced accuracy, precision, recall, F1-macro, F1-micro, AUROC, confusion matrix
- Computational profiling: runtime and peak memory
- Automated JSON reporting
- CI/CD with GitHub Actions (pytest + Docker build)
- Reproducible via Dockerfile
- pip-installable package

---

## Dataset

| Property | Value |
|----------|-------|
| Source | SCoPE2 (Specht et al., Genome Biology 2021) |
| Package | Bioconductor scpdata |
| Proteins | 3,042 |
| Cells | 1,490 |
| Classes | Macrophage (1,096), Monocyte (394) |

---

## Benchmark Results

| Model | CV F1 Macro | Test Accuracy | F1 Macro | Time (s) | Memory (MB) |
|-------|-------------|---------------|----------|----------|-------------|
| LogisticRegression | 0.9536 ± 0.0047 | 96.31% | 0.9542 | 0.31 | 28.69 |
| RandomForest | 0.9228 ± 0.0101 | 94.97% | 0.9341 | 6.04 | 14.12 |
| XGBoost | 0.9636 ± 0.0216 | 96.98% | 0.9617 | 14.51 | 0.16 |
| LightGBM | 0.9696 ± 0.0178 | 98.32% | 0.9786 | 34.01 | 2.21 |
| PyTorch MLP | 0.9607 ± 0.0043 | 95.97% | — | 11.26 | 13.86 |

Best model: LightGBM — 98.32% test accuracy.

### Visualizations

The benchmark produces eight figures in `results/figures/`:

- Test accuracy by model
- Cross-validated F1 by model
- F1 macro by model
- Training runtime by model
- Peak memory by model
- Confusion matrices
- Summary heatmap
- Feature importance

Exploratory analyses are available in `notebooks/`.

---

## Scientific Validation

| Analysis | Result |
|----------|--------|
| Majority-class baseline | 73.56% |
| Real model balanced accuracy | 98.00% |
| Permuted balanced accuracy | 49.59% ± 2.86% |
| Signal gap | +48.4% |

### Feature Importance

Top protein: Vimentin (P08670), a known macrophage activation marker.

---

## Quick Start

git clone https://github.com/AIResearcher20/scp-bench.git
cd scp-bench
pip install -r requirements.txt
python benchmarks/run_benchmark.py data/processed/scope2_real.csv

---

## Docker

docker build -t scp-bench .
docker run scp-bench

---

## Testing

pytest tests/ -v --cov=src

---

## Project Structure

scp-bench/
├── src/
│   ├── preprocessing/
│   ├── evaluation/
│   ├── models/
│   └── data_loader/
├── benchmarks/
├── tests/
├── notebooks/
├── results/
│   └── figures/
├── .github/workflows/
├── Dockerfile
├── setup.py
└── requirements.txt

---

## Technologies

Python · scikit-learn · XGBoost · LightGBM · PyTorch · pandas · NumPy · matplotlib · seaborn · pytest · GitHub Actions · Docker

---

## Citation

Specht H, Emmott E, Petelski AA, et al. Single-cell proteomic and transcriptomic analysis of macrophage heterogeneity using SCoPE2. Genome Biology 22, 50 (2021).

---

## License

MIT
