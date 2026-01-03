# Disclaimer

This methodology and software are **not financial advice**. The IBAML system is experimental, intended for scientific and research purposes only. Results and outputs should not be used for investment decisions.

This code and methodology are to be published as part of the paper:

**An Explainable ML System for Forecasting Investment Strategy Performance based on IBA Reduction**

# IBAML: Investment Target Forecasting with IBA + XGBoost

A Python framework for machine learning-driven investment target forecasting using Interpolative Boolean Algebra (IBA) feature reduction and XGBoost with custom objective function.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/your-repo/ibaml)

## Overview

**IBAML** implements Interpolative Boolean Algebra (IBA) for feature reduction and trains ML models on top of those features. It combines:

- **IBA (Interpolative Boolean Algebra)**: Boolean polynomial-based feature aggregation
- **Generalized Boolean Polynomials (GBP)**: Product-based feature combinations with negation
- **XGBoost**: Gradient boosting with custom objective function
- **Expanding Window CV**: Time-series aware cross-validation
- **Multi-factor Search**: Exhaustive exploration of factor group combinations

## Key Features

- Time-series aware: Expanding window CV prevents data leakage
- Custom loss function: SE + ME_neg + ME_pos + MSE
- PCA Benchmark: Built-in PCA baseline for comparison
- Reporting: HTML/Markdown reports with figures and tables

## Installation


### Requirements
- Python 3.11+ recommended
- Dependencies managed via `pyproject.toml`

# IBAML: Investment Target Forecasting with IBA + XGBoost

> Experimental research code — not financial advice.

This repository implements Interpolative Boolean Algebra (IBA) feature reduction combined with XGBoost and a custom objective for investment target forecasting. It is intended for research and reproducibility of the accompanying paper.

Badges
-------

- Python: 3.11+
- License: Apache 2.0
- Version: 0.3.1


Quick installation
------------------

Recommended: use the provided bootstrap script which creates a Python 3.11 venv and installs dev dependencies.

```bash
bash env_bootstrap.sh
source .venv/bin/activate
```

Or manually:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[dev]'
```

Usage
-----

Run the main pipeline:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XGBOOST_NUM_THREADS=1

PYTHONPATH=src python -m ibaml.cli configs/config.yml --outdir artifacts --n-jobs-targets 8 --n-jobs-masks 4 --n-jobs-combos 2 --top-k 1 --log-level INFO --html
```

Benchmarks / hyperopt:

```bash
PYTHONPATH=src python -m ibaml.benchmarks.pca configs/config.yml --hyperopt --outdir benchmarking_artifacts
```

Tests
-----

Run the test suite:

```bash
pytest -q
```

Citation
--------

Please cite this work when using the code or methodology. Machine-readable citation metadata is included in the `CITATION.cff` file. A recommended citation format is:

Radosavčević, Aleksa (2026). Investment target forecasting using Interpolative Boolean Algebra + XGBoost. Version 0.3.1. https://github.com/aleksa-r/ibaml

Recommended BibTeX:

```bibtex
@misc{radosavcevic2026investment,
	author = {Aleksa Radosavčević},
	title = {Investment target forecasting using Interpolative Boolean Algebra + XGBoost},
	year = {2026},
	howpublished = {GitHub repository},
	note = {version 0.3.1, https://github.com/aleksa-r/ibaml}
}
```


License
-------

This project is released under the Apache License 2.0. See the `LICENSE` file for details.

Author
------

Aleksa Radosavčević — aleksaradosavcevic@gmail.com

