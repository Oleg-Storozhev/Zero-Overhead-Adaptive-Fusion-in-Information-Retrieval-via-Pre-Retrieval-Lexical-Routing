# Zero-Overhead Adaptive Fusion in Information Retrieval via Pre-Retrieval Lexical Routing

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-13.1-76b900)
![BEIR](https://img.shields.io/badge/Benchmark-BEIR-2f855a)
![License](https://img.shields.io/badge/Status-Research%20Code-6b7280)

## Overview

This repository implements **AlphaRouter**, a lightweight dynamic fusion framework for hybrid Information Retrieval (IR). The system predicts a **query-specific interpolation weight** `alpha` for combining:

- **Dense retrieval** scores from `sentence-transformers/all-MiniLM-L6-v2`
- **Sparse retrieval** scores from BM25 (`bm25s`)

Unlike standard hybrid search, which applies a **single static fusion weight** to all queries, AlphaRouter performs **pre-retrieval lexical routing** using only inexpensive query-side features. The design target is explicit: **adaptive fusion without transformer-scale inference overhead**.

The project includes two router architectures:

- `AlphaRouterMLP`: a compact MLP-based router intended as the primary zero-overhead model
- `AlphaRouter`: a larger gated residual router for comparison

Both models are trained from query-level meta-features extracted before fusion. The feature vector contains **17 lexical, statistical, and lightweight NLP signals**, including:

- query length
- IDF statistics
- rare-token and OOV ratios
- IDF skewness
- query entropy
- digit, uppercase, and punctuation ratios
- stopword and POS-tag ratios from SpaCy
- average token length

Training follows a **learning-to-rank formulation** rather than pseudo-label regression. Instead of regressing to oracle `alpha` values with MSE, the routers are optimized end-to-end using a **listwise InfoNCE objective** over positive and negative retrieval scores. This directly trains the model to improve ranking quality under hybrid fusion.

## Method Summary

The end-to-end pipeline is:

1. Download BEIR datasets and run dense and sparse retrieval.
2. Min-max normalize dense and sparse scores per query.
3. Compute per-dataset lexical statistics such as IDF dictionaries and vocabularies.
4. Extract 17 query meta-features.
5. Construct training triplets from relevant and non-relevant documents.
6. Train query routers with an InfoNCE hybrid ranking loss.
7. Evaluate zero-shot dynamic routing against dataset-level static alpha baselines.
8. Generate SHAP-based feature attributions for interpretability.

## Installation

### Requirements

The codebase depends on the following core packages:

- `torch`
- `spacy`
- `beir`
- `bm25s`
- `shap`
- `matplotlib`
- `numpy`
- `scipy`

Dense retrieval is performed through the BEIR stack and uses **SentenceTransformers** internally. In practice, `sentence-transformers` and its transformer dependencies must be available in the environment.

Recommended runtime target:

- `CUDA 13.1`

### Setup

Create a Python environment and install the project requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers
python -m spacy download en_core_web_sm
```

### Notes

- A CUDA-enabled PyTorch installation is recommended for training and evaluation, but the code falls back to CPU automatically.
- The intended GPU environment for this repository is `CUDA 13.1`.
- The first data-preparation run will download BEIR datasets into `datasets/`.
- Dense retrieval will also download the `all-MiniLM-L6-v2` model on first use.

## Usage

Run the pipeline from the repository root in the following order.

### 1. Prepare Retrieval Caches

This step downloads BEIR datasets, runs dense and sparse retrieval, normalizes scores, computes IDF statistics, and writes cached artifacts for subsequent training and evaluation.

```bash
python scripts/01_prepare_data.py
```

Example with explicit seed and cache directory:

```bash
python scripts/01_prepare_data.py \
  --seed 42 \
  --cache-dir cached_data
```

Expected outputs:

- cached retrieval artifacts for train and test splits
- `cached_data/data_preparation_manifest.json`
- BEIR datasets stored under `datasets/`

### 2. Train Query Routers

This step loads cached training data, builds training triplets, extracts the 17-dimensional query feature vectors, and trains both router architectures with the InfoNCE hybrid ranking loss.

```bash
python scripts/02_train_models.py
```

Example with explicit hyperparameters:

```bash
python scripts/02_train_models.py \
  --cache-path cached_data/train_retrieval_data_cached_v1.pkl \
  --batch-size 64 \
  --epochs 62 \
  --seed 42
```

Expected outputs:

- `models/alpharouter_advanced_v1.pth`
- `models/alpharouter_mlp_v1.pth`
- `results/training_summary.json`

### 3. Evaluate Zero-Shot Routing

This step loads the trained models and cached test data, runs zero-shot inference across BEIR datasets, compares dynamic routing against static dataset-level alpha sweeps, and generates interpretability plots.

```bash
python scripts/03_evaluate.py
```

Example with explicit artifact locations:

```bash
python scripts/03_evaluate.py \
  --cache-path cached_data/test_retrieval_data_cached_v1.pkl \
  --models-dir models \
  --output-dir results/plots \
  --seed 42
```

Expected outputs:

- static oracle benchmark plots in `results/plots/`
- SHAP feature-attribution plots in `results/plots/`
- `results/plots/evaluation_summary.json`
- printed NDCG@10 and MRR@10 metrics for static and dynamic settings

## Repository Structure

```text
AdaptiveSparseDense/
├── README.md
├── requirements.txt
├── hybrid_benchmark.pdf
├── cached_data/
│   └── *.pkl
├── datasets/
│   └── <beir-datasets>/
├── models/
│   └── *.pth
├── notebooks/
│   └── work_pipeline.ipynb
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train_models.py
│   └── 03_evaluate.py
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── evaluation.py
    ├── features.py
    ├── loss.py
    └── models.py
```

### Module Roles

- `src/data_loader.py`: BEIR download, dense retrieval, BM25 retrieval, score normalization, and dataset artifact preparation
- `src/features.py`: extraction of the 17 lexical and statistical query features
- `src/loss.py`: listwise InfoNCE objective for hybrid fusion training
- `src/models.py`: dataset wrapper and router architectures
- `src/evaluation.py`: static oracle evaluation, zero-shot dynamic evaluation, and SHAP analysis

## Training Objective

The training loss is implemented in [`src/loss.py`](src/loss.py). For each query, the router predicts `alpha`, which is used to interpolate dense and sparse scores:

```text
hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
```

Positive and negative hybrid scores are then optimized with an InfoNCE loss. This design aligns optimization with ranking behavior more directly than regression to oracle fusion weights.

## Evaluation Protocol

The evaluation code supports two complementary settings:

- **Static Oracle Evaluation**: grid-search over fixed dataset-level `alpha` values in `[0.0, 1.0]` with step `0.05`
- **Dynamic Routing Evaluation**: per-query `alpha` prediction using the trained router in zero-shot transfer

Reported metrics include:

- `NDCG@10`
- `MRR@10`

The default test suite evaluates the following BEIR datasets:

- `scifact`
- `fiqa`
- `nfcorpus`
- `scidocs`
- `nq`
- `quora`
- `msmarco`

## SHAP Interpretability

The repository includes a SHAP-based post hoc analysis pipeline in [`src/evaluation.py`](/src/evaluation.py). After zero-shot evaluation, the code samples query feature vectors across datasets and computes feature attributions for the trained router.

This analysis is intended to quantify which lexical cues most strongly influence routing decisions, including IDF-driven specificity features, entropy, and lightweight POS-derived signals.

## Reproducibility Notes

- Run all commands from the repository root.
- Ensure the SpaCy English model `en_core_web_sm` is installed before running any feature extraction step.
- Data downloads and model downloads occur lazily during the first execution of the pipeline.
- Use the `--seed` flag across data preparation, training, and evaluation to make sampling behavior reproducible.
- The executable script directory is currently named `scripts/` in the repository and the commands above reflect the repository as-is.
