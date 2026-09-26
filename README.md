# Amazon ML Challenge 2026 – Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, high-performance machine learning repository for the **Amazon ML Challenge 2026 – Business Entity Resolution Challenge**.

---

## Problem Statement

In large-scale commercial platforms, business identity data arrives from multiple independent sources, each contributing partial, noisy fragments of information about the same real-world entities.

The objective is to determine which records from **Source 2** and **Source 3** correspond to deduplicated reference entities in **Source 1** despite:
- **Name Variations**: Legal suffixes (`Inc`, `Corp`, `LLC`, `Pvt Ltd`, `SA`), abbreviations, word-order transpositions, transliterations (e.g. Hindi/Devanagari, French), and typos.
- **Address Inconsistencies**: Missing postal codes, informal landmark references, municipal formatting differences, and localized variations.
- **Out-of-Distribution Shift**: The training data contains US and India records, whereas the test set introduces records from France.
- **Precision-Weighted Evaluation**: Evaluated using macro-averaged **$F_{0.5}$**, where precision is weighted 2× over recall, penalizing false merges twice as heavily as missed links. Singletons (entities with 0 matches) must be correctly identified.

---

## Pipeline Architecture

```text
Raw datasets (TSV)
      ↓
Preprocessing & Normalization (Person 1)
      ↓
Inverted Index & Blocking (Person 3)
      ↓
Candidate Generation (`candidate_pairs.tsv`) (Person 3)
      ↓
Pairwise Feature Engineering (Person 2)
      ↓
ML Matching Model (Person 2)
      ↓
Decision Thresholding & Singleton Handling (Person 2)
      ↓
Final Submission (`matching_results.tsv`)
```

---

## Team Division of Responsibilities

This repository is designed so that three team members can develop and experiment in parallel without blocking each other.

| Team Member | Module Ownership | Core Responsibilities |
| :--- | :--- | :--- |
| **Person 1**<br>*(Data / Preprocessing)* | `src/preprocessing/` | • Dataset profiling and summary statistics<br>• Chunked TSV streaming utilities<br>• Unicode NFKC & text normalization<br>• Business name & address standardization<br>• Legal suffix stripping<br>• Open-set country handling<br>• Ground-truth singleton analysis |
| **Person 2**<br>*(Matching / ML)* | `src/matching/` | • Pairwise similarity feature extraction<br>• Character, token-sort, and Jaccard metrics<br>• ML matching classifier training<br>• Inference and probability scoring<br>• Optimal decision threshold tuning ($\tau$)<br>• Macro-averaged $F_{0.5}$ evaluation<br>• No-match / singleton classification |
| **Person 3**<br>*(Blocking / Candidate Gen)* | `src/blocking/` | • High-recall inverted token indexing<br>• Name-based and address-based blocking<br>• Candidate set reduction & pruning<br>• Candidate recall ceiling evaluation<br>• Generation of competition `candidate_pairs.tsv` |

---

## Repository Structure

```text
amazon-ml-entity-resolution/
│
├── README.md                      # Comprehensive project guide and instructions
├── requirements.txt              # Core project dependencies
├── .gitignore                    # Robust gitignore (ignores datasets, binaries, cache)
├── .env.example                  # Environment configuration template
│
├── configs/
│   └── config.yaml               # Centralized parameters, paths, and thresholds
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/            # [Person 1] Data cleaning & normalization
│   │   ├── __init__.py
│   │   ├── normalize.py          # Name, address, country normalization
│   │   └── profiling.py          # TSV chunking, profiling, ground-truth analysis
│   │
│   ├── blocking/                 # [Person 3] Candidate generation & indexing
│   │   ├── __init__.py
│   │   ├── name_blocking.py      # Token-based name inverted index
│   │   ├── address_blocking.py   # Address & postal code blocking
│   │   └── candidate_generator.py # Candidate merging, recall scoring, TSV export
│   │
│   ├── matching/                 # [Person 2] ML matching & evaluation
│   │   ├── __init__.py
│   │   ├── features.py           # Pairwise similarity feature extraction
│   │   ├── train.py              # Classifier training interface
│   │   ├── predict.py            # Scoring, formatting, results export
│   │   └── threshold.py          # Macro F0.5 score calculation & threshold search
│   │
│   └── pipeline.py               # End-to-end integration orchestrator
│
├── tests/                        # Automated unit tests for all modules
│   ├── __init__.py
│   ├── test_preprocessing.py     # Tests for normalization and loading
│   ├── test_blocking.py          # Tests for indexing and candidate generation
│   └── test_matching.py          # Tests for features, metric, and thresholding
│
├── notebooks/                    # Exploratory analysis and prototyping
│   └── README.md                 # Notebook guidelines and git hygiene
│
├── docs/                         # Detailed documentation and logs
│   ├── data_analysis.md          # Dataset profiling, distributions, observations
│   ├── blocking.md               # Blocking strategies and reduction metrics
│   ├── matching.md               # Model architectures and similarity metrics
│   └── experiments.md            # Shared experiment tracking table
│
└── output/
    └── .gitkeep                  # Preserved directory for competition outputs
```

---

## Required Submission Output Files

The pipeline generates two tab-separated files in `output/`:

1. **`matching_results.tsv`** *(Scored on Leaderboard)*
   ```tsv
   source1_entity_id	matched_entity_ids
   S1-00001	S2-00047,S2-00193,S3-00812
   S1-00002	S3-00004
   S1-00003	
   ```
   *Every Source 1 entity must have exactly one row. Singletons have an empty list.*

2. **`candidate_pairs.tsv`** *(Audit Deliverable)*
   ```tsv
   source1_entity_id	candidate_entity_ids
   S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
   S1-00002	S3-00004
   S1-00003	
   ```
   *Represents candidate pairs from blocking. Every match in `matching_results.tsv` must appear here.*

---

## Development Workflow

### Git Branches

Each team member must work on their dedicated feature branch:
- `main` — Stable, tested integration branch.
- `person1-data` — Feature development for preprocessing and profiling.
- `person2-model` — Feature development for similarity metrics and ML matching.
- `person3-blocking` — Feature development for inverted indexes and blocking.

### Important Git Rules

To keep the repository clean and under Git storage limits, **NEVER COMMIT**:
- Raw or processed dataset files (`dataset/`, `*.tsv`, `*.csv`)
- Generated large candidate files or outputs (`output/*.tsv`)
- Model binaries or pickle files (`*.pkl`, `*.joblib`)
- Secrets, credentials, or `.env` files
- Temporary experiment scratch files or large notebook outputs

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd amazon-ml-entity-resolution

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Run Pipeline Check

```bash
python3 -m src.pipeline --config configs/config.yaml
```
