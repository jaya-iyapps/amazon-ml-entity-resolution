# Matching & Machine Learning Model Guide

**Amazon ML Challenge 2026 – Business Entity Resolution**  
*Owner: Person 2 (Matching / ML)*

---

## 1. Feature Engineering

For each candidate pair $(S_1, S_k)$ produced by blocking, compute pairwise similarity features:

| Feature Name | Description | Range |
| :--- | :--- | :--- |
| `name_ratio` | Standard character sequence matching ratio | $[0.0, 1.0]$ |
| `name_token_sort_ratio` | Token-sorted character similarity (invariance to word ordering) | $[0.0, 1.0]$ |
| `name_jaccard` | Word-level Jaccard overlap of distinctive tokens | $[0.0, 1.0]$ |
| `name_exact_match` | Binary indicator for exact string match after normalization | $\{0, 1\}$ |
| `address_ratio` | Character similarity across address strings | $[0.0, 1.0]$ |
| `address_token_sort_ratio`| Token-sorted address similarity | $[0.0, 1.0]$ |
| `address_jaccard` | Word-level Jaccard overlap of address tokens | $[0.0, 1.0]$ |
| `country_match` | Exact country match (1.0 = match, 0.0 = conflict, 0.5 = missing) | $\{0.0, 0.5, 1.0\}$ |

---

## 2. Similarity Metrics

1. **Token Set Ratio / Token Sort Ratio (`rapidfuzz`)**: Essential for addresses and business names where word positions vary (e.g. "Hospital Memorial" vs "Memorial Hospital").
2. **Levenshtein / Jaro-Winkler**: Highly effective for catching single-letter typos or transliteration discrepancies.
3. **TF-IDF Cosine Overlap**: Weights rare terms (e.g., specific brand names) higher than generic terms (e.g., "enterprises").

---

## 3. Model Experiments

- **Baseline**: Logistic Regression with class-weight balancing.
- **Tree-based**: LightGBM / XGBoost / GBDT for non-linear interactions between name and address similarities.
- **Bi-Encoder / Transformer (Optional, within 8B param budget)**: Cross-lingual or multilingual embeddings (e.g., MiniLM, BGE) fine-tuned on entity pairs.

---

## 4. Evaluation Metric ($F_{0.5}$)

The competition evaluates submissions on macro-averaged $F_{0.5}$:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- Precision is weighted **2×** over recall.
- False merges are penalized much more heavily than missed links.
- Singletons are fully included in the macro average.

---

## 5. Threshold Selection & No-Match Handling

- Predictions are filtered by a decision threshold: $\text{score} \ge \tau$.
- Because of precision weighting, optimal thresholds are typically higher than 0.5 (e.g., $\tau \in [0.55, 0.70]$).
- **Singletons (No Match)**: If no candidate exceeds the threshold $\tau$, the entity is predicted with an empty match list (`""`), receiving a score of **1.0** if it was indeed a true singleton.
- **Tuning**: Grid-search $\tau$ over a held-out validation set using `select_threshold()`.

---

## 6. Error Analysis

- **False Positives (False Merges)**: Two different entities mistakenly linked. Check if address or country mismatches were ignored.
- **False Negatives (Missed Links)**: True matches missed due to extreme transliteration, excessive abbreviations, or candidates omitted by blocking.
