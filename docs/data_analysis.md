# Dataset Analysis & Profiling

**Amazon ML Challenge 2026 – Business Entity Resolution**  
*Owner: Person 1 (Data / Preprocessing)*

---

## 1. Dataset Overview

The challenge aims to resolve business records across 3 independent, noisy sources:
- **Source 1 (`*_source1.tsv`)**: Deduplicated reference entity source.
- **Source 2 (`*_source2.tsv`)**: Secondary entity source with noisy names and addresses.
- **Source 3 (`*_source3.tsv`)**: Tertiary entity source with noisy names and addresses.
- **Ground Truth (`train_ground_truth.tsv`)**: Mapping of each Source 1 ID to comma-separated matched IDs from Source 2 and Source 3.

### Record Counts & File Sizes (Verified)

| File | Size (MB) | Rows | Unique Entities |
| :--- | :--- | :--- | :--- |
| `train_source1.tsv` | ~200.3 MB | 2,206,821 | S1 references |
| `train_source2.tsv` | ~466.6 MB | 5,034,616 | S2 candidates |
| `train_source3.tsv` | ~480.4 MB | 5,285,603 | S3 candidates |
| `train_ground_truth.tsv` | ~121.1 MB | 2,206,821 | Ground truth mapping |
| `test_source1.tsv` | ~166.9 MB | 1,732,544 | S1 queries |
| `test_source2.tsv` | ~485.9 MB | 4,887,273 | S2 candidates |
| `test_source3.tsv` | ~482.6 MB | 5,082,316 | S3 candidates |

---

## 2. Column Analysis

All entity source files have standard 4 tab-separated columns:

1. `entity_id`: Prefix identifier (`S1-`, `S2-`, `S3-`). Must be preserved as strings.
2. `business_name`: Free-form company name (contains legal entity suffixes, punctuation, typos, abbreviations, Hindi/Devanagari scripts).
3. `business_address`: Free-form address (contains street names, landmark references, PIN/postal codes, missing components).
4. `country`: Country label.

Ground truth format:
1. `source1_entity_id`: S1 record ID.
2. `matched_entity_ids`: Comma-separated list of S2 and S3 IDs (empty string for singletons).

---

## 3. Missing Values

- All files verified to have strict 4-column tab separation.
- Empty fields in `business_address` or `business_name` must be cast to empty string `""` (never `NaN` or string `"nan"`).
- In ground truth, empty `matched_entity_ids` denotes singletons (entities with 0 matches).

---

## 4. Country Distribution

### Train Set
- **US**: ~1,323,633 (60.0%) in S1
- **India**: ~883,188 (40.0%) in S1

### Test Set
- **India**: ~809,986 (46.8%) in S1
- **US**: ~663,106 (38.3%) in S1
- **France**: ~259,452 (15.0%) in S1

> **CRITICAL INSIGHT**: France appears ONLY in the test set. The preprocessing and matching pipelines must treat `country` as an open set. Never hard-code filters or one-hot vectors restricted to `{US, India}`.

---

## 5. Duplicate Analysis

- Source 1 is strictly deduplicated (each record is a distinct real-world entity).
- Sources 2 and 3 can contain multiple records representing the same Source 1 business entity (1-to-many relationship).

---

## 6. Ground-Truth Analysis

From our verification on the 2,206,821 training entities:
- **Singletons (0 matches)**: 123,247 entities (~5.6%).
  - Correctly predicting an empty match for a singleton earns a full score of **1.0**.
  - Incorrectly predicting a false match drops the entity score to **0.0**.
- **Entities with matches ($\ge 1$)**: 2,083,574 entities (~94.4%).
- **Total true matched links**: 7,638,365 pairs.
- **Average matches per non-singleton entity**: ~3.67.

---

## 7. Observations & Key Takeaways

1. **Precision Dominance**: The metric is $F_{0.5}$, penalizing false merges 2× as much as missed links.
2. **Multilingual Text**: Hindi transliterations and Devanagari script appear in Indian records; French characters and address formats appear in test records. Unicode NFKC normalization is necessary.
3. **Legal Suffixes**: Variations like `Pvt Ltd`, `Private Limited`, `Inc`, `Corp`, `LLC`, `SA`, `SAS` should be cleaned to reveal core entity stems.
