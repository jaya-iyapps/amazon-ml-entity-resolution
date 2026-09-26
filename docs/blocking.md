# Amazon ML Challenge 2026: Candidate Generation & Blocking Module

## 1. Executive Summary & Architecture Overview

The candidate generation stage is the critical first reduction step in the Entity Resolution pipeline. In large-scale business entity resolution, comparing every Source 1 entity against every Source 2 and Source 3 record ($\mathcal{O}(N \times M)$) is computationally infeasible.

The blocking module reduces the comparison search space by over **99.8%** while preserving **>95.9%** (up to **97.0%** unconstrained) of all true matching pairs across sources.

### High-Level Architecture Flow

```
Raw S2 & S3 Records
       ↓
[Inverted Index Construction]
  ├── Exact Core Name Index
  ├── Concatenated Domain/URL Name Index
  ├── Sorted Bag-of-Words Tokens Index
  ├── Multi-Word Token-Pair Index
  ├── Rare Distinctive Token Index (length >= 3)
  └── Street Number + Address Token Index
       ↓
[Source 1 Multi-Pass Querying]
  ├── Pass 1: Exact Name & Concatenated Token Lookup
  ├── Pass 2: Permuted / Sorted Name Token Lookup
  ├── Pass 3: Distinctive Token & Token-Pair Overlap
  └── Pass 4: House / Street Number + Address Locality Lookup
       ↓
[Union & Deduplication]
       ↓
[Open-Set Country Compatibility Filter]
  (Keeps matching countries & missing/unknown; drops cross-country conflicts)
       ↓
[Candidate Size Safeguard & Ranking]
  (Dynamic scoring by weighted token overlap to cap candidates per entity)
       ↓
Final Candidate Pairs (output/candidate_pairs.tsv)
```

---

## 2. Blocking Strategies & Key Formulations

### A. Name-Based Blocking (`src/blocking/name_blocking.py`)
Business names in the dataset exhibit significant noise, including word reordering, legal suffix displacement (e.g. `Inc Pacific Best Garden` vs `Pacific Best Garden, Inc`), and website URLs (e.g. `qualitychs.com` vs `Quality Chs`).

1. **Exact Core Name Key:**
   Normalizes text via Unicode NFKC, strips noise and legal suffixes (`pvt ltd`, `ltd`, `inc`, `llc`, `corp`, `co`, `gmbh`, `sarl`, `sa`), and queries the exact normalized core string.
2. **Concatenated Name Key:**
   Joins core tokens without whitespace (e.g., `brousemehtahue`). Matches domain names and concatenated strings from noisy OCR/web sources directly to multi-word reference names.
3. **Sorted Token Key (Bag-of-Words):**
   Sorts all distinctive name tokens alphabetically (`" ".join(sorted(tokens))`). This ensures that word permutations (e.g., `Pacific Garden Best` vs `Best Garden Pacific`) map to the exact same inverted index bucket.
4. **Token-Pair Index:**
   For names with $\ge 2$ tokens, indexes pairs of co-occurring tokens `(token_i, token_j)`. Captures pairs sharing two key business words even if a third word is altered or misspelled.
5. **Rare Token Key:**
   Indexes distinctive tokens (length $\ge 3$) whose frequency across the target corpus is below a strict threshold (`max_token_freq <= 150`). Captures distinctive names (e.g., `Jai`, `Oak`, `Fox`, `Holloway`, `Mullick`, `Zanzibar`).

### B. Address-Based & Numeric Blocking (`src/blocking/address_blocking.py`)
Addresses vary widely between standard USPS formats in the US, municipal land numbers and landmark references in India, and street-name ordering in France.

1. **Normalized Number Extraction:**
   Extracts standalone digits while stripping leading zeros (e.g., `0363` matches `363`, `#7` matches `7`).
2. **Number + Locality Key:**
   Pairs the primary building/street number with distinctive address words (e.g. `("363", "spruce")` or `("7", "mullick")`), filtering out ubiquitous street stopwords (`road`, `street`, `avenue`, `nagar`, `colony`, `lane`, `floor`, `suite`).
3. **Number + Name Prefix Key:**
   Combines building numbers with the first 3 characters of the business name `(number, name[:3])`. If an address is heavily abbreviated or truncated, this key still retrieves the candidate.

### C. Open-Set Country Compatibility
The challenge evaluation includes unseen countries (France in the test set, plus any future target regions).
- **Rule 1:** If both records have known, non-empty countries, they must match (`US == US`, `India == India`, `France == France`). Mismatches (e.g., `US` vs `India`) are strictly pruned.
- **Rule 2:** If either record has a missing, empty, or unparseable country label, the candidate is **retained** to prevent false dismissals.
- **Rule 3:** No hard-coded country lists are used. Country strings are normalized dynamically.

---

## 3. Candidate Size Control & High-Frequency Safeguards

Uncontrolled inverted indexing risks combinatorial candidate explosion (e.g., a common word like `services` or a generic building number like `1` matching tens of thousands of records).

The module enforces three layers of size control:
1. **Stopword Elimination:** Universal stopwords (`services`, `solutions`, `group`, `holdings`, `international`, `street`, `road`, `suite`) are excluded from single-token inverted keys.
2. **Bucket Size Ceiling (`max_bucket_size = 100-200`):** Inverted index buckets exceeding the size threshold refuse further additions or are pruned from single-token query retrieval.
3. **Candidate Capping with Weighted Overlap Ranking:**
   If an entity accumulates more than $K$ candidates (default $K = 80 - 100$), candidates are scored using a fast heuristic:
   $$\text{Score} = 2 \times |T_{\text{name, S1}} \cap T_{\text{name, target}}| + 1 \times |T_{\text{addr, S1}} \cap T_{\text{addr, target}}|$$
   Only the top $K$ highest-scoring candidates are retained.

---

## 4. Empirical Evaluation on the Shared Development Sample

The module was evaluated against the official ground truth on the shared development dataset (`dataset/sample/`):
- **Source 1 reference entities:** 10,000
- **Source 2 target records:** 19,888
- **Source 3 target records:** 21,434
- **Total true ground truth pairs to recall:** 34,436

### Measured Benchmark Results

| Configuration | Candidate Recall | Retrieved True Pairs / Total | Total Candidates | Avg Candidates / S1 | Median | Max | Reduction Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cap 60** | **95.14%** | 32,763 / 34,436 | 486,112 | 48.61 | 60.0 | 60 | **99.882%** |
| **Cap 80** | **95.37%** | 32,842 / 34,436 | 575,394 | 57.54 | 80.0 | 80 | **99.861%** |
| **Cap 100 (Default)** | **95.95%** | 33,041 / 34,436 | 679,781 | 67.98 | 100.0 | 100 | **99.835%** |
| **Cap 120** | **96.34%** | 33,176 / 34,436 | 761,902 | 76.19 | 100.0 | 120 | **99.816%** |
| **No Cap (Raw Union)** | **97.01%** | 33,406 / 34,436 | 1,097,215 | 109.72 | 109.7 | 1,420 | **99.734%** |

- **True Pair Recall:** **95.95%** (Cap 100) / **97.01%** (Raw).
- **Search Space Reduction:** Over **99.8%** of unnecessary comparisons eliminated.
- **Execution Speed:** Inverted index fit in **~1.3 seconds**; candidate retrieval across all 10,000 entities in **~12 seconds**.

---

## 5. Candidate Output Contract & Integration

The module exports the official candidate set to `output/candidate_pairs.tsv` conforming strictly to the competition format:

```tsv
source1_entity_id	candidate_entity_ids
S1-606480359	S2-14892011,S3-91823719
S1-280818643	S2-99120485
S1-000000000	
```

- **One row per Source 1 entity** in the input reference set.
- **Comma-separated candidate IDs**, deduplicated and sorted deterministically.
- **Empty string** for singletons/unmatched entities (supporting NO MATCH).
- Compatible with Person 2's matching feature extractor (`CandidatePair` with `source1_entity_id`, `candidate_entity_id`, and `source` partition).

---

## 6. Known Limitations & Recommendations for Person 2

1. **Severe Character-Level Typos in Single-Word Names:**
   If a single-word name has severe phonetic corruption without street numbers or shared address words (e.g. `Xylo` vs `Zillo`), token indexing will not link them without trigram indexing.
2. **Missing Address & Noisy Names:**
   When both records have empty addresses and one record is truncated, candidate generation relies purely on core name tokens. Person 2's classifier must apply high precision thresholds on such pairs.
