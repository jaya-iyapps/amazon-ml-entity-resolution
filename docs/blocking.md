# Blocking & Candidate Generation Guide

**Amazon ML Challenge 2026 – Business Entity Resolution**  
*Owner: Person 3 (Blocking / Candidate Generation)*

---

## 1. Blocking Objective

Comparing every Source 1 entity against all records in Source 2 and Source 3 results in:
$$1.73 \times 10^6 \times (4.89 \times 10^6 + 5.08 \times 10^6) \approx 1.7 \times 10^{13} \text{ pairs}$$
Evaluating pairwise ML models on 17 trillion pairs is computationally infeasible.

The goal of **blocking** is to:
1. Filter the comparison space by orders of magnitude (e.g. down to 20–50 candidates per entity).
2. Maximize **candidate recall ceiling** (capturing $\ge 95\%$ of true matches).
3. Produce the competition deliverable: `candidate_pairs.tsv`.

---

## 2. Candidate Generation Strategies

### A. Name-Based Inverted Index
- Tokenize normalized business names (character length $\ge 3$).
- Build inverted token index: `token -> {entity_ids}`.
- Query tokens of Source 1 entity to retrieve top matching targets.
- Token frequency dampening: ignore extremely frequent stop tokens (e.g., "solutions", "services", "store").

### B. Address-Based Indexing
- Extract postal codes / PIN codes, city tokens, and street identifiers.
- Constrain address candidates to matching countries.

### C. Locality-Sensitive Hashing (LSH) / MinHash
- Character n-grams (3-grams, 4-grams) hashed with MinHash.
- LSH bands group similar names even with minor spelling typos and transliteration variations.

---

## 3. Candidate Reduction

- Filter pairs across differing countries (unless country is ambiguous/unknown).
- Rank candidates by token overlap or lightweight Jaccard score before taking top $K$ (e.g., $K=50$).
- Enforce strict candidate pool caps to prevent memory explosion during inference.

---

## 4. Candidate Recall Evaluation

Candidate recall is calculated using `evaluate_candidate_recall()`:

$$\text{Candidate Recall} = \frac{|\text{True Ground Truth Matches in Candidates}|}{|\text{Total Ground Truth Matches}|}$$

*Note: Any true match not captured in `candidate_pairs.tsv` can never be recovered by the downstream matching model.*

---

## 5. Experiments Log

| Exp ID | Strategy | Max Candidates / Entity | Total Candidates | Candidate Recall | Reduction Ratio | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BLK-01 | Baseline inverted index (name tokens) | 50 | TBD | TBD | TBD | Initial baseline |
| BLK-02 | Name tokens + address PIN codes | 50 | TBD | TBD | TBD | Added address keys |

---

## 6. Final Strategy & Verification

The final candidate pool will be exported to:
`output/candidate_pairs.tsv`

Format:
```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```
Verified via `utils/validate_submission.py`.
