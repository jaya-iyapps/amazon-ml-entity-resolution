# Data Analysis — Amazon ML Challenge 2026: Business Entity Resolution

## 1. Dataset Structure

The challenge archive (`mldataset.zip`) extracts to `student_resource/` and contains:

```
student_resource/
├── README.md
├── Documentation_template.md
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv       ~200 MB uncompressed
│   │   ├── train_source2.tsv       ~467 MB uncompressed
│   │   ├── train_source3.tsv       ~480 MB uncompressed
│   │   └── train_ground_truth.tsv  ~121 MB uncompressed
│   └── test/
│       ├── test_source1.tsv        ~167 MB uncompressed
│       ├── test_source2.tsv        ~486 MB uncompressed
│       └── test_source3.tsv        ~483 MB uncompressed
└── utils/
    └── validate_submission.py
```

**Total uncompressed data: ~2.5 GB across all 7 TSV files.**

---

## 2. Source Roles

| Source | Role |
|--------|------|
| Source 1 (S1) | Deduplicated reference source. Canonical business records. |
| Source 2 (S2) | Noisy / partial duplicate of real records. May have abbreviations, typos, transliterations. |
| Source 3 (S3) | Same as S2 — a second noisy/partial source. |

The matching task is: for each S1 record, find all S2 and S3 records that refer to the same real-world business entity. S1→S1 matching is not required. S2→S3 matching is not required.

---

## 3. Source Column Schema

All three source files (S1, S2, S3) share the same four columns:

| Column | Type | Description |
|--------|------|-------------|
| `entity_id` | string | Unique ID with source prefix: `S1-`, `S2-`, `S3-` |
| `business_name` | string | Business name — may include abbreviations, typos, transliterations, missing words |
| `business_address` | string | Business address — partial, landmark-based, missing components, non-standard formats |
| `country` | string | Country label — **open-set** (not limited to a fixed list) |

**Critical format note:** Files are TSV (tab-separated), not CSV. Commas appear inside data fields (e.g., in addresses) and in the ground truth `matched_entity_ids` field. Always parse with `sep='\t'`.

---

## 4. Ground Truth Format

File: `train_ground_truth.tsv`

| Column | Type | Description |
|--------|------|-------------|
| `source1_entity_id` | string | An S1-prefixed ID |
| `matched_entity_ids` | string | Comma-separated list of S2-/S3- IDs that match this S1 record |

**Corner case:** When an S1 entity has no matches, `matched_entity_ids` is an **empty string** (not absent, not NaN). Parsers must handle this explicitly — do not confuse an empty string with a missing value.

Example rows:
```
source1_entity_id   matched_entity_ids
S1-00001            S2-01234,S3-05678
S1-00002            S2-09999
S1-00003            (empty string)
```

---

## 5. Approximate Dataset Scale

| File | Uncompressed Size |
|------|------------------|
| `train_source1.tsv` | ~200 MB |
| `train_source2.tsv` | ~467 MB |
| `train_source3.tsv` | ~480 MB |
| `train_ground_truth.tsv` | ~121 MB |
| `test_source1.tsv` | ~167 MB |
| `test_source2.tsv` | ~486 MB |
| `test_source3.tsv` | ~483 MB |
| **Total** | **~2.5 GB** |

Exact row counts within each file have not yet been measured (that requires reading the actual data). The sizes above are from the ZIP's stored metadata.

---

## 6. Observed Noise Patterns

The following noise patterns are documented in the challenge README:

### Business names
- **Abbreviations:** "St" for "Street" within names, "Corp" / "Corporation", "Ltd" / "Limited", "Inc" / "Incorporated", regional equivalents ("Pvt Ltd", "GmbH", "SARL")
- **Typos and OCR errors:** character substitutions, transpositions, missing or extra characters
- **Transliterations:** the same business name rendered in Latin script vs. Devanagari (Hindi), Arabic, or other scripts
- **Reordered tokens:** "Acme Trading Company" vs. "Trading Company Acme"
- **Missing tokens:** partial names, missing articles, dropped words
- **Unicode variants:** composed vs. decomposed characters, fullwidth vs. halfwidth digits

### Addresses
- **Partial addresses:** only city + PIN code, or only a landmark description
- **Landmark-based:** "Near Bus Stand", "Opposite SBI Bank" — no street number
- **Missing components:** no house number, no postal code, no state
- **Transliterated:** same address in Latin and Devanagari script
- **Non-standard formats:** no universal address structure across countries or even within one country

---

## 7. Unicode Considerations

- Data contains at minimum: English (Latin), Hindi (Devanagari), and French (accented Latin). Test data introduces French; more scripts may be present.
- **Do not transliterate to ASCII.** Transliteration is lossy and destroys the ability to match records across scripts (the two copies of the same transliterated name may or may not agree).
- Apply **NFC normalization** (canonical composition) universally. The same character can be represented as a precomposed codepoint (U+00E9 "é") or as a base letter + combining accent (U+0065 + U+0301). NFC unifies these.
- Strip **invisible formatting characters** (zero-width non-joiner, soft hyphen, BOM, direction marks) that may separate otherwise-identical strings.
- **Do not use `\W` with `re.UNICODE` for token splitting.** Python's `\W` (non-word characters) with `re.UNICODE` incorrectly classifies Devanagari combining marks — nukta (U+093C), anusvara (U+0902), and virama (U+094D) — as non-word characters (Unicode category Mn), which splits Hindi words at those marks. The implementation splits only on whitespace and ASCII punctuation using `re.split(r"[\s\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]+", s)`. This range covers standard ASCII separators (space through DEL range) without touching any non-ASCII codepoints, preserving Devanagari word integrity while still splitting on meaningful delimiters.

---

## 8. Address Considerations

- **No standard format.** Indian addresses may be "village – tehsil – district – state – PIN". French addresses are "number rue name". US addresses are "number street, city, state ZIP". Addresses from other countries follow their own conventions.
- **Preserve all numeric tokens.** House numbers, unit numbers, floor numbers, PIN codes, and postal codes are high-signal identifiers.
- **Do not parse into components.** A universal address parser does not exist given this open-country dataset. Use the address as a bag of tokens + numeric tokens.
- **Aggressive normalization is dangerous.** Removing digits to "clean" an address removes the most discriminative tokens.

---

## 9. Country / Open-Set Consideration

- Training data contains at least: **US**, **India**
- Test data adds at least: **France**
- Additional countries may appear in test data that are not in training data.
- **Never hardcode the country list.** Country normalization must handle arbitrary country values by passing them through cleanly rather than mapping unknown values to a sentinel.
- Country labels themselves may appear with variants: "USA" / "US" / "United States of America"; "IN" / "India" / "Bharat". Canonical alias mapping handles known variants; unknown variants pass through.

---

## 10. Memory and Streaming Considerations

- File sizes for S2 and S3 (train and test) exceed **460 MB each** uncompressed.
- Loading a single source file in full at once would require ~500 MB RAM for the raw DataFrame, plus additional memory for derived columns, copies, and overhead. On a machine with 8 GB or less of RAM, loading all files simultaneously is impractical.
- **Design decision:** All source files must be processed in chunks (`chunksize=50,000` rows by default, tunable). Only the ground truth (~121 MB) is small enough to load fully.
- The profiling utilities are chunk-compatible: `profile_source_chunk()` and `profile_ground_truth_chunk()` operate on a single chunk and return dicts; `merge_profiles()` aggregates across all chunks.
- The preprocessing functions (`preprocess_dataframe()`) are designed to work on a chunk DataFrame rather than requiring the full file.

---

## 11. Preprocessing Design Decisions

| Decision | Rationale |
|----------|-----------|
| NFC normalization applied first | Unifies composed/decomposed Unicode before any further processing; prevents false inequality between semantically identical strings |
| Unicode preserved, not ASCII-stripped | Transliteration is lossy; stripping destroys cross-script matching signal |
| Lowercase applied after NFC | Case folding is safe for matching; applying it after NFC avoids ordering issues with case-sensitive combining marks |
| Invisible characters removed | Zero-width chars, BOM, direction marks can split otherwise-identical strings |
| Legal suffix variants normalized to canonical | Reduces token vocabulary; both unsuffixed and suffixed forms are kept for flexibility |
| Address abbreviations expanded (English subset only) | "St" → "street" reduces vocabulary without losing meaning |
| Numeric tokens extracted separately | High-signal identifiers (postal codes, house numbers) benefit from exact-match comparison; separate extraction avoids polluting fuzzy matching |
| Empty `matched_entity_ids` → empty list | Prevents confusion between "no matches" (valid) and "missing data" (invalid) |
| `entity_id` read as string dtype | Prevents pandas from silently interpreting "S1-123" as float NaN or numeric |
| `keep_default_na=False`, `na_values=[]` | Prevents "NA", "None", "null" from being misread as NaN — these may appear in legitimate business names or addresses |
| Chunk size 50,000 rows | At ~200 chars/row average, this is ~10 MB/chunk — safe on any reasonable machine |
| Original fields preserved alongside derived fields | Downstream consumers (Person 2, Person 3) can choose to use raw or normalized values per use case |
| No global state | Functions are pure and stateless; safe to use in parallel processing across chunks |

---

## 12. Evaluation Metric

**F_0.5 macro-averaged** across all S1 entities:

```
F_0.5 = (1 + 0.5²) × Precision × Recall / (0.5² × Precision + Recall)
       = 1.25 × Precision × Recall / (0.25 × Precision + Recall)
```

F_0.5 weights **precision more than recall**: a false merge (incorrectly linking two different businesses) is penalized more heavily than a missed match. This means:
- Over-aggressive matching (merging too many records) hurts the score more than under-matching.
- Preprocessing should avoid transformations that collapse distinct businesses into identical-looking tokens.

**Only `matching_results.tsv` is scored.** `candidate_pairs.tsv` is a validator check, not a scored output.
