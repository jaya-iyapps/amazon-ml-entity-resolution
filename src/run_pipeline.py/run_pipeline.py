"""Full-scale end-to-end pipeline execution for ML Challenge 2026.

Executes the complete entity resolution pipeline:
1. Preprocessing (Person 1)
2. Blocking & Candidate Generation (Person 3)
3. ML Matching & Threshold Classification (Person 2)
4. Official TSV export: matching_results_test.tsv & candidate_pairs_test.tsv

Optimized for multi-core streaming execution with country partitioning,
in-memory shared index workers, intermediate checkpointing, and zero data leakage.
"""

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rapidfuzz.distance import Levenshtein as _rf_lev
from rapidfuzz.fuzz import ratio as _rf_ratio

import src.blocking.candidate_generator as cg
from src.blocking.address_blocking import (
    AddressInvertedIndex,
    extract_distinctive_address_tokens,
    extract_normalized_numbers,
)
from src.blocking.candidate_generator import CandidateGenerator, is_country_compatible
from src.blocking.name_blocking import (
    COMMON_STOPWORDS,
    NameInvertedIndex,
    strip_domains_and_noise,
)
from src.preprocessing.normalize import _SUFFIX_VARIANTS_SORTED
from src.matching.features import (
    clean_address,
    clean_business_name,
    extract_numeric_tokens,
    extract_tokens,
    lcs_similarity,
    normalize_country,
    strip_all_suffixes,
)
from src.matching.train import MatchingClassifier

# Precompiled fast legal suffix regex (60x faster than iterative search)
_FAST_SUFFIX_RE = re.compile(
    r"(?:[\s,]+)(?:" + "|".join(re.escape(v) for v in _SUFFIX_VARIANTS_SORTED) + r")\s*$"
)


def fast_strip_legal_suffixes(text: Optional[str]) -> str:
    """Fast regex-based legal suffix stripping matching normalize.py behavior."""
    s = (text or "").strip()
    if not s:
        return ""
    m = _FAST_SUFFIX_RE.search(s)
    if m:
        s = s[: m.start()].strip(".,;: ")
    return s


def fast_extract_core_name_tokens(raw_name: Optional[str]) -> List[str]:
    """Fast token extraction stripping domains, legal suffixes and stopwords."""
    if not raw_name:
        return []
    clean_url = strip_domains_and_noise(raw_name)
    core = fast_strip_legal_suffixes(clean_url)
    tokens = [t for t in extract_tokens(core) if len(t) >= 2]
    return [t for t in tokens if t not in COMMON_STOPWORDS]


def _prep_single_addr(rec: Dict[str, str]) -> Tuple[List[str], List[str], Optional[str]]:
    """Extract numeric tokens and distinctive tokens for address indexing."""
    addr = rec.get("business_address")
    name = rec.get("business_name")
    nums = extract_normalized_numbers(addr)
    addr_tokens = extract_distinctive_address_tokens(addr)
    first_3 = None
    if nums and name:
        name_clean = clean_address(name)
        name_tokens = [tok for tok in extract_tokens(name_clean) if len(tok) >= 3]
        if name_tokens:
            first_3 = name_tokens[0][:3]
    return (nums, addr_tokens, first_3)


def build_fast_name_index(
    records: List[Dict[str, str]], max_token_freq: int = 150
) -> NameInvertedIndex:
    """Build NameInvertedIndex with streaming tokenization and frequency safeguard."""
    idx = NameInvertedIndex(max_token_freq=max_token_freq)

    # Pass 1: Streaming token frequency counting without large array retention
    token_freqs = idx.token_frequencies
    for r in records:
        toks = fast_extract_core_name_tokens(r.get("business_name"))
        for t in set(toks):
            token_freqs[t] += 1
    idx.indexed_count = len(records)

    # Pass 2: Populate inverted indexes with high-frequency token protection
    exact_idx = idx.exact_name_idx
    concat_idx = idx.concat_name_idx
    sorted_idx = idx.sorted_tokens_idx
    pair_idx = idx.token_pair_idx
    rare_idx = idx.rare_token_idx
    max_b = idx.max_bucket_size
    max_tf = idx.max_token_freq

    for r in records:
        tokens = fast_extract_core_name_tokens(r.get("business_name"))
        if not tokens:
            continue
        eid = r["entity_id"]

        exact_key = " ".join(tokens)
        if len(exact_idx[exact_key]) < max_b:
            exact_idx[exact_key].append(eid)

        concat_key = "".join(tokens[:4])
        if len(concat_key) >= 5 and len(concat_idx[concat_key]) < max_b:
            concat_idx[concat_key].append(eid)

        if len(tokens) > 1:
            sorted_key = " ".join(sorted(tokens))
            if len(sorted_idx[sorted_key]) < max_b:
                sorted_idx[sorted_key].append(eid)

            unique_sorted = sorted(set(tokens))
            for i in range(len(unique_sorted)):
                for j in range(i + 1, min(i + 4, len(unique_sorted))):
                    pair = (unique_sorted[i], unique_sorted[j])
                    if len(pair_idx[pair]) < max_b:
                        pair_idx[pair].append(eid)

        for t in set(tokens):
            freq = token_freqs[t]
            if (len(t) >= 3 and freq <= max_tf) or (len(t) >= 6 and freq <= max_tf * 2):
                if len(rare_idx[t]) < max_b:
                    rare_idx[t].append(eid)

    return idx


def build_fast_address_index(
    records: List[Dict[str, str]], max_bucket_size: int = 100
) -> AddressInvertedIndex:
    """Build AddressInvertedIndex in a single streaming pass with minimal memory overhead."""
    idx = AddressInvertedIndex(max_bucket_size=max_bucket_size)

    for r in records:
        eid = r["entity_id"]
        nums, addr_tokens, first_3 = _prep_single_addr(r)

        if nums and addr_tokens:
            for n in nums[:2]:
                for a in addr_tokens:
                    if len(a) >= 4:
                        pair = (n, a)
                        if len(idx.num_word_idx[pair]) < idx.max_bucket_size:
                            idx.num_word_idx[pair].append(eid)

        if nums and first_3:
            prefix_pair = (nums[0], first_3)
            if len(idx.num_name_prefix_idx[prefix_pair]) < idx.max_bucket_size:
                idx.num_name_prefix_idx[prefix_pair].append(eid)

    return idx


def precompute_record(rec: Dict[str, str]) -> Tuple:
    """Precompute normalized strings, token sets, ngrams, and numerics for fast pairwise scoring."""
    raw_name = rec.get("business_name") or ""
    raw_addr = rec.get("business_address") or ""
    raw_country = rec.get("country") or ""

    name = clean_business_name(raw_name)
    core = strip_all_suffixes(name)
    addr = clean_address(raw_addr)
    c = normalize_country(raw_country)

    tokens_name = extract_tokens(name)
    tokens_addr = extract_tokens(addr)

    num = set(extract_numeric_tokens(addr) + extract_numeric_tokens(name))
    first_tok = tokens_name[0] if tokens_name else ""

    ngrams_name = (
        {name[i : i + 3] for i in range(len(name) - 2)}
        if len(name) >= 3
        else ({name} if name else set())
    )
    ngrams_addr = (
        {addr[i : i + 3] for i in range(len(addr) - 2)}
        if len(addr) >= 3
        else ({addr} if addr else set())
    )

    return (
        name,
        core,
        addr,
        c,
        tokens_name,
        set(tokens_name),
        tokens_addr,
        set(tokens_addr),
        num,
        first_tok,
        len(name),
        1.0 if not name else 0.0,
        1.0 if not addr else 0.0,
        ngrams_name,
        ngrams_addr,
    )


# Globals for worker processes
_G_GEN: Optional[CandidateGenerator] = None
_G_TARGET_PREP: Optional[Dict[str, Tuple]] = None
_G_WEIGHTS: Optional[List[float]] = None
_G_BIAS: float = 0.0
_G_MEANS: Optional[List[float]] = None
_G_STDS: Optional[List[float]] = None
_G_THRESHOLD: float = 0.98


def _compute_pair_score(p1: Tuple, p2: Tuple) -> float:
    """Compute calibrated match probability using fast precomputed feature representation."""
    global _G_WEIGHTS, _G_BIAS, _G_MEANS, _G_STDS
    (
        n1,
        c1,
        a1,
        cntry1,
        tok_n1,
        s_n1,
        tok_a1,
        s_a1,
        num1,
        ft1,
        len1,
        n_m1,
        a_m1,
        ng_n1,
        ng_a1,
    ) = p1
    (
        n2,
        c2,
        a2,
        cntry2,
        tok_n2,
        s_n2,
        tok_a2,
        s_a2,
        num2,
        ft2,
        len2,
        n_m2,
        a_m2,
        ng_n2,
        ng_a2,
    ) = p2

    n_inter = len(s_n1 & s_n2)
    n_union = len(s_n1 | s_n2)
    name_jaccard = (n_inter / n_union) if n_union else 0.0
    name_dice = (
        (2.0 * n_inter / (len(s_n1) + len(s_n2))) if (len(s_n1) + len(s_n2)) else 0.0
    )

    a_inter = len(s_a1 & s_a2)
    a_union = len(s_a1 | s_a2)
    addr_jaccard = (a_inter / a_union) if a_union else 0.0
    addr_dice = (
        (2.0 * a_inter / (len(s_a1) + len(s_a2))) if (len(s_a1) + len(s_a2)) else 0.0
    )

    ng_n_inter = len(ng_n1 & ng_n2)
    ng_n_union = len(ng_n1 | ng_n2)
    name_ngram = (ng_n_inter / ng_n_union) if ng_n_union else 0.0

    ng_a_inter = len(ng_a1 & ng_a2)
    ng_a_union = len(ng_a1 | ng_a2)
    addr_ngram = (ng_a_inter / ng_a_union) if ng_a_union else 0.0

    name_lev = (
        1.0
        if n1 == n2
        else (0.0 if not n1 or not n2 else float(_rf_lev.normalized_similarity(n1, n2)))
    )
    addr_lev = (
        1.0
        if a1 == a2
        else (0.0 if not a1 or not a2 else float(_rf_lev.normalized_similarity(a1, a2)))
    )

    name_lcs = lcs_similarity(n1, n2)

    num_common = len(num1 & num2)
    tot_num = len(num1 | num2)
    num_jacc = (num_common / tot_num) if tot_num else 0.0
    num_dis = 1.0 if (num1 and num2 and not num_common) else 0.0

    sn1 = " ".join(sorted(tok_n1))
    sn2 = " ".join(sorted(tok_n2))
    tok_sort = 1.0 if sn1 == sn2 else float(_rf_ratio(sn1, sn2)) / 100.0

    max_l = max(len1, len2)
    len_ratio = abs(len1 - len2) / max_l if max_l else 0.0

    if cntry1 and cntry2:
        c_exact = 1.0 if cntry1 == cntry2 else 0.0
        c_mismatch = 1.0 if cntry1 != cntry2 else 0.0
    else:
        c_exact = 0.0
        c_mismatch = 0.0
    c_missing = 1.0 if (not cntry1 or not cntry2) else 0.0

    raw_features = [
        1.0 if n1 and n1 == n2 else 0.0,
        1.0 if c1 and c1 == c2 else 0.0,
        name_jaccard,
        name_dice,
        tok_sort,
        name_lev,
        name_lcs,
        name_ngram,
        len_ratio,
        1.0 if ft1 and ft1 == ft2 else 0.0,
        1.0 if a1 and a1 == a2 else 0.0,
        addr_jaccard,
        addr_dice,
        addr_lev,
        addr_ngram,
        float(num_common),
        num_jacc,
        num_dis,
        c_exact,
        c_mismatch,
        c_missing,
        n_m1,
        n_m2,
        a_m1,
        a_m2,
    ]

    p = len(_G_MEANS)
    z = _G_BIAS
    for j in range(p):
        std_j = _G_STDS[j] if _G_STDS[j] > 1e-8 else 1.0
        scaled_j = (raw_features[j] - _G_MEANS[j]) / std_j
        z += _G_WEIGHTS[j] * scaled_j

    z = max(-35.0, min(35.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _process_s1_batch(
    s1_records: List[Dict[str, str]],
) -> List[Tuple[str, List[str], List[str]]]:
    """Process a chunk of S1 records: query candidates, score each, return results."""
    global _G_GEN, _G_TARGET_PREP, _G_THRESHOLD
    results = []

    for s1_rec in s1_records:
        s1_id = s1_rec.get("entity_id", "")
        cands = _G_GEN.generate_candidates_for_record(s1_rec)
        if not cands:
            results.append((s1_id, [], []))
            continue

        p1 = precompute_record(s1_rec)
        matched = []
        for cid in cands:
            p2 = _G_TARGET_PREP.get(cid)
            if p2 is None:
                trec = _G_GEN.target_records_cache.get(cid)
                if trec is None:
                    continue
                p2 = precompute_record(trec)
                _G_TARGET_PREP[cid] = p2

            prob = _compute_pair_score(p1, p2)
            if prob >= _G_THRESHOLD:
                matched.append((prob, cid))

        matched.sort(key=lambda x: x[0], reverse=True)
        matched_ids = []
        seen = set()
        for _, cid in matched:
            if cid not in seen:
                seen.add(cid)
                matched_ids.append(cid)

        results.append((s1_id, cands, matched_ids))

    return results


def run_pipeline(
    test_dir: str = "student_resource/dataset/test",
    out_matching: str = "output/matching_results_test.tsv",
    out_candidate: str = "output/candidate_pairs_test.tsv",
    checkpoint_dir: str = "output/checkpoints_test",
    model_path: str = "matching_model_improved.json",
    threshold: float = 0.98,
    max_candidates: int = 80,
    num_workers: int = 10,
    limit: Optional[int] = None,
):
    """Execute full entity resolution pipeline on test dataset with intermediate checkpoints."""
    t_start = time.time()
    print("=" * 75)
    print("AMAZON ML CHALLENGE 2026: OFFICIAL FULL PIPELINE RUNNER")
    print(f"Test Directory       : {test_dir}")
    print(f"Output Matching TSV  : {out_matching}")
    print(f"Output Candidate TSV : {out_candidate}")
    print(f"Checkpoint Directory : {checkpoint_dir}")
    print(f"Model Artifact       : {model_path}")
    print(f"Decision Threshold   : {threshold:.2f}")
    print(f"Max Candidates       : {max_candidates}")
    print(f"Worker Processes     : {num_workers}")
    if limit:
        print(f"Entity Limit (Debug) : {limit:,}")
    print("=" * 75)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for path, name in [
        (s1_path, "Source 1"),
        (s2_path, "Source 2"),
        (s3_path, "Source 3"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing official test file: {path} ({name})")

    # Load Model Weights
    print("\n[Step 1/5] Loading Matching Classifier...")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Train model first."
        )
    with open(model_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)

    weights = m_data["weights"]
    bias = m_data["bias"]
    means = m_data["means"]
    stds = m_data["stds"]
    print(f"  Loaded model: bias={bias:.4f}, {len(weights)} feature weights")

    # Scan test_source1 to preserve exact ordering and group by country
    print("\n[Step 2/5] Indexing Reference S1 Entities by Country...")
    t0 = time.time()
    s1_by_country: Dict[str, List[Dict[str, str]]] = {
        "France": [],
        "US": [],
        "India": [],
    }
    all_s1_ids: List[str] = []

    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        id_idx = header.index("entity_id")
        name_idx = header.index("business_name")
        addr_idx = header.index("business_address")
        cntry_idx = header.index("country")

        count = 0
        for row in reader:
            if not row or len(row) <= id_idx:
                continue
            eid = row[id_idx].strip()
            all_s1_ids.append(eid)
            rec = {
                "entity_id": eid,
                "business_name": row[name_idx] if len(row) > name_idx else "",
                "business_address": row[addr_idx] if len(row) > addr_idx else "",
                "country": row[cntry_idx] if len(row) > cntry_idx else "",
            }
            country = rec["country"].strip()
            if country in s1_by_country:
                s1_by_country[country].append(rec)
            else:
                s1_by_country.setdefault(country, []).append(rec)
            count += 1
            if limit and count >= limit:
                break

    print(f"  Indexed {len(all_s1_ids):,} S1 entities in {time.time()-t0:.2f}s:")
    for ctry, lst in s1_by_country.items():
        print(f"    • {ctry:8s}: {len(lst):,} entities")

    os.makedirs(checkpoint_dir, exist_ok=True)
    all_results: Dict[str, Tuple[str, str]] = {}

    # Process Partition by Partition: France first (smallest), then US, then India
    countries_to_process = [
        c for c in ["France", "US", "India"] if s1_by_country.get(c)
    ]
    for c_idx, country in enumerate(countries_to_process, 1):
        print("\n" + "-" * 75)
        print(f"[Step 3/5.{c_idx}] Processing Partition: {country.upper()}")
        print("-" * 75)

        s1_partition = s1_by_country[country]
        ckpt_file = os.path.join(checkpoint_dir, f"{country}_results.tsv")

        # Check if already processed in checkpoint
        if os.path.isfile(ckpt_file):
            print(f"  Loading existing checkpoint: {ckpt_file}...")
            loaded_count = 0
            with open(ckpt_file, "r", encoding="utf-8") as f_ckpt:
                rdr = csv.reader(f_ckpt, delimiter="\t")
                for r in rdr:
                    if r and len(r) >= 3:
                        all_results[r[0]] = (r[1], r[2])
                        loaded_count += 1
            if loaded_count >= len(s1_partition):
                print(f"  Checkpoint complete: {loaded_count:,} entities loaded. Skipping computation.")
                continue
            else:
                remaining_s1 = [
                    rec for rec in s1_partition if rec["entity_id"] not in all_results
                ]
                print(
                    f"  Partial checkpoint loaded: {loaded_count:,}/{len(s1_partition):,} entities. "
                    f"Resuming computation for {len(remaining_s1):,} remaining entities."
                )
                s1_partition = remaining_s1

        print(f"  Loading Source 2 & 3 target records for {country}...")
        t_load = time.time()
        target_records: List[Dict[str, str]] = []

        for src_path in [s2_path, s3_path]:
            with open(src_path, "r", encoding="utf-8") as f:
                rdr = csv.reader(f, delimiter="\t")
                hdr = next(rdr)
                t_id_idx = hdr.index("entity_id")
                t_name_idx = hdr.index("business_name")
                t_addr_idx = hdr.index("business_address")
                t_cntry_idx = hdr.index("country")

                for row in rdr:
                    if not row or len(row) <= t_id_idx:
                        continue
                    r_cntry = (
                        row[t_cntry_idx].strip() if len(row) > t_cntry_idx else ""
                    )
                    if r_cntry == country:
                        target_records.append(
                            {
                                "entity_id": row[t_id_idx].strip(),
                                "business_name": (
                                    row[t_name_idx] if len(row) > t_name_idx else ""
                                ),
                                "business_address": (
                                    row[t_addr_idx] if len(row) > t_addr_idx else ""
                                ),
                                "country": r_cntry,
                            }
                        )

        print(
            f"  Loaded {len(target_records):,} target records in {time.time()-t_load:.2f}s."
        )

        print(f"  Fitting Inverted Indexes for {country}...")
        t_fit = time.time()
        name_idx = build_fast_name_index(target_records)
        addr_idx = build_fast_address_index(target_records)

        gen = CandidateGenerator(max_candidates_per_entity=max_candidates)
        gen.name_index = name_idx
        gen.address_index = addr_idx
        gen.target_records_cache = {rec["entity_id"]: rec for rec in target_records}

        # Fast candidate ranking decoration with pass-priority scoring
        def make_fast(g: CandidateGenerator):
            n_idx = g.name_index
            a_idx = g.address_index
            max_k = g.max_candidates_per_entity or 80

            def fast_gen(s1_rec: Dict[str, Any]) -> List[str]:
                name = s1_rec.get("business_name")
                addr = s1_rec.get("business_address")
                scores = defaultdict(int)

                tokens = fast_extract_core_name_tokens(name)
                if tokens:
                    exact_key = " ".join(tokens)
                    for cid in n_idx.exact_name_idx.get(exact_key, ()):
                        scores[cid] += 100
                    concat_key = "".join(tokens[:4])
                    if len(concat_key) >= 5:
                        for cid in n_idx.concat_name_idx.get(concat_key, ()):
                            scores[cid] += 50
                    if len(tokens) > 1:
                        sorted_key = " ".join(sorted(tokens))
                        for cid in n_idx.sorted_tokens_idx.get(sorted_key, ()):
                            scores[cid] += 40
                        unique_sorted = sorted(set(tokens))
                        for i in range(len(unique_sorted)):
                            for j in range(i + 1, min(i + 4, len(unique_sorted))):
                                pair = (unique_sorted[i], unique_sorted[j])
                                bucket = n_idx.token_pair_idx.get(pair)
                                if bucket and len(bucket) <= n_idx.max_bucket_size:
                                    for cid in bucket:
                                        scores[cid] += 10
                    for t in set(tokens):
                        bucket = n_idx.rare_token_idx.get(t)
                        if bucket and len(bucket) <= n_idx.max_bucket_size:
                            for cid in bucket:
                                scores[cid] += 5

                nums = extract_normalized_numbers(addr)
                addr_tokens = extract_distinctive_address_tokens(addr)
                if nums and addr_tokens:
                    for n in nums[:2]:
                        for a in addr_tokens:
                            if len(a) >= 4:
                                bucket = a_idx.num_word_idx.get((n, a))
                                if bucket and len(bucket) <= a_idx.max_bucket_size:
                                    for cid in bucket:
                                        scores[cid] += 30
                if nums and name:
                    primary_num = nums[0]
                    name_clean = clean_address(name)
                    name_tokens = [
                        tok
                        for tok in extract_tokens(name_clean)
                        if len(tok) >= 3
                    ]
                    if name_tokens:
                        pair = (primary_num, name_tokens[0][:3])
                        bucket = a_idx.num_name_prefix_idx.get(pair)
                        if bucket and len(bucket) <= a_idx.max_bucket_size:
                            for cid in bucket:
                                scores[cid] += 20

                if not scores:
                    return []
                if len(scores) <= max_k:
                    return sorted(scores.keys())
                top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
                    :max_k
                ]
                return sorted(cid for cid, _ in top)

            g.generate_candidates_for_record = fast_gen
            return g

        make_fast(gen)
        print(f"  Parallel indexing complete in {time.time()-t_fit:.2f}s.")

        # Batch S1 records for workers
        batch_size = 2000
        batches = [
            s1_partition[i : i + batch_size]
            for i in range(0, len(s1_partition), batch_size)
        ]
        print(
            f"  Evaluating {len(s1_partition):,} S1 entities across {len(batches)} batches using {num_workers} workers..."
        )
        global _G_GEN, _G_TARGET_PREP, _G_WEIGHTS, _G_BIAS, _G_MEANS, _G_STDS, _G_THRESHOLD
        _G_GEN = gen
        _G_TARGET_PREP = {}
        _G_WEIGHTS = weights
        _G_BIAS = bias
        _G_MEANS = means
        _G_STDS = stds
        _G_THRESHOLD = threshold

        t_eval = time.time()

        # Stream results directly into partition checkpoint in append mode
        with open(ckpt_file, "a", encoding="utf-8", newline="") as f_ckpt:
            writer_ckpt = csv.writer(f_ckpt, delimiter="\t", lineterminator="\n")
            
            pool = None
            try:
                if num_workers > 1:
                    ctx = mp.get_context("fork")
                    pool = ctx.Pool(processes=num_workers)
                    batch_iter = pool.imap_unordered(_process_s1_batch, batches, chunksize=1)
                else:
                    batch_iter = map(_process_s1_batch, batches)
            except Exception as e:
                print(f"  Multiprocessing pool unavailable ({e}). Using ThreadPoolExecutor with {num_workers} workers.")
                from concurrent.futures import ThreadPoolExecutor
                pool = ThreadPoolExecutor(max_workers=max(1, num_workers))
                batch_iter = pool.map(_process_s1_batch, batches)

            try:
                for b_idx, batch_res in enumerate(batch_iter, 1):
                    batch_rows = []
                    for s1_id, cands, matched in batch_res:
                        c_str = ",".join(cands)
                        m_str = ",".join(matched)
                        all_results[s1_id] = (c_str, m_str)
                        batch_rows.append((s1_id, c_str, m_str))

                    writer_ckpt.writerows(batch_rows)
                    f_ckpt.flush()

                    if b_idx % 25 == 0 or b_idx == len(batches):
                        done = min(b_idx * batch_size, len(s1_partition))
                        pct = (done / len(s1_partition)) * 100
                        elapsed = time.time() - t_eval
                        rate = done / elapsed if elapsed > 0 else 0
                        print(
                            f"    Processed {done:,}/{len(s1_partition):,} ({pct:.1f}%) in {elapsed:.1f}s ({rate:.0f} entities/s)"
                        )
            finally:
                if pool is not None:
                    if hasattr(pool, "close"):
                        pool.close()
                        pool.join()
                    elif hasattr(pool, "shutdown"):
                        pool.shutdown(wait=True)

        print(f"  Completed partition {country} in {time.time()-t_eval:.2f}s.")

        del target_records
        del gen
        del name_idx
        del addr_idx
        gc.collect()

    # Step 4: Write Final Output Files in exact Source 1 order
    print("\n" + "=" * 75)
    print("[Step 4/5] Writing Official Submission Files...")
    print("=" * 75)
    os.makedirs(os.path.dirname(os.path.abspath(out_candidate)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_matching)), exist_ok=True)

    print(f"  Writing Candidate TSV: {out_candidate}...")
    total_cands = 0
    with open(out_candidate, "w", encoding="utf-8", newline="") as f_cand:
        writer_cand = csv.writer(
            f_cand, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )
        writer_cand.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in all_s1_ids:
            cands_str = all_results.get(s1_id, ("", ""))[0]
            writer_cand.writerow([s1_id, cands_str])
            if cands_str:
                total_cands += len(cands_str.split(","))

    print(f"  Writing Matching TSV: {out_matching}...")
    total_matches = 0
    multi_matches = 0
    singletons = 0
    with open(out_matching, "w", encoding="utf-8", newline="") as f_match:
        writer_match = csv.writer(
            f_match, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )
        writer_match.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in all_s1_ids:
            match_str = all_results.get(s1_id, ("", ""))[1]
            writer_match.writerow([s1_id, match_str])
            if match_str:
                n_m = len(match_str.split(","))
                total_matches += n_m
                if n_m > 1:
                    multi_matches += 1
            else:
                singletons += 1

    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print("[Step 5/5] Pipeline Run Complete!")
    print("=" * 75)
    print(f"  Total S1 Entities Processed: {len(all_s1_ids):,}")
    print(f"  Total Candidate Pairs       : {total_cands:,}")
    print(f"  Total Matches Found         : {total_matches:,}")
    print(f"  Multi-match Entities        : {multi_matches:,}")
    print(
        f"  Singletons Preserved        : {singletons:,} ({(singletons/len(all_s1_ids))*100:.2f}%)"
    )
    print(f"  Total Wall-Clock Time       : {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Candidate TSV : {out_candidate}")
    print(f"  Matching TSV  : {out_matching}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full entity resolution pipeline on test set."
    )
    parser.add_argument(
        "--test-dir",
        default="student_resource/dataset/test",
        help="Folder containing test_source1/2/3.tsv",
    )
    parser.add_argument(
        "--out-matching",
        default="output/matching_results_test.tsv",
        help="Path to save output matching_results_test.tsv",
    )
    parser.add_argument(
        "--out-candidate",
        default="output/candidate_pairs_test.tsv",
        help="Path to save output candidate_pairs_test.tsv",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="output/checkpoints_test",
        help="Directory to save intermediate partition checkpoints",
    )
    parser.add_argument(
        "--model",
        default="matching_model_improved.json",
        help="Path to trained matching model JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Decision threshold (default: 0.98)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=80,
        help="Max candidates per S1 entity (default: 80)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker processes (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of S1 entities for dry-run testing",
    )
    args = parser.parse_args()

    run_pipeline(
        test_dir=args.test_dir,
        out_matching=args.out_matching,
        out_candidate=args.out_candidate,
        checkpoint_dir=args.checkpoint_dir,
        model_path=args.model,
        threshold=args.threshold,
        max_candidates=args.max_candidates,
        num_workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
