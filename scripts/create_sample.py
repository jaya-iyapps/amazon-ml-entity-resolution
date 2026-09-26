#!/usr/bin/env python3
"""
Amazon ML Entity Resolution Challenge 2026
Relationship-Preserving Development Sample Generator

This script samples a subset of Source 1 entities while preserving all
ground-truth relationships with Source 2 and Source 3 records. It also
includes a controlled number of negative (non-matching) records from Source 2
and Source 3 for candidate generation and matching model evaluation.

Usage:
    python scripts/create_sample.py --n-entities 10000
    python scripts/create_sample.py --n-entities 1000 --seed 42
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a relationship-preserving sample dataset for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/full",
        help="Path to directory containing original training TSV files (default: dataset/full).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/sample",
        help="Path to directory where sampled TSV files will be saved (default: dataset/sample).",
    )
    parser.add_argument(
        "--n-entities",
        type=int,
        default=10000,
        help="Number of Source 1 entities to sample (default: 10000).",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.20,
        help="Ratio of additional non-matching S2/S3 records to include relative to positive matches (default: 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


def resolve_input_file(input_dir: str, filename: str) -> str:
    path = os.path.join(input_dir, filename)
    if os.path.isfile(path):
        return path
    # Fallback to student_resource/dataset/train
    fallback = os.path.join("student_resource", "dataset", "train", filename)
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(f"Could not find required file {filename} in {input_dir} or {fallback}")


def sample_source1(
    s1_path: str, n_entities: int, seed: int
) -> Tuple[Set[str], List[List[str]], List[str], Counter]:
    """Sample n_entities from Source 1 reproducibly."""
    print(f"[1/5] Scanning Source 1 entity IDs from {s1_path}...")
    all_s1_ids = []
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if row:
                all_s1_ids.append(row[0])

    total_s1 = len(all_s1_ids)
    print(f"      Total Source 1 entities found: {total_s1:,}")
    if n_entities > total_s1:
        n_entities = total_s1

    rng = random.Random(seed)
    selected_s1_set = set(rng.sample(all_s1_ids, n_entities))

    # Second pass: stream and collect records for selected entities
    selected_records = []
    country_counts = Counter()
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader)
        for row in reader:
            if row and row[0] in selected_s1_set:
                selected_records.append(row)
                if len(row) > 3:
                    country_counts[row[3]] += 1

    print(f"      Selected {len(selected_records):,} Source 1 entities.")
    return selected_s1_set, selected_records, header, country_counts


def extract_ground_truth(
    gt_path: str, selected_s1_set: Set[str]
) -> Tuple[List[List[str]], List[str], Set[str], Set[str], Counter]:
    """Extract ground truth rows for selected S1 entities and identify all positive S2 and S3 IDs."""
    print(f"[2/5] Extracting ground truth links from {gt_path}...")
    selected_gt_rows = []
    pos_s2_ids = set()
    pos_s3_ids = set()
    match_distribution = Counter()

    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            if s1_id in selected_s1_set:
                selected_gt_rows.append(row)
                matched_str = row[1].strip() if len(row) > 1 else ""
                if not matched_str:
                    match_distribution["zero matches (singletons)"] += 1
                else:
                    mids = [m.strip() for m in matched_str.split(",") if m.strip()]
                    if len(mids) == 1:
                        match_distribution["one match"] += 1
                    else:
                        match_distribution["multiple matches"] += 1

                    for mid in mids:
                        if mid.startswith("S2-"):
                            pos_s2_ids.add(mid)
                        elif mid.startswith("S3-"):
                            pos_s3_ids.add(mid)

    print(f"      Extracted {len(selected_gt_rows):,} ground truth entries.")
    print(f"      Positive matches: {len(pos_s2_ids):,} in Source 2, {len(pos_s3_ids):,} in Source 3.")
    return selected_gt_rows, header, pos_s2_ids, pos_s3_ids, match_distribution


def stream_source_with_negatives(
    source_path: str,
    pos_ids: Set[str],
    n_negatives: int,
    source_label: str,
    seed: int,
) -> Tuple[List[List[str]], List[str], int, int]:
    """Extract all positive records and select a controlled reservoir of negative records."""
    print(f"[{source_label}] Streaming records from {source_path}...")
    pos_records = []
    neg_reservoir = []
    seen_neg_count = 0
    rng = random.Random(seed)

    with open(source_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            eid = row[0]
            if eid in pos_ids:
                pos_records.append(row)
            else:
                # Reservoir sampling for negative records
                seen_neg_count += 1
                if len(neg_reservoir) < n_negatives:
                    neg_reservoir.append(row)
                else:
                    idx = rng.randint(0, seen_neg_count - 1)
                    if idx < n_negatives:
                        neg_reservoir[idx] = row

    all_records = pos_records + neg_reservoir
    print(f"      {source_label}: Extracted {len(pos_records):,} positive + {len(neg_reservoir):,} negative records (Total: {len(all_records):,}).")
    return all_records, header, len(pos_records), len(neg_reservoir)


def write_tsv(file_path: str, header: List[str], rows: List[List[str]]):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    args = parse_args()
    print("=" * 70)
    print(" Amazon ML Challenge 2026 - Sample Dataset Generator")
    print(f" Target Source 1 Entities : {args.n_entities:,}")
    print(f" Negative Record Ratio   : {args.negative_ratio:.1%}")
    print(f" Random Seed             : {args.seed}")
    print(f" Input Directory         : {args.input_dir}")
    print(f" Output Directory        : {args.output_dir}")
    print("=" * 70)

    # 1. Resolve file paths
    s1_file = resolve_input_file(args.input_dir, "train_source1.tsv")
    s2_file = resolve_input_file(args.input_dir, "train_source2.tsv")
    s3_file = resolve_input_file(args.input_dir, "train_source3.tsv")
    gt_file = resolve_input_file(args.input_dir, "train_ground_truth.tsv")

    # 2. Sample Source 1
    selected_s1_set, s1_records, s1_header, country_counts = sample_source1(
        s1_file, args.n_entities, args.seed
    )

    # 3. Extract Ground Truth
    gt_rows, gt_header, pos_s2_ids, pos_s3_ids, match_dist = extract_ground_truth(
        gt_file, selected_s1_set
    )

    # 4. Stream Source 2 with controlled negatives
    n_neg_s2 = int(len(pos_s2_ids) * args.negative_ratio)
    s2_records, s2_header, n_pos_s2, n_neg_s2_actual = stream_source_with_negatives(
        s2_file, pos_s2_ids, n_neg_s2, "3/5 Source 2", args.seed + 1
    )

    # 5. Stream Source 3 with controlled negatives
    n_neg_s3 = int(len(pos_s3_ids) * args.negative_ratio)
    s3_records, s3_header, n_pos_s3, n_neg_s3_actual = stream_source_with_negatives(
        s3_file, pos_s3_ids, n_neg_s3, "4/5 Source 3", args.seed + 2
    )

    # 6. Save sample files
    print(f"[5/5] Writing output files to {args.output_dir}...")
    out_s1 = os.path.join(args.output_dir, "train_source1.tsv")
    out_s2 = os.path.join(args.output_dir, "train_source2.tsv")
    out_s3 = os.path.join(args.output_dir, "train_source3.tsv")
    out_gt = os.path.join(args.output_dir, "train_ground_truth.tsv")

    write_tsv(out_s1, s1_header, s1_records)
    write_tsv(out_s2, s2_header, s2_records)
    write_tsv(out_s3, s3_header, s3_records)
    write_tsv(out_gt, gt_header, gt_rows)

    # 7. Verification and Integrity Check
    print("\n" + "=" * 70)
    print(" VERIFICATION & INTEGRITY REPORT")
    print("=" * 70)
    print(f"Output files generated in: {args.output_dir}/")
    print(f"  • train_source1.tsv      : {len(s1_records):,} rows")
    print(f"  • train_source2.tsv      : {len(s2_records):,} rows ({n_pos_s2:,} pos + {n_neg_s2_actual:,} neg)")
    print(f"  • train_source3.tsv      : {len(s3_records):,} rows ({n_pos_s3:,} pos + {n_neg_s3_actual:,} neg)")
    print(f"  • train_ground_truth.tsv : {len(gt_rows):,} rows")

    # Integrity assertions
    sample_s1_ids = {r[0] for r in s1_records}
    sample_s2_ids = {r[0] for r in s2_records}
    sample_s3_ids = {r[0] for r in s3_records}

    assert len(sample_s1_ids) == len(s1_records), "Duplicate IDs found in sample Source 1!"
    assert len(sample_s2_ids) == len(s2_records), "Duplicate IDs found in sample Source 2!"
    assert len(sample_s3_ids) == len(s3_records), "Duplicate IDs found in sample Source 3!"

    # Verify all ground truth references exist
    missing_s2 = pos_s2_ids - sample_s2_ids
    missing_s3 = pos_s3_ids - sample_s3_ids
    assert len(missing_s2) == 0, f"Integrity Failure: Missing S2 IDs in sample: {len(missing_s2)}"
    assert len(missing_s3) == 0, f"Integrity Failure: Missing S3 IDs in sample: {len(missing_s3)}"

    print("\n✓ Integrity Check PASSED:")
    print("  ✓ 100% of ground-truth Source 2 matches exist in sample Source 2.")
    print("  ✓ 100% of ground-truth Source 3 matches exist in sample Source 3.")
    print("  ✓ No dangling pointers or missing entities.")
    print("  ✓ No duplicate entity IDs.")
    print("  ✓ Headers and schema exactly preserved.")

    print("\nSample Distribution Details:")
    print(f"  Country representation in Source 1: {dict(country_counts)}")
    print(f"  Match distribution in Source 1:")
    for cat, count in match_dist.items():
        print(f"    - {cat}: {count:,} ({count / len(selected_s1_set):.1%})")
    print("=" * 70)


if __name__ == "__main__":
    main()
