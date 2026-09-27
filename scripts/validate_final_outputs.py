"""Official submission validation and integrity verification script."""

import csv
import os
import subprocess
import sys
import time

def validate():
    print("=" * 70)
    print("SUBMISSION VERIFICATION & INTEGRITY CHECK")
    print("=" * 70)

    matching_path = "output/matching_results_test.tsv"
    candidate_path = "output/candidate_pairs_test.tsv"
    test_dir = "dataset/test"

    for p in [matching_path, candidate_path]:
        if not os.path.isfile(p):
            print(f"[FAIL] Missing output file: {p}")
            return False

    # Check 1: Reference S1 entities
    s1_ids = []
    with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if row:
                s1_ids.append(row[0].strip())
    expected_s1_count = len(s1_ids)
    expected_s1_set = set(s1_ids)
    print(f"• Expected Reference S1 Entities: {expected_s1_count:,}")

    # Check 2: Target IDs
    target_ids = set()
    for s_file in ["test_source2.tsv", "test_source3.tsv"]:
        with open(os.path.join(test_dir, s_file), "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            for row in reader:
                if row:
                    target_ids.add(row[0].strip())
    print(f"• Known Valid Target Entities: {len(target_ids):,}")

    # Check Candidate TSV
    print(f"\nChecking candidate pairs TSV: {candidate_path}...")
    cand_s1_ids = []
    cand_by_s1 = {}
    total_cands = 0
    with open(candidate_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            print(f"[FAIL] Invalid candidate header: {header}")
            return False
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            eid = row[0].strip()
            cand_s1_ids.append(eid)
            cands_str = row[1].strip() if len(row) > 1 else ""
            cands = set(c.strip() for c in cands_str.split(",") if c.strip()) if cands_str else set()
            cand_by_s1[eid] = cands
            total_cands += len(cands)

    cand_unique = set(cand_s1_ids)
    if len(cand_s1_ids) != expected_s1_count:
        print(f"[FAIL] Candidate row count mismatch: found {len(cand_s1_ids):,}, expected {expected_s1_count:,}")
        return False
    if len(cand_unique) != expected_s1_count:
        print(f"[FAIL] Duplicate S1 IDs found in candidate TSV! ({len(cand_s1_ids) - len(cand_unique)} duplicates)")
        return False
    if cand_unique != expected_s1_set:
        print("[FAIL] Candidate S1 ID set does not match test_source1!")
        return False
    print(f"  [PASS] Candidate row count: {len(cand_s1_ids):,} (100% unique, exact test match)")
    print(f"  [PASS] Total candidate pairs: {total_cands:,}")

    # Check Matching TSV
    print(f"\nChecking matching results TSV: {matching_path}...")
    match_s1_ids = []
    total_matches = 0
    empty_matches = 0
    match_not_in_cand = 0
    invalid_targets = 0

    with open(matching_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header != ["source1_entity_id", "matched_entity_ids"]:
            print(f"[FAIL] Invalid matching header: {header}")
            return False
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            eid = row[0].strip()
            match_s1_ids.append(eid)
            m_str = row[1].strip() if len(row) > 1 else ""
            if not m_str:
                empty_matches += 1
            else:
                mids = [m.strip() for m in m_str.split(",") if m.strip()]
                total_matches += len(mids)
                s1_cands = cand_by_s1.get(eid, set())
                for mid in mids:
                    if mid not in s1_cands:
                        match_not_in_cand += 1
                    if mid not in target_ids:
                        invalid_targets += 1

    match_unique = set(match_s1_ids)
    if len(match_s1_ids) != expected_s1_count:
        print(f"[FAIL] Matching row count mismatch: found {len(match_s1_ids):,}, expected {expected_s1_count:,}")
        return False
    if len(match_unique) != expected_s1_count:
        print(f"[FAIL] Duplicate S1 IDs in matching TSV! ({len(match_s1_ids) - len(match_unique)} duplicates)")
        return False
    if match_unique != expected_s1_set:
        print("[FAIL] Matching S1 ID set does not match test_source1!")
        return False
    if match_not_in_cand > 0:
        print(f"[FAIL] {match_not_in_cand} matched IDs not in entity's candidate list!")
        return False
    if invalid_targets > 0:
        print(f"[FAIL] {invalid_targets} matched IDs not found in target sources!")
        return False

    print(f"  [PASS] Matching row count: {len(match_s1_ids):,} (100% unique, exact test match)")
    print(f"  [PASS] Total predicted matches: {total_matches:,}")
    print(f"  [PASS] Empty predictions (singletons): {empty_matches:,} ({(empty_matches/expected_s1_count)*100:.2f}%)")
    print(f"  [PASS] All matched IDs are in candidate list (0 violations)")
    print(f"  [PASS] All matched IDs exist in test target datasets")

    # Run official validator
    print("\nRunning official submission validator...")
    cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode == 0:
        print("=" * 70)
        print("ALL SUBMISSION VALIDATION CHECKS PASSED (100% COMPLIANT)!")
        print("=" * 70)
        return True
    else:
        print("Official validator stderr:")
        print(res.stderr)
        return False

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
