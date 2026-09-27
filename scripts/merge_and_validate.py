"""Merge all checkpoints into official submission TSVs and execute full validation."""

import csv
import glob
import os
import subprocess
import sys
import time

def run_merge():
    t_start = time.time()
    print("=" * 70)
    print("STEP 3: MERGING CHECKPOINTS INTO FINAL OFFICIAL SUBMISSION TSVS")
    print("=" * 70)

    checkpoint_dir = "output/checkpoints_test"
    s1_path = "dataset/test/test_source1.tsv"
    out_matching = "output/matching_results_test.tsv"
    out_candidate = "output/candidate_pairs_test.tsv"

    # Step 1: Load all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*_results.tsv"))
    print(f"Found {len(checkpoint_files)} checkpoint file(s):")
    for f in checkpoint_files:
        print(f"  • {f} ({os.path.getsize(f)/1e6:.1f} MB)")

    checkpoint_data = {}
    total_loaded = 0
    for ckpt_file in checkpoint_files:
        print(f"Loading {ckpt_file}...")
        with open(ckpt_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row or len(row) < 1:
                    continue
                s1_id = row[0].strip()
                if s1_id == "source1_entity_id":
                    continue
                cands_str = row[1].strip() if len(row) > 1 else ""
                match_str = row[2].strip() if len(row) > 2 else ""
                checkpoint_data[s1_id] = (cands_str, match_str)
                total_loaded += 1

    print(f"Total unique checkpointed S1 entities loaded: {len(checkpoint_data):,}")

    # Step 2: Stream test_source1 and generate matching & candidate TSVs
    print(f"\nStreaming reference entities from {s1_path}...")
    os.makedirs("output", exist_ok=True)

    total_s1 = 0
    total_cands = 0
    total_matches = 0
    empty_matches = 0

    with open(s1_path, "r", encoding="utf-8") as f_in, \
         open(out_matching, "w", encoding="utf-8", newline="") as f_match, \
         open(out_candidate, "w", encoding="utf-8", newline="") as f_cand:

        reader = csv.reader(f_in, delimiter="\t")
        header = next(reader)
        id_idx = header.index("entity_id")

        # Write required competition headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for row in reader:
            if not row or len(row) <= id_idx:
                continue
            s1_id = row[id_idx].strip()
            total_s1 += 1

            if s1_id in checkpoint_data:
                cands_str, match_str = checkpoint_data[s1_id]
            else:
                cands_str = ""
                match_str = ""

            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cands_str}\n")

            if match_str:
                total_matches += len(match_str.split(","))
            else:
                empty_matches += 1

            if cands_str:
                total_cands += len(cands_str.split(","))

    print(f"\nFinal submission generation completed in {time.time()-t_start:.2f}s:")
    print(f"  • Total Source 1 Rows Written : {total_s1:,}")
    print(f"  • Total Candidates Pairs      : {total_cands:,}")
    print(f"  • Total Matches Predicted     : {total_matches:,}")
    print(f"  • Empty / Singleton Count     : {empty_matches:,} ({(empty_matches/total_s1)*100:.2f}%)")
    print(f"  • Matching TSV: {out_matching} ({os.path.getsize(out_matching)/1e6:.2f} MB)")
    print(f"  • Candidate TSV: {out_candidate} ({os.path.getsize(out_candidate)/1e6:.2f} MB)")

    # Step 3: Run comprehensive validation
    print("\n" + "=" * 70)
    print("STEP 4: OFFICIAL VALIDATION AND VERIFICATION")
    print("=" * 70)

    val_res = subprocess.run([sys.executable, "scripts/validate_final_outputs.py"], capture_output=True, text=True)
    print(val_res.stdout)
    if val_res.stderr:
        print("Validation Stderr:\n", val_res.stderr)

    return val_res.returncode == 0

if __name__ == "__main__":
    success = run_merge()
    sys.exit(0 if success else 1)
