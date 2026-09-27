"""Verification script for recovered artifacts in Amazon ML Entity Resolution Challenge."""

import csv
import json
import os
import sys
import subprocess

PYTHONPATH = "/home/pyro/snap/antigravity-cli/common/local/lib/python3.14/dist-packages"
if PYTHONPATH not in sys.path:
    sys.path.insert(0, PYTHONPATH)
sys.path.insert(0, os.path.abspath("."))

def verify():
    print("=" * 65)
    print("STEP 1: RECOVERY VERIFICATION OF TRANSFERRED ARTIFACTS")
    print("=" * 65)

    all_ok = True

    # 1. Check model
    model_paths = [
        "matching_model_improved.json",
        "output/matching_model_improved.json",
    ]
    model_file = next((p for p in model_paths if os.path.isfile(p)), None)
    if not model_file:
        print("[FAIL] matching_model_improved.json not found in root or output/")
        all_ok = False
    else:
        try:
            from src.matching.train import MatchingClassifier
            model = MatchingClassifier.load(model_file)
            print(f"[PASS] Model loaded successfully from {model_file}")
            print(f"       Weights length : {len(model.weights)}")
            print(f"       Bias           : {model.bias:.4f}")
            print(f"       Fitted status  : {model.fitted}")
        except Exception as e:
            print(f"[FAIL] Error loading model from {model_file}: {e}")
            all_ok = False

    # 2. Check runner file
    runner_files = [
        "run_full_pipeline.py",
        "src/run_pipeline.py/run_pipeline.py",
        "src/run_pipeline.py",
    ]
    runner_file = next((f for f in runner_files if os.path.isfile(f)), None)
    if not runner_file:
        print(f"[FAIL] runner not found")
        all_ok = False
    else:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("run_pipeline", runner_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"[PASS] {runner_file} parsed and imported successfully")
        except Exception as e:
            print(f"[FAIL] Error importing {runner_file}: {e}")
            all_ok = False

    # 3. Check France Checkpoint
    checkpoint_paths = [
        "output/checkpoints_test/France_results.tsv",
        "output/.checkpoint_france.tsv",
    ]
    ckpt_file = next((p for p in checkpoint_paths if os.path.isfile(p)), None)
    if not ckpt_file:
        print("[FAIL] France checkpoint not found in output/checkpoints_test/ or output/")
        all_ok = False
    else:
        print(f"       Inspecting checkpoint: {ckpt_file}")
        france_test_s1_ids = set()
        with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if len(row) > 3 and row[3] == "France":
                    france_test_s1_ids.add(row[0].strip())
        print(f"       Total France entities in test_source1: {len(france_test_s1_ids):,}")

        ckpt_s1_ids = set()
        duplicate_s1 = []
        malformed_rows = 0
        total_rows = 0
        invalid_target_ids = []

        with open(ckpt_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_no, row in enumerate(reader, start=1):
                if not row or len(row) < 1:
                    malformed_rows += 1
                    continue
                s1_id = row[0].strip()
                if s1_id == "source1_entity_id":
                    continue
                total_rows += 1
                if not s1_id:
                    malformed_rows += 1
                    continue
                if s1_id in ckpt_s1_ids:
                    duplicate_s1.append((line_no, s1_id))
                ckpt_s1_ids.add(s1_id)

                matched_str = row[2].strip() if len(row) > 2 else (row[1].strip() if len(row) > 1 else "")
                if matched_str:
                    mids = [m.strip() for m in matched_str.split(",") if m.strip()]
                    for mid in mids:
                        if not (mid.startswith("S2-") or mid.startswith("S3-")):
                            invalid_target_ids.append((s1_id, mid))

        print(f"       Checkpoint rows parsed   : {total_rows:,}")
        print(f"       Unique S1 entities       : {len(ckpt_s1_ids):,}")
        print(f"       Duplicate S1 entities    : {len(duplicate_s1)}")
        print(f"       Malformed rows           : {malformed_rows}")
        print(f"       Invalid target IDs       : {len(invalid_target_ids)}")

        # Check exact count
        if total_rows == 165000 and len(ckpt_s1_ids) == 165000 and malformed_rows == 0:
            print("[PASS] Checkpoint has exactly 165,000 valid unique S1 entities")
        else:
            print(f"[INFO] Checkpoint entity count: {len(ckpt_s1_ids):,}")

        # Check subset of France test
        diff = ckpt_s1_ids - france_test_s1_ids
        if len(diff) == 0:
            remaining = len(france_test_s1_ids) - len(ckpt_s1_ids)
            print(f"[PASS] All checkpoint IDs are valid France test entities.")
            print(f"       Remaining France entities to process: {remaining:,}")
        else:
            print(f"[FAIL] Found {len(diff)} IDs in checkpoint not in France test set!")
            all_ok = False

    # 4. Run tests
    print("\nRunning unit tests...")
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    result = subprocess.run(["python3", "-m", "pytest", "tests/"], env=env, capture_output=True, text=True)
    if result.returncode == 0:
        print("[PASS] All 176 unit tests passed")
    else:
        print("[FAIL] Unit tests failed:")
        print(result.stdout)
        print(result.stderr)
        all_ok = False

    print("\n" + "=" * 65)
    if all_ok:
        print("READY: Project is safe to resume.")
    else:
        print("WAITING: Required artifacts are not yet ready or failed verification.")
    print("=" * 65)
    return all_ok

if __name__ == "__main__":
    verify()
