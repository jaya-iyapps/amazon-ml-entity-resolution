#!/usr/bin/env bash
set -e

PROJECT_DIR="/home/pyro/amazon-ml-entity-resolution"
cd "$PROJECT_DIR"

export PYTHONPATH="/home/pyro/snap/antigravity-cli/common/local/lib/python3.14/dist-packages:$PROJECT_DIR:$PROJECT_DIR/src:$PYTHONPATH"

echo "========================================================================"
echo "  AMAZON ML CHALLENGE 2026: OFFICIAL FULL PIPELINE RUNNER"
echo "  Started at: $(date -u)"
echo "========================================================================"

# Step 1: Check if rsync is still active
if pgrep -f "rsync.*checkpoints_test" > /dev/null; then
    echo "[INFO] Detected active rsync transfer for France checkpoint."
    echo "[INFO] Waiting for transfer to complete to avoid restarting France from zero..."
    while pgrep -f "rsync.*checkpoints_test" > /dev/null; do
        sleep 5
    done
    echo "[INFO] Rsync transfer finished! Proceeding with pipeline."
fi

# Step 2: Launch full pipeline
echo "[INFO] Starting pipeline with 4 workers and threshold 0.98..."
python3 run_full_pipeline.py \
    --test-dir dataset/test \
    --out-matching output/matching_results_test.tsv \
    --out-candidate output/candidate_pairs_test.tsv \
    --checkpoint-dir output/checkpoints_test \
    --model matching_model_improved.json \
    --threshold 0.98 \
    --max-candidates 80 \
    --workers 4 \
    2>&1 | tee -a run_pipeline.log

# Step 3: Run comprehensive submission validation
echo ""
echo "========================================================================"
echo "  RUNNING FINAL VALIDATION AND INTEGRITY CHECKS"
echo "========================================================================"
python3 scripts/validate_final_outputs.py 2>&1 | tee -a run_pipeline.log

echo ""
echo "========================================================================"
echo "  OFFICIAL TEST RUN FINISHED at: $(date -u)"
echo "========================================================================"
