"""End-to-End Pipeline orchestrator for Amazon ML Challenge 2026.

Integrates:
1. Preprocessing (Person 1)
2. Blocking & Candidate Generation (Person 3)
3. Pairwise Features & ML Matching (Person 2)
4. Evaluation & Submission Generation
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)
try:
    from src.blocking.candidate_generator import (
        generate_candidates,
        save_candidate_pairs,
    )
except ImportError:
    from src.blocking.candidate_generator import (
        generate_candidates,
        export_candidate_pairs_tsv as save_candidate_pairs,
    )

from src.matching.features import compute_pair_features, extract_pair_features
from src.matching.predict import (
    format_matching_results,
    predict_matches,
    save_matching_results,
    write_matching_results_tsv,
)
from src.matching.threshold import evaluate_predictions
from src.matching.train import MatchingClassifier, load_records_tsv, load_ground_truth_tsv


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load pipeline YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class EntityResolutionPipeline:
    """Modular Entity Resolution Pipeline integrating Preprocessing, Blocking, and Matching."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get("matching", {}).get("threshold", 0.98)
        self.model: Optional[MatchingClassifier] = None
        model_path = self.config.get("matching", {}).get("model_path", "output/matching_model.json")
        if os.path.isfile(model_path):
            try:
                self.model = MatchingClassifier.load(model_path)
            except Exception:
                self.model = None

    def preprocess_record(self, record: Dict[str, str]) -> Dict[str, str]:
        """Normalize fields of a single record using Person 1's preprocessing module."""
        return {
            "entity_id": record.get("entity_id", ""),
            "business_name": normalize_business_name(record.get("business_name", "")),
            "business_address": normalize_address(record.get("business_address", "")),
            "country": normalize_country(record.get("country", "")),
        }

    def preprocess_records(self, records: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Normalize a collection of records."""
        return [self.preprocess_record(r) for r in records]

    def generate_candidate_pairs(
        self,
        s1_records: List[Dict[str, str]],
        target_records: List[Dict[str, str]],
    ) -> Dict[str, List[str]]:
        """Run blocking stage using Person 3's blocking module."""
        max_cands = self.config.get("blocking", {}).get("max_candidates_per_entity", 80)
        return generate_candidates(
            s1_records=s1_records,
            target_records=target_records,
            max_candidates_per_entity=max_cands,
        )

    def score_candidate_pairs(
        self,
        s1_records_by_id: Dict[str, Dict[str, str]],
        target_records_by_id: Dict[str, Dict[str, str]],
        candidate_pairs: Dict[str, List[str]],
        model: Optional[Any] = None,
    ) -> List[tuple]:
        """Compute features and generate match probabilities using Person 2's matching module."""
        active_model = model or self.model
        scored_pairs = []

        if active_model is not None and hasattr(active_model, "predict_proba"):
            batch_size = 50000
            current_batch = []
            for s1_id, cand_ids in candidate_pairs.items():
                s1_rec = s1_records_by_id.get(s1_id)
                if not s1_rec:
                    continue
                for cand_id in cand_ids:
                    target_rec = target_records_by_id.get(cand_id)
                    if not target_rec:
                        continue
                    current_batch.append((s1_id, cand_id, s1_rec, target_rec))
                    if len(current_batch) >= batch_size:
                        features = [extract_pair_features(p[2], p[3]) for p in current_batch]
                        probs = active_model.predict_proba(features)
                        for (s1, cid, _, _), prob in zip(current_batch, probs):
                            scored_pairs.append((s1, cid, float(prob)))
                        current_batch = []

            if current_batch:
                features = [extract_pair_features(p[2], p[3]) for p in current_batch]
                probs = active_model.predict_proba(features)
                for (s1, cid, _, _), prob in zip(current_batch, probs):
                    scored_pairs.append((s1, cid, float(prob)))
            return scored_pairs

        # Baseline heuristic fallback
        for s1_id, cand_ids in candidate_pairs.items():
            s1_rec = s1_records_by_id.get(s1_id)
            if not s1_rec:
                continue

            for cand_id in cand_ids:
                target_rec = target_records_by_id.get(cand_id)
                if not target_rec:
                    continue

                features = compute_pair_features(s1_rec, target_rec)
                baseline_score = (
                    0.6 * features.get("name_token_sort_ratio", 0.0)
                    + 0.3 * features.get("address_token_sort_ratio", 0.0)
                    + 0.1 * features.get("country_match", 0.0)
                )
                scored_pairs.append((s1_id, cand_id, float(baseline_score)))

        return scored_pairs

    def run(
        self,
        s1_path: str,
        s2_path: str,
        s3_path: str,
        output_candidate_path: Optional[str] = None,
        output_matching_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute full pipeline: Preprocessing -> Blocking -> Matching -> Prediction Export."""
        print("=" * 65)
        print("END-TO-END ENTITY RESOLUTION PIPELINE")
        print("=" * 65)

        tau = threshold if threshold is not None else self.threshold
        out_cand = output_candidate_path or self.config.get("output", {}).get("candidate_pairs_file", "output/candidate_pairs.tsv")
        out_match = output_matching_path or self.config.get("output", {}).get("matching_results_file", "output/matching_results.tsv")

        print("[Stage 1/4] Loading and Preprocessing Records...")
        s1_dict = load_records_tsv(s1_path)
        s2_dict = load_records_tsv(s2_path)
        s3_dict = load_records_tsv(s3_path)
        all_s1_ids = list(s1_dict.keys())
        target_dict = {**s2_dict, **s3_dict}
        print(f"            Reference entities (S1) : {len(s1_dict):,}")
        print(f"            Target entities (S2+S3) : {len(target_dict):,}")

        # Check if pre-computed candidates file already exists to optimize execution
        if out_cand and os.path.isfile(out_cand):
            print(f"[Stage 2/4] Reusing verified candidate pairs: {out_cand}")
            candidate_pairs: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}
            with open(out_cand, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader, None)
                for row in reader:
                    if row and row[0] in candidate_pairs:
                        cands = [c.strip() for c in row[1].split(",") if c.strip()] if len(row) > 1 and row[1] else []
                        candidate_pairs[row[0]] = cands
        else:
            print("[Stage 2/4] Running Candidate Generation (Blocking)...")
            s1_list = list(s1_dict.values())
            target_list = list(target_dict.values())
            candidate_pairs = self.generate_candidate_pairs(s1_list, target_list)
            if out_cand:
                save_candidate_pairs(candidate_pairs, all_s1_ids, out_cand)
                print(f"            Saved candidate pairs to: {out_cand}")

        n_cands = sum(len(v) for v in candidate_pairs.values())
        print(f"            Total candidates to evaluate: {n_cands:,}")

        print(f"[Stage 3/4] Scoring Candidate Pairs (threshold = {tau:.2f})...")
        scored_pairs = self.score_candidate_pairs(s1_dict, target_dict, candidate_pairs)
        predictions = format_matching_results(all_s1_ids, scored_pairs, threshold=tau)

        print(f"[Stage 4/4] Writing Final Predictions to: {out_match}")
        write_matching_results_tsv(predictions, all_s1_ids, out_match)

        results = {
            "num_s1": len(all_s1_ids),
            "num_candidates": n_cands,
            "threshold": tau,
            "candidate_output": out_cand,
            "matching_output": out_match,
            "predictions": predictions,
        }

        # Optional Ground Truth Evaluation
        if ground_truth_path and os.path.isfile(ground_truth_path):
            gt = load_ground_truth_tsv(ground_truth_path)
            pred_sets = {s1: set(predictions.get(s1, [])) for s1 in all_s1_ids}
            gt_sets = {s1: gt.get(s1, set()) for s1 in all_s1_ids}
            metrics = evaluate_predictions(pred_sets, gt_sets, all_s1_ids)
            results["evaluation"] = metrics
            print("\n" + "=" * 65)
            print("EVALUATION ON GROUND TRUTH")
            print("=" * 65)
            print(f"  • Macro F0.5 Score   : {metrics['macro_f05']:.4f}")
            print(f"  • Macro Precision    : {metrics['macro_precision']:.4f}")
            print(f"  • Macro Recall       : {metrics['macro_recall']:.4f}")
            print(f"  • Singleton Accuracy : {metrics['singleton_accuracy']:.2%} ({metrics['singleton_count']} singletons)")
            print(f"  • Non-singleton F0.5 : {metrics['non_singleton_f05']:.4f}")
            print("=" * 65)

        return results


def main() -> None:
    import csv
    parser = argparse.ArgumentParser(description="Run entity resolution pipeline.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--run", action="store_true", help="Execute the end-to-end pipeline.")
    parser.add_argument("--data-dir", default="dataset/sample", help="Directory containing dataset TSVs.")
    parser.add_argument("--threshold", type=float, default=None, help="Matching decision threshold.")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = EntityResolutionPipeline(config)

    if args.run:
        s1_path = os.path.join(args.data_dir, "train_source1.tsv")
        s2_path = os.path.join(args.data_dir, "train_source2.tsv")
        s3_path = os.path.join(args.data_dir, "train_source3.tsv")
        gt_path = os.path.join(args.data_dir, "train_ground_truth.tsv")

        pipeline.run(
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            ground_truth_path=gt_path if os.path.isfile(gt_path) else None,
            threshold=args.threshold,
        )
    else:
        print(f"Loaded configuration from {args.config}")
        print(f"Candidate output : {config.get('output', {}).get('candidate_pairs_file')}")
        print(f"Matching output  : {config.get('output', {}).get('matching_results_file')}")
        print(f"Default threshold: {config.get('matching', {}).get('threshold', 0.5)}")


if __name__ == "__main__":
    main()
