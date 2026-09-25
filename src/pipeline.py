"""End-to-End Pipeline orchestrator for Amazon ML Challenge 2026.

Integrates:
1. Preprocessing (Person 1)
2. Blocking & Candidate Generation (Person 3)
3. Pairwise Features & ML Matching (Person 2)
4. Evaluation & Submission Generation
"""

import argparse
import os
from typing import Any, Dict, List, Optional
import yaml

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)
from src.blocking.candidate_generator import (
    generate_candidates,
    save_candidate_pairs,
)
from src.matching.features import compute_pair_features
from src.matching.predict import (
    format_matching_results,
    save_matching_results,
)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load pipeline YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class EntityResolutionPipeline:
    """Modular Entity Resolution Pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get("matching", {}).get("threshold", 0.5)

    def preprocess_record(self, record: Dict[str, str]) -> Dict[str, str]:
        """Normalize fields of a single record using Person 1's preprocessing module."""
        return {
            "entity_id": record.get("entity_id", ""),
            "business_name": normalize_business_name(record.get("business_name", "")),
            "business_address": normalize_address(record.get("business_address", "")),
            "country": normalize_country(record.get("country", "")),
        }

    def generate_candidate_pairs(
        self,
        s1_records: List[Dict[str, str]],
        target_records: List[Dict[str, str]],
    ) -> Dict[str, List[str]]:
        """Run blocking stage using Person 3's blocking module."""
        max_cands = self.config.get("blocking", {}).get("max_candidates_per_entity", 50)
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
        scored_pairs = []
        for s1_id, cand_ids in candidate_pairs.items():
            s1_rec = s1_records_by_id.get(s1_id)
            if not s1_rec:
                continue

            for cand_id in cand_ids:
                target_rec = target_records_by_id.get(cand_id)
                if not target_rec:
                    continue

                features = compute_pair_features(s1_rec, target_rec)
                # Baseline heuristic score combining name & address similarity
                baseline_score = (
                    0.6 * features["name_token_sort_ratio"]
                    + 0.3 * features["address_token_sort_ratio"]
                    + 0.1 * features["country_match"]
                )
                scored_pairs.append((s1_id, cand_id, baseline_score))

        return scored_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run entity resolution pipeline.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Candidate output: {config.get('output', {}).get('candidate_pairs_file')}")
    print(f"Matching output: {config.get('output', {}).get('matching_results_file')}")


if __name__ == "__main__":
    main()
