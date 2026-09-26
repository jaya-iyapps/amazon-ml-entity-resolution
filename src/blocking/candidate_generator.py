"""
Amazon ML Entity Resolution Challenge 2026 - Candidate Generator & Evaluator

This module coordinates multi-pass candidate generation (name + address + numeric),
enforces open-set country compatibility, applies candidate-size safeguards,
and exports the official output/candidate_pairs.tsv.
"""

import csv
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.blocking.address_blocking import (
    AddressInvertedIndex,
    build_address_index,
    extract_distinctive_address_tokens,
    generate_address_candidates,
)
from src.blocking.name_blocking import (
    NameInvertedIndex,
    build_name_index,
    extract_core_name_tokens,
    generate_name_candidates,
    normalize_country,
)


@dataclass(frozen=True)
class CandidatePair:
    """Structured representation of a single candidate pair."""

    source1_entity_id: str
    candidate_entity_id: str
    source: str  # 'source2' or 'source3'


def determine_source_label(entity_id: str) -> str:
    """Infer the source partition from the entity ID prefix."""
    if entity_id.startswith("S2-"):
        return "source2"
    if entity_id.startswith("S3-"):
        return "source3"
    if entity_id.startswith("S1-"):
        return "source1"
    return "unknown"


def is_country_compatible(country1: Optional[str], country2: Optional[str]) -> bool:
    """Check country compatibility under open-set rules.

    Rules:
    - If both records have known, non-empty countries, they must match.
    - If either country is missing, empty, or unknown, they are compatible.
    - Works for US, India, France, and any unseen country string.
    """
    c1 = normalize_country(country1)
    c2 = normalize_country(country2)
    if not c1 or not c2:
        return True
    return c1.lower() == c2.lower()


class CandidateGenerator:
    """Coordinates indexing, multi-pass candidate retrieval, country filtering, and pruning."""

    def __init__(
        self,
        max_candidates_per_entity: Optional[int] = 80,
        max_name_token_freq: int = 150,
        max_address_key_freq: int = 100,
    ):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.max_name_token_freq = max_name_token_freq
        self.max_address_key_freq = max_address_key_freq
        self.name_index: Optional[NameInvertedIndex] = None
        self.address_index: Optional[AddressInvertedIndex] = None
        self.target_records_cache: Dict[str, Dict[str, Any]] = {}

    def fit(self, target_records: Sequence[Dict[str, Any]]) -> "CandidateGenerator":
        """Index Source 2 and Source 3 records."""
        self.target_records_cache = {rec["entity_id"]: rec for rec in target_records}
        self.name_index = build_name_index(target_records, max_token_freq=self.max_name_token_freq)
        self.address_index = build_address_index(target_records, max_key_freq=self.max_address_key_freq)
        return self

    def generate_candidates_for_record(self, s1_rec: Dict[str, Any]) -> List[str]:
        """Generate deduplicated, country-filtered, size-controlled candidate IDs for one S1 record."""
        if not self.name_index or not self.address_index:
            raise RuntimeError("CandidateGenerator must be fitted before generating candidates.")

        name = s1_rec.get("business_name")
        addr = s1_rec.get("business_address")
        s1_country = s1_rec.get("country")

        # 1. Multi-pass candidate union
        cands: Set[str] = set()
        cands.update(self.name_index.query(name))
        cands.update(self.address_index.query(addr, name))

        if not cands:
            return []

        # 2. Open-set Country Compatibility Filter
        country_filtered: List[str] = []
        for cid in cands:
            target_rec = self.target_records_cache.get(cid)
            target_country = target_rec.get("country") if target_rec else None
            if is_country_compatible(s1_country, target_country):
                country_filtered.append(cid)

        # 3. Candidate Size Control & Pruning
        if (
            self.max_candidates_per_entity is not None
            and len(country_filtered) > self.max_candidates_per_entity
        ):
            s1_name_toks = set(extract_core_name_tokens(name))
            s1_addr_toks = set(extract_distinctive_address_tokens(addr))

            def score_candidate(cand_id: str) -> Tuple[int, str]:
                t_rec = self.target_records_cache.get(cand_id, {})
                t_name_toks = set(extract_core_name_tokens(t_rec.get("business_name")))
                t_addr_toks = set(extract_distinctive_address_tokens(t_rec.get("business_address")))
                # Name overlap weighted 2x, address overlap 1x
                score = 2 * len(s1_name_toks & t_name_toks) + len(s1_addr_toks & t_addr_toks)
                return (score, cand_id)

            ranked = sorted(country_filtered, key=score_candidate, reverse=True)
            country_filtered = [item for item in ranked[: self.max_candidates_per_entity]]

        # Deterministic sort for reproducibility
        return sorted(set(country_filtered))

    def generate(
        self, s1_records: Sequence[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Generate candidate ID lists for all Source 1 records."""
        candidates_map: Dict[str, List[str]] = {}
        for s1_rec in s1_records:
            s1_id = s1_rec["entity_id"]
            candidates_map[s1_id] = self.generate_candidates_for_record(s1_rec)
        return candidates_map

    def generate_pairs(
        self, s1_records: Sequence[Dict[str, Any]]
    ) -> List[CandidatePair]:
        """Generate flat structured CandidatePair objects."""
        candidates_map = self.generate(s1_records)
        pairs: List[CandidatePair] = []
        for s1_id, cand_ids in candidates_map.items():
            for cid in cand_ids:
                pairs.append(
                    CandidatePair(
                        source1_entity_id=s1_id,
                        candidate_entity_id=cid,
                        source=determine_source_label(cid),
                    )
                )
        return pairs


def generate_candidates(
    s1_records: Sequence[Dict[str, Any]],
    target_records: Sequence[Dict[str, Any]],
    max_candidates_per_entity: Optional[int] = 80,
) -> Dict[str, List[str]]:
    """Convenience function to fit and generate candidates end-to-end."""
    generator = CandidateGenerator(max_candidates_per_entity=max_candidates_per_entity)
    generator.fit(target_records)
    return generator.generate(s1_records)


def evaluate_blocking(
    candidates: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Set[str]],
    total_target_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate candidate recall and reduction ratio against training ground truth."""
    total_true_pairs = 0
    retrieved_true_pairs = 0

    for s1_id, true_matches in ground_truth.items():
        total_true_pairs += len(true_matches)
        cand_set = set(candidates.get(s1_id, ()))
        retrieved_true_pairs += len(true_matches & cand_set)

    candidate_lengths = [len(cands) for cands in candidates.values()]
    total_candidates = sum(candidate_lengths)
    n_s1 = len(candidates) if candidates else 1

    recall = (retrieved_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    avg_cands = total_candidates / n_s1
    median_cands = statistics.median(candidate_lengths) if candidate_lengths else 0.0
    min_cands = min(candidate_lengths) if candidate_lengths else 0
    max_cands = max(candidate_lengths) if candidate_lengths else 0

    # Reduction ratio: 1 - (total candidates / total possible pairwise comparisons)
    if total_target_records and total_target_records > 0:
        total_possible = n_s1 * total_target_records
        reduction_ratio = 1.0 - (total_candidates / total_possible)
    else:
        reduction_ratio = 0.0

    return {
        "candidate_recall": recall,
        "retrieved_true_pairs": retrieved_true_pairs,
        "total_true_pairs": total_true_pairs,
        "total_candidates": total_candidates,
        "num_s1_entities": n_s1,
        "avg_candidates_per_s1": avg_cands,
        "median_candidates_per_s1": median_cands,
        "min_candidates_per_s1": min_cands,
        "max_candidates_per_s1": max_cands,
        "reduction_ratio": reduction_ratio,
    }


def export_candidate_pairs_tsv(
    candidates: Dict[str, Sequence[str]],
    all_s1_ids: Sequence[str],
    output_path: str,
) -> None:
    """Export candidates strictly to output/candidate_pairs.tsv conforming to the challenge schema."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, ())
            # Deduplicate and sort deterministically
            cand_str = ",".join(sorted(set(cands)))
            writer.writerow([s1_id, cand_str])
