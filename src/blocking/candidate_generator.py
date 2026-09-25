"""Candidate generation pipeline and recall evaluation.

Owned by: Person 3 (Blocking / Candidate Generation)

IMPORTANT:
Candidate generation must execute BEFORE expensive pairwise ML scoring.
The output candidate pairs define the recall ceiling for the matching model.
Every true match missed at this stage is permanently lost.
"""

from typing import Dict, Iterable, List, Optional, Set

from src.blocking.address_blocking import (
    build_address_inverted_index,
    generate_address_candidates,
)
from src.blocking.name_blocking import (
    build_name_inverted_index,
    generate_name_candidates,
)


def generate_candidates(
    s1_records: Iterable[Dict[str, str]],
    target_records: Iterable[Dict[str, str]],
    max_candidates_per_entity: int = 50,
) -> Dict[str, List[str]]:
    """Generate candidate pairs combining name and address blocking.

    Args:
        s1_records: Iterable of Source 1 records.
        target_records: Iterable of Source 2 and Source 3 records.
        max_candidates_per_entity: Maximum number of candidates per Source 1 entity.

    Returns:
        Mapping of source1_entity_id -> list of candidate target entity IDs.
    """
    target_list = list(target_records)

    # Build inverted indexes over target records
    name_index = build_name_inverted_index(target_list)
    address_index = build_address_inverted_index(target_list)

    candidates_map: Dict[str, List[str]] = {}

    for s1 in s1_records:
        s1_id = s1.get("entity_id", "")
        if not s1_id:
            continue

        name_cands = generate_name_candidates(
            s1, name_index, max_candidates=max_candidates_per_entity
        )
        addr_cands = generate_address_candidates(
            s1, address_index, max_candidates=max_candidates_per_entity
        )

        # Union candidate sets and enforce maximum count
        combined = list(name_cands | addr_cands)
        candidates_map[s1_id] = combined[:max_candidates_per_entity]

    return candidates_map


def evaluate_candidate_recall(
    candidate_pairs: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """Calculate the candidate recall ceiling achieved by the blocking stage.

    Recall ceiling = (true ground truth pairs present in candidate pairs) / (total true ground truth pairs).

    Args:
        candidate_pairs: Mapping of s1_id -> list of candidate target IDs.
        ground_truth: Mapping of s1_id -> list of true matching target IDs.

    Returns:
        Dictionary with total_ground_truth_links, captured_links, and candidate_recall.
    """
    total_true_links = 0
    captured_links = 0

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        cands_set = set(candidate_pairs.get(s1_id, []))
        for match_id in true_matches:
            total_true_links += 1
            if match_id in cands_set:
                captured_links += 1

    recall = (captured_links / total_true_links) if total_true_links > 0 else 1.0
    return {
        "total_true_links": float(total_true_links),
        "captured_links": float(captured_links),
        "candidate_recall": recall,
    }


def save_candidate_pairs(
    candidate_pairs: Dict[str, List[str]],
    output_path: str,
) -> None:
    """Save candidate pairs in the required competition TSV format.

    Format:
        source1_entity_id\\tcandidate_entity_ids
        S1-00001\\tS2-00047,S2-00193,S3-00812
        S1-00002\\t

    Args:
        candidate_pairs: Mapping of s1_id -> candidate IDs.
        output_path: Destination TSV filepath.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in candidate_pairs.items():
            cands_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cands_str}\n")
