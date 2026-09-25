"""Candidate generation and blocking module.

Owned by: Person 3 (Blocking / Candidate Generation)
"""

from src.blocking.name_blocking import (
    generate_name_candidates,
    build_name_inverted_index,
)
from src.blocking.address_blocking import (
    generate_address_candidates,
    build_address_inverted_index,
)
from src.blocking.candidate_generator import (
    generate_candidates,
    evaluate_candidate_recall,
    save_candidate_pairs,
)

__all__ = [
    "generate_name_candidates",
    "build_name_inverted_index",
    "generate_address_candidates",
    "build_address_inverted_index",
    "generate_candidates",
    "evaluate_candidate_recall",
    "save_candidate_pairs",
]
