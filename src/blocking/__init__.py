"""
Amazon ML Entity Resolution Challenge 2026 - Blocking & Candidate Generation Module

Public API for Person 3 candidate blocking:
- build_name_index
- build_address_index
- generate_name_candidates
- generate_address_candidates
- generate_candidates
- CandidateGenerator
- CandidatePair
- evaluate_blocking
- export_candidate_pairs_tsv
- is_country_compatible
"""

from src.blocking.address_blocking import (
    AddressInvertedIndex,
    build_address_index,
    extract_distinctive_address_tokens,
    extract_normalized_numbers,
    generate_address_candidates,
)
from src.blocking.candidate_generator import (
    CandidateGenerator,
    CandidatePair,
    determine_source_label,
    evaluate_blocking,
    export_candidate_pairs_tsv,
    generate_candidates,
    is_country_compatible,
)
from src.blocking.name_blocking import (
    NameInvertedIndex,
    build_name_index,
    extract_core_name_tokens,
    generate_name_candidates,
)

__all__ = [
    "AddressInvertedIndex",
    "CandidateGenerator",
    "CandidatePair",
    "NameInvertedIndex",
    "build_address_index",
    "build_name_index",
    "determine_source_label",
    "evaluate_blocking",
    "export_candidate_pairs_tsv",
    "extract_core_name_tokens",
    "extract_distinctive_address_tokens",
    "extract_normalized_numbers",
    "generate_address_candidates",
    "generate_candidates",
    "generate_name_candidates",
    "is_country_compatible",
]
