"""Name-based blocking and inverted index strategies for candidate generation.

Owned by: Person 3 (Blocking / Candidate Generation)

Candidate generation must happen BEFORE expensive pairwise ML scoring.
Blocking drastically reduces the O(N * M) comparison space into a small,
high-recall candidate pool.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Set


def extract_name_tokens(normalized_name: str, min_length: int = 3) -> List[str]:
    """Extract candidate index tokens (words and prefixes) from normalized business names."""
    tokens = [t for t in normalized_name.split() if len(t) >= min_length]
    return tokens


def build_name_inverted_index(
    records: Iterable[Dict[str, str]],
    name_field: str = "business_name",
    id_field: str = "entity_id",
    min_token_len: int = 3,
) -> Dict[str, Set[str]]:
    """Build an inverted index mapping name tokens to entity IDs.

    Args:
        records: Iterable of record dictionaries (must contain id_field and name_field).
        name_field: Key for the business name.
        id_field: Key for the entity identifier.
        min_token_len: Minimum token character length.

    Returns:
        Mapping of token -> set of entity IDs.
    """
    index: Dict[str, Set[str]] = defaultdict(set)
    for record in records:
        entity_id = record.get(id_field, "")
        name = record.get(name_field, "")
        if not entity_id or not name:
            continue
        for token in extract_name_tokens(name, min_length=min_token_len):
            index[token].add(entity_id)
    return dict(index)


def generate_name_candidates(
    s1_record: Dict[str, str],
    name_index: Dict[str, Set[str]],
    max_candidates: int = 100,
    min_token_len: int = 3,
) -> Set[str]:
    """Retrieve candidate entity IDs sharing significant name tokens with a Source 1 record.

    Args:
        s1_record: Source 1 entity dictionary.
        name_index: Inverted index built from target sources (Source 2 / Source 3).
        max_candidates: Maximum candidates to retrieve for this record.
        min_token_len: Minimum token character length.

    Returns:
        Set of candidate entity IDs.
    """
    name = s1_record.get("business_name", "")
    candidates: Set[str] = set()

    for token in extract_name_tokens(name, min_length=min_token_len):
        matched_ids = name_index.get(token, set())
        candidates.update(matched_ids)
        if len(candidates) >= max_candidates:
            break

    return set(list(candidates)[:max_candidates])
