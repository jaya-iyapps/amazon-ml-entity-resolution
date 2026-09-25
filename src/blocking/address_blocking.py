"""Address-based blocking strategies for candidate generation.

Owned by: Person 3 (Blocking / Candidate Generation)

Candidate generation must happen BEFORE expensive pairwise ML scoring.
Blocking drastically reduces comparison pairs by grouping records sharing
location components (postal codes, street tokens, city keys).
"""

from collections import defaultdict
import re
from typing import Dict, Iterable, List, Set


def extract_address_tokens(normalized_address: str, min_length: int = 4) -> List[str]:
    """Extract location tokens, postal numbers, and street keywords from address."""
    # Split on whitespace and extract alphanumeric tokens with at least min_length
    tokens = [t for t in normalized_address.split() if len(t) >= min_length]
    return tokens


def build_address_inverted_index(
    records: Iterable[Dict[str, str]],
    address_field: str = "business_address",
    id_field: str = "entity_id",
    min_token_len: int = 4,
) -> Dict[str, Set[str]]:
    """Build an inverted index mapping address tokens to entity IDs.

    Args:
        records: Iterable of record dictionaries.
        address_field: Key for the address field.
        id_field: Key for entity identifier.
        min_token_len: Minimum token character length.

    Returns:
        Mapping of address token -> set of entity IDs.
    """
    index: Dict[str, Set[str]] = defaultdict(set)
    for record in records:
        entity_id = record.get(id_field, "")
        address = record.get(address_field, "")
        if not entity_id or not address:
            continue
        for token in extract_address_tokens(address, min_length=min_token_len):
            index[token].add(entity_id)
    return dict(index)


def generate_address_candidates(
    s1_record: Dict[str, str],
    address_index: Dict[str, Set[str]],
    max_candidates: int = 100,
    min_token_len: int = 4,
) -> Set[str]:
    """Retrieve candidate entity IDs sharing address tokens with a Source 1 record.

    Args:
        s1_record: Source 1 entity dictionary.
        address_index: Inverted index built from target sources (Source 2 / Source 3).
        max_candidates: Maximum candidates to retrieve.
        min_token_len: Minimum token length.

    Returns:
        Set of candidate entity IDs.
    """
    address = s1_record.get("business_address", "")
    candidates: Set[str] = set()

    for token in extract_address_tokens(address, min_length=min_token_len):
        matched_ids = address_index.get(token, set())
        candidates.update(matched_ids)
        if len(candidates) >= max_candidates:
            break

    return set(list(candidates)[:max_candidates])
