"""Pairwise similarity feature engineering for entity resolution.

Owned by: Person 2 (Matching / ML)
"""

import difflib
from typing import Dict, List, Set

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


def _token_jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection) / float(union) if union > 0 else 0.0


def _string_similarity(s1: str, s2: str) -> float:
    """Compute normalized character sequence similarity [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    if RAPIDFUZZ_AVAILABLE:
        return float(fuzz.ratio(s1, s2)) / 100.0
    else:
        matcher = difflib.SequenceMatcher(None, s1, s2)
        return float(matcher.ratio())


def _token_sort_similarity(s1: str, s2: str) -> float:
    """Compute token-sorted similarity [0.0, 1.0] to handle word order permutations."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    if RAPIDFUZZ_AVAILABLE:
        return float(fuzz.token_sort_ratio(s1, s2)) / 100.0
    else:
        tokens1 = " ".join(sorted(s1.split()))
        tokens2 = " ".join(sorted(s2.split()))
        return _string_similarity(tokens1, tokens2)


def compute_pair_features(record1: Dict[str, str], record2: Dict[str, str]) -> Dict[str, float]:
    """Compute pairwise similarity features between two records.

    Features generated:
    - name_ratio: Overall character similarity of business names
    - name_token_sort_ratio: Token-sorted similarity (robust to word order)
    - name_jaccard: Jaccard similarity of distinct name tokens
    - name_exact_match: Binary indicator for exact string equality
    - address_ratio: Overall character similarity of addresses
    - address_token_sort_ratio: Token-sorted address similarity
    - address_jaccard: Jaccard similarity of distinct address tokens
    - country_match: 1.0 if identical countries (or either is UNKNOWN), 0.0 if mismatched

    Args:
        record1: First record dictionary (e.g. from Source 1).
        record2: Second record dictionary (e.g. from Source 2 or 3).

    Returns:
        Dictionary of numeric feature names mapped to float values.
    """
    name1 = record1.get("business_name", "")
    name2 = record2.get("business_name", "")
    addr1 = record1.get("business_address", "")
    addr2 = record2.get("business_address", "")
    c1 = record1.get("country", "").upper()
    c2 = record2.get("country", "").upper()

    tokens_name1 = set(name1.split())
    tokens_name2 = set(name2.split())
    tokens_addr1 = set(addr1.split())
    tokens_addr2 = set(addr2.split())

    # Country matching: 1.0 if match, 0.0 if mismatch. Note: France is present in test.
    if not c1 or not c2 or c1 == "UNKNOWN" or c2 == "UNKNOWN":
        country_score = 0.5  # Neutral when missing
    elif c1 == c2:
        country_score = 1.0
    else:
        country_score = 0.0

    return {
        "name_ratio": _string_similarity(name1, name2),
        "name_token_sort_ratio": _token_sort_similarity(name1, name2),
        "name_jaccard": _token_jaccard_similarity(tokens_name1, tokens_name2),
        "name_exact_match": 1.0 if name1 and name1 == name2 else 0.0,
        "address_ratio": _string_similarity(addr1, addr2),
        "address_token_sort_ratio": _token_sort_similarity(addr1, addr2),
        "address_jaccard": _token_jaccard_similarity(tokens_addr1, tokens_addr2),
        "country_match": country_score,
    }
