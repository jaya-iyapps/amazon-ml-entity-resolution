"""
Amazon ML Entity Resolution Challenge 2026 - Matching Feature Extraction

This module implements pairwise feature extraction for candidate entity pairs
(Source 1 vs Source 2/3), including character similarities, token similarities,
numeric token agreement, country comparisons, and missing value indicators.
"""

import difflib
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Attempt to reuse Person 1's preprocessing API when available
try:
    from src.preprocessing.normalize import (
        clean_address,
        clean_business_name,
        extract_numeric_tokens,
        extract_tokens,
        normalize_country,
        strip_legal_suffixes,
    )
except ImportError:
    # Graceful fallback conforming strictly to Person 1's API contract
    import unicodedata

    def _nfkc(text: Optional[str]) -> str:
        return unicodedata.normalize("NFKC", text or "").strip().lower()

    def clean_business_name(name: Optional[str]) -> str:
        s = _nfkc(name)
        s = re.sub(r"[^\w\s&]", " ", s)  # replace non-alphanumeric except & with space
        s = re.sub(r"[\s\-_/]+", " ", s)
        s = s.replace("&", " and ")
        return s.strip()

    def strip_legal_suffixes(name: Optional[str]) -> str:
        s = clean_business_name(name)
        suffixes = (
            " pvt ltd", " private limited", " limited", " ltd",
            " inc", " incorporated", " llc", " llp", " corp",
            " corporation", " co", " company", " gmbh", " sarl", " sas", " sa",
        )
        changed = True
        while changed:
            changed = False
            for suf in suffixes:
                if s.endswith(suf):
                    s = s[: -len(suf)].strip()
                    changed = True
        return s

    def clean_address(address: Optional[str]) -> str:
        s = _nfkc(address)
        return re.sub(r"\s+", " ", s).strip()

    def normalize_country(country: Optional[str]) -> str:
        s = (country or "").strip().lower()
        if s in ("us", "usa", "united states", "united states of america"):
            return "US"
        if s in ("india", "ind"):
            return "India"
        if s in ("france", "fra", "fr"):
            return "France"
        return country.strip() if country else ""

    def extract_tokens(text: Optional[str], min_len: int = 1) -> List[str]:
        if not text:
            return []
        return [tok for tok in re.findall(r"\w+", text.lower()) if len(tok) >= min_len]

    def extract_numeric_tokens(text: Optional[str]) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b\d+\b", text)


def strip_all_suffixes(text: Optional[str]) -> str:
    """Iteratively strip legal suffixes until fixed point."""
    curr = clean_business_name(text)
    prev = None
    while prev != curr:
        prev = curr
        curr = strip_legal_suffixes(curr)
    return curr


FEATURE_NAMES = [
    # Name similarities
    "name_exact_match",
    "name_core_match",
    "name_token_jaccard",
    "name_token_dice",
    "name_token_sort_ratio",
    "name_levenshtein_sim",
    "name_lcs_sim",
    "name_ngram_jaccard",
    "name_len_diff_ratio",
    "name_first_token_match",
    # Address similarities
    "addr_exact_match",
    "addr_token_jaccard",
    "addr_token_dice",
    "addr_levenshtein_sim",
    "addr_ngram_jaccard",
    # Numeric token agreement
    "num_common_count",
    "num_jaccard",
    "num_disagreement",
    # Country comparison
    "country_exact_match",
    "country_mismatch",
    "country_missing",
    # Missing value indicators
    "name_missing_1",
    "name_missing_2",
    "addr_missing_1",
    "addr_missing_2",
]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate normalized Levenshtein similarity in [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row
    dist = previous_row[-1]
    return max(0.0, 1.0 - (dist / max(len(s1), len(s2))))


def lcs_similarity(s1: str, s2: str) -> float:
    """Calculate Longest Common Substring similarity ratio."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    matcher = difflib.SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    return (2.0 * match.size) / (len(s1) + len(s2))


def token_sort_similarity(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Calculate token-sort similarity ratio."""
    if not tokens1 or not tokens2:
        return 0.0
    sorted1 = " ".join(sorted(tokens1))
    sorted2 = " ".join(sorted(tokens2))
    if sorted1 == sorted2:
        return 1.0
    return difflib.SequenceMatcher(None, sorted1, sorted2).ratio()


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Calculate character n-gram Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    ngrams1 = {s1[i : i + n] for i in range(len(s1) - n + 1)} if len(s1) >= n else {s1}
    ngrams2 = {s2[i : i + n] for i in range(len(s2) - n + 1)} if len(s2) >= n else {s2}
    union_len = len(ngrams1 | ngrams2)
    if union_len == 0:
        return 0.0
    return len(ngrams1 & ngrams2) / union_len


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    """Calculate Jaccard similarity of two sets."""
    if not s1 and not s2:
        return 0.0
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


def dice_similarity(s1: Set[str], s2: Set[str]) -> float:
    """Calculate Dice similarity of two sets."""
    total = len(s1) + len(s2)
    if total == 0:
        return 0.0
    return (2.0 * len(s1 & s2)) / total


def extract_pair_features(rec1: Dict[str, Any], rec2: Dict[str, Any]) -> List[float]:
    """Extract a 25-dimensional feature vector for a pair of business records.

    rec1 is from Source 1, rec2 is from Source 2 or Source 3.
    Expected fields: 'business_name', 'business_address', 'country'.
    """
    raw_name1 = rec1.get("business_name") or ""
    raw_name2 = rec2.get("business_name") or ""
    raw_addr1 = rec1.get("business_address") or ""
    raw_addr2 = rec2.get("business_address") or ""
    raw_country1 = rec1.get("country") or ""
    raw_country2 = rec2.get("country") or ""

    # Normalization
    name1 = clean_business_name(raw_name1)
    name2 = clean_business_name(raw_name2)
    core1 = strip_all_suffixes(name1)
    core2 = strip_all_suffixes(name2)

    addr1 = clean_address(raw_addr1)
    addr2 = clean_address(raw_addr2)

    c1 = normalize_country(raw_country1)
    c2 = normalize_country(raw_country2)

    # Tokens
    tokens_name1 = extract_tokens(name1)
    tokens_name2 = extract_tokens(name2)
    set_name1 = set(tokens_name1)
    set_name2 = set(tokens_name2)

    tokens_addr1 = extract_tokens(addr1)
    tokens_addr2 = extract_tokens(addr2)
    set_addr1 = set(tokens_addr1)
    set_addr2 = set(tokens_addr2)

    # Numerics from address + name
    num1 = set(extract_numeric_tokens(addr1) + extract_numeric_tokens(name1))
    num2 = set(extract_numeric_tokens(addr2) + extract_numeric_tokens(name2))

    # Missing value flags
    name_miss1 = 1.0 if not name1 else 0.0
    name_miss2 = 1.0 if not name2 else 0.0
    addr_miss1 = 1.0 if not addr1 else 0.0
    addr_miss2 = 1.0 if not addr2 else 0.0
    c_missing = 1.0 if (not c1 or not c2) else 0.0

    # Name features
    name_exact = 1.0 if (name1 and name1 == name2) else 0.0
    name_core = 1.0 if (core1 and core1 == core2) else 0.0
    name_jaccard = jaccard_similarity(set_name1, set_name2)
    name_dice = dice_similarity(set_name1, set_name2)
    name_token_sort = token_sort_similarity(tokens_name1, tokens_name2)
    name_lev = levenshtein_similarity(name1, name2)
    name_lcs = lcs_similarity(name1, name2)
    name_ngram = char_ngram_jaccard(name1, name2, n=3)
    max_name_len = max(len(name1), len(name2), 1)
    name_len_diff = abs(len(name1) - len(name2)) / max_name_len
    name_first_match = (
        1.0
        if (tokens_name1 and tokens_name2 and tokens_name1[0] == tokens_name2[0])
        else 0.0
    )

    # Address features
    addr_exact = 1.0 if (addr1 and addr1 == addr2) else 0.0
    addr_jaccard = jaccard_similarity(set_addr1, set_addr2)
    addr_dice = dice_similarity(set_addr1, set_addr2)
    addr_lev = levenshtein_similarity(addr1, addr2)
    addr_ngram = char_ngram_jaccard(addr1, addr2, n=3)

    # Numeric features
    common_num = num1 & num2
    num_common_count = float(len(common_num))
    num_jacc = jaccard_similarity(num1, num2)
    # Disagreement: both have numbers, but share none
    num_disagree = 1.0 if (num1 and num2 and not common_num) else 0.0

    # Country features (open-set comparison)
    if c1 and c2:
        c_exact = 1.0 if c1 == c2 else 0.0
        c_mismatch = 1.0 if c1 != c2 else 0.0
    else:
        c_exact = 0.0
        c_mismatch = 0.0

    return [
        name_exact,
        name_core,
        name_jaccard,
        name_dice,
        name_token_sort,
        name_lev,
        name_lcs,
        name_ngram,
        name_len_diff,
        name_first_match,
        addr_exact,
        addr_jaccard,
        addr_dice,
        addr_lev,
        addr_ngram,
        num_common_count,
        num_jacc,
        num_disagree,
        c_exact,
        c_mismatch,
        c_missing,
        name_miss1,
        name_miss2,
        addr_miss1,
        addr_miss2,
    ]


def compute_pair_features(
    record1: Dict[str, str], record2: Dict[str, str]
) -> Dict[str, float]:
    """Compute pairwise similarity features dictionary between two records.

    Provided for backward compatibility with pipeline orchestrator and legacy tests.
    """
    name1 = record1.get("business_name", "")
    name2 = record2.get("business_name", "")
    addr1 = record1.get("business_address", "")
    addr2 = record2.get("business_address", "")
    c1 = (record1.get("country") or "").upper()
    c2 = (record2.get("country") or "").upper()

    tokens_name1 = set(name1.split())
    tokens_name2 = set(name2.split())
    tokens_addr1 = set(addr1.split())
    tokens_addr2 = set(addr2.split())

    if not c1 or not c2 or c1 == "UNKNOWN" or c2 == "UNKNOWN":
        country_score = 0.5
    elif c1 == c2:
        country_score = 1.0
    else:
        country_score = 0.0

    return {
        "name_ratio": levenshtein_similarity(name1, name2),
        "name_token_sort_ratio": token_sort_similarity(list(tokens_name1), list(tokens_name2)),
        "name_jaccard": jaccard_similarity(tokens_name1, tokens_name2),
        "name_exact_match": 1.0 if name1 and name1 == name2 else 0.0,
        "address_ratio": levenshtein_similarity(addr1, addr2),
        "address_token_sort_ratio": token_sort_similarity(list(tokens_addr1), list(tokens_addr2)),
        "address_jaccard": jaccard_similarity(tokens_addr1, tokens_addr2),
        "country_match": country_score,
    }

