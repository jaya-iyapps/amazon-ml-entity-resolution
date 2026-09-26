"""
Amazon ML Entity Resolution Challenge 2026 - Name-Based Blocking

This module implements inverted indexes and multi-pass candidate generation
using normalized business names, core name tokens, and prefix signatures.
Includes safeguards against high-frequency token explosion.
"""

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Attempt to reuse Person 1's preprocessing API when available
try:
    from src.preprocessing.normalize import (
        clean_business_name,
        extract_tokens,
        normalize_country,
        strip_legal_suffixes,
    )
except ImportError:
    # Graceful fallback strictly conforming to Person 1's API contract
    def _nfkc(text: Optional[str]) -> str:
        return unicodedata.normalize("NFKC", text or "").strip().lower()

    def clean_business_name(name: Optional[str]) -> str:
        s = _nfkc(name)
        s = re.sub(r"[^\w\s&]", " ", s)
        s = re.sub(r"[\s\-_/]+", " ", s)
        s = s.replace("&", " and ")
        return s.strip()

    def strip_legal_suffixes(name: Optional[str]) -> str:
        s = clean_business_name(name)
        suffixes = (
            " pvt ltd", " private limited", " limited", " ltd",
            " inc", " incorporated", " llc", " llp", " corp",
            " corporation", " co", " company", " gmbh", " sarl",
            " sas", " sa", " dba",
        )
        changed = True
        while changed:
            changed = False
            for suf in suffixes:
                if s.endswith(suf):
                    s = s[: -len(suf)].strip()
                    changed = True
        return s

    def normalize_country(country: Optional[str]) -> str:
        s = (country or "").strip().lower()
        if s in ("us", "usa", "united states", "united states of america"):
            return "US"
        if s in ("india", "ind"):
            return "India"
        if s in ("france", "fra", "fr"):
            return "France"
        return country.strip() if country else ""

    def extract_tokens(text: Optional[str]) -> List[str]:
        if not text:
            return []
        return [tok for tok in re.findall(r"\w+", text.lower())]


COMMON_STOPWORDS = frozenset({
    "the", "and", "of", "for", "in", "to", "at", "a", "an", "on", "by",
    "services", "solutions", "traders", "enterprise", "enterprises",
    "international", "retail", "group", "holdings", "management",
    "industries", "associates", "global", "national", "center", "centre",
})


def strip_domains_and_noise(text: Optional[str]) -> str:
    """Strip website prefixes (www.), top-level domains (.com, .org, etc.), and clean punctuation."""
    if not text:
        return ""
    s = re.sub(r"^(https?://)?(www\.)?", "", text.lower())
    s = re.sub(r"(\.com|\.org|\.net|\.in|\.co|\.io|\.biz|\.info|\.us|\.fr)\b", " ", s)
    return s


def extract_core_name_tokens(raw_name: Optional[str]) -> List[str]:
    """Normalize name, strip domains and legal suffixes, and extract distinctive tokens."""
    if not raw_name:
        return []
    clean_url_name = strip_domains_and_noise(raw_name)
    core = strip_legal_suffixes(clean_url_name)
    tokens = [t for t in extract_tokens(core) if len(t) >= 2]
    return [t for t in tokens if t not in COMMON_STOPWORDS]


class NameInvertedIndex:
    """Multi-key inverted index for business names."""

    def __init__(self, max_token_freq: int = 150, max_bucket_size: int = 200):
        self.max_token_freq = max_token_freq
        self.max_bucket_size = max_bucket_size
        self.exact_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.sorted_tokens_idx: Dict[str, List[str]] = defaultdict(list)
        self.concat_name_idx: Dict[str, List[str]] = defaultdict(list)
        self.token_pair_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.rare_token_idx: Dict[str, List[str]] = defaultdict(list)
        self.token_frequencies: Counter = Counter()
        self.indexed_count: int = 0

    def fit(self, records: Sequence[Dict[str, Any]]) -> "NameInvertedIndex":
        """Build the inverted indexes from records."""
        # 1. Count token frequencies across the target corpus
        for rec in records:
            name = rec.get("business_name")
            tokens = extract_core_name_tokens(name)
            for t in set(tokens):
                self.token_frequencies[t] += 1
            self.indexed_count += 1

        # 2. Populate inverted indexes with high-frequency protection
        for rec in records:
            eid = rec["entity_id"]
            name = rec.get("business_name")
            tokens = extract_core_name_tokens(name)
            if not tokens:
                continue

            # Key 1: Exact core name string
            exact_key = " ".join(tokens)
            if len(self.exact_name_idx[exact_key]) < self.max_bucket_size:
                self.exact_name_idx[exact_key].append(eid)

            # Key 2: Concatenated tokens (matches URLs like qualitychs.com to Quality Chs)
            concat_key = "".join(tokens[:4])
            if len(concat_key) >= 5 and len(self.concat_name_idx[concat_key]) < self.max_bucket_size:
                self.concat_name_idx[concat_key].append(eid)

            # Key 3: Bag of words / Sorted tokens (handles word order permutations)
            if len(tokens) > 1:
                sorted_key = " ".join(sorted(tokens))
                if len(self.sorted_tokens_idx[sorted_key]) < self.max_bucket_size:
                    self.sorted_tokens_idx[sorted_key].append(eid)

                # Key 4: Token pairs for multi-word names
                unique_sorted = sorted(set(tokens))
                for i in range(len(unique_sorted)):
                    for j in range(i + 1, min(i + 4, len(unique_sorted))):
                        pair = (unique_sorted[i], unique_sorted[j])
                        if len(self.token_pair_idx[pair]) < self.max_bucket_size:
                            self.token_pair_idx[pair].append(eid)

            # Key 5: Rare / distinctive single tokens (length >= 3)
            for t in set(tokens):
                freq = self.token_frequencies[t]
                if (len(t) >= 3 and freq <= self.max_token_freq) or (
                    len(t) >= 6 and freq <= self.max_token_freq * 2
                ):
                    if len(self.rare_token_idx[t]) < self.max_bucket_size:
                        self.rare_token_idx[t].append(eid)

        return self

    def query(self, raw_name: Optional[str]) -> Set[str]:
        """Query the index for candidate entity IDs matching a business name."""
        tokens = extract_core_name_tokens(raw_name)
        if not tokens:
            return set()

        candidates: Set[str] = set()

        # Pass 1: Exact core name
        exact_key = " ".join(tokens)
        candidates.update(self.exact_name_idx.get(exact_key, ()))

        # Pass 2: Concatenated tokens
        concat_key = "".join(tokens[:4])
        if len(concat_key) >= 5:
            candidates.update(self.concat_name_idx.get(concat_key, ()))

        # Pass 3: Sorted tokens
        if len(tokens) > 1:
            sorted_key = " ".join(sorted(tokens))
            candidates.update(self.sorted_tokens_idx.get(sorted_key, ()))

            # Pass 4: Token pairs
            unique_sorted = sorted(set(tokens))
            for i in range(len(unique_sorted)):
                for j in range(i + 1, min(i + 4, len(unique_sorted))):
                    pair = (unique_sorted[i], unique_sorted[j])
                    bucket = self.token_pair_idx.get(pair)
                    if bucket and len(bucket) <= self.max_bucket_size:
                        candidates.update(bucket)

        # Pass 5: Rare tokens
        for t in set(tokens):
            bucket = self.rare_token_idx.get(t)
            if bucket and len(bucket) <= self.max_bucket_size:
                candidates.update(bucket)

        return candidates


def build_name_index(
    records: Sequence[Dict[str, Any]], max_token_freq: int = 150
) -> NameInvertedIndex:
    """Build and return a NameInvertedIndex from records."""
    index = NameInvertedIndex(max_token_freq=max_token_freq)
    index.fit(records)
    return index


def generate_name_candidates(
    s1_records: Sequence[Dict[str, Any]], name_index: NameInvertedIndex
) -> Dict[str, Set[str]]:
    """Generate candidate entity IDs for Source 1 records using the name index."""
    candidates_by_s1: Dict[str, Set[str]] = {}
    for rec in s1_records:
        s1_id = rec["entity_id"]
        candidates_by_s1[s1_id] = name_index.query(rec.get("business_name"))
    return candidates_by_s1
