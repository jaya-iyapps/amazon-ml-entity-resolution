"""
Amazon ML Entity Resolution Challenge 2026 - Address and Numeric Blocking

This module implements inverted indexes on normalized address tokens,
street/postal numbers, and combined numeric-alphabetic signatures.
Includes safeguards against ubiquitous address tokens and building numbers.
"""

import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Attempt to reuse Person 1's preprocessing API when available
try:
    from src.preprocessing.normalize import (
        clean_address,
        extract_numeric_tokens,
        extract_tokens,
    )
except ImportError:
    def _nfkc(text: Optional[str]) -> str:
        return unicodedata.normalize("NFKC", text or "").strip().lower()

    def clean_address(address: Optional[str]) -> str:
        s = _nfkc(address)
        return re.sub(r"\s+", " ", s).strip()

    def extract_tokens(text: Optional[str], min_len: int = 1) -> List[str]:
        if not text:
            return []
        return [tok for tok in re.findall(r"\w+", text.lower()) if len(tok) >= min_len]

    def extract_numeric_tokens(text: Optional[str]) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b\d+\b", text)


ADDRESS_STOPWORDS = frozenset({
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr",
    "lane", "ln", "blvd", "boulevard", "court", "ct", "way", "highway", "hwy",
    "floor", "fl", "suite", "ste", "unit", "apt", "apartment", "building", "bldg",
    "room", "rm", "nagar", "colony", "marg", "chowk", "bazar", "bazaar",
    "district", "dist", "state", "city", "near", "opposite", "opp", "behind",
    "north", "south", "east", "west", "po", "box", "pobox", "pvt", "ltd",
})


def extract_normalized_numbers(raw_address: Optional[str]) -> List[str]:
    """Extract standalone numeric sequences, stripping leading zeros."""
    raw_nums = extract_numeric_tokens(raw_address)
    clean_nums: List[str] = []
    for n in raw_nums:
        # Strip leading zeros, but preserve '0' if the number itself is 0
        norm = n.lstrip("0") or "0"
        if norm not in clean_nums:
            clean_nums.append(norm)
    return clean_nums


def extract_distinctive_address_tokens(raw_address: Optional[str]) -> List[str]:
    """Extract non-numeric, non-stopword address tokens."""
    tokens = extract_tokens(raw_address, min_len=3)
    return [t for t in tokens if t not in ADDRESS_STOPWORDS and not t.isdigit()]


class AddressInvertedIndex:
    """Multi-key inverted index for addresses and numeric signals."""

    def __init__(self, max_bucket_size: int = 100):
        self.max_bucket_size = max_bucket_size
        self.num_word_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.num_name_prefix_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    def fit(self, records: Sequence[Dict[str, Any]]) -> "AddressInvertedIndex":
        """Build the inverted index from target records."""
        for rec in records:
            eid = rec["entity_id"]
            addr = rec.get("business_address")
            name = rec.get("business_name")

            nums = extract_normalized_numbers(addr)
            addr_tokens = extract_distinctive_address_tokens(addr)

            # Key 1: Primary number + distinctive address token
            if nums and addr_tokens:
                for n in nums[:2]:  # Check first two numbers (e.g. house number & zip code)
                    for a in addr_tokens:
                        if len(a) >= 4:
                            pair = (n, a)
                            if len(self.num_word_idx[pair]) < self.max_bucket_size:
                                self.num_word_idx[pair].append(eid)

            # Key 2: Primary number + business name first 3 characters
            if nums and name:
                primary_num = nums[0]
                name_clean = clean_address(name)
                name_tokens = [tok for tok in extract_tokens(name_clean) if len(tok) >= 3]
                if name_tokens:
                    first_3 = name_tokens[0][:3]
                    prefix_pair = (primary_num, first_3)
                    if len(self.num_name_prefix_idx[prefix_pair]) < self.max_bucket_size:
                        self.num_name_prefix_idx[prefix_pair].append(eid)

        return self

    def query(self, raw_address: Optional[str], raw_name: Optional[str] = None) -> Set[str]:
        """Query the index for candidate entity IDs matching address and numeric signals."""
        nums = extract_normalized_numbers(raw_address)
        addr_tokens = extract_distinctive_address_tokens(raw_address)

        candidates: Set[str] = set()

        if nums and addr_tokens:
            for n in nums[:2]:
                for a in addr_tokens:
                    if len(a) >= 4:
                        bucket = self.num_word_idx.get((n, a))
                        if bucket and len(bucket) <= self.max_bucket_size:
                            candidates.update(bucket)

        if nums and raw_name:
            primary_num = nums[0]
            name_clean = clean_address(raw_name)
            name_tokens = [tok for tok in extract_tokens(name_clean) if len(tok) >= 3]
            if name_tokens:
                first_3 = name_tokens[0][:3]
                bucket = self.num_name_prefix_idx.get((primary_num, first_3))
                if bucket and len(bucket) <= self.max_bucket_size:
                    candidates.update(bucket)

        return candidates


def build_address_index(
    records: Sequence[Dict[str, Any]], max_key_freq: int = 100
) -> AddressInvertedIndex:
    """Build and return an AddressInvertedIndex from records."""
    index = AddressInvertedIndex(max_bucket_size=max_key_freq)
    index.fit(records)
    return index


def generate_address_candidates(
    s1_records: Sequence[Dict[str, Any]], address_index: AddressInvertedIndex
) -> Dict[str, Set[str]]:
    """Generate candidate entity IDs for Source 1 records using the address index."""
    candidates_by_s1: Dict[str, Set[str]] = {}
    for rec in s1_records:
        s1_id = rec["entity_id"]
        candidates_by_s1[s1_id] = address_index.query(
            rec.get("business_address"), rec.get("business_name")
        )
    return candidates_by_s1
