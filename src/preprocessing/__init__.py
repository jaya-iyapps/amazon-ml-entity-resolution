"""
Preprocessing package for Amazon Entity Resolution.

Public API — safe to import from blocking and matching modules.
"""

from .normalize import (
    clean_text,
    clean_business_name,
    normalize_business_name,
    strip_legal_suffixes,
    clean_address,
    normalize_address,
    normalize_country,
    extract_tokens,
    extract_numeric_tokens,
    preprocess_record,
    preprocess_dataframe,
)
from .io_utils import (
    read_tsv_chunks,
    read_tsv_full,
    load_ground_truth,
    write_tsv,
)
from .profiling import (
    profile_source_chunk,
    profile_ground_truth_chunk,
    merge_profiles,
    print_profile,
)

__all__ = [
    # normalize
    "clean_text",
    "clean_business_name",
    "normalize_business_name",
    "strip_legal_suffixes",
    "clean_address",
    "normalize_address",
    "normalize_country",
    "extract_tokens",
    "extract_numeric_tokens",
    "preprocess_record",
    "preprocess_dataframe",
    # io_utils
    "read_tsv_chunks",
    "read_tsv_full",
    "load_ground_truth",
    "write_tsv",
    # profiling
    "profile_source_chunk",
    "profile_ground_truth_chunk",
    "merge_profiles",
    "print_profile",
]
