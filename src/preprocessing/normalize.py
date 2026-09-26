"""
Text normalization for business entity resolution.

Design contract for Person 2 / Person 3 consumers:
- All functions are pure: no side effects, no global state.
- None / NaN / empty string are accepted everywhere; they return "" or [].
- Unicode is PRESERVED, not stripped to ASCII.
- Normalization form is NFC throughout (canonical composition).
- All transformations are deterministic (no randomness, no ordering dependence).
- Original entity_id values are never touched here.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_str(value: Any) -> str:
    """Convert any value to str, returning '' for None/NaN/non-string."""
    if value is None:
        return ""
    if isinstance(value, float) and (value != value):  # NaN check
        return ""
    return str(value)


def _nfc(text: str) -> str:
    """Apply NFC Unicode normalization."""
    return unicodedata.normalize("NFC", text)


# Collapse runs of whitespace (including non-breaking space U+00A0, U+2009, etc.)
_WHITESPACE_RE = re.compile(r"[\s  　]+")

# Repeated punctuation collapser — keeps one of: . , - / ( )
_MULTI_PUNCT_RE = re.compile(r"([.,\-/()])(?:\s*\1)+")

# Characters that are safe to remove without losing meaning:
# zero-width space, zero-width non-joiner, zero-width joiner, soft hyphen,
# BOM, left-to-right mark, right-to-left mark, directional embedding chars.
_INVISIBLE_RE = re.compile(
    r"[​‌‍­﻿‎‏‪-‮]"
)


# ---------------------------------------------------------------------------
# Legal / corporate suffix normalization
# ---------------------------------------------------------------------------

# Ordered from longest to shortest to prevent partial matches.
# Values are (canonical_form, set_of_variants).
# Variants are matched AFTER lowercasing and stripping trailing punctuation.
_LEGAL_SUFFIX_MAP: dict[str, list[str]] = {
    "limited liability company": [
        "limited liability company", "limited liability co",
        "l.l.c", "llc", "l l c",
    ],
    "limited liability partnership": [
        "limited liability partnership", "l.l.p", "llp", "l l p",
    ],
    "incorporated": [
        "incorporated", "inc", "inc.",
    ],
    "corporation": [
        "corporation", "corp", "corp.",
    ],
    "limited": [
        "limited", "ltd", "ltd.",
    ],
    "company": [
        "company", "co", "co.",
    ],
    "private limited": [
        "private limited", "pvt ltd", "pvt. ltd", "pvt. ltd.", "pvt ltd.",
        "private ltd", "p ltd", "p. ltd",
    ],
    "public limited company": [
        "public limited company", "plc", "p.l.c",
    ],
    "gesellschaft mit beschränkter haftung": [
        "gesellschaft mit beschränkter haftung", "gmbh", "g.m.b.h",
    ],
    "société à responsabilité limitée": [
        "société à responsabilité limitée", "sarl", "s.a.r.l",
    ],
    "société anonyme": [
        "société anonyme", "sa", "s.a",
    ],
    "enterprises": ["enterprises", "enterprise", "ent", "ents"],
    "brothers": ["brothers", "bros", "bros."],
    # "& sons" variant removed: clean_business_name() converts & -> and before
    # strip_legal_suffixes() is called, so "& sons" can never match at that point.
    "and sons": ["and sons"],
    "trading": ["trading", "trdg"],
    "industries": ["industries", "ind", "inds"],
    "services": ["services", "svc", "svcs"],
    "solutions": ["solutions", "soln", "solns"],
    "international": ["international", "intl", "int'l"],
    "associates": ["associates", "assoc", "assocs"],
    "group": ["group", "grp"],
}

# Build reverse lookup: variant -> canonical
_SUFFIX_VARIANTS: dict[str, str] = {}
for _canonical, _variants in _LEGAL_SUFFIX_MAP.items():
    for _v in _variants:
        _SUFFIX_VARIANTS[_v] = _canonical

# Longest variant first so greedy match works
_SUFFIX_VARIANTS_SORTED = sorted(_SUFFIX_VARIANTS.keys(), key=len, reverse=True)

# Trailing suffix pattern: optional comma/space then a known suffix at end of string
_TRAILING_PUNCT_RE = re.compile(r"[.,;:\s]+$")


# ---------------------------------------------------------------------------
# Address abbreviation expansion — compiled at module level for performance
# ---------------------------------------------------------------------------

# Each entry is (compiled_pattern, expansion_string).
# Compiled once at import time instead of per-call.
_ADDR_ABBREVS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bst\b\.?"),   "street"),
    (re.compile(r"\bave?\b\.?"), "avenue"),
    (re.compile(r"\bblvd\b\.?"), "boulevard"),
    (re.compile(r"\brd\b\.?"),   "road"),
    (re.compile(r"\bdr\b\.?"),   "drive"),
    (re.compile(r"\bln\b\.?"),   "lane"),
    (re.compile(r"\bct\b\.?"),   "court"),
    (re.compile(r"\bpl\b\.?"),   "place"),
    (re.compile(r"\bsq\b\.?"),   "square"),
    (re.compile(r"\bpky\b\.?"),  "parkway"),
    (re.compile(r"\bpkwy\b\.?"), "parkway"),
    (re.compile(r"\bflr\b\.?"),  "floor"),
    (re.compile(r"\bapt\b\.?"),  "apartment"),
    (re.compile(r"\bste\b\.?"),  "suite"),
    (re.compile(r"\bno\b\.?(?=\s*\d)", re.IGNORECASE), "number"),
]


# ---------------------------------------------------------------------------
# Country alias mapping — module-level dict, not rebuilt per call
# ---------------------------------------------------------------------------

# Canonical aliases (extend as new countries are observed in data).
# Open-set: unknown values fall through to clean_text().lower().
_COUNTRY_ALIASES: dict[str, str] = {
    # United States
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "united states": "united states",
    "america": "united states",
    # India
    "india": "india",
    "in": "india",
    "bharat": "india",
    "ind": "india",
    # France
    "france": "france",
    "fr": "france",
    "french republic": "france",
    # United Kingdom
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "united kingdom": "united kingdom",
    "great britain": "united kingdom",
    "england": "united kingdom",
    "britain": "united kingdom",
    # Germany
    "germany": "germany",
    "de": "germany",
    "deutschland": "germany",
    # Canada
    "canada": "canada",
    "ca": "canada",
    # Australia
    "australia": "australia",
    "au": "australia",
    # China
    "china": "china",
    "cn": "china",
    "prc": "china",
    # Japan
    "japan": "japan",
    "jp": "japan",
}


# ---------------------------------------------------------------------------
# Public normalization functions
# ---------------------------------------------------------------------------

def clean_text(text: Any) -> str:
    """
    Base text cleaner applied universally.

    Steps (in order):
    1. Coerce to str; return '' for None/NaN.
    2. Strip invisible control characters.
    3. Apply NFC normalization.
    4. Collapse whitespace (including exotic Unicode spaces).
    5. Strip leading/trailing whitespace.

    Unicode characters are preserved. Case is NOT changed here.
    """
    s = _to_str(text)
    if not s:
        return ""
    s = _INVISIBLE_RE.sub("", s)
    s = _nfc(s)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()


def clean_business_name(text: Any) -> str:
    """
    Normalize a business name for matching/tokenization.

    Steps:
    1. clean_text (Unicode preserve, NFC, whitespace collapse).
    2. Lowercase.
    3. Collapse repeated punctuation (e.g. "..." -> ".").
    4. Normalize common separators: '&' -> 'and' only at word boundaries.
    5. Strip trailing/leading punctuation that carries no meaning at boundaries.
    6. Re-strip whitespace.

    Does NOT transliterate, does NOT remove non-ASCII, does NOT strip
    accented letters. The output is suitable for token extraction and
    suffix normalization.
    """
    s = clean_text(text)
    if not s:
        return ""
    s = s.lower()
    # Collapse repeated identical punctuation
    s = _MULTI_PUNCT_RE.sub(r"\1", s)
    # & -> and (with spaces) — only standalone ampersands
    s = re.sub(r"(?<=\s)&(?=\s)|^&(?=\s)|(?<=\s)&$", "and", s)
    # Strip boundary punctuation that is not part of a token
    s = s.strip(".,;:!?\"'")
    return s.strip()


def normalize_business_name(text: Any) -> str:
    """
    Normalize a business name for matching/tokenization.

    Compatibility wrapper delegating to clean_business_name.
    """
    return clean_business_name(text)


def strip_legal_suffixes(text: Any) -> str:
    """
    Remove known legal/corporate suffixes from a CLEANED business name.

    Input should already be lowercased (clean_business_name output is fine).
    Returns the name with the trailing suffix removed; the suffix itself is
    lost (use both the original and stripped version for matching flexibility).

    Examples:
        "acme corporation" -> "acme"
        "acme inc"        -> "acme"
        "the quick brown fox" -> "the quick brown fox"  (no suffix, unchanged)

    Normalization: variants are mapped to their canonical form, then the
    canonical form is stripped from the end of the string.
    """
    s = _to_str(text).strip()
    if not s:
        return ""

    # Try to match and remove trailing suffix variants (longest first)
    for variant in _SUFFIX_VARIANTS_SORTED:
        # Build a pattern: optional separator then variant at string end
        pattern = r"(?:[\s,]+)" + re.escape(variant) + r"\s*$"
        if re.search(pattern, s):
            s = re.sub(pattern, "", s).strip()
            return s.strip(".,;: ")

    return s


def clean_address(text: Any) -> str:
    """
    Normalize an address field for matching/tokenization.

    Deliberately conservative: preserves numeric tokens (house numbers,
    postal codes, PIN codes, unit numbers) and non-ASCII characters.

    Steps:
    1. clean_text (Unicode, NFC, whitespace collapse).
    2. Lowercase.
    3. Normalize common address abbreviations to canonical long form.
    4. Collapse extra punctuation.
    5. Re-strip whitespace.

    Does NOT assume any country-specific address format.
    Does NOT remove digits.
    Does NOT reorder components.
    """
    s = clean_text(text)
    if not s:
        return ""
    s = s.lower()

    # Expand common English address abbreviations using pre-compiled patterns
    for pattern, expansion in _ADDR_ABBREVS:
        s = pattern.sub(expansion, s)

    # Collapse repeated punctuation
    s = _MULTI_PUNCT_RE.sub(r"\1", s)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()


def normalize_address(text: Any) -> str:
    """
    Normalize an address field for matching/tokenization.

    Compatibility wrapper delegating to clean_address.
    """
    return clean_address(text)


def normalize_country(text: Any) -> str:
    """
    Normalize a country field to a consistent canonical string.

    OPEN-SET: does not hardcode {US, India}. Any country value is accepted.
    Returns the cleaned/lowercased/NFC-normalized country name.
    Unknown / new countries pass through rather than being dropped or errored.

    Common aliases for well-known countries are mapped to a single canonical
    form (e.g. "USA", "United States of America" -> "united states").
    All other values are returned as clean_text().lower().
    """
    s = clean_text(text)
    if not s:
        return ""
    s_lower = s.lower()

    if s_lower in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[s_lower]

    # Unknown country: preserve NFC + lowercase (do not drop)
    return s_lower


def extract_tokens(text: Any) -> list[str]:
    """
    Tokenize cleaned text into a list of non-empty Unicode word tokens.

    Splits on whitespace and ASCII punctuation (not on Unicode combining marks,
    so Devanagari nukta/chandrabindu/virama are preserved within tokens).
    Returns a sorted list of lowercase NFC tokens, no duplicates.

    Input: raw or pre-cleaned string (either is fine).
    Output: sorted list of lowercase NFC tokens, no duplicates.
    """
    s = clean_text(text)
    if not s:
        return []
    s = s.lower()
    s = _nfc(s)
    # Split only on whitespace and ASCII punctuation/separators.
    # Using \W with re.UNICODE incorrectly splits Devanagari (nukta U+093C,
    # chandrabindu U+0902, virama U+094D are category Mn — marked as non-word
    # by some regex engines). Splitting on [\s\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]
    # is safer: only ASCII punctuation + whitespace as delimiters.
    raw_tokens = re.split(r"[\s\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]+", s)
    seen: set[str] = set()
    result: list[str] = []
    for tok in raw_tokens:
        tok = tok.strip()
        if tok and tok not in seen:
            seen.add(tok)
            result.append(tok)
    return sorted(result)


def extract_numeric_tokens(text: Any) -> list[str]:
    """
    Extract tokens that are purely numeric (digits only).

    Useful for: house numbers, postal codes, PIN codes, unit numbers,
    phone fragments, route numbers.

    Returns a sorted list of distinct digit-only strings.
    Preserves leading zeros (e.g. PIN "007001" stays "007001").
    """
    s = clean_text(text)
    if not s:
        return []
    # Find all contiguous digit runs
    tokens = re.findall(r"\d+", s)
    seen: set[str] = set()
    result: list[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            result.append(tok)
    return sorted(result)


# ---------------------------------------------------------------------------
# Record-level and DataFrame-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Preprocess a single source-file record (dict with entity_id, business_name,
    business_address, country keys).

    Returns a NEW dict with:
    - all original keys preserved unchanged (entity_id, business_name, etc.)
    - additional derived keys:
        norm_business_name      : clean_business_name result
        norm_business_name_no_suffix : strip_legal_suffixes applied after clean
        norm_address            : clean_address result
        norm_country            : normalize_country result
        name_tokens             : extract_tokens(norm_business_name)
        name_tokens_no_suffix   : extract_tokens(norm_business_name_no_suffix)
        name_numeric_tokens     : extract_numeric_tokens(norm_business_name)
        address_tokens          : extract_tokens(norm_address)
        address_numeric_tokens  : extract_numeric_tokens(norm_address)

    entity_id is untouched. Original fields are untouched.
    """
    out = dict(record)  # shallow copy to not mutate caller's dict

    raw_name = record.get("business_name", "")
    raw_addr = record.get("business_address", "")
    raw_country = record.get("country", "")

    norm_name = clean_business_name(raw_name)
    norm_name_ns = strip_legal_suffixes(norm_name)
    norm_addr = clean_address(raw_addr)
    norm_ctry = normalize_country(raw_country)

    out["norm_business_name"] = norm_name
    out["norm_business_name_no_suffix"] = norm_name_ns
    out["norm_address"] = norm_addr
    out["norm_country"] = norm_ctry
    out["name_tokens"] = extract_tokens(norm_name)
    out["name_tokens_no_suffix"] = extract_tokens(norm_name_ns)
    out["name_numeric_tokens"] = extract_numeric_tokens(norm_name)
    out["address_tokens"] = extract_tokens(norm_addr)
    out["address_numeric_tokens"] = extract_numeric_tokens(norm_addr)

    return out


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to a DataFrame chunk from a source file.

    Adds derived columns (norm_*, *_tokens) in-place on a COPY of df.
    Does not modify the input DataFrame.
    entity_id column is preserved as-is.

    Expects columns: entity_id, business_name, business_address, country.
    Missing columns produce empty strings for the corresponding derived columns.
    """
    df = df.copy()

    _name_col = (
        df["business_name"]
        if "business_name" in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )
    _addr_col = (
        df["business_address"]
        if "business_address" in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )
    _ctry_col = (
        df["country"]
        if "country" in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )

    df["norm_business_name"] = _name_col.apply(clean_business_name)
    df["norm_business_name_no_suffix"] = df["norm_business_name"].apply(
        strip_legal_suffixes
    )
    df["norm_address"] = _addr_col.apply(clean_address)
    df["norm_country"] = _ctry_col.apply(normalize_country)
    df["name_tokens"] = df["norm_business_name"].apply(extract_tokens)
    df["name_tokens_no_suffix"] = df["norm_business_name_no_suffix"].apply(
        extract_tokens
    )
    df["name_numeric_tokens"] = df["norm_business_name"].apply(extract_numeric_tokens)
    df["address_tokens"] = df["norm_address"].apply(extract_tokens)
    df["address_numeric_tokens"] = df["norm_address"].apply(extract_numeric_tokens)

    return df
