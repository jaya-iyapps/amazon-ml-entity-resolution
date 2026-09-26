"""Text and field normalization utilities for entity resolution.

Owned by: Person 1 (Data / Preprocessing)
"""

import re
import unicodedata
from typing import Optional

# Common business legal suffixes across US, India, France, and international entities
LEGAL_SUFFIXES = [
    r"\bpvt\.?\s*ltd\.?\b",
    r"\bprivate\s+limited\b",
    r"\bltd\.?\b",
    r"\blimited\b",
    r"\binc\.?\b",
    r"\bincorporated\b",
    r"\bcorp\.?\b",
    r"\bcorporation\b",
    r"\bllc\.?\b",
    r"\bllp\.?\b",
    r"\bco\.?\b",
    r"\bcompany\b",
    r"\bsa\b",
    r"\bsas\b",
    r"\bsarl\b",
    r"\bgmbh\b",
    r"\bplc\b",
]

_LEGAL_SUFFIX_REGEX = re.compile(
    r"|".join(LEGAL_SUFFIXES),
    flags=re.IGNORECASE,
)

# Common address abbreviations
ADDRESS_ABBREVIATIONS = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bct\.?\b": "court",
    r"\bpkwy\.?\b": "parkway",
    r"\bapt\.?\b": "apartment",
    r"\bste\.?\b": "suite",
    r"\bfl\.?\b": "floor",
    r"\bbldg\.?\b": "building",
    r"\bhwy\.?\b": "highway",
}

_ADDRESS_REGEXES = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items()
]


def normalize_text(text: Optional[str]) -> str:
    """Perform baseline text normalization.

    - Handles None/empty input
    - Decomposes unicode characters to NFKD representation
    - Converts to lowercase
    - Normalizes internal whitespace and strips ends

    Args:
        text: Input string or None.

    Returns:
        Cleaned, normalized string (empty string if input is None or blank).
    """
    if text is None:
        return ""
    text_str = str(text).strip()
    if not text_str or text_str.lower() in ("nan", "none", "null"):
        return ""

    # Normalize unicode (keeps Latin/Devanagari scripts valid while decomposing accents)
    normalized = unicodedata.normalize("NFKC", text_str)
    # Lowercase
    lowered = normalized.lower()
    # Normalize multiple whitespace characters to single space
    cleaned = re.sub(r"\s+", " ", lowered).strip()
    return cleaned


def normalize_business_name(name: Optional[str], strip_suffixes: bool = True) -> str:
    """Normalize a business name for candidate matching.

    - Cleans whitespace and unicode
    - Removes punctuation (except alphanumerics across scripts)
    - Strips common corporate/legal suffixes (Pvt Ltd, Inc, Corp, etc.) if requested

    Args:
        name: Raw business name string.
        strip_suffixes: Whether to strip common legal entity suffixes.

    Returns:
        Normalized business name string.
    """
    clean_name = normalize_text(name)
    if not clean_name:
        return ""

    # Replace ampersands with 'and'
    clean_name = re.sub(r"&", " and ", clean_name)

    # Optionally remove common legal entity suffixes
    if strip_suffixes:
        clean_name = _LEGAL_SUFFIX_REGEX.sub(" ", clean_name)

    # Remove non-alphanumeric characters (preserving unicode letters/digits for multilingual names)
    clean_name = re.sub(r"[^\w\s]", " ", clean_name, flags=re.UNICODE)

    # Collapse whitespace
    clean_name = re.sub(r"\s+", " ", clean_name).strip()
    return clean_name


def normalize_address(address: Optional[str]) -> str:
    """Normalize a business address string.

    - Cleans punctuation and whitespace
    - Expands common abbreviations (st -> street, rd -> road, etc.)
    - Retains postal codes and city names

    Args:
        address: Raw address string.

    Returns:
        Normalized address string.
    """
    clean_addr = normalize_text(address)
    if not clean_addr:
        return ""

    # Replace punctuation with spaces (retaining hyphens in postal codes if relevant)
    clean_addr = re.sub(r"[,/\\#\(\)\[\]]", " ", clean_addr)

    # Standardize abbreviations
    for pattern, replacement in _ADDRESS_REGEXES:
        clean_addr = pattern.sub(replacement, clean_addr)

    # Remove non-word characters except hyphens and spaces
    clean_addr = re.sub(r"[^\w\s\-]", " ", clean_addr, flags=re.UNICODE)

    # Collapse whitespace
    clean_addr = re.sub(r"\s+", " ", clean_addr).strip()
    return clean_addr


def normalize_country(country: Optional[str]) -> str:
    """Normalize country identifier strings.

    Note: Training covers US and India; Test additionally contains France.
    Treat country as open set (do not hard-code to only US / India).

    Args:
        country: Raw country string.

    Returns:
        Uppercase standardized country code/name string.
    """
    clean_c = normalize_text(country).upper()
    if not clean_c:
        return "UNKNOWN"

    country_map = {
        "USA": "US",
        "UNITED STATES": "US",
        "UNITED STATES OF AMERICA": "US",
        "IND": "INDIA",
        "IN": "INDIA",
        "FR": "FRANCE",
        "FRA": "FRANCE",
    }
    return country_map.get(clean_c, clean_c)
