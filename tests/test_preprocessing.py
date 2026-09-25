"""Tests for preprocessing, normalization, and profiling modules (Person 1)."""

import pytest
from src.preprocessing.normalize import (
    normalize_text,
    normalize_business_name,
    normalize_address,
    normalize_country,
)
from src.preprocessing.profiling import profile_dataframe


def test_normalize_text_basic():
    """Verify normalize_text produces trimmed lowercase strings."""
    result = normalize_text("  Acme   Corporation \n")
    assert isinstance(result, str)
    assert result == "acme corporation"


def test_normalize_text_missing_values():
    """Verify graceful handling of null/missing/empty values."""
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
    assert normalize_text("NaN") == ""
    assert normalize_text("null") == ""


def test_normalize_business_name_suffixes():
    """Verify legal entity suffix removal."""
    assert normalize_business_name("Amazon Services LLC") == "amazon services"
    assert normalize_business_name("Reliance Industries Pvt. Ltd.") == "reliance industries"
    assert normalize_business_name("Acme Corp") == "acme"
    assert normalize_business_name("TotalEnergies SE") == "totalenergies se"


def test_normalize_business_name_special_characters():
    """Verify ampersand expansion and punctuation removal."""
    result = normalize_business_name("Barnes & Noble, Inc.")
    assert "and" in result
    assert "," not in result
    assert "." not in result


def test_normalize_address_abbreviations():
    """Verify expansion of standard street abbreviations."""
    addr = "123 Main St., Suite 4, High Rd."
    normalized = normalize_address(addr)
    assert "street" in normalized
    assert "road" in normalized


def test_normalize_country():
    """Verify country normalization handles open set including France."""
    assert normalize_country("US") == "US"
    assert normalize_country("usa") == "US"
    assert normalize_country("United States") == "US"
    assert normalize_country("India") == "INDIA"
    assert normalize_country("IND") == "INDIA"
    assert normalize_country("France") == "FRANCE"
    assert normalize_country("FR") == "FRANCE"
    assert normalize_country("Germany") == "GERMANY"  # Open-set preserved
    assert normalize_country(None) == "UNKNOWN"


def test_profile_dataframe_dict():
    """Verify summary metrics profiling on sample records."""
    toy_records = [
        {"entity_id": "S1-1", "business_name": "Acme Inc", "country": "US"},
        {"entity_id": "S1-2", "business_name": "", "country": "India"},
    ]
    summary = profile_dataframe(toy_records)
    assert summary["num_rows"] == 2
    assert summary["missing_counts"]["business_name"] == 1
    assert summary["missing_counts"]["entity_id"] == 0
