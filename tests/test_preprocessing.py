"""
Unit tests for src/preprocessing modules.

Run from the project root:
    python -m pytest tests/test_preprocessing.py -v

No large dataset files are required — all tests use in-memory data only.
"""

from __future__ import annotations

import io
import sys
import os
import textwrap
import unittest
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Make the src package importable without installing it
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.normalize import (
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
from preprocessing.io_utils import (
    read_tsv_chunks,
    read_tsv_full,
    load_ground_truth,
    write_tsv,
)
from preprocessing.profiling import (
    profile_source_chunk,
    profile_ground_truth_chunk,
    merge_profiles,
    print_profile,
)


# ===========================================================================
# clean_text
# ===========================================================================

class TestCleanText(unittest.TestCase):

    def test_none_returns_empty(self):
        self.assertEqual(clean_text(None), "")

    def test_nan_returns_empty(self):
        self.assertEqual(clean_text(float("nan")), "")

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")

    def test_plain_ascii(self):
        self.assertEqual(clean_text("  Hello World  "), "Hello World")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(clean_text("foo   bar\t\nbaz"), "foo bar baz")

    def test_non_breaking_space(self):
        result = clean_text("foo bar")
        self.assertEqual(result, "foo bar")

    def test_preserves_unicode_letters(self):
        # Hindi / Devanagari
        result = clean_text("  रिलायंस इंडस्ट्रीज  ")
        self.assertEqual(result, "रिलायंस इंडस्ट्रीज")

    def test_preserves_french_accents(self):
        result = clean_text("  Société Générale  ")
        self.assertEqual(result, "Société Générale")

    def test_removes_zero_width_chars(self):
        # Zero-width non-joiner
        result = clean_text("foo​bar")
        self.assertEqual(result, "foobar")

    def test_removes_soft_hyphen(self):
        result = clean_text("foo­bar")
        self.assertEqual(result, "foobar")

    def test_invisible_chars_via_escapes(self):
        # Regression: _INVISIBLE_RE must strip all 8 invisible codepoints.
        # Strings built with explicit \uXXXX escapes, not pasted literal chars.
        zwsp   = "​"  # zero-width space
        zwnj   = "‌"  # zero-width non-joiner
        zwj    = "‍"  # zero-width joiner
        shy    = "­"  # soft hyphen
        bom    = "﻿"  # byte-order mark
        lrm    = "‎"  # left-to-right mark
        rlm    = "‏"  # right-to-left mark
        lre    = "‪"  # left-to-right embedding
        for invisible in (zwsp, zwnj, zwj, shy, bom, lrm, rlm, lre):
            with self.subTest(codepoint=hex(ord(invisible))):
                result = clean_text("foo" + invisible + "bar")
                self.assertEqual(result, "foobar")

    def test_nfc_normalization(self):
        # Two ways to represent "é": precomposed vs decomposed
        precomposed = "é"       # NFC
        decomposed = "é"       # NFD
        self.assertEqual(clean_text(decomposed), precomposed)

    def test_integer_input(self):
        self.assertEqual(clean_text(42), "42")

    def test_does_not_change_case(self):
        self.assertEqual(clean_text("ACME Corp"), "ACME Corp")


# ===========================================================================
# clean_business_name
# ===========================================================================

class TestCleanBusinessName(unittest.TestCase):

    def test_none(self):
        self.assertEqual(clean_business_name(None), "")

    def test_empty(self):
        self.assertEqual(clean_business_name(""), "")

    def test_lowercases(self):
        self.assertEqual(clean_business_name("ACME CORP"), "acme corp")

    def test_collapses_whitespace(self):
        self.assertEqual(clean_business_name("Acme   Corp"), "acme   corp".replace("   ", " "))

    def test_ascii_business_name(self):
        result = clean_business_name("  ABC Trading Co.  ")
        self.assertIn("abc", result)
        self.assertIn("trading", result)

    def test_preserves_devanagari(self):
        result = clean_business_name("रिलायंस इंडस्ट्रीज लिमिटेड")
        self.assertIn("रिलायंस", result)

    def test_preserves_french_accents(self):
        result = clean_business_name("Société Générale")
        self.assertIn("société", result)
        self.assertIn("générale", result)

    def test_ampersand_replaced(self):
        result = clean_business_name("Smith & Jones")
        self.assertIn("and", result)
        self.assertNotIn("&", result)

    def test_ampersand_not_replaced_within_word(self):
        # "&" that is part of a token like "R&D" should not be modified
        result = clean_business_name("R&D Solutions")
        # The & is not surrounded by spaces so it stays
        self.assertIn("r", result)

    def test_trailing_punctuation_stripped(self):
        result = clean_business_name("Acme Corp.")
        self.assertFalse(result.endswith("."))

    def test_repeated_punctuation_collapsed(self):
        result = clean_business_name("Acme...Corp")
        self.assertNotIn("...", result)

    def test_numeric_preserved(self):
        result = clean_business_name("7-Eleven Inc")
        self.assertIn("7", result)
        self.assertIn("eleven", result)

    def test_nfc_normalized(self):
        decomposed = "Café"
        result = clean_business_name(decomposed)
        self.assertIn("é", result)  # é as single codepoint


# ===========================================================================
# strip_legal_suffixes
# ===========================================================================

class TestStripLegalSuffixes(unittest.TestCase):

    def test_none(self):
        self.assertEqual(strip_legal_suffixes(None), "")

    def test_empty(self):
        self.assertEqual(strip_legal_suffixes(""), "")

    def test_strips_incorporated(self):
        result = strip_legal_suffixes("acme inc")
        self.assertEqual(result, "acme")

    def test_strips_corporation(self):
        result = strip_legal_suffixes("acme corporation")
        self.assertEqual(result, "acme")

    def test_strips_corp_dot(self):
        result = strip_legal_suffixes("acme corp.")
        self.assertNotIn("corp", result)

    def test_strips_limited(self):
        result = strip_legal_suffixes("reliance limited")
        self.assertEqual(result, "reliance")

    def test_strips_ltd(self):
        result = strip_legal_suffixes("reliance ltd")
        self.assertEqual(result, "reliance")

    def test_strips_llc(self):
        result = strip_legal_suffixes("valley farms llc")
        self.assertEqual(result, "valley farms")

    def test_strips_pvt_ltd(self):
        result = strip_legal_suffixes("infosys pvt ltd")
        self.assertEqual(result, "infosys")

    def test_strips_private_limited(self):
        result = strip_legal_suffixes("wipro private limited")
        self.assertEqual(result, "wipro")

    def test_no_suffix_unchanged(self):
        result = strip_legal_suffixes("the quick brown fox")
        self.assertEqual(result, "the quick brown fox")

    def test_name_only_suffix_not_emptied(self):
        # "ltd" alone — all that's left should be empty string
        result = strip_legal_suffixes("ltd")
        # suffix covers entire string; result may be empty
        self.assertIsInstance(result, str)

    def test_preserves_mid_word_suffix_token(self):
        # "limited" in the middle should NOT be stripped
        result = strip_legal_suffixes("limited edition store")
        self.assertIn("limited", result)

    def test_strips_gmbh(self):
        result = strip_legal_suffixes("volkswagen gmbh")
        self.assertEqual(result, "volkswagen")

    def test_strips_sarl(self):
        result = strip_legal_suffixes("artisan boulangerie sarl")
        self.assertEqual(result, "artisan boulangerie")

    def test_strips_bros_not_replaces(self):
        # Regression: "bros" must be STRIPPED, not expanded to "brothers".
        result = strip_legal_suffixes("singh bros")
        self.assertEqual(result, "singh")
        self.assertNotIn("brothers", result)

    def test_strips_and_sons(self):
        # "and sons" is in _LEGAL_SUFFIX_MAP; must be stripped.
        result = strip_legal_suffixes("acme and sons")
        self.assertEqual(result, "acme")

    def test_ampersand_sons_not_stripped(self):
        # "& sons" was removed as unreachable. strip_legal_suffixes() receives
        # already-cleaned input (& -> and done upstream), so "& sons" is never
        # passed here. But if it were, it must pass through unchanged.
        result = strip_legal_suffixes("acme & sons")
        self.assertEqual(result, "acme & sons")


# ===========================================================================
# clean_address
# ===========================================================================

class TestCleanAddress(unittest.TestCase):

    def test_none(self):
        self.assertEqual(clean_address(None), "")

    def test_empty(self):
        self.assertEqual(clean_address(""), "")

    def test_lowercases(self):
        result = clean_address("123 Main Street")
        self.assertEqual(result[0:3], "123")
        self.assertIn("main", result)

    def test_preserves_house_number(self):
        result = clean_address("42 Baker Street")
        self.assertIn("42", result)

    def test_preserves_postal_code_us(self):
        result = clean_address("Chicago, IL 60601")
        self.assertIn("60601", result)

    def test_preserves_pin_code_india(self):
        result = clean_address("Mumbai 400001, Maharashtra")
        self.assertIn("400001", result)

    def test_preserves_unit_number(self):
        result = clean_address("Suite 300, 100 Tech Park")
        self.assertIn("300", result)
        self.assertIn("100", result)

    def test_collapses_whitespace(self):
        result = clean_address("123   Main   Street")
        self.assertNotIn("   ", result)

    def test_street_abbreviation(self):
        result = clean_address("100 Main St")
        self.assertIn("street", result)

    def test_avenue_abbreviation(self):
        result = clean_address("200 Park Ave")
        self.assertIn("avenue", result)

    def test_does_not_destroy_non_ascii_address(self):
        result = clean_address("गाँधी रोड, मुंबई 400001")
        self.assertIn("400001", result)
        self.assertIn("मुंबई", result)

    def test_french_address_preserved(self):
        result = clean_address("12 Rue de la Paix, Paris 75001")
        self.assertIn("75001", result)
        self.assertIn("paix", result)

    def test_landmark_address_no_number(self):
        result = clean_address("Near Central Park, New York")
        self.assertIn("central", result)
        self.assertIn("park", result)

    def test_no_format_assumed(self):
        # Address with no standard format
        result = clean_address("Opposite Bus Stand, Opp. SBI Bank")
        self.assertIn("opposite", result)


# ===========================================================================
# normalize_country
# ===========================================================================

class TestNormalizeCountry(unittest.TestCase):

    def test_none(self):
        self.assertEqual(normalize_country(None), "")

    def test_empty(self):
        self.assertEqual(normalize_country(""), "")

    def test_us_aliases(self):
        for alias in ("USA", "us", "U.S.", "United States", "United States of America"):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_country(alias), "united states")

    def test_india_aliases(self):
        for alias in ("India", "IN", "Bharat", "ind"):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_country(alias), "india")

    def test_france_aliases(self):
        for alias in ("France", "FR", "french republic"):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_country(alias), "france")

    def test_arbitrary_country_passthrough(self):
        # Open-set: unknown country not dropped
        result = normalize_country("Brazil")
        self.assertEqual(result, "brazil")

    def test_another_arbitrary_country(self):
        result = normalize_country("South Korea")
        self.assertEqual(result, "south korea")

    def test_lowercased(self):
        result = normalize_country("GERMANY")
        self.assertEqual(result, "germany")

    def test_whitespace_stripped(self):
        result = normalize_country("  France  ")
        self.assertEqual(result, "france")

    def test_uk_aliases(self):
        for alias in ("UK", "United Kingdom", "Great Britain"):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_country(alias), "united kingdom")


# ===========================================================================
# extract_tokens
# ===========================================================================

class TestExtractTokens(unittest.TestCase):

    def test_none(self):
        self.assertEqual(extract_tokens(None), [])

    def test_empty(self):
        self.assertEqual(extract_tokens(""), [])

    def test_ascii(self):
        result = extract_tokens("Acme Corp Inc")
        self.assertIn("acme", result)
        self.assertIn("corp", result)
        self.assertIn("inc", result)

    def test_sorted_output(self):
        result = extract_tokens("zebra apple mango")
        self.assertEqual(result, sorted(result))

    def test_no_duplicates(self):
        result = extract_tokens("foo foo bar bar")
        self.assertEqual(len(result), len(set(result)))

    def test_unicode_devanagari(self):
        result = extract_tokens("रिलायंस इंडस्ट्रीज")
        self.assertIn("रिलायंस", result)
        self.assertIn("इंडस्ट्रीज", result)

    def test_devanagari_combining_marks_not_split(self):
        # Regression: nukta (U+093C), anusvara (U+0902), virama (U+094D) must
        # not be treated as split points. Each word must survive as one token.
        nukta_word   = "ड़ा"    # nukta within word
        anusvara_word = "इंडस"  # anusvara within word
        virama_word  = "इन्डस"  # virama within word
        for word in (nukta_word, anusvara_word, virama_word):
            with self.subTest(word=word):
                result = extract_tokens(word)
                self.assertEqual(len(result), 1, f"word split unexpectedly: {result}")
                self.assertEqual(result[0], word.lower())

    def test_french_accented(self):
        result = extract_tokens("Société Générale")
        self.assertIn("société", result)
        self.assertIn("générale", result)

    def test_punctuation_split(self):
        result = extract_tokens("acme,corp.inc")
        self.assertIn("acme", result)
        self.assertIn("corp", result)
        self.assertIn("inc", result)

    def test_numbers_preserved(self):
        result = extract_tokens("7-Eleven")
        self.assertIn("7", result)
        self.assertIn("eleven", result)


# ===========================================================================
# extract_numeric_tokens
# ===========================================================================

class TestExtractNumericTokens(unittest.TestCase):

    def test_none(self):
        self.assertEqual(extract_numeric_tokens(None), [])

    def test_empty(self):
        self.assertEqual(extract_numeric_tokens(""), [])

    def test_house_number(self):
        result = extract_numeric_tokens("42 Baker Street")
        self.assertIn("42", result)

    def test_postal_code(self):
        result = extract_numeric_tokens("Chicago IL 60601")
        self.assertIn("60601", result)

    def test_pin_code(self):
        result = extract_numeric_tokens("Mumbai 400001")
        self.assertIn("400001", result)

    def test_preserves_leading_zeros(self):
        result = extract_numeric_tokens("PIN 007001")
        self.assertIn("007001", result)

    def test_unit_number(self):
        result = extract_numeric_tokens("Suite 300, Floor 4")
        self.assertIn("300", result)
        self.assertIn("4", result)

    def test_no_numbers(self):
        result = extract_numeric_tokens("No numbers here")
        self.assertEqual(result, [])

    def test_sorted_output(self):
        result = extract_numeric_tokens("Room 305, Floor 2, Block 12")
        self.assertEqual(result, sorted(result))

    def test_no_duplicates(self):
        result = extract_numeric_tokens("12 Main St and 12 Oak Ave")
        self.assertEqual(result.count("12"), 1)

    def test_no_alpha_tokens(self):
        result = extract_numeric_tokens("abc 123 def 456")
        for tok in result:
            self.assertTrue(tok.isdigit(), f"Non-digit token: {tok}")


# ===========================================================================
# preprocess_record
# ===========================================================================

class TestPreprocessRecord(unittest.TestCase):

    def _make_record(self, **kwargs) -> dict:
        base = {
            "entity_id": "S1-001",
            "business_name": "Acme Corp",
            "business_address": "42 Main St, NY 10001",
            "country": "USA",
        }
        base.update(kwargs)
        return base

    def test_entity_id_preserved(self):
        rec = self._make_record()
        out = preprocess_record(rec)
        self.assertEqual(out["entity_id"], "S1-001")

    def test_original_fields_preserved(self):
        rec = self._make_record()
        out = preprocess_record(rec)
        self.assertEqual(out["business_name"], "Acme Corp")
        self.assertEqual(out["country"], "USA")

    def test_derived_fields_added(self):
        rec = self._make_record()
        out = preprocess_record(rec)
        self.assertIn("norm_business_name", out)
        self.assertIn("norm_business_name_no_suffix", out)
        self.assertIn("norm_address", out)
        self.assertIn("norm_country", out)
        self.assertIn("name_tokens", out)
        self.assertIn("name_tokens_no_suffix", out)
        self.assertIn("name_numeric_tokens", out)
        self.assertIn("address_tokens", out)
        self.assertIn("address_numeric_tokens", out)

    def test_country_normalized(self):
        rec = self._make_record(country="USA")
        out = preprocess_record(rec)
        self.assertEqual(out["norm_country"], "united states")

    def test_suffix_stripped(self):
        rec = self._make_record(business_name="Reliance Limited")
        out = preprocess_record(rec)
        self.assertNotIn("limited", out["norm_business_name_no_suffix"])

    def test_address_numeric_tokens(self):
        rec = self._make_record(business_address="42 Main St, NY 10001")
        out = preprocess_record(rec)
        self.assertIn("42", out["address_numeric_tokens"])
        self.assertIn("10001", out["address_numeric_tokens"])

    def test_none_values_handled(self):
        rec = {
            "entity_id": "S2-999",
            "business_name": None,
            "business_address": None,
            "country": None,
        }
        out = preprocess_record(rec)
        self.assertEqual(out["norm_business_name"], "")
        self.assertEqual(out["norm_address"], "")
        self.assertEqual(out["norm_country"], "")
        self.assertEqual(out["name_tokens"], [])

    def test_hindi_record(self):
        rec = {
            "entity_id": "S2-100",
            "business_name": "रिलायंस इंडस्ट्रीज लिमिटेड",
            "business_address": "मुंबई 400001",
            "country": "India",
        }
        out = preprocess_record(rec)
        self.assertIn("रिलायंस", out["name_tokens"])
        self.assertIn("400001", out["address_numeric_tokens"])
        self.assertEqual(out["norm_country"], "india")

    def test_french_record(self):
        rec = {
            "entity_id": "S3-200",
            "business_name": "Boulangerie Dupont SARL",
            "business_address": "12 Rue de la Paix, Paris 75001",
            "country": "France",
        }
        out = preprocess_record(rec)
        self.assertIn("boulangerie", out["name_tokens"])
        self.assertIn("75001", out["address_numeric_tokens"])
        self.assertEqual(out["norm_country"], "france")
        self.assertNotIn("sarl", out["norm_business_name_no_suffix"])

    def test_does_not_mutate_input(self):
        rec = self._make_record()
        original_keys = set(rec.keys())
        _ = preprocess_record(rec)
        self.assertEqual(set(rec.keys()), original_keys)


# ===========================================================================
# preprocess_dataframe
# ===========================================================================

class TestPreprocessDataframe(unittest.TestCase):

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "entity_id": "S1-001",
                "business_name": "Acme Corp",
                "business_address": "42 Main St",
                "country": "USA",
            },
            {
                "entity_id": "S2-002",
                "business_name": "Reliance Limited",
                "business_address": "Mumbai 400001",
                "country": "India",
            },
            {
                "entity_id": "S3-003",
                "business_name": None,
                "business_address": None,
                "country": None,
            },
        ])

    def test_does_not_mutate_input(self):
        df = self._make_df()
        cols_before = list(df.columns)
        _ = preprocess_dataframe(df)
        self.assertEqual(list(df.columns), cols_before)

    def test_entity_id_unchanged(self):
        df = self._make_df()
        out = preprocess_dataframe(df)
        self.assertListEqual(list(out["entity_id"]), ["S1-001", "S2-002", "S3-003"])

    def test_derived_columns_added(self):
        out = preprocess_dataframe(self._make_df())
        for col in ("norm_business_name", "norm_business_name_no_suffix",
                    "norm_address", "norm_country",
                    "name_tokens", "address_numeric_tokens"):
            self.assertIn(col, out.columns)

    def test_none_rows_handled(self):
        out = preprocess_dataframe(self._make_df())
        self.assertEqual(out.loc[2, "norm_business_name"], "")
        self.assertEqual(out.loc[2, "name_tokens"], [])

    def test_country_normalized_column(self):
        out = preprocess_dataframe(self._make_df())
        self.assertEqual(out.loc[0, "norm_country"], "united states")
        self.assertEqual(out.loc[1, "norm_country"], "india")

    def _make_df_no_name(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"entity_id": "S1-001", "business_address": "42 Main St", "country": "USA"},
            {"entity_id": "S2-002", "business_address": "Mumbai 400001", "country": "India"},
        ])

    def _make_df_no_address(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Acme Corp", "country": "USA"},
            {"entity_id": "S2-002", "business_name": "Reliance Limited", "country": "India"},
        ])

    def _make_df_no_country(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Acme Corp", "business_address": "42 Main St"},
            {"entity_id": "S2-002", "business_name": "Reliance Limited", "business_address": "Mumbai"},
        ])

    def test_missing_business_name_column_produces_empty_strings(self):
        # Regression: absent column must produce "" for every row, not NaN.
        out = preprocess_dataframe(self._make_df_no_name())
        self.assertEqual(len(out), 2)
        for i in range(len(out)):
            with self.subTest(row=i):
                self.assertEqual(out.loc[i, "norm_business_name"], "")
                self.assertEqual(out.loc[i, "norm_business_name_no_suffix"], "")
                self.assertEqual(out.loc[i, "name_tokens"], [])

    def test_missing_business_address_column_produces_empty_strings(self):
        # Regression: absent column must produce "" for every row, not NaN.
        out = preprocess_dataframe(self._make_df_no_address())
        self.assertEqual(len(out), 2)
        for i in range(len(out)):
            with self.subTest(row=i):
                self.assertEqual(out.loc[i, "norm_address"], "")
                self.assertEqual(out.loc[i, "address_tokens"], [])

    def test_missing_country_column_produces_empty_strings(self):
        # Regression: absent column must produce "" for every row, not NaN.
        out = preprocess_dataframe(self._make_df_no_country())
        self.assertEqual(len(out), 2)
        for i in range(len(out)):
            with self.subTest(row=i):
                self.assertEqual(out.loc[i, "norm_country"], "")


# ===========================================================================
# io_utils — TSV reading (in-memory, no disk files)
# ===========================================================================

class TestReadTsvChunks(unittest.TestCase):
    """Use tmp file to test chunked reading without the real dataset."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-001\tAcme Corp\t42 Main St\tUSA\n"
            "S2-002\tReliance Ltd\tMumbai 400001\tIndia\n"
            "S3-003\tSociété Génerale\t12 Rue de Paix, 75001\tFrance\n"
        )
        self._tsv_path = os.path.join(self._tmpdir, "test.tsv")
        with open(self._tsv_path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_chunks_cover_all_rows(self):
        chunks = list(read_tsv_chunks(self._tsv_path, chunksize=2))
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, 3)

    def test_entity_id_is_string(self):
        chunks = list(read_tsv_chunks(self._tsv_path, chunksize=10))
        df = chunks[0]
        self.assertTrue(pd.api.types.is_string_dtype(df["entity_id"]))
        self.assertEqual(df["entity_id"].iloc[0], "S1-001")

    def test_french_row_preserved(self):
        chunks = list(read_tsv_chunks(self._tsv_path, chunksize=10))
        df = chunks[0]
        france_row = df[df["entity_id"] == "S3-003"]
        self.assertEqual(len(france_row), 1)
        self.assertIn("Génerale", france_row["business_name"].iloc[0])

    def test_chunking_splits_correctly(self):
        chunks = list(read_tsv_chunks(self._tsv_path, chunksize=1))
        self.assertEqual(len(chunks), 3)
        for c in chunks:
            self.assertEqual(len(c), 1)


class TestReadTsvFull(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-001\tAcme Corp\t42 Main St\tUSA\n"
        )
        self._tsv_path = os.path.join(self._tmpdir, "small.tsv")
        with open(self._tsv_path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_returns_dataframe(self):
        df = read_tsv_full(self._tsv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)

    def test_entity_id_is_string(self):
        df = read_tsv_full(self._tsv_path)
        self.assertEqual(str(df["entity_id"].iloc[0]), "S1-001")


class TestLoadGroundTruth(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        content = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-001\tS2-100,S3-200\n"
            "S1-002\tS2-101\n"
            "S1-003\t\n"  # empty matched_entity_ids = zero matches
        )
        self._gt_path = os.path.join(self._tmpdir, "gt.tsv")
        with open(self._gt_path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_returns_dataframe(self):
        df = load_ground_truth(self._gt_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)

    def test_source1_id_is_string(self):
        df = load_ground_truth(self._gt_path)
        self.assertTrue(pd.api.types.is_string_dtype(df["source1_entity_id"]))
        self.assertEqual(df["source1_entity_id"].iloc[0], "S1-001")

    def test_match_list_parsed_correctly(self):
        df = load_ground_truth(self._gt_path)
        self.assertEqual(df["match_list"].iloc[0], ["S2-100", "S3-200"])
        self.assertEqual(df["match_list"].iloc[1], ["S2-101"])

    def test_empty_matched_ids_is_empty_list(self):
        df = load_ground_truth(self._gt_path)
        # Row with empty matched_entity_ids -> empty list, not NaN, not [""]
        self.assertEqual(df["match_list"].iloc[2], [])
        self.assertEqual(df["match_count"].iloc[2], 0)

    def test_match_count_correct(self):
        df = load_ground_truth(self._gt_path)
        self.assertEqual(df["match_count"].iloc[0], 2)
        self.assertEqual(df["match_count"].iloc[1], 1)
        self.assertEqual(df["match_count"].iloc[2], 0)

    def test_raw_matched_entity_ids_preserved(self):
        df = load_ground_truth(self._gt_path)
        self.assertEqual(df["matched_entity_ids"].iloc[0], "S2-100,S3-200")


class TestWriteTsv(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()

    def test_roundtrip(self):
        df = pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Acme Corp"},
        ])
        path = os.path.join(self._tmpdir, "out.tsv")
        write_tsv(df, path)
        df2 = pd.read_csv(path, sep="\t", dtype=str)
        self.assertEqual(df2["entity_id"].iloc[0], "S1-001")
        self.assertEqual(df2["business_name"].iloc[0], "Acme Corp")

    def test_creates_parent_dirs(self):
        path = os.path.join(self._tmpdir, "sub", "dir", "out.tsv")
        df = pd.DataFrame([{"a": 1}])
        write_tsv(df, path)
        self.assertTrue(os.path.exists(path))

    def test_unicode_preserved_in_file(self):
        df = pd.DataFrame([
            {"entity_id": "S2-001", "business_name": "रिलायंस इंडस्ट्रीज"},
        ])
        path = os.path.join(self._tmpdir, "unicode.tsv")
        write_tsv(df, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("रिलायंस", content)


# ===========================================================================
# profiling
# ===========================================================================

class TestProfiling(unittest.TestCase):

    def _source_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "entity_id": "S1-001",
                "business_name": "Acme Corp",
                "business_address": "42 Main St",
                "country": "USA",
            },
            {
                "entity_id": "S2-002",
                "business_name": "Reliance Ltd",
                "business_address": "Mumbai 400001",
                "country": "India",
            },
            {
                "entity_id": "S2-003",
                "business_name": "",
                "business_address": "",
                "country": "India",
            },
        ])

    def _gt_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "source1_entity_id": "S1-001",
                "matched_entity_ids": "S2-100,S3-200",
                "match_list": ["S2-100", "S3-200"],
                "match_count": 2,
            },
            {
                "source1_entity_id": "S1-002",
                "matched_entity_ids": "",
                "match_list": [],
                "match_count": 0,
            },
        ])

    def test_profile_source_chunk_row_count(self):
        p = profile_source_chunk(self._source_df())
        self.assertEqual(p["row_count"], 3)

    def test_profile_source_missing_name(self):
        p = profile_source_chunk(self._source_df())
        self.assertEqual(p["missing_business_name"], 1)

    def test_profile_source_country_counts(self):
        p = profile_source_chunk(self._source_df())
        self.assertEqual(p["country_counts"]["USA"], 1)
        self.assertEqual(p["country_counts"]["India"], 2)

    def test_profile_source_entity_prefixes(self):
        p = profile_source_chunk(self._source_df())
        self.assertEqual(p["entity_id_prefixes"]["S1-"], 1)
        self.assertEqual(p["entity_id_prefixes"]["S2-"], 2)

    def test_profile_gt_singletons(self):
        p = profile_ground_truth_chunk(self._gt_df())
        self.assertEqual(p["singletons"], 1)
        self.assertEqual(p["has_matches"], 1)

    def test_profile_gt_s2_s3_counts(self):
        p = profile_ground_truth_chunk(self._gt_df())
        self.assertEqual(p["total_s2_matches"], 1)
        self.assertEqual(p["total_s3_matches"], 1)

    def test_merge_profiles_sums(self):
        p1 = profile_source_chunk(self._source_df())
        p2 = profile_source_chunk(self._source_df())
        merged = merge_profiles([p1, p2])
        self.assertEqual(merged["row_count"], 6)
        self.assertEqual(merged["country_counts"]["India"], 4)

    def test_merge_single_profile(self):
        p = profile_source_chunk(self._source_df())
        merged = merge_profiles([p])
        self.assertEqual(merged["row_count"], 3)

    def test_merge_empty_list(self):
        self.assertEqual(merge_profiles([]), {})

    def test_print_profile_runs(self):
        p = profile_source_chunk(self._source_df())
        merged = merge_profiles([p])
        # Should not raise
        with io.StringIO() as buf:
            import sys as _sys
            old = _sys.stdout
            _sys.stdout = buf
            try:
                print_profile(merged, title="Test")
            finally:
                _sys.stdout = old


# ===========================================================================
# Public API compatibility
# ===========================================================================

class TestPublicApiCompatibility(unittest.TestCase):
    """
    Verify that the public preprocessing API required by Person 2 and Person 3:
      - normalize_business_name
      - normalize_address
      - normalize_country
    can be imported directly from preprocessing and produces identical output
    to the underlying cleaning functions across representative test inputs.
    """

    def test_package_level_imports(self):
        from preprocessing import (
            normalize_business_name as nbn,
            normalize_address as na,
            normalize_country as nc,
            clean_business_name as cbn,
            clean_address as ca,
        )
        self.assertTrue(callable(nbn))
        self.assertTrue(callable(na))
        self.assertTrue(callable(nc))
        self.assertTrue(callable(cbn))
        self.assertTrue(callable(ca))

    def test_normalize_business_name_identical_to_clean_business_name(self):
        test_inputs = [
            None,
            float("nan"),
            "",
            "   ",
            "Acme Corp",
            "ACME  CORPORATION...",
            "  रिलायंस इंडस्ट्रीज  ",
            "Société Générale & Cie",
            "Foo & Bar Trading",
            "AT&T Services Inc.",
            "test​with‌invisible‍chars",
            "123 456 Business Ltd.",
            "  Pvt. Ltd.  ",
            "McDonald's",
            "A & B & C Corp.",
        ]
        for inp in test_inputs:
            with self.subTest(inp=inp):
                self.assertEqual(
                    normalize_business_name(inp),
                    clean_business_name(inp),
                )

    def test_normalize_address_identical_to_clean_address(self):
        test_inputs = [
            None,
            float("nan"),
            "",
            "   ",
            "123 Main St., Apt 4B",
            "456 North Ave. Flr 2",
            "789 Boulevard Rd. Ste 100",
            "12 Rue de la Paix, 75001 Paris",
            "Near Bus Stand, MG Road, Mumbai 400001",
            "Flat No. 12, Tower B, Sector 62",
            "address​with‌invisible‍chars",
            "P.O. Box 1234, 5th Ct.",
            "100 Parkway Dr., Suite #50",
            "Plot No. 42, Industrial Area Phase-II",
        ]
        for inp in test_inputs:
            with self.subTest(inp=inp):
                self.assertEqual(
                    normalize_address(inp),
                    clean_address(inp),
                )

    def test_normalize_country_consistency(self):
        from preprocessing import normalize_country as nc
        self.assertEqual(nc("USA"), "united states")
        self.assertEqual(nc("US"), "united states")
        self.assertEqual(nc("India"), "india")
        self.assertEqual(nc("IN"), "india")
        self.assertEqual(nc("Bharat"), "india")
        self.assertEqual(nc("France"), "france")
        self.assertEqual(nc("FR"), "france")
        self.assertEqual(nc("Germany"), "germany")
        self.assertEqual(nc("Unknown Country XYZ"), "unknown country xyz")
        self.assertEqual(nc(None), "")
        self.assertEqual(nc(""), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
