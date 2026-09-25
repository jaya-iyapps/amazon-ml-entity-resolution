"""Tests for matching, similarity features, and F0.5 evaluation (Person 2)."""

import math
import os
import tempfile
import pytest

from src.matching.features import compute_pair_features
from src.matching.threshold import (
    calculate_f_beta,
    score_single_entity_f05,
    calculate_macro_f05,
    select_threshold,
)
from src.matching.predict import format_matching_results, save_matching_results


def test_pairwise_feature_similarities():
    """Verify similarity values for identical vs distinct records."""
    rec1 = {
        "business_name": "alpha tech solutions",
        "business_address": "100 silicon way san jose ca",
        "country": "US",
    }
    # Identical record
    rec_same = {
        "business_name": "alpha tech solutions",
        "business_address": "100 silicon way san jose ca",
        "country": "US",
    }
    # Distinct record
    rec_diff = {
        "business_name": "zebra fashion store",
        "business_address": "999 broadway new york ny",
        "country": "INDIA",
    }

    feat_same = compute_pair_features(rec1, rec_same)
    feat_diff = compute_pair_features(rec1, rec_diff)

    # Identical should have max similarity
    assert feat_same["name_ratio"] == 1.0
    assert feat_same["name_exact_match"] == 1.0
    assert feat_same["country_match"] == 1.0

    # Distinct should have lower similarity
    assert feat_diff["name_ratio"] < feat_same["name_ratio"]
    assert feat_diff["name_exact_match"] == 0.0
    assert feat_diff["country_match"] == 0.0


def test_f05_formula_precision_weighting():
    """Verify that F_0.5 formula places 2x weight on precision."""
    # When Precision=1.0 and Recall=0.5:
    # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    f05_high_prec = calculate_f_beta(precision=1.0, recall=0.5, beta=0.5)
    assert math.isclose(f05_high_prec, 5.0 / 6.0, rel_tol=1e-3)

    # When Precision=0.5 and Recall=1.0:
    # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555
    f05_low_prec = calculate_f_beta(precision=0.5, recall=1.0, beta=0.5)
    assert math.isclose(f05_low_prec, 5.0 / 9.0, rel_tol=1e-3)

    # High precision must yield a higher F_0.5 score than high recall
    assert f05_high_prec > f05_low_prec


def test_single_entity_f05_singletons():
    """Verify singleton credit (1.0 for empty match) and penalty (0.0 for false merge)."""
    # Case A: True singleton, predicted empty -> 1.0
    assert score_single_entity_f05(predicted_ids=[], true_ids=[]) == 1.0

    # Case B: True singleton, predicted false match -> 0.0
    assert score_single_entity_f05(predicted_ids=["S2-999"], true_ids=[]) == 0.0

    # Case C: True entity with matches, predicted empty -> 0.0
    assert score_single_entity_f05(predicted_ids=[], true_ids=["S2-101"]) == 0.0

    # Case D: Exact match -> 1.0
    assert score_single_entity_f05(predicted_ids=["S2-101"], true_ids=["S2-101"]) == 1.0


def test_calculate_macro_f05():
    """Verify macro-averaging across entities."""
    ground_truth = {
        "S1-1": ["S2-101"],
        "S1-2": [],  # singleton
    }
    # Perfect predictions: S1-1 gets 1.0, S1-2 gets 1.0 -> Macro = 1.0
    perfect_preds = {
        "S1-1": ["S2-101"],
        "S1-2": [],
    }
    assert calculate_macro_f05(perfect_preds, ground_truth) == 1.0

    # One failed singleton: S1-1 gets 1.0, S1-2 false merge gets 0.0 -> Macro = 0.5
    imperfect_preds = {
        "S1-1": ["S2-101"],
        "S1-2": ["S2-999"],
    }
    assert calculate_macro_f05(imperfect_preds, ground_truth) == 0.5


def test_select_threshold():
    """Verify threshold tuning chooses threshold that maximizes F0.5."""
    candidate_scores = [
        ("S1-1", "S2-101", 0.9),  # true match
        ("S1-1", "S2-999", 0.3),  # false candidate
        ("S1-2", "S2-888", 0.35), # false candidate for singleton
    ]
    ground_truth = {
        "S1-1": ["S2-101"],
        "S1-2": [],
    }
    best_thresh, best_score = select_threshold(
        candidate_scores, ground_truth, threshold_candidates=[0.2, 0.5, 0.8]
    )
    # At thresh=0.5: S1-1 gets S2-101 (score 1.0), S1-2 gets empty (score 1.0) -> Macro F0.5 = 1.0
    assert best_thresh == 0.5
    assert best_score == 1.0


def test_format_and_save_matching_results():
    """Verify formatting and TSV serialization for final submission."""
    all_s1 = ["S1-1", "S1-2"]
    predicted_pairs = [("S1-1", "S2-101", 0.8), ("S1-1", "S3-202", 0.7)]
    results = format_matching_results(all_s1, predicted_pairs, threshold=0.5)

    assert results["S1-1"] == ["S2-101", "S3-202"]
    assert results["S1-2"] == []  # Singleton

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tmp:
        tmp_path = tmp.name

    try:
        save_matching_results(results, tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\r\n").split("\t") for l in f]
        assert lines[0] == ["source1_entity_id", "matched_entity_ids"]
        assert lines[1] == ["S1-1", "S2-101,S3-202"]
        assert lines[2] == ["S1-2", ""]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
