"""Tests for blocking and candidate generation modules (Person 3)."""

import os
import tempfile
import pytest

from src.blocking.name_blocking import (
    build_name_inverted_index,
    generate_name_candidates,
)
from src.blocking.address_blocking import (
    build_address_inverted_index,
    generate_address_candidates,
)
from src.blocking.candidate_generator import (
    generate_candidates,
    evaluate_candidate_recall,
    save_candidate_pairs,
)


def test_build_name_inverted_index_and_retrieval():
    """Verify inverted index accurately indexes and retrieves name tokens."""
    target_records = [
        {"entity_id": "S2-101", "business_name": "apex logistics solutions"},
        {"entity_id": "S3-202", "business_name": "quantum computing lab"},
    ]
    index = build_name_inverted_index(target_records)
    assert "apex" in index
    assert "S2-101" in index["apex"]
    assert "quantum" in index
    assert "S3-202" in index["quantum"]

    query = {"entity_id": "S1-001", "business_name": "apex global"}
    candidates = generate_name_candidates(query, index)
    assert "S2-101" in candidates
    assert "S3-202" not in candidates


def test_address_inverted_index_and_retrieval():
    """Verify address-based candidate generation."""
    target_records = [
        {"entity_id": "S2-101", "business_address": "123 Market Street San Francisco"},
        {"entity_id": "S3-202", "business_address": "456 Main Road Austin"},
    ]
    addr_index = build_address_inverted_index(target_records)
    query = {"entity_id": "S1-001", "business_address": "999 Market Avenue"}
    candidates = generate_address_candidates(query, addr_index)
    assert "S2-101" in candidates
    assert "S3-202" not in candidates


def test_generate_candidates_pipeline():
    """Verify end-to-end candidate generation combining name and address."""
    s1_records = [
        {"entity_id": "S1-001", "business_name": "omega technologies", "business_address": "seattle"},
        {"entity_id": "S1-002", "business_name": "isolated store", "business_address": "nowhere"},
    ]
    targets = [
        {"entity_id": "S2-501", "business_name": "omega solutions", "business_address": "boston"},
        {"entity_id": "S3-601", "business_name": "different corp", "business_address": "seattle"},
    ]
    candidates = generate_candidates(s1_records, targets, max_candidates_per_entity=10)

    assert "S1-001" in candidates
    # S1-001 should capture S2-501 (via name 'omega') and S3-601 (via address 'seattle')
    assert "S2-501" in candidates["S1-001"]
    assert "S3-601" in candidates["S1-001"]


def test_evaluate_candidate_recall():
    """Verify candidate recall ceiling metric calculation."""
    candidate_pairs = {
        "S1-1": ["S2-10", "S2-11"],
        "S1-2": ["S3-20"],
    }
    # S1-1 has 2 true matches (both found). S1-2 has 2 true matches (1 found, 1 missed).
    ground_truth = {
        "S1-1": ["S2-10", "S2-11"],
        "S1-2": ["S3-20", "S3-99"],
    }
    recall_stats = evaluate_candidate_recall(candidate_pairs, ground_truth)
    assert recall_stats["total_true_links"] == 4.0
    assert recall_stats["captured_links"] == 3.0
    assert recall_stats["candidate_recall"] == 0.75


def test_save_candidate_pairs_format():
    """Verify candidate pairs output matches competition TSV format."""
    candidate_pairs = {
        "S1-001": ["S2-10", "S3-20"],
        "S1-002": [],
    }
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tmp:
        tmp_path = tmp.name

    try:
        save_candidate_pairs(candidate_pairs, tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n").split("\t") for line in f]
        assert lines[0] == ["source1_entity_id", "candidate_entity_ids"]
        assert lines[1] == ["S1-001", "S2-10,S3-20"]
        assert lines[2] == ["S1-002", ""]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
