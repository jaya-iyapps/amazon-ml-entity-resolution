"""
Unit tests for Person 3 - Blocking and Candidate Generation Module.

Covers all 16 required test cases:
1. Exact name blocking
2. Address blocking
3. Numeric blocking
4. Multiple blocking passes
5. Candidate union
6. Duplicate removal
7. Country compatibility
8. Missing country handling
9. France support
10. Unseen country support
11. High-frequency key protection
12. Source 2 / Source 3 separation
13. No dangling candidate IDs
14. Singleton/no-match handling
15. Candidate recall evaluation
16. Deterministic output
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.blocking import (
    AddressInvertedIndex,
    CandidateGenerator,
    CandidatePair,
    NameInvertedIndex,
    build_address_index,
    build_name_index,
    determine_source_label,
    evaluate_blocking,
    export_candidate_pairs_tsv,
    extract_distinctive_address_tokens,
    extract_normalized_numbers,
    generate_address_candidates,
    generate_candidates,
    generate_name_candidates,
    is_country_compatible,
)


class TestBlockingModule(unittest.TestCase):

    def setUp(self):
        # Synthetic target records from Source 2 and Source 3
        self.target_records = [
            {
                "entity_id": "S2-101",
                "business_name": "Pacific Best Garden, Inc",
                "business_address": "363 Blue Spruce Road, Clinton, AR",
                "country": "US",
            },
            {
                "entity_id": "S3-201",
                "business_name": "Inc Pacific Best Garden",
                "business_address": "0363 Blue Spruce Rd, Clinton, Arkansas",
                "country": "US",
            },
            {
                "entity_id": "S2-102",
                "business_name": "Green Logistics Private Limited",
                "business_address": "7 Kashinath Mullick Lane, Calcutta, WB",
                "country": "India",
            },
            {
                "entity_id": "S3-202",
                "business_name": "Green Logistics Center",
                "business_address": "#7 Kashinath Mullick Lane, Calcutta",
                "country": "India",
            },
            {
                "entity_id": "S2-103",
                "business_name": "Boulangerie Patisserie SAS",
                "business_address": "15 Rue de Paris, Lyon",
                "country": "France",
            },
            {
                "entity_id": "S3-203",
                "business_name": "Boulangerie Lyon",
                "business_address": "15 Rue de Paris",
                "country": "France",
            },
            {
                "entity_id": "S2-104",
                "business_name": "Nordic Solutions GmbH",
                "business_address": "42 Fjord Way, Oslo",
                "country": "Norway",  # Unseen country
            },
            {
                "entity_id": "S2-105",
                "business_name": "Mystery Traders",
                "business_address": "Unknown Road",
                "country": "",  # Missing country
            },
        ]

    # 1. Exact name blocking
    def test_exact_name_blocking(self):
        name_idx = build_name_index(self.target_records)
        s1_rec = {
            "entity_id": "S1-001",
            "business_name": "Pacific Best Garden LLC",
            "business_address": "Different Street",
            "country": "US",
        }
        cands = generate_name_candidates([s1_rec], name_idx)
        self.assertIn("S2-101", cands["S1-001"])
        self.assertIn("S3-201", cands["S1-001"])

    # 2. Address blocking
    def test_address_blocking(self):
        addr_idx = build_address_index(self.target_records)
        s1_rec = {
            "entity_id": "S1-002",
            "business_name": "Completely Different Name",
            "business_address": "7 Kashinath Mullick Lane, Calcutta",
            "country": "India",
        }
        cands = generate_address_candidates([s1_rec], addr_idx)
        self.assertIn("S2-102", cands["S1-002"])

    # 3. Numeric blocking
    def test_numeric_blocking(self):
        # Tests leading zero normalization: 363 matches 0363
        nums1 = extract_normalized_numbers("363 Blue Spruce Road")
        nums2 = extract_normalized_numbers("0363 Blue Spruce Rd")
        self.assertEqual(nums1, ["363"])
        self.assertEqual(nums2, ["363"])

        addr_idx = build_address_index(self.target_records)
        s1_rec = {
            "entity_id": "S1-003",
            "business_name": "Pacific Nursery",
            "business_address": "363 Spruce Rd",
            "country": "US",
        }
        cands = generate_address_candidates([s1_rec], addr_idx)
        self.assertTrue(len(cands["S1-003"]) > 0)
        self.assertIn("S2-101", cands["S1-003"])

    # 4. Multiple blocking passes
    def test_multiple_blocking_passes(self):
        gen = CandidateGenerator(max_candidates_per_entity=50)
        gen.fit(self.target_records)

        # Record where name matches S2-101 and address matches S2-102
        s1_rec = {
            "entity_id": "S1-004",
            "business_name": "Pacific Best Garden Inc",
            "business_address": "7 Kashinath Mullick Lane",
            "country": "",  # Missing country allows both
        }
        cands = gen.generate_candidates_for_record(s1_rec)
        self.assertIn("S2-101", cands)  # From name pass
        self.assertIn("S2-102", cands)  # From address pass

    # 5. Candidate union
    def test_candidate_union(self):
        gen = CandidateGenerator()
        gen.fit(self.target_records)
        s1_rec = {
            "entity_id": "S1-005",
            "business_name": "Pacific Best Garden",
            "business_address": "15 Rue de Paris",
            "country": "",
        }
        cands = set(gen.generate_candidates_for_record(s1_rec))
        # Should union name candidates (S2-101) and address candidates (S2-103, S3-203)
        self.assertIn("S2-101", cands)
        self.assertIn("S2-103", cands)

    # 6. Duplicate removal
    def test_duplicate_removal(self):
        gen = CandidateGenerator()
        gen.fit(self.target_records)
        s1_rec = {
            "entity_id": "S1-006",
            "business_name": "Pacific Best Garden",
            "business_address": "363 Blue Spruce Road",
            "country": "US",
        }
        cand_list = gen.generate_candidates_for_record(s1_rec)
        # S2-101 matches BOTH name and address, but must appear exactly once
        self.assertEqual(len(cand_list), len(set(cand_list)))
        self.assertEqual(cand_list.count("S2-101"), 1)

    # 7. Country compatibility
    def test_country_compatibility(self):
        self.assertTrue(is_country_compatible("US", "US"))
        self.assertTrue(is_country_compatible("United States", "US"))
        self.assertFalse(is_country_compatible("US", "India"))
        self.assertFalse(is_country_compatible("US", "France"))

        gen = CandidateGenerator()
        gen.fit(self.target_records)
        s1_rec = {
            "entity_id": "S1-007",
            "business_name": "Pacific Best Garden",
            "business_address": "363 Blue Spruce Road",
            "country": "India",  # Conflicting country with US target
        }
        cands = gen.generate_candidates_for_record(s1_rec)
        self.assertNotIn("S2-101", cands)  # S2-101 is US, so it must be pruned

    # 8. Missing country handling
    def test_missing_country_handling(self):
        self.assertTrue(is_country_compatible("US", ""))
        self.assertTrue(is_country_compatible("", "India"))
        self.assertTrue(is_country_compatible(None, "France"))
        self.assertTrue(is_country_compatible(None, None))

        gen = CandidateGenerator()
        gen.fit(self.target_records)
        s1_rec = {
            "entity_id": "S1-008",
            "business_name": "Mystery Traders",
            "business_address": "Unknown Road",
            "country": "US",
        }
        cands = gen.generate_candidates_for_record(s1_rec)
        # Target S2-105 has missing country, so it should not be discarded
        self.assertIn("S2-105", cands)

    # 9. France support
    def test_france_support(self):
        self.assertTrue(is_country_compatible("France", "France"))
        self.assertTrue(is_country_compatible("FR", "France"))

        gen = CandidateGenerator()
        gen.fit(self.target_records)
        s1_rec = {
            "entity_id": "S1-009",
            "business_name": "Boulangerie Patisserie",
            "business_address": "15 Rue de Paris, Lyon",
            "country": "France",
        }
        cands = gen.generate_candidates_for_record(s1_rec)
        self.assertIn("S2-103", cands)
        self.assertIn("S3-203", cands)

    # 10. Unseen country support
    def test_unseen_country_support(self):
        self.assertTrue(is_country_compatible("Norway", "Norway"))
        self.assertTrue(is_country_compatible("Brazil", "Brazil"))
        self.assertFalse(is_country_compatible("Norway", "Brazil"))

        gen = CandidateGenerator()
        gen.fit(self.target_records)
        s1_rec = {
            "entity_id": "S1-010",
            "business_name": "Nordic Solutions",
            "business_address": "42 Fjord Way",
            "country": "Norway",
        }
        cands = gen.generate_candidates_for_record(s1_rec)
        self.assertIn("S2-104", cands)

    # 11. High-frequency key protection
    def test_high_frequency_key_protection(self):
        # Create corpus with a ubiquitous word
        noisy_records = [
            {"entity_id": f"S2-noisy-{i}", "business_name": f"Universal Services {i}", "business_address": "Road", "country": "US"}
            for i in range(50)
        ]
        noisy_records.append(
            {"entity_id": "S2-target", "business_name": "Zanzibar Special Craft", "business_address": "Road", "country": "US"}
        )
        # With max_token_freq=10, 'universal' (freq 50) won't be indexed as rare single token
        name_idx = build_name_index(noisy_records, max_token_freq=10)
        self.assertNotIn("services", name_idx.rare_token_idx)
        self.assertIn("zanzibar", name_idx.rare_token_idx)

    # 12. Source 2 / Source 3 separation
    def test_source2_source3_separation(self):
        self.assertEqual(determine_source_label("S2-12345"), "source2")
        self.assertEqual(determine_source_label("S3-98765"), "source3")
        self.assertEqual(determine_source_label("S1-00001"), "source1")

        gen = CandidateGenerator()
        gen.fit(self.target_records)
        s1_rec = {
            "entity_id": "S1-012",
            "business_name": "Pacific Best Garden",
            "business_address": "363 Blue Spruce",
            "country": "US",
        }
        pairs = gen.generate_pairs([s1_rec])
        sources = {p.source for p in pairs}
        self.assertIn("source2", sources)
        self.assertIn("source3", sources)

    # 13. No dangling candidate IDs
    def test_no_dangling_candidate_ids(self):
        gen = CandidateGenerator()
        gen.fit(self.target_records)
        s1_rec = {
            "entity_id": "S1-013",
            "business_name": "Pacific Best Garden Inc",
            "business_address": "363 Blue Spruce",
            "country": "US",
        }
        cands = gen.generate_candidates_for_record(s1_rec)
        target_ids = {r["entity_id"] for r in self.target_records}
        for cid in cands:
            self.assertIn(cid, target_ids)

    # 14. Singleton / no-match handling
    def test_singleton_no_match_handling(self):
        gen = CandidateGenerator()
        gen.fit(self.target_records)
        s1_singleton = {
            "entity_id": "S1-singleton",
            "business_name": "Xylophone Quantum Quasar 9999",
            "business_address": "Outer Space Station Alpha",
            "country": "US",
        }
        cands = gen.generate_candidates_for_record(s1_singleton)
        # Must return empty list cleanly without throwing errors or hallucinating
        self.assertEqual(cands, [])

    # 15. Candidate recall evaluation
    def test_candidate_recall_evaluation(self):
        candidates = {
            "S1-1": ["S2-A", "S2-B"],
            "S1-2": ["S3-C"],
            "S1-3": [],  # Singleton
        }
        ground_truth = {
            "S1-1": {"S2-A", "S2-B"},
            "S1-2": {"S3-C", "S3-D"},  # Missed S3-D
            "S1-3": set(),  # True singleton
        }
        metrics = evaluate_blocking(candidates, ground_truth, total_target_records=10)
        # 3 retrieved out of 4 true pairs -> recall = 0.75
        self.assertEqual(metrics["retrieved_true_pairs"], 3)
        self.assertEqual(metrics["total_true_pairs"], 4)
        self.assertAlmostEqual(metrics["candidate_recall"], 0.75)
        self.assertEqual(metrics["total_candidates"], 3)
        self.assertEqual(metrics["num_s1_entities"], 3)
        self.assertAlmostEqual(metrics["avg_candidates_per_s1"], 1.0)
        self.assertAlmostEqual(metrics["reduction_ratio"], 0.9)

    # 16. Deterministic output
    def test_deterministic_output(self):
        gen = CandidateGenerator(max_candidates_per_entity=10)
        gen.fit(self.target_records)
        s1_rec = {
            "entity_id": "S1-016",
            "business_name": "Pacific Best Garden",
            "business_address": "363 Blue Spruce",
            "country": "US",
        }
        run1 = gen.generate_candidates_for_record(s1_rec)
        run2 = gen.generate_candidates_for_record(s1_rec)
        run3 = gen.generate_candidates_for_record(s1_rec)
        self.assertEqual(run1, run2)
        self.assertEqual(run2, run3)

        # Test TSV exporter format
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tsv = os.path.join(tmpdir, "candidate_pairs.tsv")
            export_candidate_pairs_tsv({"S1-001": run1, "S1-002": []}, ["S1-001", "S1-002"], out_tsv)
            with open(out_tsv, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\n").split("\t") for line in f]
            self.assertEqual(lines[0], ["source1_entity_id", "candidate_entity_ids"])
            self.assertEqual(lines[1][0], "S1-001")
            self.assertEqual(lines[2], ["S1-002", ""])


if __name__ == "__main__":
    unittest.main()
