"""
Amazon ML Entity Resolution Challenge 2026 - Matching Unit Tests

Tests cover:
1. Feature calculations (Jaccard, Levenshtein, LCS, token-sort, character n-grams)
2. Missing values handling and indicator flags
3. Identical records similarity (expect near 1.0)
4. Clearly different records similarity (expect near 0.0)
5. Numeric token agreement and disagreement
6. Open-set country agreement and mismatch
7. Model prediction and probability calibration
8. Threshold selection and F0.5 optimization
9. No-match (singleton) behavior
10. Multiple matches per Source 1 entity
11. Output formatting compliance (single tab, no header distortion, no duplicates)
"""

import os
import sys
import tempfile
import unittest

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.matching.features import (
    FEATURE_NAMES,
    char_ngram_jaccard,
    compute_pair_features,
    dice_similarity,
    extract_pair_features,
    jaccard_similarity,
    levenshtein_similarity,
    lcs_similarity,
    token_sort_similarity,
)
from src.matching.predict import (
    format_matching_results,
    predict_matches,
    save_matching_results,
    write_matching_results_tsv,
)
from src.matching.threshold import (
    calculate_f_beta,
    calculate_macro_f05,
    compute_entity_f05,
    evaluate_predictions,
    find_optimal_threshold,
    score_single_entity_f05,
    select_threshold,
)
from src.matching.train import (
    MatchingClassifier,
    grouped_train_val_split,
    train_model,
)


class TestMatchingFeatures(unittest.TestCase):
    """Test pairwise similarity metrics and feature vector construction."""

    def test_feature_names_length(self):
        self.assertEqual(len(FEATURE_NAMES), 25)

    def test_identical_records(self):
        r1 = {
            "business_name": "Apex Technology Solutions Inc.",
            "business_address": "100 Innovation Way, Suite 400, Austin, TX",
            "country": "US",
        }
        r2 = {
            "business_name": "Apex Technology Solutions",
            "business_address": "100 Innovation Way, Suite 400, Austin, TX",
            "country": "US",
        }
        feats = extract_pair_features(r1, r2)
        idx_core = FEATURE_NAMES.index("name_core_match")
        idx_addr = FEATURE_NAMES.index("addr_exact_match")
        idx_c_exact = FEATURE_NAMES.index("country_exact_match")
        idx_num_jacc = FEATURE_NAMES.index("num_jaccard")

        self.assertEqual(feats[idx_core], 1.0)
        self.assertEqual(feats[idx_addr], 1.0)
        self.assertEqual(feats[idx_c_exact], 1.0)
        self.assertGreaterEqual(feats[idx_num_jacc], 0.9)

    def test_clearly_different_records(self):
        r1 = {
            "business_name": "Green Valley Grocery",
            "business_address": "45 Elm Street, Portland, ME",
            "country": "US",
        }
        r2 = {
            "business_name": "Sharma Electronics Private Limited",
            "business_address": "MG Road, Sector 14, Gurgaon, Haryana",
            "country": "India",
        }
        feats = extract_pair_features(r1, r2)
        idx_name_jacc = FEATURE_NAMES.index("name_token_jaccard")
        idx_c_mismatch = FEATURE_NAMES.index("country_mismatch")
        idx_num_disagree = FEATURE_NAMES.index("num_disagreement")

        self.assertEqual(feats[idx_name_jacc], 0.0)
        self.assertEqual(feats[idx_c_mismatch], 1.0)
        self.assertEqual(feats[idx_num_disagree], 1.0)

    def test_missing_values(self):
        r1 = {"business_name": "", "business_address": None, "country": ""}
        r2 = {"business_name": "Test Co", "business_address": "123 Main", "country": "US"}
        feats = extract_pair_features(r1, r2)

        self.assertEqual(feats[FEATURE_NAMES.index("name_missing_1")], 1.0)
        self.assertEqual(feats[FEATURE_NAMES.index("name_missing_2")], 0.0)
        self.assertEqual(feats[FEATURE_NAMES.index("addr_missing_1")], 1.0)
        self.assertEqual(feats[FEATURE_NAMES.index("country_missing")], 1.0)

    def test_numeric_agreement_and_disagreement(self):
        # Case A: Same numbers (105, 27203)
        r1 = {"business_name": "A", "business_address": "105 Elm St, 27203", "country": "US"}
        r2 = {"business_name": "A", "business_address": "Apt 105, Zip 27203", "country": "US"}
        f_agree = extract_pair_features(r1, r2)
        self.assertEqual(f_agree[FEATURE_NAMES.index("num_disagreement")], 0.0)
        self.assertGreaterEqual(f_agree[FEATURE_NAMES.index("num_common_count")], 2.0)

        # Case B: Disagreeing numbers (105 vs 999)
        r3 = {"business_name": "A", "business_address": "999 Oak St, 90210", "country": "US"}
        f_disagree = extract_pair_features(r1, r3)
        self.assertEqual(f_disagree[FEATURE_NAMES.index("num_disagreement")], 1.0)
        self.assertEqual(f_disagree[FEATURE_NAMES.index("num_common_count")], 0.0)

    def test_string_similarity_primitives(self):
        self.assertEqual(levenshtein_similarity("kitten", "sitting"), 1.0 - (3.0 / 7.0))
        self.assertEqual(levenshtein_similarity("apple", "apple"), 1.0)
        self.assertAlmostEqual(jaccard_similarity({"a", "b"}, {"b", "c"}), 1.0 / 3.0)
        self.assertAlmostEqual(dice_similarity({"a", "b"}, {"b", "c"}), 2.0 / 4.0)
        self.assertGreater(lcs_similarity("testing", "tester"), 0.5)
        self.assertGreater(token_sort_similarity(["corp", "apex"], ["apex", "corp"]), 0.99)
        self.assertGreater(char_ngram_jaccard("corporation", "corporate", n=3), 0.5)


class TestThresholdAndF05(unittest.TestCase):
    """Test macro F0.5 evaluation, singleton handling, and threshold grid search."""

    def test_singleton_correctness(self):
        # True is empty, pred is empty -> F0.5 = 1.0
        f05, prec, rec = compute_entity_f05(set(), set())
        self.assertEqual(f05, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)

    def test_singleton_false_positive_penalty(self):
        # True is empty, pred has matches -> F0.5 = 0.0
        f05, prec, rec = compute_entity_f05(set(), {"S2-99"})
        self.assertEqual(f05, 0.0)

    def test_entity_f05_formula(self):
        # True: {A, B}, Pred: {A, B, C} -> TP=2, Prec=2/3, Rec=1.0
        # F0.5 = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = 0.8333 / 1.1667 = 0.7142857
        f05, prec, rec = compute_entity_f05({"S2-1", "S2-2"}, {"S2-1", "S2-2", "S2-3"})
        self.assertAlmostEqual(prec, 2.0 / 3.0, places=4)
        self.assertAlmostEqual(rec, 1.0, places=4)
        expected_f05 = (1.25 * (2.0 / 3.0) * 1.0) / (0.25 * (2.0 / 3.0) + 1.0)
        self.assertAlmostEqual(f05, expected_f05, places=4)

    def test_find_optimal_threshold(self):
        candidate_scores = [
            {"source1_entity_id": "S1-1", "source2_entity_id": "S2-A", "score": 0.92},
            {"source1_entity_id": "S1-1", "source2_entity_id": "S2-B", "score": 0.45},
            {"source1_entity_id": "S1-2", "source2_entity_id": "S2-C", "score": 0.88},
            {"source1_entity_id": "S1-3", "source2_entity_id": "S2-D", "score": 0.35},
        ]
        gt = {
            "S1-1": {"S2-A"},
            "S1-2": {"S2-C"},
            "S1-3": set(),  # singleton
        }
        all_s1 = ["S1-1", "S1-2", "S1-3"]
        res = find_optimal_threshold(
            candidate_scores, gt, all_s1, min_threshold=0.50, max_threshold=0.90, step=0.05
        )
        self.assertGreaterEqual(res["best_macro_f05"], 0.99)
        self.assertGreaterEqual(res["best_threshold"], 0.50)


class TestModelTrainingAndValidation(unittest.TestCase):
    """Test grouped validation without entity leakage and classifier training."""

    def test_grouped_train_val_split_zero_leakage(self):
        pairs = [
            {"source1_entity_id": "S1-1", "source2_entity_id": "S2-1"},
            {"source1_entity_id": "S1-1", "source2_entity_id": "S2-2"},
            {"source1_entity_id": "S1-2", "source2_entity_id": "S2-3"},
            {"source1_entity_id": "S1-3", "source2_entity_id": "S2-4"},
            {"source1_entity_id": "S1-4", "source2_entity_id": "S2-5"},
        ]
        train_p, val_p, train_s1, val_s1 = grouped_train_val_split(pairs, val_ratio=0.5, seed=42)

        # Confirm sets of S1 entities are mutually exclusive
        self.assertEqual(len(train_s1 & val_s1), 0)
        self.assertEqual(len(train_p) + len(val_p), len(pairs))

    def test_classifier_fit_predict_save_load(self):
        clf = MatchingClassifier(l2_reg=1e-4, learning_rate=0.1, n_epochs=15)
        # Synthetic separable data: 25 features
        X_pos = [[1.0] * 25 for _ in range(30)]
        X_neg = [[0.0] * 25 for _ in range(60)]
        X = X_pos + X_neg
        y = [1] * 30 + [0] * 60

        clf.fit(X, y)
        probs_pos = clf.predict_proba([[1.0] * 25])
        probs_neg = clf.predict_proba([[0.0] * 25])

        self.assertGreater(probs_pos[0], 0.7)
        self.assertLess(probs_neg[0], 0.3)

        # Test serialization
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            clf.save(tmp_path)
            loaded_clf = MatchingClassifier.load(tmp_path)
            loaded_prob = loaded_clf.predict_proba([[1.0] * 25])
            self.assertAlmostEqual(probs_pos[0], loaded_prob[0], places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestPredictionAndOutputContract(unittest.TestCase):
    """Test prediction streaming, no-match handling, and output TSV formatting."""

    def test_no_match_and_multiple_matches(self):
        # Create a trained classifier on synthetic positive vs negative examples
        model = MatchingClassifier(l2_reg=1e-4, learning_rate=0.1, n_epochs=20)
        # Feature vector for high match vs low match
        feat_high = [1.0] * 15 + [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        feat_low = [0.0] * 15 + [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.fit([feat_high] * 20 + [feat_low] * 20, [1] * 20 + [0] * 20)

        all_s1 = ["S1-SINGLETON", "S1-MULTI", "S1-ZERO-CAND"]
        s1_records = {
            "S1-SINGLETON": {"business_name": "Alpha Corp", "business_address": "123 Main", "country": "US"},
            "S1-MULTI": {"business_name": "Beta LLC", "business_address": "456 Oak", "country": "US"},
            "S1-ZERO-CAND": {"business_name": "Gamma Inc", "business_address": "789 Pine", "country": "US"},
        }
        target_records = {
            "S2-LOW": {"business_name": "Completely Different", "business_address": "000 Far", "country": "India"},
            "S2-HIGH1": {"business_name": "Beta LLC", "business_address": "456 Oak", "country": "US"},
            "S3-HIGH2": {"business_name": "Beta Limited", "business_address": "456 Oak", "country": "US"},
        }
        pairs_stream = [
            [
                {"source1_entity_id": "S1-SINGLETON", "source2_entity_id": "S2-LOW", "source": "S2"},
                {"source1_entity_id": "S1-MULTI", "source2_entity_id": "S2-HIGH1", "source": "S2"},
                {"source1_entity_id": "S1-MULTI", "source2_entity_id": "S3-HIGH2", "source": "S3"},
                {"source1_entity_id": "S1-ZERO-CAND", "source2_entity_id": "", "source": ""},
            ]
        ]

        preds = predict_matches(
            model=model,
            threshold=0.80,
            all_s1_ids=all_s1,
            candidate_pairs_stream=iter(pairs_stream),
            s1_records=s1_records,
            target_records=target_records,
        )

        # S1-SINGLETON should have empty matches because S2-LOW is below threshold
        self.assertEqual(preds["S1-SINGLETON"], [])
        # S1-ZERO-CAND had no candidates -> empty matches
        self.assertEqual(preds["S1-ZERO-CAND"], [])
        # S1-MULTI should have both matches
        self.assertEqual(set(preds["S1-MULTI"]), {"S2-HIGH1", "S3-HIGH2"})

    def test_output_tsv_contract(self):
        preds = {
            "S1-0001": ["S2-0047", "S3-0812"],
            "S1-0002": ["S3-0004"],
            "S1-0003": [],  # singleton
        }
        all_s1 = ["S1-0001", "S1-0002", "S1-0003"]

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            tmp_tsv = f.name

        try:
            write_matching_results_tsv(preds, all_s1, tmp_tsv)
            with open(tmp_tsv, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

            # Verify header exactly tab separated
            self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
            self.assertEqual(lines[1], "S1-0001\tS2-0047,S3-0812")
            self.assertEqual(lines[2], "S1-0002\tS3-0004")
            self.assertEqual(lines[3], "S1-0003\t")
            self.assertEqual(len(lines), 4)
        finally:
            if os.path.exists(tmp_tsv):
                os.remove(tmp_tsv)


class TestLegacyMatchingInterface(unittest.TestCase):
    """Verify backwards-compatible functions from initial repository interfaces."""

    def test_pairwise_feature_similarities(self):
        rec1 = {
            "business_name": "alpha tech solutions",
            "business_address": "100 silicon way san jose ca",
            "country": "US",
        }
        rec_same = {
            "business_name": "alpha tech solutions",
            "business_address": "100 silicon way san jose ca",
            "country": "US",
        }
        rec_diff = {
            "business_name": "zebra fashion store",
            "business_address": "999 broadway new york ny",
            "country": "INDIA",
        }
        feat_same = compute_pair_features(rec1, rec_same)
        feat_diff = compute_pair_features(rec1, rec_diff)

        self.assertEqual(feat_same["name_ratio"], 1.0)
        self.assertEqual(feat_same["name_exact_match"], 1.0)
        self.assertEqual(feat_same["country_match"], 1.0)
        self.assertLess(feat_diff["name_ratio"], feat_same["name_ratio"])
        self.assertEqual(feat_diff["name_exact_match"], 0.0)
        self.assertEqual(feat_diff["country_match"], 0.0)

    def test_f05_formula_precision_weighting(self):
        f05_high_prec = calculate_f_beta(precision=1.0, recall=0.5, beta=0.5)
        self.assertAlmostEqual(f05_high_prec, 5.0 / 6.0, places=3)
        f05_low_prec = calculate_f_beta(precision=0.5, recall=1.0, beta=0.5)
        self.assertAlmostEqual(f05_low_prec, 5.0 / 9.0, places=3)
        self.assertGreater(f05_high_prec, f05_low_prec)

    def test_single_entity_f05_singletons(self):
        self.assertEqual(score_single_entity_f05(predicted_ids=[], true_ids=[]), 1.0)
        self.assertEqual(score_single_entity_f05(predicted_ids=["S2-999"], true_ids=[]), 0.0)
        self.assertEqual(score_single_entity_f05(predicted_ids=[], true_ids=["S2-101"]), 0.0)
        self.assertEqual(score_single_entity_f05(predicted_ids=["S2-101"], true_ids=["S2-101"]), 1.0)

    def test_calculate_macro_f05(self):
        ground_truth = {"S1-1": ["S2-101"], "S1-2": []}
        perfect_preds = {"S1-1": ["S2-101"], "S1-2": []}
        self.assertEqual(calculate_macro_f05(perfect_preds, ground_truth), 1.0)
        imperfect_preds = {"S1-1": ["S2-101"], "S1-2": ["S2-999"]}
        self.assertEqual(calculate_macro_f05(imperfect_preds, ground_truth), 0.5)

    def test_select_threshold(self):
        candidate_scores = [
            ("S1-1", "S2-101", 0.9),
            ("S1-1", "S2-999", 0.3),
            ("S1-2", "S2-888", 0.35),
        ]
        ground_truth = {"S1-1": ["S2-101"], "S1-2": []}
        best_thresh, best_score = select_threshold(
            candidate_scores, ground_truth, threshold_candidates=[0.2, 0.5, 0.8]
        )
        self.assertEqual(best_thresh, 0.5)
        self.assertEqual(best_score, 1.0)

    def test_format_and_save_matching_results(self):
        all_s1 = ["S1-1", "S1-2"]
        predicted_pairs = [("S1-1", "S2-101", 0.8), ("S1-1", "S3-202", 0.7)]
        results = format_matching_results(all_s1, predicted_pairs, threshold=0.5)
        self.assertEqual(results["S1-1"], ["S2-101", "S3-202"])
        self.assertEqual(results["S1-2"], [])

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as tmp:
            tmp_path = tmp.name
        try:
            save_matching_results(results, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\r\n").split("\t") for line in f]
            self.assertEqual(lines[0], ["source1_entity_id", "matched_entity_ids"])
            self.assertEqual(lines[1], ["S1-1", "S2-101,S3-202"])
            self.assertEqual(lines[2], ["S1-2", ""])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestEdgeCasesAndRobustness(unittest.TestCase):
    """Test unseen countries, unicode variations, empty candidates, and formatting robustness."""

    def test_unseen_country_generalization(self):
        # Case A: Same unseen country (France, Germany, Norway)
        r1 = {"business_name": "Boulangerie Parisienne", "business_address": "15 Rue de Paris", "country": "France"}
        r2 = {"business_name": "Boulangerie Parisienne", "business_address": "15 Rue de Paris", "country": "France"}
        f_same = extract_pair_features(r1, r2)
        idx_c_exact = FEATURE_NAMES.index("country_exact_match")
        idx_c_mismatch = FEATURE_NAMES.index("country_mismatch")
        self.assertEqual(f_same[idx_c_exact], 1.0)
        self.assertEqual(f_same[idx_c_mismatch], 0.0)

        # Case B: Cross-country mismatch between unseen countries
        r3 = {"business_name": "Boulangerie Parisienne", "business_address": "15 Rue de Paris", "country": "Germany"}
        f_diff = extract_pair_features(r1, r3)
        self.assertEqual(f_diff[idx_c_exact], 0.0)
        self.assertEqual(f_diff[idx_c_mismatch], 1.0)

    def test_unicode_hindi_and_accents(self):
        # Business names in Devanagari or French accents
        r1 = {"business_name": "रिलायंस इंडस्ट्रीज लिमिटेड", "business_address": "मुंबई", "country": "India"}
        r2 = {"business_name": "रिलायंस इंडस्ट्रीज", "business_address": "मुंबई", "country": "India"}
        f = extract_pair_features(r1, r2)
        self.assertGreater(f[FEATURE_NAMES.index("name_token_jaccard")], 0.5)

        r_fr1 = {"business_name": "Société Générale", "business_address": "29 Bd Haussmann", "country": "France"}
        r_fr2 = {"business_name": "Societe Generale", "business_address": "29 Boulevard Haussmann", "country": "France"}
        f_fr = extract_pair_features(r_fr1, r_fr2)
        self.assertGreaterEqual(f_fr[FEATURE_NAMES.index("name_levenshtein_sim")], 0.75)

    def test_complete_no_match_singleton_pipeline(self):
        # When an S1 entity has zero candidates or all candidates below threshold
        all_s1 = ["S1-ISOLATED"]
        s1_records = {"S1-ISOLATED": {"business_name": "Solo Venture", "business_address": "Nowhere", "country": "US"}}
        target_records = {"S2-FAR": {"business_name": "Completely Other", "business_address": "Elsewhere", "country": "India"}}
        pairs_stream = [[{"source1_entity_id": "S1-ISOLATED", "source2_entity_id": "S2-FAR", "source": "S2"}]]

        model = MatchingClassifier()
        # Train on dummy negative
        model.fit([[0.0] * 25 for _ in range(10)] + [[1.0] * 25 for _ in range(10)], [0] * 10 + [1] * 10)
        preds = predict_matches(
            model=model,
            threshold=0.85,
            all_s1_ids=all_s1,
            candidate_pairs_stream=iter(pairs_stream),
            s1_records=s1_records,
            target_records=target_records,
        )
        self.assertEqual(preds["S1-ISOLATED"], [])

        # Ground truth has it as singleton -> F0.5 should be 1.0
        eval_res = evaluate_predictions(preds, {"S1-ISOLATED": set()}, all_s1)
        self.assertEqual(eval_res["macro_f05"], 1.0)
        self.assertEqual(eval_res["singleton_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

