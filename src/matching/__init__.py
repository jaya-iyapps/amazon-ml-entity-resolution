"""
Amazon ML Entity Resolution Challenge 2026 - Matching Module

This package contains components for pairwise feature extraction,
model training, threshold selection, and match prediction.
"""

from .features import FEATURE_NAMES, compute_pair_features, extract_pair_features
from .predict import (
    format_matching_results,
    predict_matches,
    run_prediction_pipeline,
    save_matching_results,
    stream_candidate_pairs_from_tsv,
    write_matching_results_tsv,
)
from .threshold import (
    calculate_f_beta,
    calculate_macro_f05,
    compute_entity_f05,
    evaluate_predictions,
    find_optimal_threshold,
    score_single_entity_f05,
    select_threshold,
)
from .train import (
    MatchingClassifier,
    grouped_train_val_split,
    train_matching_pipeline,
    train_model,
)

__all__ = [
    "FEATURE_NAMES",
    "extract_pair_features",
    "compute_pair_features",
    "compute_entity_f05",
    "calculate_f_beta",
    "score_single_entity_f05",
    "calculate_macro_f05",
    "evaluate_predictions",
    "find_optimal_threshold",
    "select_threshold",
    "MatchingClassifier",
    "grouped_train_val_split",
    "train_matching_pipeline",
    "train_model",
    "predict_matches",
    "format_matching_results",
    "save_matching_results",
    "run_prediction_pipeline",
    "stream_candidate_pairs_from_tsv",
    "write_matching_results_tsv",
]
