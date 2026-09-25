"""Machine learning matching module for pairwise features, training, inference, and thresholding.

Owned by: Person 2 (Matching / ML)
"""

from src.matching.features import compute_pair_features
from src.matching.train import train_model
from src.matching.predict import (
    predict_scores,
    format_matching_results,
    save_matching_results,
)
from src.matching.threshold import (
    calculate_f_beta,
    score_single_entity_f05,
    calculate_macro_f05,
    select_threshold,
)

__all__ = [
    "compute_pair_features",
    "train_model",
    "predict_scores",
    "format_matching_results",
    "save_matching_results",
    "calculate_f_beta",
    "score_single_entity_f05",
    "calculate_macro_f05",
    "select_threshold",
]
