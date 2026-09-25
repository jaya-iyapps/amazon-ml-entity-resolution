"""Competition metric evaluation (Macro F_0.5) and decision threshold selection.

Owned by: Person 2 (Matching / ML)

Evaluation Metric: Macro-averaged F_0.5 score across all Source 1 entities.
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Weights precision 2x over recall (penalizes false merges more than missed links).
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple


def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate F_beta score given precision and recall.

    For beta = 0.5:
        weight = beta^2 = 0.25
        (1 + 0.25) * P * R / (0.25 * P + R) = 1.25 * P * R / (0.25 * P + R)
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    return numerator / denominator if denominator > 0.0 else 0.0


def score_single_entity_f05(
    predicted_ids: Iterable[str],
    true_ids: Iterable[str],
) -> float:
    """Score a single Source 1 entity using the competition F_0.5 rule.

    Crucial Singleton Handling:
    - If an entity has no true matches (singleton):
        - Predicting empty list -> 1.0 (correctly identified singleton)
        - Predicting any match -> 0.0 (false merge penalty)
    - If an entity has true matches:
        - Predicting empty list -> 0.0
        - Predicting matches -> standard F_0.5 based on TP, FP, FN

    Args:
        predicted_ids: Iterable of predicted matching IDs.
        true_ids: Iterable of true matching IDs.

    Returns:
        F_0.5 score [0.0, 1.0] for this entity.
    """
    pred_set = set(predicted_ids)
    true_set = set(true_ids)

    # Singleton case (no true matches)
    if not true_set:
        return 1.0 if not pred_set else 0.0

    # Non-singleton case but model predicted no matches
    if not pred_set:
        return 0.0

    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0

    precision = float(tp) / float(len(pred_set))
    recall = float(tp) / float(len(true_set))

    return calculate_f_beta(precision, recall, beta=0.5)


def calculate_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> float:
    """Calculate macro-averaged F_0.5 across all Source 1 entities in ground truth.

    Args:
        predictions: Mapping of source1_entity_id -> list of predicted IDs.
        ground_truth: Mapping of source1_entity_id -> list of true matching IDs.

    Returns:
        Macro-averaged F_0.5 score.
    """
    if not ground_truth:
        return 0.0

    total_score = 0.0
    for s1_id, true_matches in ground_truth.items():
        preds = predictions.get(s1_id, [])
        score = score_single_entity_f05(preds, true_matches)
        total_score += score

    return total_score / float(len(ground_truth))


def select_threshold(
    candidate_scores: List[Tuple[str, str, float]],
    ground_truth: Dict[str, List[str]],
    threshold_candidates: Optional[List[float]] = None,
    beta: float = 0.5,
) -> Tuple[float, float]:
    """Find the probability threshold that maximizes validation macro F_0.5.

    Args:
        candidate_scores: List of tuples (source1_id, target_id, score).
        ground_truth: Validation ground truth mapping s1_id -> [matched_ids].
        threshold_candidates: List of candidate thresholds to evaluate (defaults to 0.1 to 0.9).
        beta: F-beta parameter (default 0.5).

    Returns:
        Tuple of (best_threshold, best_macro_f05).
    """
    if threshold_candidates is None:
        threshold_candidates = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]

    best_thresh = 0.5
    best_score = -1.0

    for thresh in threshold_candidates:
        preds: Dict[str, List[str]] = {s1_id: [] for s1_id in ground_truth.keys()}
        for s1_id, target_id, score in candidate_scores:
            if score >= thresh and s1_id in preds:
                preds[s1_id].append(target_id)

        macro_f = calculate_macro_f05(preds, ground_truth)
        if macro_f > best_score:
            best_score = macro_f
            best_thresh = thresh

    return best_thresh, best_score
