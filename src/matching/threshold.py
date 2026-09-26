"""
Amazon ML Entity Resolution Challenge 2026 - Threshold Selection & F0.5 Evaluation

This module implements macro-averaged F0.5 evaluation, singleton/no-match handling,
and threshold grid search to optimize precision-heavy entity resolution decisions.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def compute_entity_f05(
    true_ids: Set[str], pred_ids: Set[str]
) -> Tuple[float, float, float]:
    """Calculate F0.5, Precision, and Recall for a single Source 1 entity.

    Handles edge cases and singletons:
    - true=empty, pred=empty -> 1.0 (correct singleton)
    - true=empty, pred=non-empty -> 0.0 (false merge on singleton)
    - true=non-empty, pred=empty -> 0.0 (missed match)
    """
    n_true = len(true_ids)
    n_pred = len(pred_ids)

    # Singleton cases
    if n_true == 0:
        if n_pred == 0:
            return 1.0, 1.0, 1.0  # Correct singleton identification
        return 0.0, 0.0, 0.0  # False merge on singleton

    if n_pred == 0:
        return 0.0, 0.0, 0.0  # Missed all matches

    # Non-empty matches
    tp = len(true_ids & pred_ids)
    precision = tp / n_pred
    recall = tp / n_true

    denom = (0.25 * precision) + recall
    if denom == 0.0 or (precision == 0.0 and recall == 0.0):
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return f05, precision, recall


def evaluate_predictions(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Sequence[str],
) -> Dict[str, float]:
    """Calculate macro-averaged F0.5, precision, recall, and singleton accuracy.

    all_s1_ids defines the exact evaluation set of Source 1 entities.
    """
    total_entities = len(all_s1_ids)
    if total_entities == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "singleton_count": 0,
            "total_entities": 0,
        }

    sum_f05 = 0.0
    sum_prec = 0.0
    sum_rec = 0.0
    singleton_count = 0
    correct_singletons = 0
    non_singleton_count = 0
    sum_non_singleton_f05 = 0.0

    for s1 in all_s1_ids:
        true_set = ground_truth.get(s1, set())
        pred_set = predictions.get(s1, set())

        f05, prec, rec = compute_entity_f05(true_set, pred_set)
        sum_f05 += f05
        sum_prec += prec
        sum_rec += rec

        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                correct_singletons += 1
        else:
            non_singleton_count += 1
            sum_non_singleton_f05 += f05

    singleton_acc = (
        correct_singletons / singleton_count if singleton_count > 0 else 1.0
    )
    non_singleton_f05 = (
        sum_non_singleton_f05 / non_singleton_count if non_singleton_count > 0 else 0.0
    )

    return {
        "macro_f05": sum_f05 / total_entities,
        "macro_precision": sum_prec / total_entities,
        "macro_recall": sum_rec / total_entities,
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_acc,
        "non_singleton_count": non_singleton_count,
        "non_singleton_f05": non_singleton_f05,
        "total_entities": total_entities,
    }


def find_optimal_threshold(
    candidate_scores: Sequence[Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Sequence[str],
    min_threshold: float = 0.40,
    max_threshold: float = 0.95,
    step: float = 0.02,
) -> Dict[str, Any]:
    """Find the threshold tau in [min_threshold, max_threshold] maximizing Macro F0.5.

    candidate_scores must contain dicts with:
    - 'source1_entity_id' (or 's1')
    - 'source2_entity_id' (or 'candidate_id' or 's2')
    - 'score' (probability in [0.0, 1.0])
    """
    # Group candidates by Source 1 entity
    by_s1: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in all_s1_ids}
    for item in candidate_scores:
        s1 = item.get("source1_entity_id") or item.get("s1")
        cand = (
            item.get("source2_entity_id")
            or item.get("candidate_id")
            or item.get("candidate_entity_id")
            or item.get("s2")
        )
        score = float(item["score"])
        if s1 in by_s1 and cand:
            by_s1[s1].append((cand, score))

    best_threshold = min_threshold
    best_f05 = -1.0
    best_metrics: Dict[str, float] = {}
    grid_results = []

    # Grid search
    curr = min_threshold
    while curr <= max_threshold + 1e-9:
        tau = round(curr, 4)
        # Generate predictions under threshold tau
        preds: Dict[str, Set[str]] = {}
        for s1, cands in by_s1.items():
            matches = {cand for cand, score in cands if score >= tau}
            preds[s1] = matches

        metrics = evaluate_predictions(preds, ground_truth, all_s1_ids)
        metrics["threshold"] = tau
        grid_results.append(metrics)

        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_threshold = tau
            best_metrics = metrics

        curr += step

    return {
        "best_threshold": best_threshold,
        "best_macro_f05": best_f05,
        "best_metrics": best_metrics,
        "grid_results": grid_results,
    }


def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate F_beta score given precision and recall."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denom


def score_single_entity_f05(
    predicted_ids: Sequence[str], true_ids: Sequence[str]
) -> float:
    """Score a single entity using F0.5 with singleton handling."""
    f05, _, _ = compute_entity_f05(set(true_ids), set(predicted_ids))
    return f05


def calculate_macro_f05(
    predictions: Dict[str, Sequence[str]], ground_truth: Dict[str, Sequence[str]]
) -> float:
    """Calculate macro-averaged F0.5 across entities in ground truth."""
    all_s1 = sorted(list(ground_truth.keys()))
    pred_sets = {s1: set(predictions.get(s1, [])) for s1 in all_s1}
    gt_sets = {s1: set(ground_truth.get(s1, [])) for s1 in all_s1}
    res = evaluate_predictions(pred_sets, gt_sets, all_s1)
    return res["macro_f05"]


def select_threshold(
    candidate_scores: Sequence[Any],
    ground_truth: Dict[str, Sequence[str]],
    threshold_candidates: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Select threshold from candidates that maximizes macro F0.5."""
    if threshold_candidates is None:
        threshold_candidates = [round(0.05 * i, 2) for i in range(1, 20)]

    all_s1 = sorted(list(ground_truth.keys()))
    gt_sets = {s1: set(ground_truth.get(s1, [])) for s1 in all_s1}

    by_s1: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in all_s1}
    for item in candidate_scores:
        if isinstance(item, (tuple, list)):
            s1, target, score = item[0], item[1], float(item[2])
        else:
            s1 = item.get("source1_entity_id") or item.get("s1")
            target = item.get("source2_entity_id") or item.get("s2")
            score = float(item.get("score", 0.0))
        if s1 in by_s1 and target:
            by_s1[s1].append((target, score))

    best_thresh = threshold_candidates[0]
    best_score = -1.0

    for thresh in threshold_candidates:
        preds = {s1: {cand for cand, sc in cands if sc >= thresh} for s1, cands in by_s1.items()}
        res = evaluate_predictions(preds, gt_sets, all_s1)
        if res["macro_f05"] > best_score:
            best_score = res["macro_f05"]
            best_thresh = thresh

    return best_thresh, best_score

