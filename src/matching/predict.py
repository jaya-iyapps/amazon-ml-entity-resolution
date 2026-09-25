"""Inference and score generation for candidate pairs.

Owned by: Person 2 (Matching / ML)

Produces final matching decisions (MATCH vs NO MATCH) and formats output.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple


def predict_scores(
    model: Any,
    X_features: List[List[float]],
) -> List[float]:
    """Generate match probability scores for feature vectors.

    Args:
        model: Fitted scikit-learn compatible classifier.
        X_features: Feature matrix for candidate pairs.

    Returns:
        List of predicted match probabilities [0.0, 1.0].
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_features)
        # Probability of class 1 (match)
        return [float(p[1]) for p in probabilities]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_features)
        return [float(s) for s in scores]
    else:
        preds = model.predict(X_features)
        return [float(p) for p in preds]


def format_matching_results(
    all_source1_ids: Iterable[str],
    predicted_pairs: List[Tuple[str, str, float]],
    threshold: float = 0.5,
) -> Dict[str, List[str]]:
    """Convert candidate prediction scores into final entity match lists.

    Applies the decision threshold. Any pair with score >= threshold is classified
    as a MATCH. Entities with no pairs meeting the threshold become singletons (NO MATCH).

    Args:
        all_source1_ids: Complete set of Source 1 entity IDs in evaluation/test set.
        predicted_pairs: List of tuples (s1_id, candidate_target_id, match_probability).
        threshold: Decision threshold for positive match.

    Returns:
        Dictionary mapping every source1_entity_id -> list of matched target IDs.
    """
    matches_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_source1_ids}

    for s1_id, target_id, score in predicted_pairs:
        if score >= threshold and s1_id in matches_map:
            if target_id not in matches_map[s1_id]:
                matches_map[s1_id].append(target_id)

    return matches_map


def save_matching_results(
    matching_results: Dict[str, List[str]],
    output_path: str,
) -> None:
    """Save final entity matches in the official submission TSV format.

    Format:
        source1_entity_id\\tmatched_entity_ids
        S1-00001\\tS2-00047,S2-00193,S3-00812
        S1-00002\\tS3-00004
        S1-00003\\t

    Args:
        matching_results: Mapping of source1_entity_id -> list of matched IDs.
        output_path: Path where matching_results.tsv will be written.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matches in matching_results.items():
            matches_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{matches_str}\n")
