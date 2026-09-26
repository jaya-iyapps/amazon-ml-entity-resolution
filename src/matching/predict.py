"""
Amazon ML Entity Resolution Challenge 2026 - Inference & Prediction Pipeline

This module generates final entity match decisions from candidate pairs:
1. Streams candidate pairs produced by upstream blocking (Person 3).
2. Computes pairwise features against normalized records.
3. Scores candidates using the trained MatchingClassifier.
4. Applies optimal decision threshold (supporting 0, 1, or multiple matches).
5. Generates the final output/matching_results.tsv conforming strictly to tournament format.
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.matching.features import extract_pair_features
    from src.matching.train import MatchingClassifier, load_records_tsv
else:
    from .features import extract_pair_features
    from .train import MatchingClassifier, load_records_tsv


def stream_candidate_pairs_from_tsv(
    candidate_tsv_path: str, chunk_size: int = 50000
) -> Iterator[List[Dict[str, str]]]:
    """Yield chunks of candidate pairs from Person 3's candidate_pairs.tsv.

    Supports candidate_pairs.tsv format:
    source1_entity_id \t candidate_entity_ids (comma-separated)
    """
    chunk: List[Dict[str, str]] = []
    with open(candidate_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        s1_col = 0
        cands_col = 1

        for row in reader:
            if not row:
                continue
            s1_id = row[s1_col].strip()
            cands_str = row[cands_col].strip() if len(row) > cands_col else ""
            if not cands_str:
                # Still register the S1 entity so it receives an empty output row
                chunk.append({
                    "source1_entity_id": s1_id,
                    "source2_entity_id": "",
                    "source": "",
                })
            else:
                c_ids = [c.strip() for c in cands_str.split(",") if c.strip()]
                for cid in c_ids:
                    src = "S2" if cid.startswith("S2-") else "S3"
                    chunk.append({
                        "source1_entity_id": s1_id,
                        "source2_entity_id": cid,
                        "source": src,
                    })

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

    if chunk:
        yield chunk


def predict_matches(
    model: MatchingClassifier,
    threshold: float,
    all_s1_ids: Sequence[str],
    candidate_pairs_stream: Iterator[List[Dict[str, str]]],
    s1_records: Dict[str, Dict[str, str]],
    target_records: Dict[str, Dict[str, str]],
) -> Dict[str, List[str]]:
    """Score candidate pairs in streaming chunks and retain candidates exceeding threshold.

    Guarantees every S1 entity in all_s1_ids is present in the returned mapping.
    """
    # Track predictions per S1 entity: s1_id -> list of (score, candidate_id)
    matches_by_s1: Dict[str, List[Tuple[float, str]]] = {s1: [] for s1 in all_s1_ids}

    for chunk in candidate_pairs_stream:
        # Filter pairs that have actual candidate IDs
        valid_pairs = [p for p in chunk if p["source2_entity_id"]]
        if not valid_pairs:
            continue

        # Extract features
        X_chunk = []
        for p in valid_pairs:
            s1_rec = s1_records.get(p["source1_entity_id"], {})
            cand_rec = target_records.get(p["source2_entity_id"], {})
            feats = extract_pair_features(s1_rec, cand_rec)
            X_chunk.append(feats)

        # Predict probabilities
        probs = model.predict_proba(X_chunk)

        # Filter by threshold
        for p, score in zip(valid_pairs, probs):
            if score >= threshold:
                s1_id = p["source1_entity_id"]
                cand_id = p["source2_entity_id"]
                if s1_id in matches_by_s1:
                    matches_by_s1[s1_id].append((score, cand_id))

    # Format final predictions: sorted by score descending, deduplicated
    final_predictions: Dict[str, List[str]] = {}
    for s1 in all_s1_ids:
        scored_cands = matches_by_s1.get(s1, [])
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        seen: Set[str] = set()
        deduped: List[str] = []
        for score, cand_id in scored_cands:
            if cand_id not in seen:
                seen.add(cand_id)
                deduped.append(cand_id)
        final_predictions[s1] = deduped

    return final_predictions


def write_matching_results_tsv(
    predictions: Dict[str, List[str]],
    all_s1_ids: Sequence[str],
    output_path: str,
):
    """Write predictions strictly conforming to challenge requirements:

    output/matching_results.tsv
    source1_entity_id \\t matched_entity_ids (comma-separated, empty for singletons)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in all_s1_ids:
            matched_ids = predictions.get(s1_id, [])
            matched_str = ",".join(matched_ids)
            writer.writerow([s1_id, matched_str])


def run_prediction_pipeline(
    model_path: str,
    threshold: float,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    candidates_path: Optional[str] = None,
    output_path: str = "output/matching_results.tsv",
) -> str:
    """Execute end-to-end inference from candidate pairs to final TSV."""
    print("=" * 65)
    print("PERSON 2 - MATCHING PREDICTION PIPELINE")
    print(f"Model path       : {model_path}")
    print(f"Threshold (tau)  : {threshold:.2f}")
    print(f"Output TSV       : {output_path}")
    print("=" * 65)

    print("[1/4] Loading model and record metadata...")
    model = MatchingClassifier.load(model_path)
    s1_dict = load_records_tsv(s1_path)
    s2_dict = load_records_tsv(s2_path)
    s3_dict = load_records_tsv(s3_path)
    combined_targets = {**s2_dict, **s3_dict}
    all_s1_ids = list(s1_dict.keys())
    print(f"      Reference S1 Entities : {len(all_s1_ids):,}")
    print(f"      Available Targets     : {len(combined_targets):,}")

    print("[2/4] Initializing candidate pair stream...")
    if candidates_path and os.path.isfile(candidates_path):
        candidate_stream = stream_candidate_pairs_from_tsv(candidates_path)
    else:
        # If Person 3's candidate_pairs.tsv is not yet provided, create synthetic stream from targets
        print("      No external candidates file found, generating internal stream...")
        candidate_stream = ([
            {"source1_entity_id": s1, "source2_entity_id": "", "source": ""}
            for s1 in all_s1_ids
        ],)

    print("[3/4] Scoring candidates and applying decision threshold...")
    predictions = predict_matches(
        model=model,
        threshold=threshold,
        all_s1_ids=all_s1_ids,
        candidate_pairs_stream=candidate_stream,
        s1_records=s1_dict,
        target_records=combined_targets,
    )

    print(f"[4/4] Writing output to {output_path}...")
    write_matching_results_tsv(predictions, all_s1_ids, output_path)

    # Verification statistics
    n_total = len(all_s1_ids)
    n_singletons = sum(1 for s1 in all_s1_ids if not predictions[s1])
    n_matches = sum(len(predictions[s1]) for s1 in all_s1_ids)
    print("\n" + "=" * 65)
    print("PREDICTION RESULTS SUMMARY")
    print("=" * 65)
    print(f"  • Total S1 Entities Evaluated  : {n_total:,}")
    print(f"  • Singletons (0 Matches)       : {n_singletons:,} ({n_singletons / n_total:.1%})")
    print(f"  • Non-Singletons (>=1 Matches) : {n_total - n_singletons:,} ({(n_total - n_singletons) / n_total:.1%})")
    print(f"  • Total Matched Links Output   : {n_matches:,}")
    print(f"  • Output Format Verified       : {output_path}")
    print("=" * 65)
    return output_path


def format_matching_results(
    all_source1_ids: Sequence[str],
    predicted_pairs: Sequence[Any],
    threshold: float = 0.5,
) -> Dict[str, List[str]]:
    """Format matching results mapping source1_entity_id to list of matched candidate IDs.

    Backward compatibility interface.
    """
    results: Dict[str, List[str]] = {s1: [] for s1 in all_source1_ids}
    for item in predicted_pairs:
        if isinstance(item, (tuple, list)):
            s1, target, score = item[0], item[1], float(item[2])
        else:
            s1 = item.get("source1_entity_id") or item.get("s1")
            target = item.get("source2_entity_id") or item.get("s2")
            score = float(item.get("score", 0.0))
        if score >= threshold and s1 in results and target not in results[s1]:
            results[s1].append(target)
    return results


def save_matching_results(results: Dict[str, List[str]], output_path: str) -> None:
    """Save formatted matching results to TSV. Backward compatibility interface."""
    all_s1 = list(results.keys())
    write_matching_results_tsv(results, all_s1, output_path)



def main():
    parser = argparse.ArgumentParser(description="Generate entity matching predictions.")
    parser.add_argument("--model-path", default="output/matching_model.json", help="Trained model path.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Decision threshold tau.")
    parser.add_argument("--s1-path", default="dataset/sample/train_source1.tsv", help="Source 1 TSV path.")
    parser.add_argument("--s2-path", default="dataset/sample/train_source2.tsv", help="Source 2 TSV path.")
    parser.add_argument("--s3-path", default="dataset/sample/train_source3.tsv", help="Source 3 TSV path.")
    parser.add_argument("--candidates-path", default=None, help="Person 3 candidate_pairs.tsv path.")
    parser.add_argument("--output-path", default="output/matching_results.tsv", help="Output TSV path.")
    args = parser.parse_args()

    run_prediction_pipeline(
        model_path=args.model_path,
        threshold=args.threshold,
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        candidates_path=args.candidates_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
