"""
Amazon ML Entity Resolution Challenge 2026 - Model Training & Grouped Validation

This module handles:
1. Grouped train/validation split by source1_entity_id (leakage-free).
2. Pairwise feature generation on candidate pairs.
3. Supervised model training (calibrated probabilities).
4. Validation threshold tuning to maximize macro F0.5.
5. Model serialization.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Support both direct execution and module import
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.matching.features import FEATURE_NAMES, extract_pair_features
    from src.matching.threshold import evaluate_predictions, find_optimal_threshold
else:
    from .features import FEATURE_NAMES, extract_pair_features
    from .threshold import evaluate_predictions, find_optimal_threshold


class MatchingClassifier:
    """Calibrated binary classifier for entity matching with pure-Python implementation

    and optional scikit-learn acceleration when available.
    """

    def __init__(self, l2_reg: float = 1e-4, learning_rate: float = 0.05, n_epochs: int = 25):
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights: List[float] = [0.0] * len(FEATURE_NAMES)
        self.bias: float = 0.0
        self.means: List[float] = [0.0] * len(FEATURE_NAMES)
        self.stds: List[float] = [1.0] * len(FEATURE_NAMES)
        self.fitted: bool = False
        self._sklearn_model = None

    def _fit_scaler(self, X: Sequence[Sequence[float]]):
        n = len(X)
        if n == 0:
            return
        p = len(X[0])
        means = [0.0] * p
        for row in X:
            for j in range(p):
                means[j] += row[j]
        means = [m / n for m in means]

        vars_ = [0.0] * p
        for row in X:
            for j in range(p):
                diff = row[j] - means[j]
                vars_[j] += diff * diff
        stds = [math.sqrt(v / n) if v > 1e-8 else 1.0 for v in vars_]

        self.means = means
        self.stds = stds

    def _transform_row(self, row: Sequence[float]) -> List[float]:
        return [(row[j] - self.means[j]) / self.stds[j] for j in range(len(row))]

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]):
        """Train the classifier using mini-batch SGD with Adam-like momentum."""
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("Cannot train on 0 samples.")

        # Try sklearn if installed
        try:
            from sklearn.linear_model import LogisticRegression

            self._fit_scaler(X)
            X_scaled = [self._transform_row(r) for r in X]
            clf = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
            clf.fit(X_scaled, y)
            self._sklearn_model = clf
            self.weights = [float(w) for w in clf.coef_[0]]
            self.bias = float(clf.intercept_[0])
            self.fitted = True
            return self
        except ImportError:
            pass

        self._fit_scaler(X)
        X_scaled = [self._transform_row(r) for r in X]
        p = len(self.means)

        # Class imbalance weighting
        pos_count = sum(y)
        neg_count = n_samples - pos_count
        pos_weight = (n_samples / (2.0 * max(pos_count, 1)))
        neg_weight = (n_samples / (2.0 * max(neg_count, 1)))

        weights = [0.0] * p
        bias = 0.0
        m_w = [0.0] * p
        v_w = [0.0] * p
        m_b = 0.0
        v_b = 0.0
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        # Prior initialization for bias
        if pos_count > 0 and neg_count > 0:
            bias = math.log(pos_count / neg_count)

        indices = list(range(n_samples))
        rng = random.Random(42)
        step = 0

        for epoch in range(self.n_epochs):
            rng.shuffle(indices)
            for idx in indices:
                step += 1
                row = X_scaled[idx]
                target = float(y[idx])
                w_sample = pos_weight if target == 1.0 else neg_weight

                # Forward: sigmoid(z)
                z = bias + sum(weights[j] * row[j] for j in range(p))
                z = max(-35.0, min(35.0, z))
                prob = 1.0 / (1.0 + math.exp(-z))

                err = (prob - target) * w_sample

                # Update bias
                grad_b = err
                m_b = beta1 * m_b + (1.0 - beta1) * grad_b
                v_b = beta2 * v_b + (1.0 - beta2) * (grad_b**2)
                m_b_hat = m_b / (1.0 - (beta1**step))
                v_b_hat = v_b / (1.0 - (beta2**step))
                bias -= (self.learning_rate * m_b_hat) / (math.sqrt(v_b_hat) + eps)

                # Update weights
                for j in range(p):
                    grad_w = err * row[j] + self.l2_reg * weights[j]
                    m_w[j] = beta1 * m_w[j] + (1.0 - beta1) * grad_w
                    v_w[j] = beta2 * v_w[j] + (1.0 - beta2) * (grad_w**2)
                    m_w_hat = m_w[j] / (1.0 - (beta1**step))
                    v_w_hat = v_w[j] / (1.0 - (beta2**step))
                    weights[j] -= (self.learning_rate * m_w_hat) / (math.sqrt(v_w_hat) + eps)

        self.weights = weights
        self.bias = bias
        self.fitted = True
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        """Predict calibrated match probability for each feature vector."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet.")

        if self._sklearn_model is not None:
            X_scaled = [self._transform_row(r) for r in X]
            probs = self._sklearn_model.predict_proba(X_scaled)
            return [float(p[1]) for p in probs]

        results = []
        p = len(self.means)
        for row in X:
            scaled = self._transform_row(row)
            z = self.bias + sum(self.weights[j] * scaled[j] for j in range(p))
            z = max(-35.0, min(35.0, z))
            prob = 1.0 / (1.0 + math.exp(-z))
            results.append(prob)
        return results

    def save(self, file_path: str):
        """Save model configuration and weights to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "stds": self.stds,
            "feature_names": FEATURE_NAMES,
            "fitted": self.fitted,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "MatchingClassifier":
        """Load model configuration and weights from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls()
        model.weights = data["weights"]
        model.bias = data["bias"]
        model.means = data["means"]
        model.stds = data["stds"]
        model.fitted = data["fitted"]
        return model


def load_records_tsv(file_path: str) -> Dict[str, Dict[str, str]]:
    """Load a TSV of entities into memory mapping entity_id to field dictionary."""
    records: Dict[str, Dict[str, str]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        id_idx = header.index("entity_id")
        name_idx = header.index("business_name")
        addr_idx = header.index("business_address")
        cntry_idx = header.index("country")

        for row in reader:
            if not row:
                continue
            eid = row[id_idx]
            records[eid] = {
                "entity_id": eid,
                "business_name": row[name_idx] if len(row) > name_idx else "",
                "business_address": row[addr_idx] if len(row) > addr_idx else "",
                "country": row[cntry_idx] if len(row) > cntry_idx else "",
            }
    return records


def load_ground_truth_tsv(file_path: str) -> Dict[str, Set[str]]:
    """Load ground truth TSV mapping source1_entity_id to set of matched entity IDs."""
    gt: Dict[str, Set[str]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader)
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            matched_str = row[1].strip() if len(row) > 1 else ""
            if matched_str:
                mids = {m.strip() for m in matched_str.split(",") if m.strip()}
                gt[s1_id] = mids
            else:
                gt[s1_id] = set()
    return gt


def grouped_train_val_split(
    pairs: Sequence[Dict[str, Any]],
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str], Set[str]]:
    """Split candidate pairs into train and val folds strictly grouped by source1_entity_id.

    Guaranteeing 0% entity leakage across splits.
    """
    unique_s1 = sorted(list({p["source1_entity_id"] for p in pairs}))
    rng = random.Random(seed)
    rng.shuffle(unique_s1)

    val_count = max(1, int(len(unique_s1) * val_ratio))
    val_s1 = set(unique_s1[:val_count])
    train_s1 = set(unique_s1[val_count:])

    train_pairs = [p for p in pairs if p["source1_entity_id"] in train_s1]
    val_pairs = [p for p in pairs if p["source1_entity_id"] in val_s1]

    # Verify 0% leakage
    assert len(train_s1 & val_s1) == 0, "Leakage detected between train and val S1 entities!"
    return train_pairs, val_pairs, train_s1, val_s1


def build_candidate_pairs_from_sample(
    s1_dict: Dict[str, Dict[str, str]],
    s2_dict: Dict[str, Dict[str, str]],
    s3_dict: Dict[str, Dict[str, str]],
    ground_truth: Dict[str, Set[str]],
    negatives_per_positive: int = 4,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build candidate pairs containing 100% of true ground-truth matches

    plus controlled hard and random negatives from the sample records.
    """
    pairs: List[Dict[str, Any]] = []
    all_s2_ids = list(s2_dict.keys())
    all_s3_ids = list(s3_dict.keys())
    rng = random.Random(seed)

    for s1_id in s1_dict.keys():
        true_matches = ground_truth.get(s1_id, set())

        # Add all positives
        for match_id in true_matches:
            src = "S2" if match_id.startswith("S2-") else "S3"
            pairs.append(
                {
                    "source1_entity_id": s1_id,
                    "source2_entity_id": match_id,
                    "source": src,
                    "label": 1,
                }
            )

        # Add negative candidates
        n_neg = max(2, len(true_matches) * negatives_per_positive)
        sampled_neg = 0
        attempts = 0
        while sampled_neg < n_neg and attempts < n_neg * 5:
            attempts += 1
            if rng.random() < 0.5 and all_s2_ids:
                cand_id = rng.choice(all_s2_ids)
                src = "S2"
            elif all_s3_ids:
                cand_id = rng.choice(all_s3_ids)
                src = "S3"
            else:
                continue

            if cand_id not in true_matches:
                pairs.append(
                    {
                        "source1_entity_id": s1_id,
                        "source2_entity_id": cand_id,
                        "source": src,
                        "label": 0,
                    }
                )
                sampled_neg += 1

    return pairs


def train_matching_pipeline(
    sample_dir: str = "dataset/sample",
    candidates_path: Optional[str] = "output/candidate_pairs.tsv",
    model_output_path: str = "output/matching_model.json",
    sample_entities: Optional[int] = None,
    val_ratio: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end model training, grouped validation, and threshold selection."""
    print("=" * 65)
    print("PERSON 2 - MATCHING MODEL TRAINING & EVALUATION PIPELINE")
    print(f"Data directory   : {sample_dir}")
    print(f"Candidates file  : {candidates_path}")
    print(f"Model target     : {model_output_path}")
    print("=" * 65)

    s1_path = os.path.join(sample_dir, "train_source1.tsv")
    s2_path = os.path.join(sample_dir, "train_source2.tsv")
    s3_path = os.path.join(sample_dir, "train_source3.tsv")
    gt_path = os.path.join(sample_dir, "train_ground_truth.tsv")

    print("[1/5] Loading records into memory...")
    s1_dict = load_records_tsv(s1_path)
    s2_dict = load_records_tsv(s2_path)
    s3_dict = load_records_tsv(s3_path)
    combined_cands = {**s2_dict, **s3_dict}
    ground_truth = load_ground_truth_tsv(gt_path)

    if sample_entities and sample_entities < len(s1_dict):
        print(f"      Subsampling {sample_entities:,} entities for fast training/validation...")
        rng = random.Random(seed)
        sub_s1 = set(rng.sample(list(s1_dict.keys()), sample_entities))
        s1_dict = {k: v for k, v in s1_dict.items() if k in sub_s1}
        ground_truth = {k: v for k, v in ground_truth.items() if k in sub_s1}

    print(f"      Source 1: {len(s1_dict):,} entities | Source 2: {len(s2_dict):,} | Source 3: {len(s3_dict):,}")

    print("[2/5] Constructing candidate pairs with labels...")
    pairs: List[Dict[str, Any]] = []
    if candidates_path and os.path.isfile(candidates_path):
        print(f"      Loading real blocking candidate pairs from: {candidates_path}")
        with open(candidates_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            for row in reader:
                if not row:
                    continue
                s1_id = row[0].strip()
                if s1_id not in s1_dict:
                    continue
                cands_str = row[1].strip() if len(row) > 1 else ""
                if not cands_str:
                    continue
                true_matches = ground_truth.get(s1_id, set())
                for cid in cands_str.split(","):
                    cid = cid.strip()
                    if not cid or cid not in combined_cands:
                        continue
                    src = "S2" if cid.startswith("S2-") else "S3"
                    label = 1 if cid in true_matches else 0
                    pairs.append({
                        "source1_entity_id": s1_id,
                        "source2_entity_id": cid,
                        "source": src,
                        "label": label,
                    })
    else:
        print("      No candidates file found; generating synthetic candidate pairs...")
        pairs = build_candidate_pairs_from_sample(
            s1_dict, s2_dict, s3_dict, ground_truth, negatives_per_positive=4, seed=seed
        )

    n_pos = sum(1 for p in pairs if p["label"] == 1)
    n_neg = sum(1 for p in pairs if p["label"] == 0)
    print(f"      Total candidate pairs: {len(pairs):,} ({n_pos:,} positive, {n_neg:,} negative)")

    print("[3/5] Performing Grouped Train/Validation Split (group = source1_entity_id)...")
    all_s1_list = sorted(list(s1_dict.keys()))
    rng = random.Random(seed)
    shuffled_s1 = all_s1_list.copy()
    rng.shuffle(shuffled_s1)
    val_count = max(1, int(len(shuffled_s1) * val_ratio))
    val_s1 = set(shuffled_s1[:val_count])
    train_s1 = set(shuffled_s1[val_count:])

    # Confirm 0% leakage
    assert len(train_s1 & val_s1) == 0, "Leakage detected between train and val S1 entities!"

    train_pairs = [p for p in pairs if p["source1_entity_id"] in train_s1]
    val_pairs = [p for p in pairs if p["source1_entity_id"] in val_s1]
    print(f"      Train: {len(train_s1):,} S1 entities ({len(train_pairs):,} pairs)")
    print(f"      Val  : {len(val_s1):,} S1 entities ({len(val_pairs):,} pairs)")

    # Candidate recall before matching on validation set
    val_gt = {s1: ground_truth.get(s1, set()) for s1 in val_s1}
    total_val_true = sum(len(val_gt[s1]) for s1 in val_s1)
    val_cands_by_s1 = {s1: set() for s1 in val_s1}
    for p in val_pairs:
        val_cands_by_s1[p["source1_entity_id"]].add(p["source2_entity_id"])
    retrieved_in_blocking = sum(len(val_gt[s1] & val_cands_by_s1[s1]) for s1 in val_s1)
    blocking_recall = (retrieved_in_blocking / total_val_true) if total_val_true > 0 else 1.0
    print(f"      Validation Blocking Candidate Recall: {blocking_recall:.4%} ({retrieved_in_blocking:,}/{total_val_true:,})")

    print("[4/5] Extracting pairwise features for training and validation...")
    X_train = [
        extract_pair_features(s1_dict[p["source1_entity_id"]], combined_cands[p["source2_entity_id"]])
        for p in train_pairs
    ]
    y_train = [p["label"] for p in train_pairs]

    X_val = [
        extract_pair_features(s1_dict[p["source1_entity_id"]], combined_cands[p["source2_entity_id"]])
        for p in val_pairs
    ]
    y_val = [p["label"] for p in val_pairs]

    print("      Training MatchingClassifier model...")
    model = MatchingClassifier(l2_reg=1e-4, learning_rate=0.08, n_epochs=20)
    model.fit(X_train, y_train)

    print("[5/5] Scoring validation candidates and tuning F0.5 decision threshold...")
    val_probs = model.predict_proba(X_val)

    val_candidate_scores = [
        {
            "source1_entity_id": p["source1_entity_id"],
            "source2_entity_id": p["source2_entity_id"],
            "score": val_probs[i],
            "label": p["label"],
            "source": p["source"],
        }
        for i, p in enumerate(val_pairs)
    ]

    threshold_results = find_optimal_threshold(
        val_candidate_scores, val_gt, sorted(list(val_s1)), min_threshold=0.40, max_threshold=0.98, step=0.02
    )

    best_tau = threshold_results["best_threshold"]
    best_metrics = threshold_results["best_metrics"]

    # Detailed statistics at best threshold
    preds_at_tau = {s1: set() for s1 in val_s1}
    for item in val_candidate_scores:
        if item["score"] >= best_tau:
            preds_at_tau[item["source1_entity_id"]].add(item["source2_entity_id"])

    tp = sum(len(preds_at_tau[s1] & val_gt[s1]) for s1 in val_s1)
    fp = sum(len(preds_at_tau[s1] - val_gt[s1]) for s1 in val_s1)
    fn = sum(len(val_gt[s1] - preds_at_tau[s1]) for s1 in val_s1)
    pair_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pair_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pair_f05 = (1.25 * pair_prec * pair_rec) / (0.25 * pair_prec + pair_rec) if (0.25 * pair_prec + pair_rec) > 0 else 0.0

    print("\n" + "=" * 65)
    print("VALIDATION PERFORMANCE (Macro F0.5 & Pair Metrics)")
    print("=" * 65)
    print(f"  • Candidate Recall Before Matching : {blocking_recall:.4%}")
    print(f"  • Optimal Decision Threshold (tau) : {best_tau:.2f}")
    print(f"  • Macro F0.5 Score                 : {best_metrics['macro_f05']:.4f}")
    print(f"  • Macro Precision                  : {best_metrics['macro_precision']:.4f}")
    print(f"  • Macro Recall                     : {best_metrics['macro_recall']:.4f}")
    print(f"  • Singleton Accuracy               : {best_metrics['singleton_accuracy']:.2%} ({best_metrics['singleton_count']} singletons)")
    print(f"  • Non-Singleton F0.5               : {best_metrics['non_singleton_f05']:.4f}")
    print(f"  • Pair-level TP / FP / FN          : {tp:,} / {fp:,} / {fn:,}")
    print(f"  • Pair-level Precision             : {pair_prec:.4f}")
    print(f"  • Pair-level Recall                : {pair_rec:.4f}")
    print(f"  • Pair-level F0.5                  : {pair_f05:.4f}")

    print("\nPerformance by Country:")
    for country in ["US", "India"]:
        c_s1 = [s1 for s1 in val_s1 if s1_dict[s1].get("country", "").upper() == country.upper()]
        if not c_s1:
            continue
        c_gt = {s1: val_gt[s1] for s1 in c_s1}
        c_preds = {s1: preds_at_tau[s1] for s1 in c_s1}
        c_res = evaluate_predictions(c_preds, c_gt, c_s1)
        c_tp = sum(len(c_preds[s1] & c_gt[s1]) for s1 in c_s1)
        c_fp = sum(len(c_preds[s1] - c_gt[s1]) for s1 in c_s1)
        c_fn = sum(len(c_gt[s1] - c_preds[s1]) for s1 in c_s1)
        print(f"  • {country:5s} ({len(c_s1):,} entities): Macro F0.5={c_res['macro_f05']:.4f}, Prec={c_res['macro_precision']:.4f}, Rec={c_res['macro_recall']:.4f} (TP={c_tp}, FP={c_fp}, FN={c_fn})")

    print("\nPerformance by Target Partition:")
    for src in ["S2", "S3"]:
        src_gt = {s1: {cid for cid in val_gt[s1] if cid.startswith(src + "-")} for s1 in val_s1}
        src_preds = {s1: {cid for cid in preds_at_tau[s1] if cid.startswith(src + "-")} for s1 in val_s1}
        src_res = evaluate_predictions(src_preds, src_gt, sorted(list(val_s1)))
        src_tp = sum(len(src_preds[s1] & src_gt[s1]) for s1 in val_s1)
        src_fp = sum(len(src_preds[s1] - src_gt[s1]) for s1 in val_s1)
        src_fn = sum(len(src_gt[s1] - src_preds[s1]) for s1 in val_s1)
        print(f"  • Source {src}: Macro F0.5={src_res['macro_f05']:.4f}, Prec={src_res['macro_precision']:.4f}, Rec={src_res['macro_recall']:.4f} (TP={src_tp}, FP={src_fp}, FN={src_fn})")
    print("=" * 65)

    # Save model
    model.save(model_output_path)
    print(f"\nSaved trained matching model to: {model_output_path}")

    return {
        "model": model,
        "best_threshold": best_tau,
        "metrics": best_metrics,
        "blocking_recall": blocking_recall,
        "pair_metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": pair_prec,
            "recall": pair_rec,
            "f05": pair_f05,
        },
        "model_path": model_output_path,
    }


def train_model(
    X_train: List[List[float]],
    y_train: List[int],
    model_type: str = "logistic_regression",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """Train a binary classifier for entity matching. Backwards compatibility interface."""
    clf = MatchingClassifier()
    clf.fit(X_train, y_train)
    return clf


def main():
    parser = argparse.ArgumentParser(description="Train Entity Matching model with grouped validation.")
    parser.add_argument("--data-dir", default="dataset/sample", help="Directory with TSV dataset.")
    parser.add_argument("--candidates-path", default="output/candidate_pairs.tsv", help="Person 3 candidate pairs TSV.")
    parser.add_argument("--model-out", default="output/matching_model.json", help="Path to save model.")
    parser.add_argument("--sample-entities", type=int, default=None, help="Number of S1 entities to use (None for all).")
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Validation ratio for grouped split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_matching_pipeline(
        sample_dir=args.data_dir,
        candidates_path=args.candidates_path,
        model_output_path=args.model_out,
        sample_entities=args.sample_entities,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
