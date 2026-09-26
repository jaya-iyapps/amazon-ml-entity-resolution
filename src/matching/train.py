"""Model training interface for entity matching classifier.

Owned by: Person 2 (Matching / ML)

Person 2 can develop using small sampled candidate pairs and does not need
to load the entire dataset.
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def train_model(
    X_train: List[List[float]],
    y_train: List[int],
    model_type: str = "logistic_regression",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """Train a binary classifier to predict match probability for candidate pairs.

    Args:
        X_train: Feature matrix where each row represents pairwise similarity features.
        y_train: Target labels (1 for match, 0 for non-match).
        model_type: Classifier architecture name (e.g. 'logistic_regression', 'lightgbm').
        hyperparameters: Dictionary of model parameters.

    Returns:
        Fitted model object.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required to train the matching model.")

    params = hyperparameters or {}

    if model_type == "logistic_regression":
        # Logistic Regression default with class balance to handle imbalance
        clf = LogisticRegression(
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 1000),
            random_state=params.get("random_state", 42),
        )
    else:
        # Placeholder for tree-based models (e.g., LightGBM / XGBoost / RandomForest)
        clf = LogisticRegression(class_weight="balanced", random_state=42)

    clf.fit(X_train, y_train)
    return clf
