"""Small metric helpers shared by examples and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Evaluate probabilistic edge predictions with thresholded and ranking metrics."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_probability = np.asarray(y_probability, dtype=float).reshape(-1)
    if y_true.shape != y_probability.shape:
        raise ValueError("y_true and y_probability must have matching shapes")
    if np.unique(y_true).size < 2:
        raise ValueError("ROC/PR AUC require both positive and negative labels")

    y_pred = (y_probability >= threshold).astype(int)
    return BinaryMetrics(
        roc_auc=float(roc_auc_score(y_true, y_probability)),
        pr_auc=float(average_precision_score(y_true, y_probability)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )
