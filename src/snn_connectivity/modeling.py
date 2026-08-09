"""Reusable supervised-learning helpers for edge classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline


DEFAULT_EDGE_FEATURES = ("mean", "min", "max", "std", "median")


def grouped_network_split(
    frame: pd.DataFrame,
    *,
    group_column: str = "network_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split edge rows while keeping entire network instances together.

    This is a small but important modeling detail. Thousands of candidate edges
    can come from one simulated network; treating those rows as independent in a
    random split would leak network-specific information into the test set.
    """
    if group_column not in frame.columns:
        raise KeyError(f"Missing grouping column: {group_column}")
    if frame[group_column].nunique() < 2:
        raise ValueError("At least two network groups are required for a grouped split")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = next(
        splitter.split(frame, groups=frame[group_column].to_numpy())
    )
    return train_index, test_index


def make_edge_classifier(
    *,
    random_state: int = 42,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    subsample: float = 0.8,
) -> Pipeline:
    """Build the gradient-boosted edge classifier used by the inference stage."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
        ),
    )
