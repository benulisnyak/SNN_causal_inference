"""Feature engineering for turning learned weight samples into edge-level rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


_FEATURE_NAMES = ("mean", "std", "min", "max", "median")


def build_directed_edge_features(
    learned_matrices: np.ndarray,
    *,
    include_self_edges: bool = False,
) -> pd.DataFrame:
    """Summarize repeated learned connectivity matrices for every directed edge.

    Parameters
    ----------
    learned_matrices:
        Array with shape ``[samples, source, target]``. Each sample is the
        activity-matching SNN's learned weight matrix for one burst.

    Returns
    -------
    pandas.DataFrame
        One row per candidate directed edge with the summary statistics used by
        the gradient-boosted classifier in the research pipeline.
    """
    matrices = np.asarray(learned_matrices, dtype=float)
    if matrices.ndim != 3:
        raise ValueError("learned_matrices must have shape [samples, N, N]")
    if matrices.shape[1] != matrices.shape[2]:
        raise ValueError("learned connectivity matrices must be square")
    if matrices.shape[0] == 0:
        raise ValueError("at least one learned matrix is required")

    n_nodes = matrices.shape[1]
    rows: list[dict[str, float | int]] = []

    for source in range(n_nodes):
        for target in range(n_nodes):
            if not include_self_edges and source == target:
                continue
            values = matrices[:, source, target]
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }
            )

    frame = pd.DataFrame(rows)
    return frame[["source", "target", *_FEATURE_NAMES]]
