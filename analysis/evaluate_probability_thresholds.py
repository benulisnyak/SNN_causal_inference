#!/usr/bin/env python3
"""
Batch threshold analysis for saved GBT connection-probability matrices.

Run this script from the main project directory. It expects:

    prob_matrices/
        prob_matrix_N100_p24_CC05_3.npy
        ...

    networks/
        network_N100_p24_CC05_3.yaml
        ...

For every probability matrix, the script sweeps classification thresholds and
calculates graph and binary-classification statistics for nine evaluation
categories that match the accompanying ROC/PR-AUC analysis workflow:

    1. total
    2. undirected
    3. directed
    4. excitatory_total
    5. excitatory_undirected
    6. excitatory_directed
    7. inhibitory_total
    8. inhibitory_undirected
    9. inhibitory_directed

Definitions
-----------
total
    All ordered off-diagonal entries of the directed adjacency matrix.

undirected
    One value per unordered neuron pair. The ground-truth label is positive if
    either direction exists. The prediction score is the mean of the two
    directional probabilities:

        A_undirected = 1[(A + A.T) != 0]
        P_undirected = (P + P.T) / 2

    Only the upper triangle is evaluated.

directed
    Uses the same definition as the reference ROC/PR-AUC script:

        A_directed = 1[(A - A.T) != 0]

    The full ordered off-diagonal probability matrix P supplies the scores.
    Therefore, at a given threshold, the predicted graph statistics for
    ``directed`` are intentionally identical to ``total``; the ground-truth
    labels and resulting confusion metrics differ.

excitatory / inhibitory
    Source-neuron type is inferred from the sign of nonzero outgoing weights in
    the signed YAML adjacency matrix. Positive outgoing weights identify an
    excitatory source neuron; negative outgoing weights identify an inhibitory
    source neuron. Mixed-sign outgoing weights raise an error.

For every category and threshold, the CSV includes:

    - predicted and ground-truth clustering coefficient;
    - predicted and ground-truth connection count;
    - predicted and ground-truth average connections per neuron;
    - predicted and ground-truth connection probability;
    - precision, TPR, FPR;
    - TP, FP, TN, FN;
    - predicted positives, actual positives, and number evaluated.

The original unprefixed output columns are retained as aliases of the ``total``
columns for backwards compatibility.

One CSV is saved immediately after a network has been fully analyzed:

    prob_matrix_analysis/
        thresholding_stats_N100_p24_CC05_3.csv

If that CSV already exists, the matching probability matrix is skipped. CSVs
are written atomically: a temporary file is created first and renamed only
after the full calculation has completed.

Threshold modes
---------------
roc
    Uses the union of ROC thresholds from every evaluation category that
    contains both classes. This ensures that category-specific ROC operating
    points are retained in the shared threshold table.

fixed
    Uses evenly spaced thresholds from 1 to 0, plus +infinity for the empty
    predicted graph.

Dependencies
------------
    numpy, pandas, PyYAML, networkx, scikit-learn
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

try:
    from yaml import CSafeLoader as YamlLoader
except ImportError:  # pragma: no cover - depends on the local PyYAML build
    from yaml import SafeLoader as YamlLoader


# =============================================================================
# Analysis defaults
# =============================================================================

PROJECT_ROOT = Path(".")
PROBABILITY_MATRIX_DIR = PROJECT_ROOT / "prob_matrices"
GROUND_TRUTH_DIR = PROJECT_ROOT / "networks"
OUTPUT_DIR = PROJECT_ROOT / "prob_matrix_analysis"

PROBABILITY_FILE_GLOB = "prob_matrix_*.npy"

# "roc" uses the union of category-specific ROC thresholds.
# "fixed" uses FIXED_THRESHOLDS.
THRESHOLD_MODE = "roc"

# True removes category-specific thresholds that are collinear in ROC space.
# False retains every distinct score threshold and can be substantially slower.
ROC_DROP_INTERMEDIATE = True

# Used only when THRESHOLD_MODE == "fixed". +infinity is added automatically.
FIXED_THRESHOLDS = np.linspace(1.0, 0.0, 101)

# Applies to categories represented as directed matrices. Undirected categories
# always use ordinary undirected NetworkX clustering.
CLUSTERING_MODE = "directed"

# Existing final CSV files are skipped unless --overwrite is supplied.
SKIP_EXISTING = True

# Predictions use probability >= threshold, matching sklearn ROC conventions.
THRESHOLD_INCLUSIVE = True

# Numerical tolerances.
PROBABILITY_TOLERANCE = 1e-7
GROUND_TRUTH_WEIGHT_EPS = 1e-12


# =============================================================================
# Filename and sorting helpers
# =============================================================================

PROBABILITY_FILENAME_RE = re.compile(
    r"^prob_matrix_(?P<stats>N\d+_p\d+_CC\d+)_(?P<number>\d+)\.npy$"
)

NETWORK_STATS_RE = re.compile(
    r"^N(?P<N>\d+)_p(?P<p>\d+)_CC(?P<cc>\d+)$"
)


def natural_sort_key(text: str) -> list[object]:
    """Sort strings naturally so that network 2 precedes network 10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


def parse_probability_matrix_filename(
    path: str | Path,
) -> tuple[str, int]:
    """Parse prob_matrix_N100_p24_CC05_3.npy."""
    path = Path(path)
    match = PROBABILITY_FILENAME_RE.fullmatch(path.name)

    if match is None:
        raise ValueError(
            f"Could not parse probability-matrix filename {path.name!r}. "
            "Expected prob_matrix_N100_p24_CC05_3.npy."
        )

    return match.group("stats"), int(match.group("number"))


def parse_network_statistics(
    stats_name: str,
) -> dict[str, int | float]:
    """Parse N, target p, and target CC from N100_p24_CC05."""
    match = NETWORK_STATS_RE.fullmatch(stats_name)

    if match is None:
        raise ValueError(
            f"Could not parse network statistics {stats_name!r}. "
            "Expected a value such as N100_p24_CC05."
        )

    n_nodes = int(match.group("N"))
    p_code = int(match.group("p"))
    cc_code = int(match.group("cc"))

    return {
        "N_from_filename": n_nodes,
        "p_code": p_code,
        "target_connection_probability": p_code / 100.0,
        "cc_code": cc_code,
        "target_clustering_coefficient": cc_code / 10.0,
    }


def output_path_for_probability_matrix(
    probability_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Build thresholding_stats_<stats>_<number>.csv."""
    stats_name, network_number = parse_probability_matrix_filename(
        probability_path
    )

    return Path(output_dir) / (
        f"thresholding_stats_{stats_name}_{network_number}.csv"
    )


# =============================================================================
# Ground-truth YAML loading and source-neuron typing
# =============================================================================


def first_present(
    mapping: dict,
    keys: Iterable[str],
    default=None,
):
    """Return the first explicitly present, non-None dictionary value."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]

    return default


def load_ground_truth_signed_connectivity_matrix(
    file_path: str | Path,
) -> tuple[np.ndarray, list]:
    """
    Load a YAML network into a signed weighted adjacency matrix.

    Positive outgoing weights identify excitatory source neurons and negative
    outgoing weights identify inhibitory source neurons.
    """
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as file_handle:
        data = yaml.load(file_handle, Loader=YamlLoader)

    if data is None:
        raise ValueError(
            f"Ground-truth YAML file is empty: {file_path}"
        )

    nodes = data.get("nodes", [])

    if isinstance(nodes, dict):
        node_list = []

        for node_id, node_data in nodes.items():
            try:
                parsed_id = int(node_id)
            except (TypeError, ValueError):
                parsed_id = node_id

            node_list.append(
                {
                    "id": parsed_id,
                    **(node_data or {}),
                }
            )

    elif isinstance(nodes, list):
        node_list = nodes

    else:
        raise ValueError(
            f"'nodes' must be a list or dictionary in {file_path}"
        )

    if not node_list:
        raise ValueError(
            f"No nodes were found in {file_path}"
        )

    id_order = [node["id"] for node in node_list]
    id_to_index = {
        node_id: index
        for index, node_id in enumerate(id_order)
    }

    n_nodes = len(id_order)
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)

    for node in node_list:
        source_id = node["id"]
        source_index = id_to_index[source_id]

        if (
            "connections" in node
            and isinstance(node["connections"], list)
        ):
            for connection in node["connections"]:
                target_id = first_present(
                    connection,
                    ["target", "to", "id"],
                )

                if target_id is None or target_id not in id_to_index:
                    continue

                weight = first_present(
                    connection,
                    ["weight", "w"],
                    default=0.0,
                )

                matrix[
                    source_index,
                    id_to_index[target_id],
                ] = float(weight)

        else:
            targets = first_present(
                node,
                ["connectedTo", "targets"],
                default=[],
            )

            weights = first_present(
                node,
                ["weights", "w"],
                default=[],
            )

            for target_id, weight in zip(targets, weights):
                if target_id in id_to_index:
                    matrix[
                        source_index,
                        id_to_index[target_id],
                    ] = float(weight)

    return matrix, id_order


def derive_neuron_type_labels_from_ground_truth(
    signed_matrix: np.ndarray,
    *,
    eps: float = GROUND_TRUTH_WEIGHT_EPS,
    unlabeled_value: float = np.nan,
) -> np.ndarray:
    """
    Infer one source-neuron type label from the signs of outgoing weights.

    Returns
    -------
    labels : ndarray, shape (N,)
        1 = excitatory source neuron
        0 = inhibitory source neuron
        NaN = no nonzero outgoing weights, so type cannot be inferred
    """
    signed_matrix = validate_square_matrix(
        np.asarray(signed_matrix, dtype=float),
        "Signed ground-truth matrix",
    )

    n_nodes = signed_matrix.shape[0]
    labels = np.full(n_nodes, unlabeled_value, dtype=float)

    for source_index in range(n_nodes):
        outgoing = signed_matrix[source_index, :].copy()
        outgoing[source_index] = 0.0
        nonzero = outgoing[np.abs(outgoing) > eps]

        if nonzero.size == 0:
            continue

        has_positive = bool(np.any(nonzero > 0))
        has_negative = bool(np.any(nonzero < 0))

        if has_positive and has_negative:
            raise ValueError(
                "Neuron has mixed-sign outgoing ground-truth weights. "
                f"Neuron index={source_index}, "
                f"positive_count={int(np.sum(nonzero > 0))}, "
                f"negative_count={int(np.sum(nonzero < 0))}."
            )

        labels[source_index] = 1.0 if has_positive else 0.0

    return labels


# =============================================================================
# Matrix validation
# =============================================================================


def validate_square_matrix(
    matrix: np.ndarray,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{name} must be square; received shape {matrix.shape}."
        )

    return matrix


def validate_probability_matrix(
    probability_matrix: np.ndarray,
    path: Path,
) -> np.ndarray:
    probability_matrix = validate_square_matrix(
        np.asarray(probability_matrix, dtype=float),
        f"Probability matrix {path}",
    )

    n_nodes = probability_matrix.shape[0]
    off_diagonal = ~np.eye(n_nodes, dtype=bool)
    off_diagonal_values = probability_matrix[off_diagonal]

    if not np.all(np.isfinite(off_diagonal_values)):
        bad_count = int(
            np.sum(~np.isfinite(off_diagonal_values))
        )
        raise ValueError(
            f"Probability matrix {path} contains {bad_count} non-finite "
            "off-diagonal values."
        )

    min_probability = float(np.min(off_diagonal_values))
    max_probability = float(np.max(off_diagonal_values))

    if (
        min_probability < -PROBABILITY_TOLERANCE
        or max_probability > 1.0 + PROBABILITY_TOLERANCE
    ):
        raise ValueError(
            f"Probability values in {path} fall outside [0, 1]: "
            f"minimum={min_probability}, maximum={max_probability}."
        )

    return np.clip(probability_matrix, 0.0, 1.0)


# =============================================================================
# Evaluation-category construction
# =============================================================================


@dataclass(frozen=True)
class EvaluationCategory:
    """One binary classification and graph-statistics evaluation stream."""

    name: str
    y_true: np.ndarray
    y_score: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    graph_is_undirected: bool
    degree_multiplier: int
    average_degree_denominator: int
    prediction_group: str

    @property
    def possible_connections(self) -> int:
        return int(self.y_true.size)


def off_diagonal_mask(n_nodes: int) -> np.ndarray:
    """Boolean mask for all ordered non-diagonal matrix entries."""
    return ~np.eye(n_nodes, dtype=bool)


def upper_triangle_mask(
    n_nodes: int,
    *,
    exclude_diagonal: bool = True,
) -> np.ndarray:
    """Boolean mask for one half of a symmetric matrix."""
    k = 1 if exclude_diagonal else 0
    return np.triu(
        np.ones((n_nodes, n_nodes), dtype=bool),
        k=k,
    )


def make_category(
    *,
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    graph_is_undirected: bool,
    degree_multiplier: int,
    average_degree_denominator: int,
    prediction_group: str,
) -> EvaluationCategory:
    """Validate and construct an EvaluationCategory."""
    y_true = np.asarray(y_true, dtype=np.uint8).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    rows = np.asarray(rows, dtype=int).ravel()
    columns = np.asarray(columns, dtype=int).ravel()

    sizes = {
        y_true.size,
        y_score.size,
        rows.size,
        columns.size,
    }

    if len(sizes) != 1:
        raise ValueError(
            f"Category {name!r} has inconsistent array lengths: "
            f"y_true={y_true.size}, y_score={y_score.size}, "
            f"rows={rows.size}, columns={columns.size}."
        )

    if not np.all(np.isfinite(y_score)):
        raise ValueError(
            f"Category {name!r} contains non-finite prediction scores."
        )

    return EvaluationCategory(
        name=name,
        y_true=y_true,
        y_score=y_score,
        rows=rows,
        columns=columns,
        graph_is_undirected=graph_is_undirected,
        degree_multiplier=int(degree_multiplier),
        average_degree_denominator=int(average_degree_denominator),
        prediction_group=prediction_group,
    )


def build_evaluation_categories(
    signed_ground_truth_matrix: np.ndarray,
    probability_matrix: np.ndarray,
    *,
    eps: float = GROUND_TRUTH_WEIGHT_EPS,
) -> tuple[dict[str, EvaluationCategory], np.ndarray]:
    """
    Build the nine threshold-evaluation categories used by the AUC reference.
    """
    signed_matrix = validate_square_matrix(
        np.asarray(signed_ground_truth_matrix, dtype=float),
        "Signed ground-truth matrix",
    )
    probability_matrix = validate_square_matrix(
        np.asarray(probability_matrix, dtype=float),
        "Probability matrix",
    )

    if signed_matrix.shape != probability_matrix.shape:
        raise ValueError(
            "Shape mismatch while constructing evaluation categories: "
            f"ground truth {signed_matrix.shape}, "
            f"probability matrix {probability_matrix.shape}."
        )

    n_nodes = signed_matrix.shape[0]
    adjacency = (np.abs(signed_matrix) > eps).astype(np.uint8)
    np.fill_diagonal(adjacency, 0)

    full_mask = off_diagonal_mask(n_nodes)
    upper_mask = upper_triangle_mask(n_nodes, exclude_diagonal=True)

    full_rows, full_columns = np.where(full_mask)
    upper_rows, upper_columns = np.where(upper_mask)

    adjacency_undirected = (
        (adjacency + adjacency.T) != 0
    ).astype(np.uint8)
    probability_undirected = (
        probability_matrix + probability_matrix.T
    ) / 2.0

    # This intentionally matches the supplied AUC reference script exactly.
    adjacency_directed = (
        (adjacency.astype(int) - adjacency.T.astype(int)) != 0
    ).astype(np.uint8)

    categories: dict[str, EvaluationCategory] = {}

    categories["total"] = make_category(
        name="total",
        y_true=adjacency[full_mask],
        y_score=probability_matrix[full_mask],
        rows=full_rows,
        columns=full_columns,
        graph_is_undirected=False,
        degree_multiplier=1,
        average_degree_denominator=n_nodes,
        prediction_group="total_ordered",
    )

    categories["undirected"] = make_category(
        name="undirected",
        y_true=adjacency_undirected[upper_mask],
        y_score=probability_undirected[upper_mask],
        rows=upper_rows,
        columns=upper_columns,
        graph_is_undirected=True,
        degree_multiplier=2,
        average_degree_denominator=n_nodes,
        prediction_group="undirected_pairs",
    )

    categories["directed"] = make_category(
        name="directed",
        y_true=adjacency_directed[full_mask],
        y_score=probability_matrix[full_mask],
        rows=full_rows,
        columns=full_columns,
        graph_is_undirected=False,
        degree_multiplier=1,
        average_degree_denominator=n_nodes,
        prediction_group="total_ordered",
    )

    neuron_type_labels = derive_neuron_type_labels_from_ground_truth(
        signed_matrix,
        eps=eps,
    )

    for type_name, type_label in (
        ("excitatory", 1.0),
        ("inhibitory", 0.0),
    ):
        source_is_type = np.isclose(
            neuron_type_labels,
            type_label,
            equal_nan=False,
        )
        num_source_neurons = int(np.sum(source_is_type))

        source_type_mask = (
            np.repeat(source_is_type[:, None], n_nodes, axis=1)
            & full_mask
        )
        source_rows, source_columns = np.where(source_type_mask)

        categories[f"{type_name}_total"] = make_category(
            name=f"{type_name}_total",
            y_true=adjacency[source_type_mask],
            y_score=probability_matrix[source_type_mask],
            rows=source_rows,
            columns=source_columns,
            graph_is_undirected=False,
            degree_multiplier=1,
            average_degree_denominator=num_source_neurons,
            prediction_group=f"{type_name}_ordered",
        )

        categories[f"{type_name}_directed"] = make_category(
            name=f"{type_name}_directed",
            y_true=adjacency_directed[source_type_mask],
            y_score=probability_matrix[source_type_mask],
            rows=source_rows,
            columns=source_columns,
            graph_is_undirected=False,
            degree_multiplier=1,
            average_degree_denominator=num_source_neurons,
            prediction_group=f"{type_name}_ordered",
        )

        undirected_truth: list[int] = []
        undirected_scores: list[float] = []
        undirected_rows: list[int] = []
        undirected_columns: list[int] = []

        for source_i in range(n_nodes):
            for target_j in range(source_i + 1, n_nodes):
                candidate_directions: list[tuple[int, int]] = []

                if source_is_type[source_i]:
                    candidate_directions.append((source_i, target_j))

                if source_is_type[target_j]:
                    candidate_directions.append((target_j, source_i))

                if not candidate_directions:
                    continue

                truth_value = int(
                    any(
                        adjacency[source, target] != 0
                        for source, target in candidate_directions
                    )
                )
                score_value = float(
                    np.mean(
                        [
                            probability_matrix[source, target]
                            for source, target in candidate_directions
                        ]
                    )
                )

                undirected_truth.append(truth_value)
                undirected_scores.append(score_value)
                undirected_rows.append(source_i)
                undirected_columns.append(target_j)

        categories[f"{type_name}_undirected"] = make_category(
            name=f"{type_name}_undirected",
            y_true=np.asarray(undirected_truth, dtype=np.uint8),
            y_score=np.asarray(undirected_scores, dtype=float),
            rows=np.asarray(undirected_rows, dtype=int),
            columns=np.asarray(undirected_columns, dtype=int),
            graph_is_undirected=True,
            degree_multiplier=2,
            average_degree_denominator=n_nodes,
            prediction_group=f"{type_name}_undirected_pairs",
        )

    ordered_names = [
        "total",
        "undirected",
        "directed",
        "excitatory_total",
        "excitatory_undirected",
        "excitatory_directed",
        "inhibitory_total",
        "inhibitory_undirected",
        "inhibitory_directed",
    ]

    return (
        {name: categories[name] for name in ordered_names},
        neuron_type_labels,
    )


# =============================================================================
# Threshold selection
# =============================================================================


def build_thresholds(
    categories: Iterable[EvaluationCategory],
    *,
    threshold_mode: str,
    roc_drop_intermediate: bool,
    fixed_thresholds: np.ndarray,
) -> np.ndarray:
    """Build thresholds in descending order."""
    threshold_mode = threshold_mode.lower()

    if threshold_mode == "roc":
        threshold_arrays: list[np.ndarray] = []
        skipped_categories: list[str] = []

        for category in categories:
            if category.y_true.size == 0 or np.unique(category.y_true).size < 2:
                skipped_categories.append(category.name)
                continue

            _, _, category_thresholds = roc_curve(
                category.y_true,
                category.y_score,
                drop_intermediate=roc_drop_intermediate,
            )
            threshold_arrays.append(
                np.asarray(category_thresholds, dtype=float)
            )

        if not threshold_arrays:
            raise ValueError(
                "ROC thresholds cannot be calculated because every evaluation "
                "category contains fewer than two ground-truth classes. Use "
                "threshold_mode='fixed' instead."
            )

        if skipped_categories:
            warnings.warn(
                "The following categories contain fewer than two ground-truth "
                "classes and did not contribute ROC thresholds: "
                + ", ".join(skipped_categories),
                stacklevel=2,
            )

        all_thresholds = np.concatenate(threshold_arrays)

        # np.unique sorts ascending. Reverse for the usual descending sweep.
        return np.unique(all_thresholds)[::-1]

    if threshold_mode == "fixed":
        thresholds = np.asarray(
            fixed_thresholds,
            dtype=float,
        ).ravel()

        if thresholds.size == 0:
            raise ValueError(
                "fixed_thresholds cannot be empty."
            )

        if not np.all(np.isfinite(thresholds)):
            raise ValueError(
                "Every fixed threshold must be finite."
            )

        thresholds = np.unique(thresholds)[::-1]

        return np.concatenate(
            (
                [np.inf],
                thresholds,
            )
        )

    raise ValueError(
        "threshold_mode must be either 'roc' or 'fixed'."
    )


# =============================================================================
# Graph statistics and binary-classification metrics
# =============================================================================


def average_clustering_from_binary_matrix(
    binary_matrix: np.ndarray,
    mode: str = CLUSTERING_MODE,
) -> float:
    """
    Compute average local clustering, including zero-degree nodes.

    directed
        NetworkX's unweighted directed clustering coefficient (Fagiolo form).

    undirected
        Symmetrize the graph first and calculate ordinary average clustering.
    """
    binary_matrix = validate_square_matrix(
        np.asarray(binary_matrix, dtype=np.uint8),
        "Binary adjacency matrix",
    ).copy()
    np.fill_diagonal(binary_matrix, 0)

    n_nodes = binary_matrix.shape[0]
    mode = mode.lower()

    if mode == "directed":
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n_nodes))
        rows, columns = np.where(binary_matrix > 0)
        graph.add_edges_from(zip(rows.tolist(), columns.tolist()))

    elif mode == "undirected":
        undirected_matrix = (
            (binary_matrix > 0)
            | (binary_matrix.T > 0)
        ).astype(np.uint8)
        np.fill_diagonal(undirected_matrix, 0)

        graph = nx.Graph()
        graph.add_nodes_from(range(n_nodes))
        rows, columns = np.where(
            np.triu(undirected_matrix, k=1) > 0
        )
        graph.add_edges_from(zip(rows.tolist(), columns.tolist()))

    else:
        raise ValueError(
            "clustering mode must be 'directed' or 'undirected'."
        )

    clustering_by_node = nx.clustering(graph)

    if not clustering_by_node:
        return 0.0

    return float(
        np.mean(list(clustering_by_node.values()))
    )


def safe_ratio(
    numerator: int | float,
    denominator: int | float,
    *,
    zero_value: float = np.nan,
) -> float:
    """Calculate a ratio safely when a denominator may be zero."""
    if denominator == 0:
        return float(zero_value)

    return float(numerator / denominator)


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return TP, FP, TN, FN."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred shapes differ: "
            f"{y_true.shape} versus {y_pred.shape}."
        )

    true_positives = int(np.count_nonzero(y_true & y_pred))
    false_positives = int(np.count_nonzero(~y_true & y_pred))
    true_negatives = int(np.count_nonzero(~y_true & ~y_pred))
    false_negatives = int(np.count_nonzero(y_true & ~y_pred))

    return (
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    )


def category_vector_to_matrix(
    values: np.ndarray,
    category: EvaluationCategory,
    n_nodes: int,
) -> np.ndarray:
    """Place one category's binary values into an N x N graph matrix."""
    values = np.asarray(values, dtype=np.uint8).ravel()

    if values.size != category.possible_connections:
        raise ValueError(
            f"Category {category.name!r} expected "
            f"{category.possible_connections} values, received {values.size}."
        )

    matrix = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    matrix[category.rows, category.columns] = values

    if category.graph_is_undirected:
        matrix[category.columns, category.rows] = values

    np.fill_diagonal(matrix, 0)
    return matrix


def summarize_category_binary_values(
    values: np.ndarray,
    category: EvaluationCategory,
    n_nodes: int,
    *,
    directed_clustering_mode: str,
) -> dict[str, float | int]:
    """Calculate graph statistics for one category's binary values."""
    values = np.asarray(values, dtype=np.uint8).ravel()
    edge_count = int(np.count_nonzero(values))

    binary_matrix = category_vector_to_matrix(
        values,
        category,
        n_nodes,
    )

    clustering_mode = (
        "undirected"
        if category.graph_is_undirected
        else directed_clustering_mode
    )

    clustering = average_clustering_from_binary_matrix(
        binary_matrix,
        mode=clustering_mode,
    )

    average_connections = safe_ratio(
        edge_count * category.degree_multiplier,
        category.average_degree_denominator,
        zero_value=np.nan,
    )

    connection_probability = safe_ratio(
        edge_count,
        category.possible_connections,
        zero_value=np.nan,
    )

    return {
        "clustering_coefficient": clustering,
        "connections": edge_count,
        "average_connections_per_neuron": average_connections,
        "connection_probability": connection_probability,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    """Calculate confusion counts and derived threshold metrics."""
    (
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    ) = confusion_counts(y_true, y_pred)

    precision = safe_ratio(
        true_positives,
        true_positives + false_positives,
        zero_value=0.0,
    )
    true_positive_rate = safe_ratio(
        true_positives,
        true_positives + false_negatives,
    )
    false_positive_rate = safe_ratio(
        false_positives,
        false_positives + true_negatives,
    )

    return {
        "precision": precision,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "predicted_positive_connections": (
            true_positives + false_positives
        ),
        "actual_positive_connections": (
            true_positives + false_negatives
        ),
        "evaluated_connections": int(np.asarray(y_true).size),
    }


def add_category_columns(
    record: dict[str, float | int | str | bool],
    category_name: str,
    classification: dict[str, float | int],
    predicted_stats: dict[str, float | int],
    ground_truth_stats: dict[str, float | int],
) -> None:
    """Add one category's complete metric set to an output row."""
    for metric_name, metric_value in predicted_stats.items():
        record[
            f"{category_name}_predicted_{metric_name}"
        ] = metric_value

    for metric_name, metric_value in classification.items():
        record[
            f"{category_name}_{metric_name}"
        ] = metric_value

    for metric_name, metric_value in ground_truth_stats.items():
        record[
            f"{category_name}_ground_truth_{metric_name}"
        ] = metric_value


# =============================================================================
# One-network analysis
# =============================================================================


def analyze_one_probability_matrix(
    probability_path: str | Path,
    *,
    ground_truth_dir: str | Path,
    threshold_mode: str,
    roc_drop_intermediate: bool,
    fixed_thresholds: np.ndarray,
    clustering_mode: str,
    ground_truth_weight_eps: float,
) -> pd.DataFrame:
    """Analyze every selected threshold for one saved probability matrix."""
    probability_path = Path(probability_path)
    ground_truth_dir = Path(ground_truth_dir)

    stats_name, network_number = parse_probability_matrix_filename(
        probability_path
    )
    parsed_stats = parse_network_statistics(stats_name)
    network_name = f"{stats_name}_{network_number}"

    ground_truth_path = (
        ground_truth_dir
        / f"network_{network_name}.yaml"
    )

    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Missing ground-truth YAML for {probability_path.name}: "
            f"{ground_truth_path}"
        )

    probability_matrix = validate_probability_matrix(
        np.load(probability_path, allow_pickle=False),
        probability_path,
    )

    signed_ground_truth_matrix, _ = (
        load_ground_truth_signed_connectivity_matrix(
            ground_truth_path
        )
    )
    signed_ground_truth_matrix = validate_square_matrix(
        np.asarray(signed_ground_truth_matrix, dtype=float),
        f"Ground-truth matrix {ground_truth_path}",
    )

    if probability_matrix.shape != signed_ground_truth_matrix.shape:
        raise ValueError(
            f"Shape mismatch for {network_name}: probability matrix "
            f"{probability_matrix.shape}, ground truth "
            f"{signed_ground_truth_matrix.shape}."
        )

    n_nodes = probability_matrix.shape[0]

    if int(parsed_stats["N_from_filename"]) != n_nodes:
        warnings.warn(
            f"Filename for {network_name} says "
            f"N={parsed_stats['N_from_filename']}, "
            f"but the matrices have N={n_nodes}. Matrix size will be used.",
            stacklevel=2,
        )

    probability_matrix = probability_matrix.copy()
    signed_ground_truth_matrix = signed_ground_truth_matrix.copy()
    np.fill_diagonal(probability_matrix, 0.0)
    np.fill_diagonal(signed_ground_truth_matrix, 0.0)

    categories, neuron_type_labels = build_evaluation_categories(
        signed_ground_truth_matrix,
        probability_matrix,
        eps=ground_truth_weight_eps,
    )

    thresholds = build_thresholds(
        categories.values(),
        threshold_mode=threshold_mode,
        roc_drop_intermediate=roc_drop_intermediate,
        fixed_thresholds=fixed_thresholds,
    )

    num_excitatory_neurons = int(
        np.sum(
            np.isclose(
                neuron_type_labels,
                1.0,
                equal_nan=False,
            )
        )
    )
    num_inhibitory_neurons = int(
        np.sum(
            np.isclose(
                neuron_type_labels,
                0.0,
                equal_nan=False,
            )
        )
    )
    num_untyped_neurons = int(
        np.sum(~np.isfinite(neuron_type_labels))
    )

    ground_truth_stats_by_category = {
        category_name: summarize_category_binary_values(
            category.y_true,
            category,
            n_nodes,
            directed_clustering_mode=clustering_mode,
        )
        for category_name, category in categories.items()
    }

    records: list[dict[str, float | int | str | bool]] = []

    for threshold_index, threshold in enumerate(thresholds):
        record: dict[str, float | int | str | bool] = {
            "network_name": network_name,
            "network_statistics": stats_name,
            "network_number": network_number,
            "N": n_nodes,
            "p_code": int(parsed_stats["p_code"]),
            "target_connection_probability": float(
                parsed_stats["target_connection_probability"]
            ),
            "target_average_connections_per_neuron": float(
                parsed_stats["target_connection_probability"]
                * (n_nodes - 1)
            ),
            "cc_code": int(parsed_stats["cc_code"]),
            "target_clustering_coefficient": float(
                parsed_stats["target_clustering_coefficient"]
            ),
            "num_excitatory_source_neurons": num_excitatory_neurons,
            "num_inhibitory_source_neurons": num_inhibitory_neurons,
            "num_untyped_source_neurons": num_untyped_neurons,
            "threshold_index": threshold_index,
            "threshold": float(threshold),
            "threshold_is_infinite": bool(np.isinf(threshold)),
            "threshold_rule": (
                ">=" if THRESHOLD_INCLUSIVE else ">"
            ),
            "threshold_mode": threshold_mode,
            "roc_drop_intermediate": (
                bool(roc_drop_intermediate)
                if threshold_mode == "roc"
                else False
            ),
            "clustering_mode_for_directed_categories": clustering_mode,
            "undirected_categories_use_undirected_clustering": True,
            "ground_truth_weight_eps": float(ground_truth_weight_eps),
        }

        # Categories sharing the same prediction_group use the same scores,
        # graph representation, and graph-statistics denominator. Cache their
        # predicted graph statistics to avoid redundant clustering calculations.
        predicted_stats_cache: dict[
            str,
            dict[str, float | int]
        ] = {}

        classification_by_category: dict[
            str,
            dict[str, float | int]
        ] = {}

        for category_name, category in categories.items():
            if THRESHOLD_INCLUSIVE:
                y_pred = category.y_score >= threshold
            else:
                y_pred = category.y_score > threshold

            classification = classification_metrics(
                category.y_true,
                y_pred,
            )
            classification_by_category[category_name] = classification

            if category.prediction_group not in predicted_stats_cache:
                predicted_stats_cache[
                    category.prediction_group
                ] = summarize_category_binary_values(
                    y_pred.astype(np.uint8),
                    category,
                    n_nodes,
                    directed_clustering_mode=clustering_mode,
                )

            add_category_columns(
                record,
                category_name,
                classification,
                predicted_stats_cache[category.prediction_group],
                ground_truth_stats_by_category[category_name],
            )

        # Preserve the original column names as aliases of the total metrics.
        total_classification = classification_by_category["total"]
        total_predicted_stats = predicted_stats_cache["total_ordered"]
        total_ground_truth_stats = ground_truth_stats_by_category["total"]

        record.update(
            {
                "clustering_mode": clustering_mode,
                "predicted_clustering_coefficient": (
                    total_predicted_stats["clustering_coefficient"]
                ),
                "predicted_connections": int(
                    total_predicted_stats["connections"]
                ),
                "predicted_average_connections_per_neuron": float(
                    total_predicted_stats[
                        "average_connections_per_neuron"
                    ]
                ),
                "predicted_connection_probability": float(
                    total_predicted_stats["connection_probability"]
                ),
                "precision": total_classification["precision"],
                "true_positive_rate": total_classification[
                    "true_positive_rate"
                ],
                "false_positive_rate": total_classification[
                    "false_positive_rate"
                ],
                "true_positives": total_classification[
                    "true_positives"
                ],
                "false_positives": total_classification[
                    "false_positives"
                ],
                "true_negatives": total_classification[
                    "true_negatives"
                ],
                "false_negatives": total_classification[
                    "false_negatives"
                ],
                "predicted_positive_connections": total_classification[
                    "predicted_positive_connections"
                ],
                "actual_positive_connections": total_classification[
                    "actual_positive_connections"
                ],
                "ground_truth_clustering_coefficient": (
                    total_ground_truth_stats["clustering_coefficient"]
                ),
                "ground_truth_connections": int(
                    total_ground_truth_stats["connections"]
                ),
                "ground_truth_average_connections_per_neuron": float(
                    total_ground_truth_stats[
                        "average_connections_per_neuron"
                    ]
                ),
                "ground_truth_connection_probability": float(
                    total_ground_truth_stats["connection_probability"]
                ),
                "probability_matrix_file": str(probability_path),
                "ground_truth_file": str(ground_truth_path),
            }
        )

        records.append(record)

    return pd.DataFrame.from_records(records)


# =============================================================================
# Atomic CSV output and batch execution
# =============================================================================


def save_dataframe_atomically(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write a complete CSV and atomically move it into its final name."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    try:
        dataframe.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def discover_probability_files(
    probability_dir: str | Path,
) -> list[Path]:
    probability_dir = Path(probability_dir)

    if not probability_dir.exists():
        raise FileNotFoundError(
            "Probability-matrix directory does not exist: "
            f"{probability_dir}"
        )

    if not probability_dir.is_dir():
        raise NotADirectoryError(
            "Expected a probability-matrix directory, got: "
            f"{probability_dir}"
        )

    candidate_files = sorted(
        probability_dir.glob(PROBABILITY_FILE_GLOB),
        key=lambda path: natural_sort_key(path.name),
    )

    valid_files: list[Path] = []
    ignored_files: list[Path] = []

    for path in candidate_files:
        if PROBABILITY_FILENAME_RE.fullmatch(path.name):
            valid_files.append(path)
        else:
            ignored_files.append(path)

    for path in ignored_files:
        warnings.warn(
            "Ignoring probability file with an unexpected name: "
            f"{path}",
            stacklevel=2,
        )

    if not valid_files:
        raise FileNotFoundError(
            "No files matching "
            "prob_matrix_N..._p..._CC..._<number>.npy "
            f"were found in {probability_dir}."
        )

    return valid_files


def determine_max_workers(
    requested_workers: int | None = None,
) -> int:
    """
    Determine how many worker processes may be used.

    When --workers is omitted, this respects process CPU affinity,
    SLURM_CPUS_PER_TASK when present, and os.cpu_count().
    """
    if requested_workers is not None:
        if requested_workers < 1:
            raise ValueError(
                "workers must be at least 1."
            )
        return int(requested_workers)

    available_counts: list[int] = []
    cpu_count = os.cpu_count()

    if cpu_count is not None and cpu_count > 0:
        available_counts.append(int(cpu_count))

    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
            if affinity_count > 0:
                available_counts.append(int(affinity_count))
        except (AttributeError, OSError):
            pass

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus:
        try:
            parsed_slurm_cpus = int(slurm_cpus)
            if parsed_slurm_cpus > 0:
                available_counts.append(parsed_slurm_cpus)
        except ValueError:
            warnings.warn(
                "Ignoring invalid "
                f"SLURM_CPUS_PER_TASK={slurm_cpus!r}.",
                stacklevel=2,
            )

    if not available_counts:
        return 1

    return max(1, min(available_counts))


def analyze_and_save_probability_matrix_worker(
    probability_path: str | Path,
    output_path: str | Path,
    *,
    ground_truth_dir: str | Path,
    threshold_mode: str,
    roc_drop_intermediate: bool,
    fixed_thresholds: np.ndarray,
    clustering_mode: str,
    ground_truth_weight_eps: float,
) -> dict[str, object]:
    """Analyze and atomically save one probability matrix inside a worker."""
    probability_path = Path(probability_path)
    output_path = Path(output_path)

    result_df = analyze_one_probability_matrix(
        probability_path,
        ground_truth_dir=ground_truth_dir,
        threshold_mode=threshold_mode,
        roc_drop_intermediate=roc_drop_intermediate,
        fixed_thresholds=fixed_thresholds,
        clustering_mode=clustering_mode,
        ground_truth_weight_eps=ground_truth_weight_eps,
    )

    save_dataframe_atomically(result_df, output_path)

    return {
        "probability_path": probability_path,
        "output_path": output_path,
        "num_threshold_rows": len(result_df),
        "num_columns": len(result_df.columns),
    }


def run_batch_analysis(
    *,
    probability_dir: str | Path,
    ground_truth_dir: str | Path,
    output_dir: str | Path,
    threshold_mode: str,
    roc_drop_intermediate: bool,
    fixed_thresholds: np.ndarray,
    clustering_mode: str,
    ground_truth_weight_eps: float,
    skip_existing: bool,
    workers: int | None,
) -> dict[str, object]:
    """Analyze probability matrices in parallel and save each CSV immediately."""
    probability_dir = Path(probability_dir)
    ground_truth_dir = Path(ground_truth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probability_files = discover_probability_files(probability_dir)

    saved_paths: list[Path] = []
    skipped_paths: list[Path] = []
    failed: list[tuple[Path, str]] = []
    pending_jobs: list[tuple[Path, Path]] = []

    print(
        f"Found {len(probability_files)} probability matrix file(s)."
    )
    print(f"Threshold mode: {threshold_mode}")

    if threshold_mode == "roc":
        print(
            "ROC thresholds: union across all valid evaluation categories"
        )
        print(
            "ROC drop_intermediate: "
            f"{roc_drop_intermediate}"
        )
    else:
        print(
            "Number of finite fixed thresholds: "
            f"{len(np.asarray(fixed_thresholds).ravel())}"
        )

    print(
        "Clustering mode for directed categories: "
        f"{clustering_mode}"
    )
    print(
        "Undirected categories always use undirected clustering."
    )
    print(
        "Ground-truth nonzero tolerance: "
        f"{ground_truth_weight_eps:g}"
    )
    print(
        "Output directory: "
        f"{output_dir.resolve()}"
    )
    print()

    for file_index, probability_path in enumerate(
        probability_files,
        start=1,
    ):
        output_path = output_path_for_probability_matrix(
            probability_path,
            output_dir,
        )

        if skip_existing and output_path.is_file():
            print(
                f"[{file_index}/{len(probability_files)}] "
                "Skipping existing output: "
                f"{output_path.name}"
            )
            skipped_paths.append(output_path)
        else:
            pending_jobs.append((probability_path, output_path))

    if pending_jobs:
        max_workers = min(
            determine_max_workers(workers),
            len(pending_jobs),
        )

        print()
        print(
            "Probability matrices remaining: "
            f"{len(pending_jobs)}"
        )
        print(f"Worker processes: {max_workers}")
        print()

        future_to_job = {}

        with ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for probability_path, output_path in pending_jobs:
                print(f"Submitting: {probability_path.name}")

                future = executor.submit(
                    analyze_and_save_probability_matrix_worker,
                    probability_path,
                    output_path,
                    ground_truth_dir=ground_truth_dir,
                    threshold_mode=threshold_mode,
                    roc_drop_intermediate=roc_drop_intermediate,
                    fixed_thresholds=fixed_thresholds,
                    clustering_mode=clustering_mode,
                    ground_truth_weight_eps=ground_truth_weight_eps,
                )

                future_to_job[future] = (
                    probability_path,
                    output_path,
                )

            completed_count = 0

            for future in as_completed(future_to_job):
                probability_path, output_path = future_to_job[future]
                completed_count += 1
                prefix = (
                    f"[{completed_count}/{len(pending_jobs)}]"
                )

                try:
                    result = future.result()
                    saved_path = Path(result["output_path"])
                    saved_paths.append(saved_path)

                    print(
                        f"{prefix} Completed "
                        f"{probability_path.name}: "
                        f"{result['num_threshold_rows']} threshold row(s), "
                        f"{result['num_columns']} column(s)."
                    )
                    print(f"    Saved: {saved_path}")

                except Exception as exc:
                    failed.append((probability_path, str(exc)))
                    print(
                        f"{prefix} ERROR for "
                        f"{probability_path.name}: {exc}"
                    )

    else:
        print(
            "No unfinished probability matrices were found."
        )

    print("\nBatch analysis complete.")
    print(f"New CSV files saved: {len(saved_paths)}")
    print(
        "Existing CSV files skipped: "
        f"{len(skipped_paths)}"
    )
    print(
        "Probability matrices that failed: "
        f"{len(failed)}"
    )

    if failed:
        print("\nFailures:")
        for probability_path, error_message in failed:
            print(
                f"  {probability_path.name}: {error_message}"
            )

    return {
        "saved_paths": saved_paths,
        "skipped_paths": skipped_paths,
        "failed": failed,
        "num_probability_files": len(probability_files),
    }


# =============================================================================
# Command-line interface
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Threshold saved GBT probability matrices and save total, "
            "undirected, directed, excitatory-source, and inhibitory-source "
            "graph and classification metrics."
        )
    )

    parser.add_argument(
        "--prob-dir",
        type=Path,
        default=PROBABILITY_MATRIX_DIR,
        help=(
            "Probability-matrix folder "
            f"(default: {PROBABILITY_MATRIX_DIR})."
        ),
    )

    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=GROUND_TRUTH_DIR,
        help=(
            "Ground-truth YAML folder "
            f"(default: {GROUND_TRUTH_DIR})."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "CSV output folder "
            f"(default: {OUTPUT_DIR})."
        ),
    )

    parser.add_argument(
        "--threshold-mode",
        choices=("roc", "fixed"),
        default=THRESHOLD_MODE,
        help=(
            "Threshold strategy "
            f"(default: {THRESHOLD_MODE})."
        ),
    )

    parser.add_argument(
        "--all-roc-thresholds",
        action="store_true",
        help=(
            "For ROC mode, retain every distinct category-specific score "
            "threshold by setting drop_intermediate=False. This can be "
            "substantially slower."
        ),
    )

    parser.add_argument(
        "--num-fixed-thresholds",
        type=int,
        default=len(FIXED_THRESHOLDS),
        help=(
            "For fixed mode, use this many evenly spaced thresholds from 1 "
            f"to 0 (default: {len(FIXED_THRESHOLDS)})."
        ),
    )

    parser.add_argument(
        "--clustering-mode",
        choices=("directed", "undirected"),
        default=CLUSTERING_MODE,
        help=(
            "Clustering definition for categories represented as directed "
            f"matrices (default: {CLUSTERING_MODE}). Undirected categories "
            "always use undirected clustering."
        ),
    )

    parser.add_argument(
        "--ground-truth-weight-eps",
        type=float,
        default=GROUND_TRUTH_WEIGHT_EPS,
        help=(
            "Absolute ground-truth weight tolerance below which an edge is "
            f"treated as absent (default: {GROUND_TRUTH_WEIGHT_EPS:g})."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recalculate networks even when the final CSV already exists."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of probability matrices to analyze concurrently. By "
            "default, use the CPU cores available to this process or "
            "scheduler job."
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.num_fixed_thresholds < 2:
        parser.error(
            "--num-fixed-thresholds must be at least 2."
        )

    if args.workers is not None and args.workers < 1:
        parser.error(
            "--workers must be at least 1."
        )

    if args.ground_truth_weight_eps < 0:
        parser.error(
            "--ground-truth-weight-eps must be non-negative."
        )

    fixed_thresholds = np.linspace(
        1.0,
        0.0,
        args.num_fixed_thresholds,
    )

    summary = run_batch_analysis(
        probability_dir=args.prob_dir,
        ground_truth_dir=args.truth_dir,
        output_dir=args.output_dir,
        threshold_mode=args.threshold_mode,
        roc_drop_intermediate=(
            False
            if args.all_roc_thresholds
            else ROC_DROP_INTERMEDIATE
        ),
        fixed_thresholds=fixed_thresholds,
        clustering_mode=args.clustering_mode,
        ground_truth_weight_eps=args.ground_truth_weight_eps,
        skip_existing=(
            False
            if args.overwrite
            else SKIP_EXISTING
        ),
        workers=args.workers,
    )

    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
