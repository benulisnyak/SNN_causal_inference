"""Inspect saved edge-probability matrices against ground-truth connectivity.

This utility pairs classifier probability matrices with their matching YAML
networks, computes per-network ROC AUC, and compares predicted graph clustering
against the ground-truth clustering coefficient across ROC operating thresholds.
It is intended for exploratory validation after ``train_edge_classifier.py`` has
written probability matrices.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from yaml import CSafeLoader as YamlLoader
except ImportError:  # pragma: no cover - depends on the local PyYAML build
    from yaml import SafeLoader as YamlLoader


PROJECT_ROOT = Path(".")
PROBABILITY_MATRIX_DIR = PROJECT_ROOT / "prob_matrices"
GROUND_TRUTH_DIR = PROJECT_ROOT / "networks"
PROBABILITY_FILE_GLOB = "prob_matrix_*.npy"
DEFAULT_DIAGNOSTIC_NETWORK = "N100_p24_CC05_6"


def natural_sort_key(text: str) -> list[object]:
    """Return a natural-sort key so numeric suffixes sort numerically."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def load_ground_truth_connectivity_matrix(
    file_path: str | Path,
) -> tuple[np.ndarray, list[object]]:
    """Load a ground-truth YAML network as a non-negative adjacency matrix."""
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=YamlLoader)

    if data is None:
        raise ValueError(f"Ground-truth YAML file is empty: {file_path}")

    nodes = data.get("nodes", [])
    if isinstance(nodes, dict):
        node_list = []
        for raw_id, attributes in nodes.items():
            try:
                node_id: object = int(raw_id)
            except (TypeError, ValueError):
                node_id = raw_id
            node_list.append({"id": node_id, **(attributes or {})})
    elif isinstance(nodes, list):
        node_list = nodes
    else:
        raise ValueError(f"'nodes' must be a list or dict in {file_path}")

    id_order = [node["id"] for node in node_list]
    id_to_index = {node_id: index for index, node_id in enumerate(id_order)}
    adjacency = np.zeros((len(id_order), len(id_order)), dtype=float)

    for node in node_list:
        source_id = node["id"]
        source_index = id_to_index[source_id]

        if "connections" in node and isinstance(node["connections"], list):
            for connection in node["connections"]:
                target_id = (
                    connection.get("target")
                    or connection.get("to")
                    or connection.get("id")
                )
                if target_id is None or target_id not in id_to_index:
                    continue
                weight = connection.get("weight", connection.get("w", 0.0))
                adjacency[source_index, id_to_index[target_id]] = abs(float(weight))
        else:
            targets = node.get("connectedTo") or node.get("targets") or []
            weights = node.get("weights") or node.get("w") or []
            for target_id, weight in zip(targets, weights):
                if target_id in id_to_index:
                    adjacency[source_index, id_to_index[target_id]] = abs(float(weight))

    return adjacency, id_order


def parse_probability_matrix_filename(probability_path: str | Path) -> tuple[str, int]:
    """Parse ``prob_matrix_<statistics>_<network>.npy`` into its identifiers."""
    probability_path = Path(probability_path)
    match = re.fullmatch(
        r"prob_matrix_(.+)_(\d+)\.npy",
        probability_path.name,
    )
    if match is None:
        raise ValueError(
            "Probability matrix filename does not match the expected pattern: "
            f"{probability_path.name}"
        )
    return match.group(1), int(match.group(2))


def load_probability_truth_pair(
    probability_path: str | Path,
    ground_truth_dir: str | Path = GROUND_TRUTH_DIR,
) -> dict[str, object]:
    """Load one probability matrix and its matching ground-truth network."""
    probability_path = Path(probability_path)
    ground_truth_dir = Path(ground_truth_dir)
    statistics_name, network_number = parse_probability_matrix_filename(
        probability_path
    )

    truth_path = ground_truth_dir / f"network_{statistics_name}_{network_number}.yaml"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing matching ground-truth file: {truth_path}")

    probability_matrix = np.load(probability_path)
    ground_truth_matrix, id_order = load_ground_truth_connectivity_matrix(truth_path)
    if probability_matrix.shape != ground_truth_matrix.shape:
        raise ValueError(
            f"Shape mismatch for {probability_path.name}: probability matrix has "
            f"shape {probability_matrix.shape}, but ground truth has shape "
            f"{ground_truth_matrix.shape}."
        )

    return {
        "name": f"{statistics_name}_{network_number}",
        "stats_name": statistics_name,
        "network_number": network_number,
        "prob_matrix": probability_matrix,
        "ground_truth_matrix": ground_truth_matrix,
        "id_order": id_order,
        "prob_path": probability_path,
        "truth_path": truth_path,
    }


def load_all_probability_truth_pairs(
    probability_dir: str | Path = PROBABILITY_MATRIX_DIR,
    ground_truth_dir: str | Path = GROUND_TRUTH_DIR,
) -> dict[str, dict[str, object]]:
    """Load every saved probability matrix with its matching ground truth."""
    probability_dir = Path(probability_dir)
    ground_truth_dir = Path(ground_truth_dir)

    if not probability_dir.exists():
        raise FileNotFoundError(
            f"Probability matrix directory does not exist: {probability_dir}"
        )
    if not probability_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {probability_dir}")

    probability_files = sorted(
        probability_dir.glob(PROBABILITY_FILE_GLOB),
        key=lambda path: natural_sort_key(path.name),
    )
    if not probability_files:
        raise FileNotFoundError(
            f"No probability matrix files were found in: {probability_dir}"
        )

    pairs: dict[str, dict[str, object]] = {}
    for probability_path in probability_files:
        pair = load_probability_truth_pair(
            probability_path,
            ground_truth_dir=ground_truth_dir,
        )
        pairs[str(pair["name"])] = pair
    return pairs


def average_directed_clustering(binary_matrix: np.ndarray) -> float:
    """Return NetworkX's mean directed clustering coefficient over all nodes."""
    adjacency = np.asarray(binary_matrix).copy()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Expected a square adjacency matrix, got {adjacency.shape}.")

    np.fill_diagonal(adjacency, 0)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    rows, columns = np.where(adjacency > 0)
    graph.add_edges_from(zip(rows, columns))

    coefficients = nx.clustering(graph)
    return float(np.mean(list(coefficients.values())))


def compute_roc_clustering_diagnostics(
    pair: dict[str, object],
) -> dict[str, np.ndarray | float]:
    """Compute ROC coordinates and predicted clustering over ROC thresholds."""
    probability_matrix = np.asarray(pair["prob_matrix"], dtype=float)
    ground_truth_matrix = np.asarray(pair["ground_truth_matrix"], dtype=float)
    binary_truth = (ground_truth_matrix > 0).astype(int)

    n_nodes = probability_matrix.shape[0]
    off_diagonal = ~np.eye(n_nodes, dtype=bool)
    y_true = binary_truth[off_diagonal]
    y_score = probability_matrix[off_diagonal]

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true,
        y_score,
        drop_intermediate=True,
    )
    roc_auc = float(roc_auc_score(y_true, y_score))
    ground_truth_clustering = average_directed_clustering(binary_truth)

    predicted_clustering = []
    for threshold in thresholds:
        predicted_adjacency = (probability_matrix >= threshold).astype(int)
        np.fill_diagonal(predicted_adjacency, 0)
        predicted_clustering.append(average_directed_clustering(predicted_adjacency))

    return {
        "fpr": false_positive_rate,
        "tpr": true_positive_rate,
        "thresholds": thresholds,
        "roc_auc": roc_auc,
        "ground_truth_clustering": ground_truth_clustering,
        "predicted_clustering": np.asarray(predicted_clustering, dtype=float),
    }


def plot_network_diagnostics(
    pair_name: str,
    diagnostics: dict[str, np.ndarray | float],
) -> None:
    """Plot ROC performance and clustering coefficient versus false-positive rate."""
    fpr = np.asarray(diagnostics["fpr"])
    tpr = np.asarray(diagnostics["tpr"])
    predicted_clustering = np.asarray(diagnostics["predicted_clustering"])
    roc_auc = float(diagnostics["roc_auc"])
    ground_truth_clustering = float(diagnostics["ground_truth_clustering"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.plot(fpr, tpr, "o", label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"ROC {pair_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, predicted_clustering, "o", label="Predicted network clustering")
    ax.axhline(
        ground_truth_clustering,
        linestyle="--",
        label=f"Ground-truth clustering = {ground_truth_clustering:.3f}",
    )
    ax.set(
        xlabel="False Positive Rate",
        ylabel="Clustering Coefficient",
        title=f"CC vs FPR for {pair_name}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()


def compute_auc_summary(
    pairs: dict[str, dict[str, object]],
) -> tuple[dict[str, float], float, float]:
    """Compute per-network ROC AUC and its mean/population standard deviation."""
    auc_results: dict[str, float] = {}

    for pair_name, pair in pairs.items():
        probability_matrix = np.asarray(pair["prob_matrix"], dtype=float)
        ground_truth_matrix = np.asarray(pair["ground_truth_matrix"], dtype=float)
        binary_truth = (ground_truth_matrix > 0).astype(int)

        off_diagonal = ~np.eye(probability_matrix.shape[0], dtype=bool)
        y_true = binary_truth[off_diagonal]
        y_score = probability_matrix[off_diagonal]

        if np.unique(y_true).size < 2:
            auc_value = np.nan
        else:
            auc_value = float(roc_auc_score(y_true, y_score))
        auc_results[pair_name] = auc_value

    valid_values = np.asarray(
        [value for value in auc_results.values() if not np.isnan(value)],
        dtype=float,
    )
    if valid_values.size == 0:
        return auc_results, np.nan, np.nan

    return auc_results, float(np.mean(valid_values)), float(np.std(valid_values))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probability-dir",
        type=Path,
        default=PROBABILITY_MATRIX_DIR,
        help="Directory containing prob_matrix_*.npy classifier outputs.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=GROUND_TRUTH_DIR,
        help="Directory containing network_<statistics>_<index>.yaml files.",
    )
    parser.add_argument(
        "--diagnostic-network",
        default=DEFAULT_DIAGNOSTIC_NETWORK,
        help="Network identifier to use for the ROC/clustering diagnostic plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Compute diagnostics without opening Matplotlib windows.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    pairs = load_all_probability_truth_pairs(
        probability_dir=args.probability_dir,
        ground_truth_dir=args.ground_truth_dir,
    )

    print(f"Loaded {len(pairs)} probability/ground-truth network pairs.")
    for name, pair in pairs.items():
        print(
            f"{name}: probability={Path(pair['prob_path']).name}, "
            f"truth={Path(pair['truth_path']).name}, "
            f"shape={np.asarray(pair['prob_matrix']).shape}"
        )

    if args.diagnostic_network in pairs:
        diagnostics = compute_roc_clustering_diagnostics(pairs[args.diagnostic_network])
        plot_network_diagnostics(args.diagnostic_network, diagnostics)
    else:
        print(
            f"Diagnostic network '{args.diagnostic_network}' was not found; "
            "skipping the two single-network plots."
        )

    auc_results, mean_auc, std_auc = compute_auc_summary(pairs)
    print("\nPer-network ROC AUC:")
    for name, auc_value in auc_results.items():
        if np.isnan(auc_value):
            print(f"  {name}: unavailable (ground truth contains one class)")
        else:
            print(f"  {name}: {auc_value:.4f}")

    valid_count = sum(not np.isnan(value) for value in auc_results.values())
    print("\nSummary across probability matrices:")
    print(f"  Valid networks: {valid_count}")
    print(f"  Mean ROC AUC: {mean_auc:.4f}")
    print(f"  ROC AUC std.: {std_auc:.4f}")

    if not args.no_show and plt.get_fignums():
        plt.show()


if __name__ == "__main__":
    main()
