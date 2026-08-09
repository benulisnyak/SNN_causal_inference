"""Train a gradient-boosted classifier for directed edge inference.

This script implements the second stage of the MSc connectivity-inference pipeline.
It loads learned SNN connectivity matrices from multiple network realizations,
constructs edge-level summary features, performs network-grouped train/test splits,
and fits a scikit-learn ``GradientBoostingClassifier`` to estimate the probability
of each directed structural connection.

The default configuration reproduces the thesis experiment over the N=100
parameter grid. Paths and model hyperparameters are collected near the top and
bottom of the file so the experiment can be changed without editing the core
feature-engineering or evaluation routines.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

try:
    from yaml import CSafeLoader as YamlLoader
except ImportError:  # pragma: no cover - depends on the local PyYAML build
    from yaml import SafeLoader as YamlLoader


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------

NETWORK_STATISTICS = [
    "N100_p12_CC01",
    "N100_p12_CC03",
    "N100_p12_CC05",
    "N100_p24_CC01",
    "N100_p24_CC03",
    "N100_p24_CC05",
    "N100_p36_CC01",
    "N100_p36_CC03",
    "N100_p36_CC05",
]

PROJECT_ROOT = Path(".")
LEARNED_CONNECTIVITY_DIR = PROJECT_ROOT / "LIFoutput_files"
GROUND_TRUTH_DIR = PROJECT_ROOT / "networks"
CONNECTIVITY_FILE_GLOB = "connectivity_matrices_*.npy"
PROBABILITY_OUTPUT_DIR = PROJECT_ROOT / "prob_matrices"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def natural_sort_key(text: str) -> list[object]:
    """Return a natural-sort key so numeric suffixes sort numerically."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def resolve_npy_dir(folder_name: str, learned_connectivity_dir: str | Path) -> Path:
    """Return the directory containing learned matrices for one network class."""
    return Path(learned_connectivity_dir) / folder_name


def load_one_npy_matrix_file(npy_path: str | Path) -> np.ndarray:

    """Load one stack of burst-specific learned connectivity matrices."""
    npy_path = Path(npy_path)
    matrices = np.load(npy_path)

    if matrices.ndim != 3:

        raise ValueError(
            f"Expected shape (num_matrices, N, N) in {npy_path}, got {matrices.shape}."
        )

    return matrices


def load_all_connectivity_runs_from_one_folder(
    npy_dir: str | Path,
    pattern: str = "connectivity_matrices_*.npy",

) -> tuple[list[np.ndarray], list[Path]]:
    """Load every learned-matrix stack for one statistical network class."""
    npy_dir = Path(npy_dir)

    if not npy_dir.exists():
        raise FileNotFoundError(f".npy directory does not exist: {npy_dir}")
    if not npy_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {npy_dir}")

    npy_files = sorted(npy_dir.glob(pattern), key=lambda p: natural_sort_key(p.name))

    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files matching '{pattern}' were found in: {npy_dir}"
        )

    # Each file stores the learned matrices produced from burst samples of one network.
    matrices = [load_one_npy_matrix_file(path) for path in npy_files]
    return matrices, npy_files


def load_ground_truth_connectivity_matrix(file_path: str | Path) -> tuple[np.ndarray, list]:
    """Load one YAML network as a non-negative directed adjacency matrix."""
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=YamlLoader)

    if data is None:
        raise ValueError(f"Ground-truth YAML file is empty: {file_path}")

    nodes = data.get("nodes", [])

    if isinstance(nodes, dict):
        node_list = []
        for k, v in nodes.items():
            try:
                nid = int(k)
            except Exception:
                nid = k
            node_list.append({"id": nid, **(v or {})})

    elif isinstance(nodes, list):
        node_list = nodes

    else:

        raise ValueError(f"'nodes' must be a list or a dict in {file_path}")

    id_order = [n["id"] for n in node_list]
    id_to_index = {nid: i for i, nid in enumerate(id_order)}
    N = len(id_order)
    matrix = np.zeros((N, N), dtype=float)

    for n in node_list:
        src = n["id"]
        src_idx = id_to_index[src]
        if "connections" in n and isinstance(n["connections"], list):
            for conn in n["connections"]:
                tgt = conn.get("target") or conn.get("to") or conn.get("id")
                if tgt is None:
                    continue

                w = conn.get("weight", conn.get("w", 0.0))

                if tgt in id_to_index:
                    matrix[src_idx, id_to_index[tgt]] = float(abs(w))

        else:
            targets = n.get("connectedTo") or n.get("targets") or []
            weights = n.get("weights") or n.get("w") or []

            for tgt, w in zip(targets, weights):
                if tgt in id_to_index:

                    matrix[src_idx, id_to_index[tgt]] = float(abs(w))

    return matrix, id_order


def load_ground_truth_matrices_for_folder(
    stats_name: str,
    num_expected: int,
    ground_truth_dir: str | Path,

) -> tuple[list[np.ndarray], list[Path], list[list]]:

    """Load the ground-truth networks corresponding to learned matrix files."""
    ground_truth_dir = Path(ground_truth_dir)

    if not ground_truth_dir.exists():

        raise FileNotFoundError(
            f"Ground-truth directory does not exist: {ground_truth_dir}"
        )
    if not ground_truth_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory for ground-truth files, got: {ground_truth_dir}"
        )


    true_matrices: list[np.ndarray] = []
    true_yaml_files: list[Path] = []
    true_id_orders: list[list] = []


    for idx in range(num_expected):

        yaml_path = ground_truth_dir / f"network_{stats_name}_{idx + 1}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing ground-truth YAML file: {yaml_path}")
        matrix, id_order = load_ground_truth_connectivity_matrix(yaml_path)
        true_matrices.append(matrix)
        true_yaml_files.append(yaml_path)
        true_id_orders.append(id_order)
    return true_matrices, true_yaml_files, true_id_orders


# -----------------------------------------------------------------------------
# Multi-network loading
# -----------------------------------------------------------------------------

def load_all_connectivity_runs_multiple_folders(
    folder_names: list[str],
    learned_connectivity_dir: str | Path,
    ground_truth_dir: str | Path,
    pattern: str = "connectivity_matrices_*.npy",
) -> dict[str, dict[str, object]]:
    """Load learned and ground-truth matrices for each statistical class."""
    results: dict[str, dict[str, object]] = {}

    for folder_name in folder_names:
        stats_name = folder_name

        npy_dir = resolve_npy_dir(folder_name, learned_connectivity_dir=learned_connectivity_dir)


        learned_matrices, npy_files = load_all_connectivity_runs_from_one_folder(
            npy_dir=npy_dir,
            pattern=pattern,
        )


        true_matrices, true_yaml_files, true_id_orders = load_ground_truth_matrices_for_folder(
            stats_name=stats_name,
            num_expected=len(learned_matrices),
            ground_truth_dir=ground_truth_dir,
        )

        if len(true_matrices) != len(learned_matrices):
            raise ValueError(
                f"Mismatch for {stats_name}: "
                f"{len(learned_matrices)} learned files but {len(true_matrices)} ground-truth files."
            )


        results[stats_name] = {
            "stats_name": stats_name,
            "npy_dir": npy_dir,
            "learned_matrices": learned_matrices,
            "npy_files": npy_files,
            "true_matrices": true_matrices,
            "true_yaml_files": true_yaml_files,
            "true_id_orders": true_id_orders,
        }
    return results


def load_default_experiment_data() -> dict[str, dict[str, object]]:
    """Load the statistical classes used by the default edge-classification run."""
    all_data = load_all_connectivity_runs_multiple_folders(
        folder_names=NETWORK_STATISTICS,
        learned_connectivity_dir=LEARNED_CONNECTIVITY_DIR,
        ground_truth_dir=GROUND_TRUTH_DIR,
        pattern=CONNECTIVITY_FILE_GLOB,
    )

    print("Loaded learned and ground-truth connectivity matrices:")
    print()

    for stats_name, data in all_data.items():

        npy_dir = data["npy_dir"]
        learned_matrices = data["learned_matrices"]
        npy_files = data["npy_files"]
        true_matrices = data["true_matrices"]
        true_yaml_files = data["true_yaml_files"]

        print(f"Stats setting: {stats_name}")
        print(f"Learned .npy directory: {Path(npy_dir).resolve()}")
        print(f"Loaded {len(learned_matrices)} learned .npy file(s)")
        print(f"Loaded {len(true_matrices)} ground-truth .yaml file(s)")

        for i, (npy_path, learned_arr, true_yaml_path, true_arr) in enumerate(
            zip(npy_files, learned_matrices, true_yaml_files, true_matrices),
            start=1,
        ):
            print(
                f"  Pair {i}: "
                f"{npy_path.name} -> learned shape={learned_arr.shape}; "
                f"{true_yaml_path.name} -> true shape={true_arr.shape}"
            )
        print()
    return all_data


# -----------------------------------------------------------------------------
# Feature construction and validation
# -----------------------------------------------------------------------------

def _validate_network_inputs(all_matrices_lists, all_A_t):

    """Validate learned and ground-truth matrix dimensions before feature extraction."""
    if len(all_matrices_lists) == 0:
        raise ValueError("all_matrices_lists is empty.")


    if len(all_matrices_lists) != len(all_A_t):

        raise ValueError(
            f"Length mismatch: len(all_matrices_lists)={len(all_matrices_lists)} "
            f"but len(all_A_t)={len(all_A_t)}"
        )


    for net_idx, matrices in enumerate(all_matrices_lists):
        if len(matrices) == 0:
            raise ValueError(f"Network {net_idx} contains no matrices.")

        network_shape = None

        for mat_idx, mat in enumerate(matrices):
            mat = np.asarray(mat)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"Matrix {mat_idx} in network {net_idx} is not square: shape={mat.shape}"
                )

            if network_shape is None:
                network_shape = mat.shape
            elif mat.shape != network_shape:
                raise ValueError(
                    f"Within network {net_idx}, learned matrix {mat_idx} has shape {mat.shape}, "
                    f"but expected {network_shape}"
                )

        A_t = np.asarray(all_A_t[net_idx])
        if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
            raise ValueError(
                f"Ground truth for network {net_idx} is not square: shape={A_t.shape}"
            )

        if A_t.shape != network_shape:
            raise ValueError(
                f"Ground truth shape mismatch at network {net_idx}: "
                f"got {A_t.shape}, expected {network_shape}"
            )


# -----------------------------------------------------------------------------
# Edge-level feature construction

def _build_edge_features_for_one_network(
    matrices,
    A_t=None,
    max_num_matrices=None,
    undirected=False,
    exclude_diagonal=True,
    use_per_matrix_features=True,
    add_summary_features=True,
):

    """Construct edge-level features and labels for one network realization."""
    mats = [np.asarray(m, dtype=float) for m in matrices]
    K_all = len(mats)
    N = mats[0].shape[0]

    if max_num_matrices is None:
        # Training fixes this width; test networks are padded to the same feature space.
        max_num_matrices = K_all


    stack_all = np.stack(mats, axis=0)  # (K_all, N, N)

    mask = np.ones((N, N), dtype=bool)

    if exclude_diagonal:

        np.fill_diagonal(mask, False)

    if undirected:

        tri_mask = np.triu(np.ones((N, N), dtype=bool), k=1 if exclude_diagonal else 0)
        mask &= tri_mask

    # Preserve source/target indices so predictions can be reconstructed as matrices.
    rows, cols = np.where(mask)


    actual_vals_all = stack_all[:, rows, cols].T  # (num_edges, K_all)


    X_parts = []
    feature_names = []


    if use_per_matrix_features:

        K_used = min(K_all, max_num_matrices)

        # Pad networks with fewer burst-specific matrices; the pipeline imputes these values.
        X_base = np.full((actual_vals_all.shape[0], max_num_matrices), np.nan, dtype=float)
        if K_used > 0:
            X_base[:, :K_used] = actual_vals_all[:, :K_used]

        X_parts.append(X_base)

        feature_names.extend([f"matrix_{i}" for i in range(max_num_matrices)])


    if add_summary_features:
        # Summary statistics aggregate each edge over all available burst-specific fits.
        X_parts.extend([
            actual_vals_all.mean(axis=1, keepdims=True),
            actual_vals_all.std(axis=1, keepdims=True),
            np.median(actual_vals_all, axis=1, keepdims=True),
            actual_vals_all.min(axis=1, keepdims=True),
            actual_vals_all.max(axis=1, keepdims=True),
        ])
        feature_names.extend(["mean", "std", "median", "min", "max"])


    if not X_parts:
        raise ValueError(
            "At least one of use_per_matrix_features or add_summary_features must be True."
        )


    X = np.hstack(X_parts)

    if A_t is None:
        return X, None, rows, cols, feature_names


    y = np.asarray(A_t)[rows, cols]


    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    rows = rows[valid]
    cols = cols[valid]
    return X, y.astype(int), rows, cols, feature_names


def _reconstruct_matrix_from_edge_values(
    edge_values,
    rows,
    cols,
    N,
    undirected=False,
    exclude_diagonal=True,
    fill_value=0.0,
    dtype=float,
):

    """Map flattened edge predictions back to an N x N matrix."""
    M = np.full((N, N), fill_value, dtype=dtype)
    M[rows, cols] = edge_values

    if undirected:
        M[cols, rows] = edge_values

    if exclude_diagonal:
        np.fill_diagonal(M, 0 if np.issubdtype(np.dtype(dtype), np.integer) else 0.0)

    return M


def _safe_binary_metrics(y_true, y_prob, y_pred):

    """Compute binary metrics while handling single-class ROC-AUC cases."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))

    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    return metrics


# -----------------------------------------------------------------------------
# Model training and evaluation
# -----------------------------------------------------------------------------
# Gradient-boosted edge classifier
def train_gbt_across_networks(
    all_matrices_lists,
    all_A_t,
    train_indices=None,
    test_indices=None,
    undirected=False,
    exclude_diagonal=True,
    use_per_matrix_features=True,
    add_summary_features=True,
    threshold=0.5,
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=1.0,
):

    """Fit and evaluate the directed-edge GBT using network-level train/test indices."""
    _validate_network_inputs(all_matrices_lists, all_A_t)


    num_networks = len(all_matrices_lists)

    if train_indices is None and test_indices is None:
        # Default to a deterministic network-level half split when indices are omitted.
        split = num_networks // 2
        train_indices = list(range(split))
        test_indices = list(range(split, num_networks))
    elif train_indices is None:
        train_indices = [i for i in range(num_networks) if i not in test_indices]
    elif test_indices is None:
        test_indices = [i for i in range(num_networks) if i not in train_indices]
    train_indices = list(train_indices)
    test_indices = list(test_indices)
    if len(set(train_indices).intersection(set(test_indices))) > 0:
        raise ValueError("train_indices and test_indices must not overlap.")
    if len(train_indices) == 0:
        raise ValueError("train_indices is empty.")

    # Derive feature width from training networks only to avoid test-set information leakage.
    max_num_matrices = max(len(all_matrices_lists[i]) for i in train_indices)

    X_train_all = []
    y_train_all = []

    feature_names = None

    for idx in train_indices:

        X_i, y_i, _, _, feature_names = _build_edge_features_for_one_network(
            matrices=all_matrices_lists[idx],
            A_t=all_A_t[idx],
            max_num_matrices=max_num_matrices,
            undirected=undirected,
            exclude_diagonal=exclude_diagonal,
            use_per_matrix_features=use_per_matrix_features,
            add_summary_features=add_summary_features,
        )
        X_train_all.append(X_i)
        y_train_all.append(y_i)

    X_train_all = np.vstack(X_train_all)
    y_train_all = np.concatenate(y_train_all)


    unique_y = np.unique(y_train_all)
    if not np.array_equal(np.sort(unique_y), np.array([0, 1])):
        raise ValueError(
            f"This function expects binary ground-truth labels 0/1. "
            f"Training labels found: {unique_y}"
        )

    # Median imputation handles padded/missing edge features before GBT fitting.
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )
    )


    model.fit(X_train_all, y_train_all)

    imputer = model.named_steps["simpleimputer"]

    gbt = model.named_steps["gradientboostingclassifier"]


    feature_names_for_importance = list(feature_names)

    if len(gbt.feature_importances_) != len(feature_names_for_importance):
        # Align feature names with columns retained by the fitted imputer.
        stats = getattr(imputer, "statistics_", None)
        if stats is not None and len(stats) == len(feature_names_for_importance):

            kept_mask = ~np.isnan(stats)
            feature_names_for_importance = list(np.asarray(feature_names_for_importance)[kept_mask])

        else:
            raise RuntimeError(
                f"Feature name / importance length mismatch: "
                f"{len(feature_names_for_importance)} names vs "
                f"{len(gbt.feature_importances_)} importances"
            )


    feature_importance = pd.DataFrame({
        "feature": feature_names_for_importance,
        "importance": gbt.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)


    test_predictions = {}
    all_test_true = []
    all_test_prob = []
    all_test_pred = []
    for idx in test_indices:
        X_test, y_test, rows, cols, _ = _build_edge_features_for_one_network(
            matrices=all_matrices_lists[idx],
            A_t=all_A_t[idx],
            max_num_matrices=max_num_matrices,
            undirected=undirected,
            exclude_diagonal=exclude_diagonal,
            use_per_matrix_features=use_per_matrix_features,
            add_summary_features=add_summary_features,
        )


        prob = model.predict_proba(X_test)[:, 1]
        # Convert probabilities to binary predictions at the requested operating threshold.
        pred = (prob >= threshold).astype(int)


        N = np.asarray(all_A_t[idx]).shape[0]

        prob_matrix = _reconstruct_matrix_from_edge_values(
            edge_values=prob,
            rows=rows,
            cols=cols,
            N=N,
            undirected=undirected,
            exclude_diagonal=exclude_diagonal,
            fill_value=0.0,
            dtype=float,
        )

        binary_matrix = _reconstruct_matrix_from_edge_values(
            edge_values=pred,
            rows=rows,
            cols=cols,
            N=N,
            undirected=undirected,
            exclude_diagonal=exclude_diagonal,
            fill_value=0,
            dtype=int,
        )


        metrics = _safe_binary_metrics(y_test, prob, pred)
        test_predictions[idx] = {
            "global_prob_matrix": prob_matrix,
            "global_binary_matrix": binary_matrix,
            "edge_probabilities": prob,
            "edge_predictions": pred,
            "edge_truth": y_test,
            "metrics": metrics,
        }


        all_test_true.append(y_test)
        all_test_prob.append(prob)
        all_test_pred.append(pred)


    all_test_true = np.concatenate(all_test_true)

    all_test_prob = np.concatenate(all_test_prob)

    all_test_pred = np.concatenate(all_test_pred)


    overall_test_metrics = _safe_binary_metrics(
        all_test_true, all_test_prob, all_test_pred
    )

    return {
        "model": model,
        "feature_importance": feature_importance,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "max_num_matrices": max_num_matrices,
        "test_predictions": test_predictions,
        "overall_test_metrics": overall_test_metrics,
    }


# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------
# Network-grouped train/test splitting
def flatten_loaded_data_by_group(all_data):
    """
    Convert the loader output 'all_data' into flat network-level lists.

    Each .npy file + corresponding ground-truth YAML file becomes one network-level example.
    """
    all_matrices_lists = []
    all_A_t = []
    group_labels = []
    network_metadata = []

    for stats_name in sorted(all_data.keys(), key=natural_sort_key):

        data = all_data[stats_name]
        learned_matrices = data["learned_matrices"]
        true_matrices = data["true_matrices"]

        npy_files = data["npy_files"]
        true_yaml_files = data["true_yaml_files"]
        if not (
            len(learned_matrices)
            == len(true_matrices)
            == len(npy_files)
            == len(true_yaml_files)
        ):
            raise ValueError(
                f"Length mismatch inside group {stats_name}: "
                f"{len(learned_matrices)=}, {len(true_matrices)=}, "
                f"{len(npy_files)=}, {len(true_yaml_files)=}"
            )
        for local_idx, (learned_arr, true_A, npy_path, yaml_path) in enumerate(
            zip(learned_matrices, true_matrices, npy_files, true_yaml_files)
        ):

            all_matrices_lists.append(np.asarray(learned_arr))
            all_A_t.append(np.asarray(true_A))
            group_labels.append(stats_name)

            # Retain file-level provenance for each network in the flattened dataset.
            network_metadata.append({
                "flat_index": len(all_matrices_lists) - 1,
                "group": stats_name,
                "group_local_index": local_idx,
                "npy_file": str(npy_path),
                "true_yaml_file": str(yaml_path),
                "num_learned_samples": int(np.asarray(learned_arr).shape[0]),
                "N": int(np.asarray(true_A).shape[0]),
            })
    return all_matrices_lists, all_A_t, group_labels, network_metadata


def make_grouped_train_test_split(
    group_labels,
    train_counts_by_group=None,
    test_counts_by_group=None,
    default_train_count=1,
    default_test_count=None,
    shuffle_within_group=True,
    random_state=42,
    require_at_least_one_train_per_group=True,
):
    """
    Build train/test flat indices while respecting group membership.

    Parameters
    ----------
    group_labels : list[str]
        One group label per flat network example.

    train_counts_by_group : dict[str, int] | None
        Exact train count for any listed group.

    test_counts_by_group : dict[str, int] | None
        Exact test count for any listed group.

    default_train_count : int
        Train count for groups not explicitly listed.

    default_test_count : int | None
        Test count for groups not explicitly listed.
        If None, use all remaining networks in that group after training selection.

    Returns
    -------
    dict with train_indices, test_indices, unused_indices, split_summary
    """
    if train_counts_by_group is None:
        train_counts_by_group = {}
    if test_counts_by_group is None:
        test_counts_by_group = {}


    group_to_indices = {}
    for idx, group in enumerate(group_labels):
        group_to_indices.setdefault(group, []).append(idx)

    # Use a local generator so the split is reproducible without changing global RNG state.
    rng = np.random.default_rng(random_state)


    train_indices = []
    test_indices = []
    unused_indices = []

    split_summary = []

    for group in sorted(group_to_indices.keys(), key=natural_sort_key):
        indices = list(group_to_indices[group])
        if shuffle_within_group:
            rng.shuffle(indices)
        n_total = len(indices)
        train_k = int(train_counts_by_group.get(group, default_train_count))
        if train_k < 0:
            raise ValueError(f"train count for group {group} cannot be negative.")
        if require_at_least_one_train_per_group and train_k < 1:
            raise ValueError(
                f"Group {group} must contribute at least one training network, "
                f"but train_k={train_k}."
            )
        if train_k > n_total:
            raise ValueError(
                f"Group {group} requested {train_k} training networks but only has {n_total}."
            )

        remaining_after_train = n_total - train_k
        if group in test_counts_by_group:
            test_k = int(test_counts_by_group[group])
        elif default_test_count is None:
            test_k = remaining_after_train
        else:
            test_k = int(default_test_count)
        if test_k < 0:
            raise ValueError(f"test count for group {group} cannot be negative.")
        if train_k + test_k > n_total:
            raise ValueError(
                f"Group {group} requested train={train_k}, test={test_k}, "
                f"but only has {n_total} total networks."
            )


        group_train = indices[:train_k]
        group_test = indices[train_k : train_k + test_k]
        group_unused = indices[train_k + test_k :]


        train_indices.extend(group_train)
        test_indices.extend(group_test)
        unused_indices.extend(group_unused)


        split_summary.append({
            "group": group,
            "n_total": n_total,
            "n_train": len(group_train),
            "n_test": len(group_test),
            "n_unused": len(group_unused),
            "train_indices": group_train,
            "test_indices": group_test,
            "unused_indices": group_unused,
        })


    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)
    unused_indices = sorted(unused_indices)


    if set(train_indices).intersection(test_indices):
        raise RuntimeError("train_indices and test_indices overlap after grouped split.")

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "unused_indices": unused_indices,
        "split_summary": split_summary,
    }


def train_gbt_across_grouped_networks(
    all_data,
    train_counts_by_group=None,
    test_counts_by_group=None,
    default_train_count=1,
    default_test_count=None,
    shuffle_within_group=True,
    split_random_state=42,
    undirected=False,
    exclude_diagonal=True,
    use_per_matrix_features=True,
    add_summary_features=True,
    threshold=0.5,
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=1.0,
):

    # Flatten the grouped loader output, then split at the network level before fitting.
    """Flatten statistical classes, build a grouped split, and train the edge classifier."""
    all_matrices_lists, all_A_t, group_labels, network_metadata = flatten_loaded_data_by_group(all_data)


    split_info = make_grouped_train_test_split(
        group_labels=group_labels,
        train_counts_by_group=train_counts_by_group,
        test_counts_by_group=test_counts_by_group,
        default_train_count=default_train_count,
        default_test_count=default_test_count,
        shuffle_within_group=shuffle_within_group,
        random_state=split_random_state,
        require_at_least_one_train_per_group=True,
    )


    results = train_gbt_across_networks(
        all_matrices_lists=all_matrices_lists,
        all_A_t=all_A_t,
        train_indices=split_info["train_indices"],
        test_indices=split_info["test_indices"],
        undirected=undirected,
        exclude_diagonal=exclude_diagonal,
        use_per_matrix_features=use_per_matrix_features,
        add_summary_features=add_summary_features,
        threshold=threshold,
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
    )


    split_summary_df = pd.DataFrame(split_info["split_summary"])


    train_metadata = [network_metadata[i] for i in split_info["train_indices"]]
    test_metadata = [network_metadata[i] for i in split_info["test_indices"]]
    unused_metadata = [network_metadata[i] for i in split_info["unused_indices"]]


    results.update({
        "all_matrices_lists": all_matrices_lists,
        "all_A_t": all_A_t,
        "group_labels": group_labels,
        "network_metadata": network_metadata,
        "split_summary": split_summary_df,
        "unused_indices": split_info["unused_indices"],
        "train_network_metadata": train_metadata,
        "test_network_metadata": test_metadata,
        "unused_network_metadata": unused_metadata,
    })
    return results


def save_edge_classifier_outputs(
    results: dict[str, object],
    output_dir: str | Path = PROBABILITY_OUTPUT_DIR,
) -> pd.DataFrame:
    """Persist held-out probability matrices and compact classifier summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_index = {
        int(item["flat_index"]): item
        for item in results["test_network_metadata"]
    }
    manifest_rows: list[dict[str, object]] = []

    for flat_index, prediction in results["test_predictions"].items():
        metadata = metadata_by_index[int(flat_index)]
        truth_stem = Path(str(metadata["true_yaml_file"])).stem
        if not truth_stem.startswith("network_"):
            raise ValueError(
                f"Unexpected ground-truth filename: {metadata['true_yaml_file']}"
            )

        network_id = truth_stem.removeprefix("network_")
        probability_path = output_dir / f"prob_matrix_{network_id}.npy"
        np.save(probability_path, np.asarray(prediction["global_prob_matrix"], dtype=float))

        manifest_rows.append(
            {
                "flat_index": int(flat_index),
                "network_id": network_id,
                "group": metadata["group"],
                "probability_matrix": str(probability_path),
                **prediction["metrics"],
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("flat_index").reset_index(drop=True)
    manifest.to_csv(output_dir / "edge_classifier_test_network_metrics.csv", index=False)
    results["feature_importance"].to_csv(
        output_dir / "edge_classifier_feature_importance.csv",
        index=False,
    )
    results["split_summary"].to_csv(
        output_dir / "edge_classifier_split_summary.csv",
        index=False,
    )
    return manifest


# -----------------------------------------------------------------------------
# Default thesis experiment
# -----------------------------------------------------------------------------

def run_default_experiment() -> dict[str, object]:
    """Run the default grouped edge-classification experiment."""
    all_data = load_default_experiment_data()

    results_gbt = train_gbt_across_grouped_networks(
        all_data=all_data,
        train_counts_by_group={
            "N100_p12_CC01": 5,
            "N100_p12_CC03": 5,
            "N100_p12_CC05": 5,
            "N100_p24_CC01": 5,
            "N100_p24_CC03": 5,
            "N100_p24_CC05": 5,
            "N100_p36_CC01": 5,
            "N100_p36_CC03": 5,
            "N100_p36_CC05": 5,

        },

        test_counts_by_group={
            "N100_p12_CC01": 6,
            "N100_p12_CC03": 6,
            "N100_p12_CC05": 6,
            "N100_p24_CC01": 6,
            "N100_p24_CC03": 6,
            "N100_p24_CC05": 6,
            "N100_p36_CC01": 6,
            "N100_p36_CC03": 6,
            "N100_p36_CC05": 6,
        },

        # Fallback counts apply only to groups not listed explicitly above.
        default_train_count=1,
        default_test_count=1,
        shuffle_within_group=True,
        split_random_state=42,

        undirected=False,
        exclude_diagonal=True,
        use_per_matrix_features=False,
        add_summary_features=True,
        threshold=0.9,
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
    )

    output_manifest = save_edge_classifier_outputs(results_gbt)

    print(results_gbt["overall_test_metrics"])
    print(results_gbt["feature_importance"].head(10))
    print(results_gbt["split_summary"])
    print(f"Saved {len(output_manifest)} held-out probability matrices to {PROBABILITY_OUTPUT_DIR}.")
    return results_gbt


if __name__ == "__main__":
    run_default_experiment()
