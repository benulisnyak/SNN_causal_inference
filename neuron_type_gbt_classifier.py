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
except ImportError:
    from yaml import SafeLoader as YamlLoader


# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------

network_stats_list = [
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

base_dir = Path(".")
learned_base_dir = base_dir / "LIFoutput_files"
ground_truth_base_dir = base_dir / "networks"
npy_pattern = "connectivity_matrices_*.npy"


# -----------------------------------------------------------------------------
# loading helpers
# -----------------------------------------------------------------------------

def natural_sort_key(text: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def resolve_npy_dir(folder_name: str, learned_base_dir: str | Path) -> Path:
    return Path(learned_base_dir) / folder_name


def load_one_npy_matrix_file(npy_path: str | Path) -> np.ndarray:
    npy_path = Path(npy_path)
    matrices = np.load(npy_path)

    if matrices.ndim != 3:
        raise ValueError(
            f"Expected shape (num_matrices, N, N) in {npy_path}, got {matrices.shape}."
        )

    return np.asarray(matrices, dtype=float)



def load_all_connectivity_runs_from_one_folder(
    npy_dir: str | Path,
    pattern: str = "connectivity_matrices_*.npy",
) -> tuple[list[np.ndarray], list[Path]]:
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

    matrices = [load_one_npy_matrix_file(path) for path in npy_files]
    return matrices, npy_files



def load_ground_truth_signed_connectivity_matrix(
    file_path: str | Path,
) -> tuple[np.ndarray, list]:
    """
    Load the ground-truth connectivity matrix and preserve the sign of each weight.

    Positive outgoing weights => excitatory source neuron.
    Negative outgoing weights => inhibitory source neuron.
    """
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
    n_nodes = len(id_order)
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)

    for node in node_list:
        src = node["id"]
        src_idx = id_to_index[src]

        if "connections" in node and isinstance(node["connections"], list):
            for conn in node["connections"]:
                tgt = conn.get("target") or conn.get("to") or conn.get("id")
                if tgt is None or tgt not in id_to_index:
                    continue
                w = conn.get("weight", conn.get("w", 0.0))
                matrix[src_idx, id_to_index[tgt]] = float(w)
        else:
            targets = node.get("connectedTo") or node.get("targets") or []
            weights = node.get("weights") or node.get("w") or []
            for tgt, w in zip(targets, weights):
                if tgt in id_to_index:
                    matrix[src_idx, id_to_index[tgt]] = float(w)

    return matrix, id_order



def load_ground_truth_matrices_for_folder(
    stats_name: str,
    num_expected: int,
    ground_truth_dir: str | Path,
) -> tuple[list[np.ndarray], list[Path], list[list]]:
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
        matrix, id_order = load_ground_truth_signed_connectivity_matrix(yaml_path)
        true_matrices.append(matrix)
        true_yaml_files.append(yaml_path)
        true_id_orders.append(id_order)

    return true_matrices, true_yaml_files, true_id_orders



def load_all_connectivity_runs_multiple_folders(
    folder_names: list[str],
    learned_base_dir: str | Path,
    ground_truth_base_dir: str | Path,
    npy_pattern: str = "connectivity_matrices_*.npy",
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}

    for folder_name in folder_names:
        stats_name = folder_name
        npy_dir = resolve_npy_dir(folder_name, learned_base_dir=learned_base_dir)

        learned_matrices, npy_files = load_all_connectivity_runs_from_one_folder(
            npy_dir=npy_dir,
            pattern=npy_pattern,
        )

        true_matrices, true_yaml_files, true_id_orders = load_ground_truth_matrices_for_folder(
            stats_name=stats_name,
            num_expected=len(learned_matrices),
            ground_truth_dir=ground_truth_base_dir,
        )

        if len(true_matrices) != len(learned_matrices):
            raise ValueError(
                f"Mismatch for {stats_name}: {len(learned_matrices)} learned files "
                f"but {len(true_matrices)} ground-truth files."
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



def main() -> dict[str, dict[str, object]]:
    all_data = load_all_connectivity_runs_multiple_folders(
        folder_names=network_stats_list,
        learned_base_dir=learned_base_dir,
        ground_truth_base_dir=ground_truth_base_dir,
        npy_pattern=npy_pattern,
    )

    print("Loaded learned and signed ground-truth connectivity matrices:")
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
                f"  Pair {i}: {npy_path.name} -> learned shape={learned_arr.shape}; "
                f"{true_yaml_path.name} -> true shape={true_arr.shape}"
            )
        print()

    return all_data


# -----------------------------------------------------------------------------
# neuron-type labels
# -----------------------------------------------------------------------------

def derive_neuron_type_labels_from_ground_truth(
    signed_A_t: np.ndarray,
    *,
    eps: float = 1e-12,
    unlabeled_value: float = np.nan,
) -> np.ndarray:
    """
    Build one binary label per neuron from the sign of its outgoing ground-truth weights.

    Returns
    -------
    labels : np.ndarray, shape (N,)
        1 = excitatory (all nonzero outgoing weights positive)
        0 = inhibitory (all nonzero outgoing weights negative)
        NaN = neuron has no nonzero outgoing weights, so label is unavailable
    """
    signed_A_t = np.asarray(signed_A_t, dtype=float)
    if signed_A_t.ndim != 2 or signed_A_t.shape[0] != signed_A_t.shape[1]:
        raise ValueError(f"signed_A_t must be square. Got shape={signed_A_t.shape}")

    n_nodes = signed_A_t.shape[0]
    labels = np.full(n_nodes, unlabeled_value, dtype=float)

    for i in range(n_nodes):
        outgoing = signed_A_t[i, :].copy()
        if i < outgoing.size:
            outgoing[i] = 0.0
        nz = outgoing[np.abs(outgoing) > eps]

        if nz.size == 0:
            continue

        has_pos = np.any(nz > 0)
        has_neg = np.any(nz < 0)

        if has_pos and has_neg:
            raise ValueError(
                "Neuron has mixed-sign outgoing ground-truth weights. "
                f"Neuron index={i}, positive_count={np.sum(nz > 0)}, negative_count={np.sum(nz < 0)}"
            )

        labels[i] = 1.0 if has_pos else 0.0

    return labels


# -----------------------------------------------------------------------------
# validation
# -----------------------------------------------------------------------------

def _validate_network_inputs(
    all_matrices_lists: list[np.ndarray],
    all_signed_A_t: list[np.ndarray],
) -> None:
    if len(all_matrices_lists) == 0:
        raise ValueError("all_matrices_lists is empty.")

    if len(all_matrices_lists) != len(all_signed_A_t):
        raise ValueError(
            f"Length mismatch: len(all_matrices_lists)={len(all_matrices_lists)} "
            f"but len(all_signed_A_t)={len(all_signed_A_t)}"
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

        A_t = np.asarray(all_signed_A_t[net_idx])
        if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
            raise ValueError(
                f"Ground truth for network {net_idx} is not square: shape={A_t.shape}"
            )
        if A_t.shape != network_shape:
            raise ValueError(
                f"Ground truth shape mismatch at network {net_idx}: got {A_t.shape}, expected {network_shape}"
            )


# -----------------------------------------------------------------------------
# feature construction
# -----------------------------------------------------------------------------

def _node_stats_for_one_matrix(
    matrix: np.ndarray,
    *,
    exclude_diagonal: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute node-level statistics for one learned connectivity matrix.

    Returns an array of shape (N, num_base_features).
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"matrix must be square. Got {M.shape}")

    n_nodes = M.shape[0]
    if exclude_diagonal:
        M = M.copy()
        np.fill_diagonal(M, np.nan)

    feature_blocks = [
        np.nanmean(M, axis=1, keepdims=True),   # outgoing mean
        np.nanstd(M, axis=1, keepdims=True),    # outgoing std
        np.nanmedian(M, axis=1, keepdims=True), # outgoing median
        np.nanmin(M, axis=1, keepdims=True),    # outgoing min
        np.nanmax(M, axis=1, keepdims=True),    # outgoing max
        np.nansum(M, axis=1, keepdims=True),    # outgoing sum
        np.nanmean(np.abs(M), axis=1, keepdims=True),  # outgoing abs mean
        np.nansum(np.abs(M), axis=1, keepdims=True),   # outgoing abs sum
        np.mean(np.abs(np.nan_to_num(M, nan=0.0)) > 0, axis=1, keepdims=True),
        np.nanmean(M, axis=0, keepdims=True).T,   # incoming mean
        np.nanstd(M, axis=0, keepdims=True).T,    # incoming std
        np.nanmedian(M, axis=0, keepdims=True).T, # incoming median
        np.nanmin(M, axis=0, keepdims=True).T,    # incoming min
        np.nanmax(M, axis=0, keepdims=True).T,    # incoming max
        np.nansum(M, axis=0, keepdims=True).T,    # incoming sum
        np.nanmean(np.abs(M), axis=0, keepdims=True).T,  # incoming abs mean
        np.nansum(np.abs(M), axis=0, keepdims=True).T,   # incoming abs sum
        np.mean(np.abs(np.nan_to_num(M, nan=0.0)) > 0, axis=0, keepdims=True).T,
    ]

    feature_names = [
        "out_mean",
        "out_std",
        "out_median",
        "out_min",
        "out_max",
        "out_sum",
        "out_abs_mean",
        "out_abs_sum",
        "out_nonzero_frac",
        "in_mean",
        "in_std",
        "in_median",
        "in_min",
        "in_max",
        "in_sum",
        "in_abs_mean",
        "in_abs_sum",
        "in_nonzero_frac",
    ]

    X = np.hstack(feature_blocks)
    if X.shape != (n_nodes, len(feature_names)):
        raise RuntimeError(
            f"Unexpected node feature shape {X.shape}, expected {(n_nodes, len(feature_names))}."
        )

    return X, feature_names



def _build_node_features_for_one_network(
    matrices: np.ndarray,
    signed_A_t: np.ndarray | None = None,
    *,
    max_num_matrices: int | None = None,
    exclude_diagonal: bool = True,
    use_per_matrix_features: bool = True,
    add_summary_features: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[str]]:
    mats = [np.asarray(m, dtype=float) for m in matrices]
    K_all = len(mats)
    n_nodes = mats[0].shape[0]

    if max_num_matrices is None:
        max_num_matrices = K_all

    per_matrix_feature_list = []
    base_feature_names = None
    for mat in mats:
        X_one, base_feature_names = _node_stats_for_one_matrix(
            mat,
            exclude_diagonal=exclude_diagonal,
        )
        per_matrix_feature_list.append(X_one)

    per_matrix_stack = np.stack(per_matrix_feature_list, axis=0)  # (K_all, N, F)
    num_base_features = per_matrix_stack.shape[2]

    X_parts = []
    feature_names = []

    if use_per_matrix_features:
        K_used = min(K_all, max_num_matrices)
        X_base = np.full(
            (n_nodes, max_num_matrices * num_base_features),
            np.nan,
            dtype=float,
        )

        if K_used > 0:
            block = per_matrix_stack[:K_used].transpose(1, 0, 2).reshape(n_nodes, K_used * num_base_features)
            X_base[:, : K_used * num_base_features] = block

        X_parts.append(X_base)
        for matrix_idx in range(max_num_matrices):
            for feat_name in base_feature_names:
                feature_names.append(f"matrix_{matrix_idx}_{feat_name}")

    if add_summary_features:
        for reducer_name, reducer in [
            ("mean", np.mean),
            ("std", np.std),
            ("median", np.median),
            ("min", np.min),
            ("max", np.max),
        ]:
            summary_block = reducer(per_matrix_stack, axis=0)
            X_parts.append(summary_block)
            for feat_name in base_feature_names:
                feature_names.append(f"{reducer_name}_{feat_name}")

    if not X_parts:
        raise ValueError(
            "At least one of use_per_matrix_features or add_summary_features must be True."
        )

    X = np.hstack(X_parts)
    neuron_indices = np.arange(n_nodes)

    if signed_A_t is None:
        return X, None, neuron_indices, feature_names

    y = derive_neuron_type_labels_from_ground_truth(signed_A_t)
    valid = ~np.isnan(y)

    X = X[valid]
    y = y[valid].astype(int)
    neuron_indices = neuron_indices[valid]
    return X, y, neuron_indices, feature_names



def _reconstruct_neuron_vector(
    values: np.ndarray,
    neuron_indices: np.ndarray,
    n_nodes: int,
    *,
    fill_value: float,
    dtype: type | np.dtype,
) -> np.ndarray:
    out = np.full(n_nodes, fill_value, dtype=dtype)
    out[neuron_indices] = values
    return out



def _safe_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
# main trainer
# -----------------------------------------------------------------------------

def train_gbt_across_networks(
    all_matrices_lists: list[np.ndarray],
    all_signed_A_t: list[np.ndarray],
    *,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    exclude_diagonal: bool = True,
    use_per_matrix_features: bool = True,
    add_summary_features: bool = True,
    threshold: float = 0.5,
    random_state: int = 42,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    subsample: float = 1.0,
) -> dict[str, object]:
    _validate_network_inputs(all_matrices_lists, all_signed_A_t)

    num_networks = len(all_matrices_lists)

    if train_indices is None and test_indices is None:
        split = num_networks // 2
        train_indices = list(range(split))
        test_indices = list(range(split, num_networks))
    elif train_indices is None:
        train_indices = [i for i in range(num_networks) if i not in test_indices]
    elif test_indices is None:
        test_indices = [i for i in range(num_networks) if i not in train_indices]

    train_indices = list(train_indices)
    test_indices = list(test_indices)

    if set(train_indices).intersection(test_indices):
        raise ValueError("train_indices and test_indices must not overlap.")
    if len(train_indices) == 0:
        raise ValueError("train_indices is empty.")

    max_num_matrices = max(len(all_matrices_lists[i]) for i in train_indices)

    X_train_all = []
    y_train_all = []
    feature_names = None

    for idx in train_indices:
        X_i, y_i, _, feature_names = _build_node_features_for_one_network(
            matrices=all_matrices_lists[idx],
            signed_A_t=all_signed_A_t[idx],
            max_num_matrices=max_num_matrices,
            exclude_diagonal=exclude_diagonal,
            use_per_matrix_features=use_per_matrix_features,
            add_summary_features=add_summary_features,
        )
        if len(y_i) == 0:
            raise ValueError(f"Training network {idx} has no labeled neurons.")
        X_train_all.append(X_i)
        y_train_all.append(y_i)

    X_train_all = np.vstack(X_train_all)
    y_train_all = np.concatenate(y_train_all)

    unique_y = np.unique(y_train_all)
    if not np.array_equal(np.sort(unique_y), np.array([0, 1])):
        raise ValueError(
            "This function expects binary neuron-type labels 0/1. "
            f"Training labels found: {unique_y}"
        )

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        ),
    )

    model.fit(X_train_all, y_train_all)

    imputer = model.named_steps["simpleimputer"]
    gbt = model.named_steps["gradientboostingclassifier"]

    feature_names_for_importance = list(feature_names)
    if len(gbt.feature_importances_) != len(feature_names_for_importance):
        stats = getattr(imputer, "statistics_", None)
        if stats is not None and len(stats) == len(feature_names_for_importance):
            kept_mask = ~np.isnan(stats)
            feature_names_for_importance = list(np.asarray(feature_names_for_importance)[kept_mask])
        else:
            raise RuntimeError(
                f"Feature name / importance length mismatch: {len(feature_names_for_importance)} names vs "
                f"{len(gbt.feature_importances_)} importances"
            )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names_for_importance,
            "importance": gbt.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    test_predictions: dict[int, dict[str, object]] = {}
    all_test_true = []
    all_test_prob = []
    all_test_pred = []

    for idx in test_indices:
        X_test_labeled, y_test, labeled_neuron_indices, _ = _build_node_features_for_one_network(
            matrices=all_matrices_lists[idx],
            signed_A_t=all_signed_A_t[idx],
            max_num_matrices=max_num_matrices,
            exclude_diagonal=exclude_diagonal,
            use_per_matrix_features=use_per_matrix_features,
            add_summary_features=add_summary_features,
        )

        X_test_all, _, all_neuron_indices, _ = _build_node_features_for_one_network(
            matrices=all_matrices_lists[idx],
            signed_A_t=None,
            max_num_matrices=max_num_matrices,
            exclude_diagonal=exclude_diagonal,
            use_per_matrix_features=use_per_matrix_features,
            add_summary_features=add_summary_features,
        )

        prob_all = model.predict_proba(X_test_all)[:, 1]
        pred_all = (prob_all >= threshold).astype(int)

        prob_labeled = model.predict_proba(X_test_labeled)[:, 1]
        pred_labeled = (prob_labeled >= threshold).astype(int)

        n_nodes = np.asarray(all_signed_A_t[idx]).shape[0]
        truth_labels_full = derive_neuron_type_labels_from_ground_truth(all_signed_A_t[idx])

        prob_vector = _reconstruct_neuron_vector(
            prob_all,
            all_neuron_indices,
            n_nodes,
            fill_value=np.nan,
            dtype=float,
        )
        binary_vector = _reconstruct_neuron_vector(
            pred_all,
            all_neuron_indices,
            n_nodes,
            fill_value=-1,
            dtype=int,
        )

        metrics = _safe_binary_metrics(y_test, prob_labeled, pred_labeled)
        test_predictions[idx] = {
            "global_prob_vector": prob_vector,
            "global_binary_vector": binary_vector,
            "global_truth_vector": truth_labels_full,
            "labeled_neuron_indices": labeled_neuron_indices,
            "node_probabilities": prob_labeled,
            "node_predictions": pred_labeled,
            "node_truth": y_test,
            "metrics": metrics,
        }

        all_test_true.append(y_test)
        all_test_prob.append(prob_labeled)
        all_test_pred.append(pred_labeled)

    all_test_true = np.concatenate(all_test_true)
    all_test_prob = np.concatenate(all_test_prob)
    all_test_pred = np.concatenate(all_test_pred)

    overall_test_metrics = _safe_binary_metrics(
        all_test_true,
        all_test_prob,
        all_test_pred,
    )

    return {
        "model": model,
        "feature_importance": feature_importance,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "max_num_matrices": max_num_matrices,
        "test_predictions": test_predictions,
        "overall_test_metrics": overall_test_metrics,
        "label_definition": {0: "inhibitory", 1: "excitatory"},
    }


# -----------------------------------------------------------------------------
# dataset preparation
# -----------------------------------------------------------------------------

def flatten_loaded_data_by_group(
    all_data: dict[str, dict[str, object]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[dict[str, object]]]:
    all_matrices_lists: list[np.ndarray] = []
    all_signed_A_t: list[np.ndarray] = []
    group_labels: list[str] = []
    network_metadata: list[dict[str, object]] = []

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
            neuron_labels = derive_neuron_type_labels_from_ground_truth(true_A)
            labeled_count = int(np.sum(~np.isnan(neuron_labels)))
            excitatory_count = int(np.sum(neuron_labels == 1))
            inhibitory_count = int(np.sum(neuron_labels == 0))

            all_matrices_lists.append(np.asarray(learned_arr, dtype=float))
            all_signed_A_t.append(np.asarray(true_A, dtype=float))
            group_labels.append(stats_name)
            network_metadata.append(
                {
                    "flat_index": len(all_matrices_lists) - 1,
                    "group": stats_name,
                    "group_local_index": local_idx,
                    "npy_file": str(npy_path),
                    "true_yaml_file": str(yaml_path),
                    "num_learned_samples": int(np.asarray(learned_arr).shape[0]),
                    "N": int(np.asarray(true_A).shape[0]),
                    "num_labeled_neurons": labeled_count,
                    "num_excitatory_neurons": excitatory_count,
                    "num_inhibitory_neurons": inhibitory_count,
                }
            )

    return all_matrices_lists, all_signed_A_t, group_labels, network_metadata



def make_grouped_train_test_split(
    group_labels: list[str],
    *,
    train_counts_by_group: dict[str, int] | None = None,
    test_counts_by_group: dict[str, int] | None = None,
    default_train_count: int = 1,
    default_test_count: int | None = None,
    shuffle_within_group: bool = True,
    random_state: int = 42,
    require_at_least_one_train_per_group: bool = True,
) -> dict[str, object]:
    if train_counts_by_group is None:
        train_counts_by_group = {}
    if test_counts_by_group is None:
        test_counts_by_group = {}

    group_to_indices: dict[str, list[int]] = {}
    for idx, group in enumerate(group_labels):
        group_to_indices.setdefault(group, []).append(idx)

    rng = np.random.default_rng(random_state)

    train_indices: list[int] = []
    test_indices: list[int] = []
    unused_indices: list[int] = []
    split_summary: list[dict[str, object]] = []

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
                f"Group {group} must contribute at least one training network, but train_k={train_k}."
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
                f"Group {group} requested train={train_k}, test={test_k}, but only has {n_total} total networks."
            )

        group_train = indices[:train_k]
        group_test = indices[train_k : train_k + test_k]
        group_unused = indices[train_k + test_k :]

        train_indices.extend(group_train)
        test_indices.extend(group_test)
        unused_indices.extend(group_unused)

        split_summary.append(
            {
                "group": group,
                "n_total": n_total,
                "n_train": len(group_train),
                "n_test": len(group_test),
                "n_unused": len(group_unused),
                "train_indices": group_train,
                "test_indices": group_test,
                "unused_indices": group_unused,
            }
        )

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
    all_data: dict[str, dict[str, object]],
    *,
    train_counts_by_group: dict[str, int] | None = None,
    test_counts_by_group: dict[str, int] | None = None,
    default_train_count: int = 1,
    default_test_count: int | None = None,
    shuffle_within_group: bool = True,
    split_random_state: int = 42,
    exclude_diagonal: bool = True,
    use_per_matrix_features: bool = True,
    add_summary_features: bool = True,
    threshold: float = 0.5,
    random_state: int = 42,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    subsample: float = 1.0,
) -> dict[str, object]:
    all_matrices_lists, all_signed_A_t, group_labels, network_metadata = flatten_loaded_data_by_group(all_data)

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
        all_signed_A_t=all_signed_A_t,
        train_indices=split_info["train_indices"],
        test_indices=split_info["test_indices"],
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

    results.update(
        {
            "all_matrices_lists": all_matrices_lists,
            "all_signed_A_t": all_signed_A_t,
            "group_labels": group_labels,
            "network_metadata": network_metadata,
            "split_summary": split_summary_df,
            "unused_indices": split_info["unused_indices"],
            "train_network_metadata": train_metadata,
            "test_network_metadata": test_metadata,
            "unused_network_metadata": unused_metadata,
        }
    )
    return results


# -----------------------------------------------------------------------------
# example run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    all_data = main()

    results_gbt = train_gbt_across_grouped_networks(
        all_data=all_data,
        train_counts_by_group={
            "N100_p12_CC01": 10,
            "N100_p12_CC03": 10,
            "N100_p12_CC05": 10,
            "N100_p24_CC01": 10,
            "N100_p24_CC03": 10,
            "N100_p24_CC05": 10,
            "N100_p36_CC01": 10,
            "N100_p36_CC03": 10,
            "N100_p36_CC05": 10,
        },
        test_counts_by_group={
            "N100_p12_CC01": 1,
            "N100_p12_CC03": 1,
            "N100_p12_CC05": 1,
            "N100_p24_CC01": 1,
            "N100_p24_CC03": 1,
            "N100_p24_CC05": 1,
            "N100_p36_CC01": 1,
            "N100_p36_CC03": 1,
            "N100_p36_CC05": 1,
        },
        default_train_count=1,
        default_test_count=1,
        shuffle_within_group=False,
        split_random_state=42,
        exclude_diagonal=True,
        use_per_matrix_features=False,
        add_summary_features=True,
        threshold=0.5,
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
    )

    print("Overall test metrics:")
    print(results_gbt["overall_test_metrics"])
    print()

    print("Top 10 feature importances:")
    print(results_gbt["feature_importance"].head(10))
    print()

    print("Split summary:")
    print(results_gbt["split_summary"])
    print()

    print("Label definition:")
    print(results_gbt["label_definition"])
