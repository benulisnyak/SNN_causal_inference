from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
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
ei_output_base_dir = base_dir / "ei_prob_matrices"


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
        1 = inhibitory (all nonzero outgoing weights negative)
        0 = excitatory (all nonzero outgoing weights positive)
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

        labels[i] = 1.0 if has_neg else 0.0

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

    Only the requested five summary statistics are used for both outgoing and
    incoming connections: mean, minimum, maximum, standard deviation, and
    median.

    Returns an array of shape (N, 10).
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
        np.nanmin(M, axis=1, keepdims=True),    # outgoing min
        np.nanmax(M, axis=1, keepdims=True),    # outgoing max
        np.nanstd(M, axis=1, keepdims=True),    # outgoing std
        np.nanmedian(M, axis=1, keepdims=True), # outgoing median
        np.nanmean(M, axis=0, keepdims=True).T,   # incoming mean
        np.nanmin(M, axis=0, keepdims=True).T,    # incoming min
        np.nanmax(M, axis=0, keepdims=True).T,    # incoming max
        np.nanstd(M, axis=0, keepdims=True).T,    # incoming std
        np.nanmedian(M, axis=0, keepdims=True).T, # incoming median
    ]

    feature_names = [
        "out_mean",
        "out_min",
        "out_max",
        "out_std",
        "out_median",
        "in_mean",
        "in_min",
        "in_max",
        "in_std",
        "in_median",
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
            block = per_matrix_stack[:K_used].transpose(1, 0, 2).reshape(
                n_nodes,
                K_used * num_base_features,
            )
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


def _safe_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(auc(pr_recall, pr_precision))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["average_precision"] = np.nan

    return metrics


def _numeric_series_without_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


# -----------------------------------------------------------------------------
# saving helpers
# -----------------------------------------------------------------------------

def save_test_prediction_arrays(
    test_predictions: dict[int, dict[str, object]],
    test_network_metadata: list[dict[str, object]],
    output_dir: str | Path = "ei_prob_matrices",
    *,
    prob_prefix: str = "ei_prob_vector",
    binary_prefix: str = "ei_binary_vector",
    truth_prefix: str = "ei_truth_vector",
    prob_dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_rows: list[dict[str, object]] = []

    for meta in test_network_metadata:
        flat_index = meta["flat_index"]
        if flat_index not in test_predictions:
            continue

        stats_name = meta["group"]
        network_number = meta["group_local_index"] + 1
        pred_entry = test_predictions[flat_index]

        prob_vector = np.asarray(pred_entry["global_prob_vector"], dtype=prob_dtype)
        binary_vector = np.asarray(pred_entry["global_binary_vector"], dtype=np.int16)
        truth_vector = np.asarray(pred_entry["global_truth_vector"], dtype=np.float32)

        prob_path = output_dir / f"{prob_prefix}_{stats_name}_{network_number}.npy"
        binary_path = output_dir / f"{binary_prefix}_{stats_name}_{network_number}.npy"
        truth_path = output_dir / f"{truth_prefix}_{stats_name}_{network_number}.npy"

        np.save(prob_path, prob_vector)
        np.save(binary_path, binary_vector)
        np.save(truth_path, truth_vector)

        saved_rows.append(
            {
                "flat_index": flat_index,
                "group": stats_name,
                "group_local_index": meta["group_local_index"],
                "network_number": network_number,
                "npy_file": meta["npy_file"],
                "true_yaml_file": meta["true_yaml_file"],
                "prob_vector_path": str(prob_path),
                "binary_vector_path": str(binary_path),
                "truth_vector_path": str(truth_path),
            }
        )

    return pd.DataFrame(saved_rows)


def build_test_network_auc_summary(
    test_predictions: dict[int, dict[str, object]],
    test_network_metadata: list[dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for meta in test_network_metadata:
        flat_index = meta["flat_index"]
        if flat_index not in test_predictions:
            continue

        pred_entry = test_predictions[flat_index]
        metrics = pred_entry["metrics"]

        rows.append(
            {
                "flat_index": flat_index,
                "group": meta["group"],
                "group_local_index": meta["group_local_index"],
                "network_number": meta["group_local_index"] + 1,
                "npy_file": meta["npy_file"],
                "true_yaml_file": meta["true_yaml_file"],
                "N": meta["N"],
                "num_labeled_neurons": meta["num_labeled_neurons"],
                "num_excitatory_neurons": meta["num_excitatory_neurons"],
                "num_inhibitory_neurons": meta["num_inhibitory_neurons"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "average_precision": metrics["average_precision"],
            }
        )

    return pd.DataFrame(rows)


def save_combination_outputs(
    results: dict[str, object],
    combo_dir: str | Path,
    *,
    train_count: int,
    test_count: int,
    prob_dtype: np.dtype = np.float32,
) -> dict[str, object]:
    combo_dir = Path(combo_dir)
    combo_dir.mkdir(parents=True, exist_ok=True)

    prediction_manifest_df = save_test_prediction_arrays(
        test_predictions=results["test_predictions"],
        test_network_metadata=results["test_network_metadata"],
        output_dir=combo_dir,
        prob_dtype=prob_dtype,
    )
    prediction_manifest_path = combo_dir / "saved_prediction_arrays.csv"
    prediction_manifest_df.to_csv(prediction_manifest_path, index=False)

    feature_importance_df = results["feature_importance"].copy()
    feature_importance_path = combo_dir / "feature_importance.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)

    split_summary_df = results["split_summary"].copy()
    split_summary_path = combo_dir / "split_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)

    overall_metrics_row = dict(results["overall_test_metrics"])
    overall_metrics_row.update(
        {
            "train_count_per_group": train_count,
            "test_count_per_group": test_count,
        }
    )
    overall_metrics_df = pd.DataFrame([overall_metrics_row])
    overall_metrics_path = combo_dir / "overall_test_metrics.csv"
    overall_metrics_df.to_csv(overall_metrics_path, index=False)

    test_auc_df = build_test_network_auc_summary(
        test_predictions=results["test_predictions"],
        test_network_metadata=results["test_network_metadata"],
    )
    test_auc_path = combo_dir / "test_network_auc_summary.csv"
    test_auc_df.to_csv(test_auc_path, index=False)

    valid_roc_auc = _numeric_series_without_nan(test_auc_df, "roc_auc")
    valid_pr_auc = _numeric_series_without_nan(test_auc_df, "pr_auc")

    auc_summary_row = {
        "train_count_per_group": train_count,
        "test_count_per_group": test_count,
        "mean_test_network_roc_auc": float(valid_roc_auc.mean()) if len(valid_roc_auc) > 0 else np.nan,
        "std_test_network_roc_auc": float(valid_roc_auc.std(ddof=0)) if len(valid_roc_auc) > 0 else np.nan,
        "valid_test_network_roc_auc_count": int(len(valid_roc_auc)),
        "mean_test_network_pr_auc": float(valid_pr_auc.mean()) if len(valid_pr_auc) > 0 else np.nan,
        "std_test_network_pr_auc": float(valid_pr_auc.std(ddof=0)) if len(valid_pr_auc) > 0 else np.nan,
        "valid_test_network_pr_auc_count": int(len(valid_pr_auc)),
        "total_test_network_count": int(len(test_auc_df)),
    }
    auc_summary_df = pd.DataFrame([auc_summary_row])
    auc_summary_path = combo_dir / "auc_summary.csv"
    auc_summary_df.to_csv(auc_summary_path, index=False)

    return {
        "train_count_per_group": train_count,
        "test_count_per_group": test_count,
        "combo_dir": str(combo_dir),
        "feature_importance_path": str(feature_importance_path),
        "split_summary_path": str(split_summary_path),
        "overall_metrics_path": str(overall_metrics_path),
        "test_auc_summary_path": str(test_auc_path),
        "auc_summary_path": str(auc_summary_path),
        "saved_prediction_arrays_path": str(prediction_manifest_path),
        **overall_metrics_row,
        **auc_summary_row,
    }


def create_auc_heatmap(
    summary_df: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    colorbar_label: str,
    output_png: str | Path,
    output_csv: str | Path,
    annotate: bool = True,
) -> None:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    pivot_df = (
        summary_df.pivot(
            index="train_count_per_group",
            columns="test_count_per_group",
            values=value_column,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    pivot_df.to_csv(output_csv, index=True)

    train_counts = list(pivot_df.index)
    test_counts = list(pivot_df.columns)
    values = pivot_df.to_numpy(dtype=float)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", origin="lower", cmap=cmap)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_xlabel("Number of test networks per statistical class")
    ax.set_ylabel("Number of train networks per statistical class")
    ax.set_xticks(np.arange(len(test_counts)))
    ax.set_xticklabels(test_counts)
    ax.set_yticks(np.arange(len(train_counts)))
    ax.set_yticklabels(train_counts)

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
        "label_definition": {0: "excitatory", 1: "inhibitory"},
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
            inhibitory_count = int(np.sum(neuron_labels == 1))
            excitatory_count = int(np.sum(neuron_labels == 0))

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
# train/test sweep across all valid combinations
# -----------------------------------------------------------------------------

def run_all_train_test_combinations(
    all_data: dict[str, dict[str, object]],
    *,
    output_root: str | Path = ei_output_base_dir,
    min_train_count: int = 1,
    max_train_count: int = 10,
    min_test_count: int = 1,
    max_test_count: int = 10,
    networks_per_group: int = 11,
    shuffle_within_group: bool = False,
    split_random_state: int = 42,
    exclude_diagonal: bool = True,
    use_per_matrix_features: bool = False,
    add_summary_features: bool = True,
    threshold: float = 0.5,
    random_state: int = 42,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    subsample: float = 0.8,
) -> dict[str, object]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    combination_rows: list[dict[str, object]] = []
    combination_results: dict[tuple[int, int], dict[str, object]] = {}
    all_pr_auc_values: list[np.ndarray] = []

    for train_count in range(min_train_count, max_train_count + 1):
        for test_count in range(min_test_count, max_test_count + 1):
            if train_count + test_count > networks_per_group:
                continue

            combo_name = f"train_{train_count}_test_{test_count}"
            combo_dir = output_root / combo_name
            print(f"Running {combo_name} ...")

            results = train_gbt_across_grouped_networks(
                all_data=all_data,
                default_train_count=train_count,
                default_test_count=test_count,
                shuffle_within_group=shuffle_within_group,
                split_random_state=split_random_state,
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

            combo_test_auc_df = build_test_network_auc_summary(
                test_predictions=results["test_predictions"],
                test_network_metadata=results["test_network_metadata"],
            )
            valid_combo_pr_auc = _numeric_series_without_nan(combo_test_auc_df, "pr_auc")
            if len(valid_combo_pr_auc) > 0:
                all_pr_auc_values.append(valid_combo_pr_auc.to_numpy(dtype=float))

            combo_summary = save_combination_outputs(
                results=results,
                combo_dir=combo_dir,
                train_count=train_count,
                test_count=test_count,
            )

            combination_rows.append(combo_summary)
            combination_results[(train_count, test_count)] = results

    if not combination_rows:
        raise RuntimeError("No valid train/test combinations were run.")

    master_summary_df = pd.DataFrame(combination_rows).sort_values(
        ["train_count_per_group", "test_count_per_group"]
    )
    master_summary_path = output_root / "all_train_test_combination_metrics.csv"
    master_summary_df.to_csv(master_summary_path, index=False)

    create_auc_heatmap(
        master_summary_df,
        value_column="mean_test_network_roc_auc",
        title="Mean test-network ROC AUC across train/test combinations",
        colorbar_label="Mean ROC AUC",
        output_png=output_root / "heatmap_mean_test_network_roc_auc.png",
        output_csv=output_root / "heatmap_mean_test_network_roc_auc_values.csv",
    )

    create_auc_heatmap(
        master_summary_df,
        value_column="std_test_network_roc_auc",
        title="Std. dev. of test-network ROC AUC across train/test combinations",
        colorbar_label="ROC AUC standard deviation",
        output_png=output_root / "heatmap_std_test_network_roc_auc.png",
        output_csv=output_root / "heatmap_std_test_network_roc_auc_values.csv",
    )

    create_auc_heatmap(
        master_summary_df,
        value_column="mean_test_network_pr_auc",
        title="Mean test-network precision-recall AUC across train/test combinations",
        colorbar_label="Mean precision-recall AUC",
        output_png=output_root / "heatmap_mean_test_network_pr_auc.png",
        output_csv=output_root / "heatmap_mean_test_network_pr_auc_values.csv",
    )

    create_auc_heatmap(
        master_summary_df,
        value_column="std_test_network_pr_auc",
        title="Std. dev. of test-network precision-recall AUC across train/test combinations",
        colorbar_label="Precision-recall AUC standard deviation",
        output_png=output_root / "heatmap_std_test_network_pr_auc.png",
        output_csv=output_root / "heatmap_std_test_network_pr_auc_values.csv",
    )

    if all_pr_auc_values:
        global_pr_auc = np.concatenate(all_pr_auc_values)
        global_pr_auc_mean = float(np.mean(global_pr_auc))
        global_pr_auc_std = float(np.std(global_pr_auc, ddof=0))
        global_pr_auc_count = int(global_pr_auc.size)
    else:
        global_pr_auc_mean = np.nan
        global_pr_auc_std = np.nan
        global_pr_auc_count = 0

    global_pr_auc_summary = pd.DataFrame(
        [
            {
                "global_mean_pr_auc": global_pr_auc_mean,
                "global_std_pr_auc": global_pr_auc_std,
                "global_valid_test_network_pr_auc_count": global_pr_auc_count,
                "num_train_test_combinations": int(len(master_summary_df)),
            }
        ]
    )
    global_pr_auc_summary_path = output_root / "global_pr_auc_summary.csv"
    global_pr_auc_summary.to_csv(global_pr_auc_summary_path, index=False)

    return {
        "master_summary": master_summary_df,
        "master_summary_path": str(master_summary_path),
        "mean_auc_heatmap_path": str(output_root / "heatmap_mean_test_network_roc_auc.png"),
        "std_auc_heatmap_path": str(output_root / "heatmap_std_test_network_roc_auc.png"),
        "mean_pr_auc_heatmap_path": str(output_root / "heatmap_mean_test_network_pr_auc.png"),
        "std_pr_auc_heatmap_path": str(output_root / "heatmap_std_test_network_pr_auc.png"),
        "global_pr_auc_mean": global_pr_auc_mean,
        "global_pr_auc_std": global_pr_auc_std,
        "global_pr_auc_count": global_pr_auc_count,
        "global_pr_auc_summary_path": str(global_pr_auc_summary_path),
        "combination_results": combination_results,
    }


# -----------------------------------------------------------------------------
# example run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    all_data = main()

    sweep_results = run_all_train_test_combinations(
        all_data=all_data,
        output_root=ei_output_base_dir,
        min_train_count=1,
        max_train_count=10,
        min_test_count=1,
        max_test_count=10,
        networks_per_group=11,
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

    print()
    print("Saved master summary:")
    print(sweep_results["master_summary_path"])
    print()

    print("Saved heatmaps:")
    print(sweep_results["mean_auc_heatmap_path"])
    print(sweep_results["std_auc_heatmap_path"])
    print(sweep_results["mean_pr_auc_heatmap_path"])
    print(sweep_results["std_pr_auc_heatmap_path"])
    print()

    print("Saved global precision-recall AUC summary:")
    print(sweep_results["global_pr_auc_summary_path"])
    print()

    print(
        "Global precision-recall AUC across all train/test combinations: "
        f"mean={sweep_results['global_pr_auc_mean']:.6f}, "
        f"std={sweep_results['global_pr_auc_std']:.6f}, "
        f"n={sweep_results['global_pr_auc_count']}"
    )
    print()

    print("Top rows of the master summary:")
    print(sweep_results["master_summary"].head())
