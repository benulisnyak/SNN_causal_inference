
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import yaml
try:

    from yaml import CSafeLoader as YamlLoader
except ImportError:

    from yaml import SafeLoader as YamlLoader

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
#############
#list of the experiment folders that the script will load
#############
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


#current working folder as the base path
base_dir = Path(".")

#learned connectivity matrices path
learned_base_dir = base_dir / "LIFoutput_files/"

#ground-truth connectivity YAML files 
ground_truth_base_dir = base_dir / "networks/"

#pattern for learned connectivity files inside each experiment folder
npy_pattern = "connectivity_matrices_*.npy"


#helper function to natural sort key, sorts strings by increasing order (2 comes before 10, etc.)
def natural_sort_key(text: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


#function to resolve npy dir
def resolve_npy_dir(folder_name: str, learned_base_dir: str | Path) -> Path:
    return Path(learned_base_dir) / folder_name


#function to load one npy matrix file
def load_one_npy_matrix_file(npy_path: str | Path) -> np.ndarray:

    npy_path = Path(npy_path)
    matrices = np.load(npy_path)

    if matrices.ndim != 3:
  
        raise ValueError(
            f"Expected shape (num_matrices, N, N) in {npy_path}, got {matrices.shape}."
        )

    return matrices


#function to load all connectivity runs from one folder
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

    # Load the matrix data from the .npy file or files
    matrices = [load_one_npy_matrix_file(path) for path in npy_files]
    return matrices, npy_files



#function to load ground truth connectivity matrix
def load_ground_truth_connectivity_matrix(file_path: str | Path) -> tuple[np.ndarray, list]:
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


#function to load ground truth matrices for folder
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

    # make empty lists to collect values
    true_matrices: list[np.ndarray] = []
    true_yaml_files: list[Path] = []
    true_id_orders: list[list] = []

    # Loop through each item
    for idx in range(num_expected):
        #build the path to the expected ground-truth YAML file
        yaml_path = ground_truth_dir / f"network_{stats_name}_{idx + 1}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing ground-truth YAML file: {yaml_path}")
        matrix, id_order = load_ground_truth_connectivity_matrix(yaml_path)
        true_matrices.append(matrix)
        true_yaml_files.append(yaml_path)
        true_id_orders.append(id_order)
    return true_matrices, true_yaml_files, true_id_orders


# -----------------------------------------------------------------------------
#combined multi-folder loader
#defines a function to load all connectivity runs multiple folders.
def load_all_connectivity_runs_multiple_folders(
    folder_names: list[str],
    learned_base_dir: str | Path,
    ground_truth_base_dir: str | Path,
    npy_pattern: str = "connectivity_matrices_*.npy",
) -> dict[str, dict[str, object]]:
  
    #create the main results dictionary for all folders
    results: dict[str, dict[str, object]] = {}

    for folder_name in folder_names:
        stats_name = folder_name
        #build the path to the folder that holds the learned .npy files
        npy_dir = resolve_npy_dir(folder_name, learned_base_dir=learned_base_dir)

        #load the learned matrices and the matching file paths
        learned_matrices, npy_files = load_all_connectivity_runs_from_one_folder(
            npy_dir=npy_dir,
            pattern=npy_pattern,
        )

        #load the matching ground-truth matrices and file paths
        true_matrices, true_yaml_files, true_id_orders = load_ground_truth_matrices_for_folder(
            stats_name=stats_name,
            num_expected=len(learned_matrices),
            ground_truth_dir=ground_truth_base_dir,
        )

        if len(true_matrices) != len(learned_matrices):
            raise ValueError(
                f"Mismatch for {stats_name}: "
                f"{len(learned_matrices)} learned files but {len(true_matrices)} ground-truth files."
            )

        #save these values for this stats setting
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


#main function to load in data
def main() -> dict[str, dict[str, object]]:
    #load all of the learned and ground-truth data into one variable
    all_data = load_all_connectivity_runs_multiple_folders(
        folder_names=network_stats_list,
        learned_base_dir=learned_base_dir,
        ground_truth_base_dir=ground_truth_base_dir,
        npy_pattern=npy_pattern,
    )

    print("Loaded learned and ground-truth connectivity matrices:")
    print()

    for stats_name, data in all_data.items():
        #build the path to the folder that holds the learned .npy files
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


if __name__ == "__main__":
    # Load all of the learned and ground-truth data into one variable.
    all_data = main()
#print(all_data["N100_p24_CC05"])

#------------------ end of loading in data ----------------------



#function to natural sort key
def natural_sort_key(text: str):
    #Import re so the script can use it later.
    import re
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]



#function to validate network inputs
#validates all matrices have the same square shape, and ground-truth matrices matches networks shapes
def _validate_network_inputs(all_matrices_lists, all_A_t):

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
#feature construction
#function to build edge features for one network
def _build_edge_features_for_one_network(
    matrices,
    A_t=None,
    max_num_matrices=None,
    undirected=False,
    exclude_diagonal=True,
    use_per_matrix_features=True,
    add_summary_features=True,
):

    mats = [np.asarray(m, dtype=float) for m in matrices]
    K_all = len(mats)
    N = mats[0].shape[0]

    if max_num_matrices is None:
        #set the maximum number of per-matrix feature columns to use
        max_num_matrices = K_all

    #stack all learned matrices so edge values are easy to slice.
    stack_all = np.stack(mats, axis=0)  # (K_all, N, N)
    #mask that tells us which edges to keep.
    mask = np.ones((N, N), dtype=bool)

    if exclude_diagonal:
        #fill the diagonal entries
        np.fill_diagonal(mask, False)

    if undirected:
        #build an upper-triangle mask for if using undirected networks 
        tri_mask = np.triu(np.ones((N, N), dtype=bool), k=1 if exclude_diagonal else 0)
        mask &= tri_mask

    #saves the row and column index positions for the chosen edges
    rows, cols = np.where(mask)

    #pulls out all edge values across all matrices for the kept edges.
    actual_vals_all = stack_all[:, rows, cols].T  # (num_edges, K_all)

    #list to hold blocks of feature columns
    X_parts = []
    feature_names = []


    if use_per_matrix_features:
        #only keep as many per-matrix columns as the model can use
        K_used = min(K_all, max_num_matrices)

        #fill missing slots with NaN
        X_base = np.full((actual_vals_all.shape[0], max_num_matrices), np.nan, dtype=float)
        if K_used > 0:
            X_base[:, :K_used] = actual_vals_all[:, :K_used]

        X_parts.append(X_base)
        #add names
        feature_names.extend([f"matrix_{i}" for i in range(max_num_matrices)])


    if add_summary_features:
        #summary feature columns built from all learned matrices
        X_parts.extend([
            actual_vals_all.mean(axis=1, keepdims=True),
            actual_vals_all.std(axis=1, keepdims=True),
            np.median(actual_vals_all, axis=1, keepdims=True),
            actual_vals_all.min(axis=1, keepdims=True),
            actual_vals_all.max(axis=1, keepdims=True),
        ])
        #add names for the features
        feature_names.extend(["mean", "std", "median", "min", "max"])


    if not X_parts:
        raise ValueError(
            "At least one of use_per_matrix_features or add_summary_features must be True."
        )

    #combine the feature blocks into one feature matrix
    X = np.hstack(X_parts)

    if A_t is None:
        return X, None, rows, cols, feature_names

    #pull out the target values for the chosen edges
    y = np.asarray(A_t)[rows, cols]

    #mark which target values are real and not missing.
    valid = ~np.isnan(y)
    #combine the feature blocks into one feature matrix.
    X = X[valid]
    y = y[valid]
    rows = rows[valid]
    cols = cols[valid]
    return X, y.astype(int), rows, cols, feature_names


#function to reconstruct matrix from edge values
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
    #new matrix that will be filled with edge values
    M = np.full((N, N), fill_value, dtype=dtype)
    M[rows, cols] = edge_values

    if undirected:
        M[cols, rows] = edge_values

    if exclude_diagonal:
        np.fill_diagonal(M, 0 if np.issubdtype(np.dtype(dtype), np.integer) else 0.0)

    return M


#function to safe binary metrics
def _safe_binary_metrics(y_true, y_prob, y_pred):
    #calculate the score values for this set of predictions
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    #use this path when the earlier condition was not met
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    return metrics


# -----------------------------------------------------------------------------
#main trainer
#function to train gbt across networks.
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

    _validate_network_inputs(all_matrices_lists, all_A_t)

    #how many full networks are available
    num_networks = len(all_matrices_lists)

    if train_indices is None and test_indices is None:
        #split the networks into two halves when no split is given
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

    #define per-matrix feature width from TRAINING networks only
    max_num_matrices = max(len(all_matrices_lists[i]) for i in train_indices)
    #start a list for X and y training
    X_train_all = []
    y_train_all = []
    #start a list for the feature names
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

    #find the unique label values in the training data
    unique_y = np.unique(y_train_all)
    if not np.array_equal(np.sort(unique_y), np.array([0, 1])):
        raise ValueError(
            f"This function expects binary ground-truth labels 0/1. "
            f"Training labels found: {unique_y}"
        )

    ### machine learning pipeline ###
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

    #train the model using the full training data
    model.fit(X_train_all, y_train_all)
    #get the imputer step out of the trained pipeline
    imputer = model.named_steps["simpleimputer"]
    #get the gradient boosting model out of the pipeline
    gbt = model.named_steps["gradientboostingclassifier"]

    #starting with the feature names that match the training input
    feature_names_for_importance = list(feature_names)

    if len(gbt.feature_importances_) != len(feature_names_for_importance):
        #reads the imputer statistics so feature alignment can be checked
        stats = getattr(imputer, "statistics_", None)
        if stats is not None and len(stats) == len(feature_names_for_importance):
            #mark which feature columns were kept after imputation checks
            kept_mask = ~np.isnan(stats)
            feature_names_for_importance = list(np.asarray(feature_names_for_importance)[kept_mask])
        #uses this path when the earlier condition was not met.
        else:
            raise RuntimeError(
                f"Feature name / importance length mismatch: "
                f"{len(feature_names_for_importance)} names vs "
                f"{len(gbt.feature_importances_)} importances"
            )

    #build a table that shows which features mattered most
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

        #saves the predicted probabilities for this test network.
        prob = model.predict_proba(X_test)[:, 1]
        #turn probabilities into 0 or 1 (binary) predictions using the threshold
        pred = (prob >= threshold).astype(int)

        #save the network size
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

        #calculate the score values for this set of predictions
        metrics = _safe_binary_metrics(y_test, prob, pred)
        test_predictions[idx] = {
            "global_prob_matrix": prob_matrix,
            "global_binary_matrix": binary_matrix,
            "edge_probabilities": prob,
            "edge_predictions": pred,
            "edge_truth": y_test,
            "metrics": metrics,
        }

        #add these test output so overall metrics can be computed later
        all_test_true.append(y_test)
        all_test_prob.append(prob)
        all_test_pred.append(pred)

    #list for all true test labels
    all_test_true = np.concatenate(all_test_true)
    #list for all predicted test probabilities
    all_test_prob = np.concatenate(all_test_prob)
    #list for all predicted test labels
    all_test_pred = np.concatenate(all_test_pred)

    #calculates one overall score across all test networks
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
#dataset preparation
#function to flatten loaded data by group
def flatten_loaded_data_by_group(all_data):
    all_matrices_lists = []
    all_A_t = []
    group_labels = []
    network_metadata = []

    for stats_name in sorted(all_data.keys(), key=natural_sort_key):
        #Reads the file content into a Python object
        data = all_data[stats_name]
        learned_matrices = data["learned_matrices"]
        true_matrices = data["true_matrices"]
        #Finds and sort the .npy files in this folder
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
            #add values to flat dataset lists
            all_matrices_lists.append(np.asarray(learned_arr))
            all_A_t.append(np.asarray(true_A))
            group_labels.append(stats_name)

            #savee metadata for this flattened network example
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


#function to make grouped train test split
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

    if train_counts_by_group is None:
        train_counts_by_group = {}
    if test_counts_by_group is None:
        test_counts_by_group = {}

    #makes a lookup that groups flat indices by group name
    group_to_indices = {}
    for idx, group in enumerate(group_labels):
        group_to_indices.setdefault(group, []).append(idx)

    #random number generator so shuffling is repeatable
    rng = np.random.default_rng(random_state)

    #save the training/test/unused indices as a plain list.
    train_indices = []
    test_indices = []
    unused_indices = []
    #list that will describe the split for each group
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
                f"but train_k={train_k}." #saves result
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

        #picks the flat indices that will be used for training/testing in this group
        group_train = indices[:train_k]
        group_test = indices[train_k:train_k + test_k]
        group_unused = indices[train_k + test_k:]

        #adds these group indices into the full split lists
        train_indices.extend(group_train)
        test_indices.extend(group_test)
        unused_indices.extend(group_unused)

        #saves a summary row for this group split
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

    #saves the indices 
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


#function to train gbt across grouped networks
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

    #flatten grouped data, choose group-aware train/test splits, then train the original GBT model
    all_matrices_lists, all_A_t, group_labels, network_metadata = flatten_loaded_data_by_group(all_data)

    #build a group-aware train and test split
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

    #save these results
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

    #turns the split summary into a pandas table (for easy viewing)
    split_summary_df = pd.DataFrame(split_info["split_summary"])

    #collect metadata for the networks
    train_metadata = [network_metadata[i] for i in split_info["train_indices"]]
    test_metadata = [network_metadata[i] for i in split_info["test_indices"]]
    unused_metadata = [network_metadata[i] for i in split_info["unused_indices"]]

    #adds the extra results and metadata into the result dictionary
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


#######
#run the grouped training step and save all output here
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

    #for any group not listed above:
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


print(results_gbt["overall_test_metrics"])
print(results_gbt["feature_importance"].head(10))
print(results_gbt["split_summary"])