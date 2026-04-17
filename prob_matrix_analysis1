from __future__ import annotations
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import re
from pathlib import Path
import networkx as nx
import numpy as np
import yaml

try:
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeLoader as YamlLoader


# -----------------------------------------------------------------------------
# paths
base_dir = Path(".")
prob_matrices_dir = base_dir / "prob_matrices"
ground_truth_dir = base_dir / "networks"


# -----------------------------------------------------------------------------
# helpers

def natural_sort_key(text: str):
    """Sort strings like 2 before 10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


# -----------------------------------------------------------------------------
# ground-truth loader

def load_ground_truth_connectivity_matrix(file_path: str | Path):
    """
    Load one ground-truth connectivity matrix from a YAML file.

    Returns
    -------
    matrix : np.ndarray
        Square connectivity matrix.
    id_order : list
        Node IDs in the row/column order used in the matrix.
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
                node_id = int(k)
            except Exception:
                node_id = k
            node_list.append({"id": node_id, **(v or {})})
    elif isinstance(nodes, list):
        node_list = nodes
    else:
        raise ValueError(f"'nodes' must be a list or dict in {file_path}")

    id_order = [node["id"] for node in node_list]
    id_to_index = {node_id: i for i, node_id in enumerate(id_order)}

    N = len(id_order)
    matrix = np.zeros((N, N), dtype=float)

    for node in node_list:
        src = node["id"]
        src_idx = id_to_index[src]

        if "connections" in node and isinstance(node["connections"], list):
            for conn in node["connections"]:
                tgt = conn.get("target") or conn.get("to") or conn.get("id")
                if tgt is None:
                    continue
                weight = conn.get("weight", conn.get("w", 0.0))
                if tgt in id_to_index:
                    matrix[src_idx, id_to_index[tgt]] = float(abs(weight))
        else:
            targets = node.get("connectedTo") or node.get("targets") or []
            weights = node.get("weights") or node.get("w") or []
            for tgt, weight in zip(targets, weights):
                if tgt in id_to_index:
                    matrix[src_idx, id_to_index[tgt]] = float(abs(weight))

    return matrix, id_order


# -----------------------------------------------------------------------------
# probability-matrix loader

def parse_probability_matrix_filename(prob_path: str | Path):
    """
    Parse a file name like:
        prob_matrix_N100_p24_CC05_1.npy

    Returns
    -------
    stats_name : str
    network_number : int
    """
    prob_path = Path(prob_path)
    match = re.fullmatch(r"prob_matrix_(.+)_(\d+)\.npy", prob_path.name)

    if match is None:
        raise ValueError(
            f"Probability matrix file name does not match expected pattern: {prob_path.name}"
        )

    stats_name = match.group(1)
    network_number = int(match.group(2))
    return stats_name, network_number



def load_one_probability_and_truth_pair(
    prob_path: str | Path,
    ground_truth_dir: str | Path = ground_truth_dir,
):
    """
    Load one probability matrix and its matching ground-truth matrix.

    Matching rule:
        prob_matrix_<stats_name>_<k>.npy
    pairs with:
        network_<stats_name>_<k>.yaml
    """
    prob_path = Path(prob_path)
    ground_truth_dir = Path(ground_truth_dir)

    stats_name, network_number = parse_probability_matrix_filename(prob_path)

    truth_path = ground_truth_dir / f"network_{stats_name}_{network_number}.yaml"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing matching ground-truth file: {truth_path}")

    prob_matrix = np.load(prob_path)
    truth_matrix, id_order = load_ground_truth_connectivity_matrix(truth_path)

    if prob_matrix.shape != truth_matrix.shape:
        raise ValueError(
            f"Shape mismatch for {prob_path.name}: "
            f"probability matrix has shape {prob_matrix.shape}, "
            f"but ground truth has shape {truth_matrix.shape}"
        )

    return {
        "name": f"{stats_name}_{network_number}",
        "stats_name": stats_name,
        "network_number": network_number,
        "prob_matrix": prob_matrix,
        "ground_truth_matrix": truth_matrix,
        "id_order": id_order,
        "prob_path": prob_path,
        "truth_path": truth_path,
    }



def load_all_probability_and_truth_pairs(
    prob_dir: str | Path = prob_matrices_dir,
    ground_truth_dir: str | Path = ground_truth_dir,
):
    """
    Load all saved probability matrices and their matching ground-truth matrices.

    Returns
    -------
    pairs : dict[str, dict]
        Dictionary keyed by names like 'N100_p24_CC05_1'.
    """
    prob_dir = Path(prob_dir)
    ground_truth_dir = Path(ground_truth_dir)

    if not prob_dir.exists():
        raise FileNotFoundError(f"Probability matrix directory does not exist: {prob_dir}")
    if not prob_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {prob_dir}")

    prob_files = sorted(prob_dir.glob("prob_matrix_*.npy"), key=lambda p: natural_sort_key(p.name))

    if not prob_files:
        raise FileNotFoundError(f"No probability matrix files were found in: {prob_dir}")

    pairs = {}
    for prob_path in prob_files:
        pair = load_one_probability_and_truth_pair(
            prob_path=prob_path,
            ground_truth_dir=ground_truth_dir,
        )
        pairs[pair["name"]] = pair

    return pairs


# -----------------------------------------------------------------------------
# example run

def main():
    pairs = load_all_probability_and_truth_pairs(
        prob_dir=prob_matrices_dir,
        ground_truth_dir=ground_truth_dir,
    )

    print("Loaded probability matrices and matching ground-truth matrices:")
    print()

    for name, pair in pairs.items():
        print(f"Name: {name}")
        print(f"  Probability file:   {pair['prob_path']}")
        print(f"  Ground-truth file:  {pair['truth_path']}")
        print(f"  Probability shape:  {pair['prob_matrix'].shape}")
        print(f"  Ground-truth shape: {pair['ground_truth_matrix'].shape}")
        print(f"  Probability dtype:  {pair['prob_matrix'].dtype}")
        print()

    # example: access one pair
    first_name = next(iter(pairs))
    first_pair = pairs[first_name]

    print("Example access:")
    print(f"  first pair name = {first_name}")
    print(f"  prob matrix[0, 1] = {first_pair['prob_matrix'][0, 1]}")
    print(f"  truth matrix[0, 1] = {first_pair['ground_truth_matrix'][0, 1]}")

    return pairs

if __name__ == "__main__":
    all_pairs = main()

# -----------------------------------------------------------------------------
# ROC curve + clustering coefficient vs false positive rate

def average_directed_clustering_from_binary_matrix(binary_matrix):
    """
    Compute the average directed clustering coefficient of a binary adjacency matrix.

    Uses NetworkX's directed clustering definition.
    """
    binary_matrix = np.asarray(binary_matrix).copy()
    N = binary_matrix.shape[0]

    # remove self-connections
    np.fill_diagonal(binary_matrix, 0)

    # build directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(N))  # keep isolated nodes in the graph
    rows, cols = np.where(binary_matrix > 0)
    G.add_edges_from(zip(rows, cols))

    # node-level directed clustering coefficients
    clustering_dict = nx.clustering(G)

    # average across all nodes
    return float(np.mean(list(clustering_dict.values())))


pair_name = "N100_p24_CC05_6"   # change to whichever network you want
pair = all_pairs[pair_name]

prob_matrix = np.asarray(pair["prob_matrix"], dtype=float)
ground_truth_matrix = np.asarray(pair["ground_truth_matrix"], dtype=float)

# convert ground truth to binary adjacency
y_true_matrix = (ground_truth_matrix > 0).astype(int)

# remove diagonal from both matrices for ROC calculations
N = prob_matrix.shape[0]
mask_offdiag = ~np.eye(N, dtype=bool)

y_true = y_true_matrix[mask_offdiag]
y_score = prob_matrix[mask_offdiag]

# ROC curve
# drop_intermediate=False keeps all ROC thresholds, which is useful here
# because you want to compute clustering at each threshold.
fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
roc_auc = roc_auc_score(y_true, y_score)

# clustering coefficient of the ground-truth directed network
gt_clustering = average_directed_clustering_from_binary_matrix(y_true_matrix)

# clustering coefficient of predicted directed network at each ROC threshold
pred_clustering = []

for thr in thresholds:
    pred_binary = (prob_matrix >= thr).astype(int)
    np.fill_diagonal(pred_binary, 0)
    cc = average_directed_clustering_from_binary_matrix(pred_binary)
    pred_clustering.append(cc)

pred_clustering = np.array(pred_clustering)

# plot ROC and clustering on the same axes
plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.plot(fpr, tpr,'o', label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC {pair_name}")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(7,6))
plt.plot(fpr, pred_clustering, 'o', label="Predicted network clustering")
plt.axhline(gt_clustering, linestyle="--", label=f"Ground-truth clustering = {gt_clustering:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("Clustering Coefficient")
plt.title(f"CC vs FPR for {pair_name}")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# compute ROC AUC for all probability matrices, then mean and standard deviation

auc_results = {}

for pair_name, pair in all_pairs.items():
    prob_matrix = np.asarray(pair["prob_matrix"], dtype=float)
    ground_truth_matrix = np.asarray(pair["ground_truth_matrix"], dtype=float)

    # turn ground truth into binary labels: 1 = connected, 0 = not connected
    y_true_matrix = (ground_truth_matrix > 0).astype(int)

    # remove diagonal entries
    N = prob_matrix.shape[0]
    mask_offdiag = ~np.eye(N, dtype=bool)

    y_true = y_true_matrix[mask_offdiag]
    y_score = prob_matrix[mask_offdiag]

    # ROC AUC is only defined if both classes are present
    if len(np.unique(y_true)) < 2:
        auc_value = np.nan
        print(f"{pair_name}: skipped (ground truth has only one class)")
    else:
        auc_value = roc_auc_score(y_true, y_score)
        print(f"{pair_name}: AUC = {auc_value:.4f}")

    auc_results[pair_name] = auc_value

# collect valid AUC values
auc_values = np.array([v for v in auc_results.values() if not np.isnan(v)], dtype=float)

if len(auc_values) == 0:
    print("\nNo valid AUC values were found.")
else:
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)          # population standard deviation
    # std_auc = np.std(auc_values, ddof=1)  # use this instead for sample std

    print("\nSummary across all probability matrices:")
    print(f"Number of valid matrices: {len(auc_values)}")
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Std AUC:  {std_auc:.4f}")
