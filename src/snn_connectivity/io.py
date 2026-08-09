"""Input helpers for network YAML files and binned spike-train matrices."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


_NETWORK_NAME = re.compile(
    r"(?:network_)?N(?P<n>\d+)_p(?P<p>\d+)_CC(?P<cc>\d+)(?:_(?P<idx>\d+))?"
)


@dataclass(frozen=True)
class NetworkData:
    """Parsed ground-truth network with metadata used throughout the project."""

    adjacency: np.ndarray
    positions: np.ndarray
    node_ids: tuple[int, ...]
    connection_probability: float | None
    clustering_coefficient: float | None
    inhibitory_fraction: float | None

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def binary_adjacency(self) -> np.ndarray:
        return (self.adjacency != 0).astype(np.uint8)

    @property
    def excitatory_sources(self) -> np.ndarray:
        return np.any(self.adjacency > 0, axis=1)

    @property
    def inhibitory_sources(self) -> np.ndarray:
        return np.any(self.adjacency < 0, axis=1)


def _normalise_nodes(raw_nodes: object) -> list[dict]:
    """Accept the list- or dict-style node layouts used by older network files."""
    if isinstance(raw_nodes, list):
        return raw_nodes
    if isinstance(raw_nodes, dict):
        nodes: list[dict] = []
        for key, value in raw_nodes.items():
            item = dict(value or {})
            item.setdefault("id", int(key) if str(key).isdigit() else key)
            nodes.append(item)
        return nodes
    raise ValueError("The YAML 'nodes' field must be a list or mapping.")


def load_network_yaml(path: str | Path) -> NetworkData:
    """Load a signed directed adjacency matrix from one project network YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")

    nodes = _normalise_nodes(payload.get("nodes", []))
    if not nodes:
        raise ValueError(f"No nodes were found in {path}.")

    node_ids = tuple(int(node["id"]) for node in nodes)
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    n_nodes = len(nodes)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    positions = np.full((n_nodes, 2), np.nan, dtype=float)

    for source_index, node in enumerate(nodes):
        position = node.get("pos")
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            positions[source_index] = [float(position[0]), float(position[1])]

        targets = node.get("connectedTo") or node.get("targets") or []
        weights = node.get("weights") or node.get("w") or []

        # The MATLAB/NEST network files use one-based neuron IDs. Mapping IDs
        # explicitly is safer than assuming IDs are always contiguous.
        for target, weight in zip(targets, weights, strict=False):
            target_id = int(target)
            if target_id in id_to_index:
                adjacency[source_index, id_to_index[target_id]] = float(weight)

    return NetworkData(
        adjacency=adjacency,
        positions=positions,
        node_ids=node_ids,
        connection_probability=_optional_float(payload.get("connectionProbability")),
        clustering_coefficient=_optional_float(payload.get("clusteringCoefficient")),
        inhibitory_fraction=_optional_float(payload.get("inhibitoryFraction")),
    )


def load_spike_matrix(path: str | Path, expected_n: int | None = None) -> np.ndarray:
    """Load the project's comma-separated [time_bin, neuron] spike occupancy matrix."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 2:
        raise ValueError(
            f"Spike file {path} is empty. Add a populated fdata matrix or use the "
            "synthetic example in examples/."
        )

    # np.genfromtxt handles both ', ' and ',' delimiters. Reading as text first
    # gives a clearer error for malformed rows than silently producing NaNs.
    matrix = np.genfromtxt(path, delimiter=",", dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"Expected a non-empty 2-D spike matrix in {path}.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Spike matrix {path} contains missing or non-numeric values.")
    if expected_n is not None and matrix.shape[1] != expected_n:
        raise ValueError(
            f"Expected {expected_n} neurons but found {matrix.shape[1]} columns in {path}."
        )
    return (matrix > 0.5).astype(np.uint8)


def parse_network_name(name: str) -> dict[str, int] | None:
    """Extract N, connection probability code, CC code and optional index."""
    match = _NETWORK_NAME.search(Path(name).stem)
    if match is None:
        return None
    values = match.groupdict()
    return {
        "N": int(values["n"]),
        "p_code": int(values["p"]),
        "cc_code": int(values["cc"]),
        "index": int(values["idx"]) if values["idx"] is not None else -1,
    }


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
