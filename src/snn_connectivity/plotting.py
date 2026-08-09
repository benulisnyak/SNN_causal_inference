"""Compact plotting helpers for example datasets and documentation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .io import NetworkData


def plot_network(network: NetworkData, output: str | Path) -> Path:
    """Render the signed ground-truth graph with excitatory/inhibitory source types."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(network.n_nodes))
    source, target = np.nonzero(network.adjacency)
    graph.add_edges_from(zip(source.tolist(), target.tolist(), strict=False))

    if np.isfinite(network.positions).all():
        positions = {i: tuple(network.positions[i]) for i in range(network.n_nodes)}
    else:
        positions = nx.spring_layout(graph, seed=7)

    inhibitory = network.inhibitory_sources
    node_values = np.where(inhibitory, 0.15, 0.85)

    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        arrows=False,
        width=0.35,
        alpha=0.18,
    )
    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=38,
        node_color=node_values,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        linewidths=0,
    )
    ax.set_title("Example directed ground-truth network")
    ax.set_axis_off()
    # A tiny colorbar makes the E/I encoding understandable without a dense legend.
    colorbar = fig.colorbar(nodes, ax=ax, fraction=0.04, pad=0.01)
    colorbar.set_ticks([0.15, 0.85], labels=["Inhibitory source", "Excitatory source"])
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_spike_raster(
    spikes: np.ndarray,
    output: str | Path,
    *,
    bin_width_ms: float = 4.0,
) -> Path:
    """Plot a binary spike-occupancy matrix as a conventional neuronal raster."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(spikes) > 0.5
    times, neurons = np.nonzero(matrix)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(times * bin_width_ms, neurons + 1, s=5, marker="|", linewidths=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")
    ax.set_title("Synthetic 4 ms binned target activity")
    ax.set_xlim(0, matrix.shape[0] * bin_width_ms)
    ax.set_ylim(0, matrix.shape[1] + 1)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
