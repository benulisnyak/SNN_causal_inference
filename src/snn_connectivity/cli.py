"""Command-line entry points for inspecting the example data and repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .burst_detection import detect_population_bursts
from .io import load_network_yaml, load_spike_matrix
from .plotting import plot_network, plot_spike_raster


def _network_summary(path: Path) -> dict[str, int | float | None]:
    network = load_network_yaml(path)
    edges = int(np.count_nonzero(network.adjacency))
    return {
        "nodes": network.n_nodes,
        "directed_edges": edges,
        "realized_density": edges / (network.n_nodes * (network.n_nodes - 1)),
        "excitatory_sources": int(network.excitatory_sources.sum()),
        "inhibitory_sources": int(network.inhibitory_sources.sum()),
        "configured_connection_probability": network.connection_probability,
        "configured_clustering_coefficient": network.clustering_coefficient,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snn-connectivity",
        description="Inspect example inputs from the SNN connectivity-inference project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_network = subparsers.add_parser("inspect-network", help="summarize a YAML network")
    inspect_network.add_argument("path", type=Path)

    inspect_spikes = subparsers.add_parser("inspect-spikes", help="summarize a binned spike file")
    inspect_spikes.add_argument("path", type=Path)
    inspect_spikes.add_argument("--expected-n", type=int, default=None)
    inspect_spikes.add_argument("--bin-width-ms", type=float, default=4.0)

    figures = subparsers.add_parser("make-example-figures", help="render example network/raster")
    figures.add_argument("--network", type=Path, required=True)
    figures.add_argument("--spikes", type=Path, required=True)
    figures.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "inspect-network":
        print(json.dumps(_network_summary(args.path), indent=2))
        return 0

    if args.command == "inspect-spikes":
        spikes = load_spike_matrix(args.path, expected_n=args.expected_n)
        bursts = detect_population_bursts(spikes, bin_width_ms=args.bin_width_ms)
        summary = {
            "time_bins": int(spikes.shape[0]),
            "neurons": int(spikes.shape[1]),
            "occupied_bins": int(spikes.sum()),
            "detected_bursts": len(bursts),
            "burst_lengths_bins": [burst.duration_bins for burst in bursts],
        }
        print(json.dumps(summary, indent=2))
        return 0

    network = load_network_yaml(args.network)
    spikes = load_spike_matrix(args.spikes, expected_n=network.n_nodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_network(network, args.output_dir / "example_ground_truth_network.png")
    plot_spike_raster(spikes, args.output_dir / "example_spike_raster.png")
    print(f"Saved example figures to {args.output_dir}")
    return 0
