"""Utilities for inspecting and evaluating SNN connectivity-inference experiments."""

from .burst_detection import Burst, detect_population_bursts
from .features import build_directed_edge_features
from .io import NetworkData, load_network_yaml, load_spike_matrix
from .metrics import BinaryMetrics, evaluate_binary_predictions
from .modeling import grouped_network_split, make_edge_classifier

__all__ = [
    "BinaryMetrics",
    "Burst",
    "NetworkData",
    "build_directed_edge_features",
    "detect_population_bursts",
    "evaluate_binary_predictions",
    "load_network_yaml",
    "load_spike_matrix",
    "grouped_network_split",
    "make_edge_classifier",
]

__version__ = "1.0.0"
