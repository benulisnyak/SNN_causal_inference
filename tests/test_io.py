from pathlib import Path

import numpy as np
import pytest

from snn_connectivity.io import load_network_yaml, load_spike_matrix, parse_network_name


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_example_network_loads_with_expected_shape_and_edge_count():
    network = load_network_yaml(REPO_ROOT / "networks/network_N100_p24_CC01_1.yaml")

    assert network.adjacency.shape == (100, 100)
    assert int(np.count_nonzero(network.adjacency)) == 2406
    assert np.all(np.diag(network.adjacency) == 0)
    assert network.connection_probability == pytest.approx(0.24)


def test_synthetic_spike_matrix_has_100_neurons():
    spikes = load_spike_matrix(
        REPO_ROOT / "examples/synthetic_fdata_N100_demo.txt",
        expected_n=100,
    )

    assert spikes.shape == (260, 100)
    assert set(np.unique(spikes)).issubset({0, 1})


def test_empty_public_fdata_placeholder_raises_clear_error():
    with pytest.raises(ValueError, match="empty"):
        load_spike_matrix(REPO_ROOT / "fdata/fdata_N100_p24_CC05_1.txt")


def test_network_name_parser():
    parsed = parse_network_name("network_N200_p36_CC05_7.yaml")
    assert parsed == {"N": 200, "p_code": 36, "cc_code": 5, "index": 7}
