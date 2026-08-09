from pathlib import Path

from snn_connectivity.burst_detection import detect_population_bursts
from snn_connectivity.io import load_spike_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_example_contains_detectable_population_bursts():
    spikes = load_spike_matrix(REPO_ROOT / "examples/synthetic_fdata_N100_demo.txt")
    bursts = detect_population_bursts(spikes)

    assert len(bursts) >= 2
    assert all(burst.duration_bins >= 1 for burst in bursts)
    assert all(burst.active_fraction_at_trigger > 0.9 for burst in bursts)


def test_short_recording_has_no_complete_detection_window():
    spikes = load_spike_matrix(REPO_ROOT / "examples/synthetic_fdata_N100_demo.txt")[:5]
    assert detect_population_bursts(spikes) == []
