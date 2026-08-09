"""Population-burst detection for 4 ms binned neuronal activity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Burst:
    """A burst cropped from first continuous activity through its activity peak."""

    start_bin: int
    peak_bin: int
    end_window: int
    active_fraction_at_trigger: float

    @property
    def duration_bins(self) -> int:
        return self.peak_bin - self.start_bin + 1


def detect_population_bursts(
    spikes: np.ndarray,
    *,
    bin_width_ms: float = 4.0,
    detection_window_ms: float = 50.0,
    min_active_fraction: float = 0.9,
) -> list[Burst]:
    """Detect bursts using the NEST-style unique-active-neuron criterion.

    The detector follows the same two-scale logic used by the training script:
    population activity is detected in non-overlapping 50 ms windows, while the
    returned training crop is resolved at the original 4 ms bin width.
    """
    x = np.asarray(spikes)
    if x.ndim != 2:
        raise ValueError("spikes must have shape [time_bins, neurons]")
    if x.shape[0] == 0 or x.shape[1] == 0:
        return []
    if bin_width_ms <= 0 or detection_window_ms <= 0:
        raise ValueError("bin and detection window widths must be positive")
    if not 0.0 <= min_active_fraction <= 1.0:
        raise ValueError("min_active_fraction must be between 0 and 1")

    binary = x > 0.5
    n_bins, n_neurons = binary.shape
    total_ms = n_bins * bin_width_ms
    n_complete_windows = int(total_ms // detection_window_ms)
    if n_complete_windows == 0:
        return []

    # Fifty milliseconds is not an integer multiple of four. Assigning each
    # 4 ms bin by its centre reproduces the deterministic convention in the
    # thesis training code at the window boundaries.
    centres_ms = (np.arange(n_bins, dtype=float) + 0.5) * bin_width_ms
    window_edges = np.arange(n_complete_windows + 1, dtype=float) * detection_window_ms
    window_for_bin = np.searchsorted(window_edges, centres_ms, side="left") - 1

    active_fractions = np.zeros(n_complete_windows, dtype=float)
    for window in range(n_complete_windows):
        in_window = window_for_bin == window
        if np.any(in_window):
            active_fractions[window] = binary[in_window].any(axis=0).mean()

    # The strict greater-than comparison is intentional: it matches the burst
    # criterion used by the NEST simulation code rather than rounding at 90%.
    is_burst_window = active_fractions > min_active_fraction
    bursts: list[Burst] = []
    window = 0

    while window < n_complete_windows:
        if not is_burst_window[window]:
            window += 1
            continue

        run_start = window
        while window + 1 < n_complete_windows and is_burst_window[window + 1]:
            window += 1
        run_end = window

        run_bins = np.flatnonzero(
            (window_for_bin >= run_start) & (window_for_bin <= run_end)
        )
        if run_bins.size:
            population_activity = binary[run_bins].sum(axis=1)
            peak_bin = int(run_bins[int(np.argmax(population_activity))])
            start_bin = peak_bin
            while start_bin > 0 and binary[start_bin - 1].any():
                start_bin -= 1
            bursts.append(
                Burst(
                    start_bin=start_bin,
                    peak_bin=peak_bin,
                    end_window=run_end,
                    active_fraction_at_trigger=float(active_fractions[run_start]),
                )
            )
        window += 1

    return bursts
