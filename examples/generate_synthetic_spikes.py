"""Generate a small, deterministic spike-occupancy matrix for the quickstart demo.

This file is deliberately synthetic. It is useful for exercising the I/O,
burst-detection and plotting code without presenting generated values as MSc
experimental data.
"""

from pathlib import Path

import numpy as np


SEED = 15
N_NEURONS = 100
N_BINS = 260
OUTPUT = Path(__file__).with_name("synthetic_fdata_N100_demo.txt")


def main() -> None:
    rng = np.random.default_rng(SEED)
    spikes = (rng.random((N_BINS, N_NEURONS)) < 0.0015).astype(np.uint8)

    # Three compact population events make the example visually recognizable as
    # bursting activity. Activity ramps toward each peak rather than switching
    # the whole network on at once.
    for start in (30, 105, 190):
        probabilities = np.linspace(0.12, 0.98, 9)
        for offset, probability in enumerate(probabilities):
            spikes[start + offset] = (
                rng.random(N_NEURONS) < probability
            ).astype(np.uint8)

    np.savetxt(OUTPUT, spikes, fmt="%d", delimiter=", ")
    print(f"Wrote {spikes.shape[0]} x {spikes.shape[1]} spike matrix to {OUTPUT}")


if __name__ == "__main__":
    main()
