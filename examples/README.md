# Runnable example data

`synthetic_fdata_N100_demo.txt` is a **synthetic** 100-neuron, 4 ms binned spike-occupancy matrix generated with a fixed random seed. It exists only so the repository's inspection, burst-detection, plotting, and CI examples can run without requiring the full thesis dataset.

Regenerate it with:

```bash
python examples/generate_synthetic_spikes.py
```

The file is not used to report thesis performance and should not be interpreted as an experimental result.
