# Target spike-train data

The activity-matching SNN expects target activity as a two-dimensional comma-separated matrix with shape `[time bins, neurons]`. Values are binary spike occupancy after binning the original NEST output at 4 ms resolution.

The public repository currently contains `fdata_N100_p24_CC05_1.txt`, but that file is an empty placeholder (newline only). It is retained unchanged so the polished repository does not silently replace research data with fabricated values.

For a working example, use `../examples/synthetic_fdata_N100_demo.txt`. Full thesis-scale runs require populated `fdata/<network-statistics>/` folders as described in `docs/data_format.md`.
