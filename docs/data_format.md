# Data formats

## Ground-truth network YAML

Network files follow names such as:

```text
network_N100_p24_CC01_1.yaml
```

The example file contains network-level metadata and a `nodes` collection. A node record has the form:

```yaml
- id: 1
  pos: [0.483, 0.599]
  connectedTo: [2, 5, 8]
  weights: [1.0, 1.0, 1.0]
```

`connectedTo` stores outgoing target neuron IDs and `weights` stores the corresponding signed synaptic weights. The project treats a non-zero weight as a structural edge. Positive outgoing weights identify excitatory source neurons and negative outgoing weights identify inhibitory source neurons.

## Binned target activity (`fdata`)

Target activity is a comma-separated matrix:

```text
0, 0, 1, 0, ...
0, 1, 0, 0, ...
...
```

Rows are time bins and columns are neurons. The training code uses 4 ms binary occupancy, so a value of `1` means that neuron emitted at least one spike within that bin.

Thesis-scale runs are organized by network-statistics folder, for example:

```text
fdata/
└── N100_p24_CC03/
    ├── fdata_N100_p24_CC03_1.txt
    ├── fdata_N100_p24_CC03_2.txt
    └── ...
```

The included `fdata/fdata_N100_p24_CC05_1.txt` is currently empty and is retained only as the placeholder present in the source repository. The runnable file under `examples/` is explicitly synthetic.

## Learned connectivity matrices

The activity-matching stage saves NumPy arrays with shape:

```text
[number_of_burst_samples, N, N]
```

Each `[N, N]` slice is one learned recurrent weight matrix. These repeated estimates are aggregated into directed edge-level features for the gradient-boosted classifier.

## Probability matrices and threshold statistics

Classifier outputs are `N x N` probability matrices. The analysis scripts can sweep thresholds and save CSV files containing ranking/classification metrics and graph-level quantities. These files are intentionally excluded from version control because thesis-scale experiments can generate many large outputs.
