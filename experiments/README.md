# Experiments

This directory contains the executable model-training workflows used in the MSc project. The scripts are intentionally separated from the reusable `src/snn_connectivity/` package: `src/` contains small testable components, while `experiments/` contains the complete experiment orchestration, file conventions, hyperparameters, and saved-output logic.

## Core pipeline

| Script | Role |
|---|---|
| `train_activity_matching_snn.py` | **Stage 1.** Detect target bursts and train a recurrent LIF SNN with surrogate gradients. Saves one learned connectivity matrix per burst sample. |
| `train_edge_classifier.py` | **Stage 2.** Aggregate learned weights into directed edge features, perform network-grouped train/test splitting, and fit the GBT structural-edge classifier. |
| `train_neuron_type_classifier.py` | Secondary experiment that predicts excitatory versus inhibitory source-neuron type from node-level summaries of learned SNN weights. |

## Stage 1: activity matching

From the repository root:

```bash
python experiments/train_activity_matching_snn.py N100_p24_CC03 --skip-trained-samples
```

The script expects target spike-train files under a statistics-specific directory such as:

```text
fdata/N100_p24_CC03/fdata_N100_p24_CC03_1.txt
```

Learned burst-specific matrices are written under `LIFoutput_files/` using the project naming convention.

## Stage 2: directed edge classification

```bash
python experiments/train_edge_classifier.py
```

The default configuration loads the N=100 statistical grid, uses five network realizations per class for training and six for testing, builds summary features from the learned weight matrices, reports held-out edge-classification metrics, and saves held-out probability matrices plus compact CSV summaries to `prob_matrices/`. The train/test split is performed at the **network level**, not by randomly splitting individual edges.

The script is structured so the default class list, data directories, and experiment hyperparameters can be changed independently from the feature-engineering and evaluation functions.

## Secondary neuron-type experiment

```bash
python experiments/train_neuron_type_classifier.py
```

This workflow derives excitatory/inhibitory labels from the sign of outgoing ground-truth synapses and evaluates whether statistics of the learned SNN weights are predictive of source-neuron type.

## Reproducibility

The activity-matching workflow sets Python, NumPy, and PyTorch random seeds. The GBT workflows use explicit split and model random states. Exact floating-point identity is not guaranteed across hardware, but the experiment definitions and split logic are deterministic for a fixed environment.
