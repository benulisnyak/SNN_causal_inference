# Analysis

This directory contains post-training evaluation and visualization workflows for the structural-connectivity predictions produced by the experiment scripts.

| Script | Role |
|---|---|
| `evaluate_probability_thresholds.py` | Sweeps classifier probability thresholds and computes classification metrics plus reconstructed-network statistics for total, undirected, directed, excitatory-source, and inhibitory-source evaluations. |
| `plot_roc_pr_confidence_intervals.py` | Aggregates threshold-analysis CSVs across independent networks, interpolates ROC/PR curves onto common grids, and plots mean curves with pointwise confidence intervals. |
| `analyze_probability_matrices.py` | Lightweight probability-matrix inspection utility for per-network ROC AUC and clustering-coefficient diagnostics. |

## Typical workflow

After running the edge classifier and saving probability matrices:

```bash
python analysis/evaluate_probability_thresholds.py
```

Then aggregate the generated threshold CSVs for a selected statistical class:

```bash
python analysis/plot_roc_pr_confidence_intervals.py \
    --results-folder prob_matrix_analysis
```

For focused inspection of probability matrices:

```bash
python analysis/analyze_probability_matrices.py \
    --probability-dir prob_matrices \
    --ground-truth-dir networks
```

Each script exposes command-line help with `--help`. Large generated outputs are excluded from version control; the repository tracks only compact example figures used in the main README.
