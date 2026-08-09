# Reproducibility notes

## Lightweight repository check

The reusable package and example data do not require PyTorch:

```bash
python -m pip install -e ".[dev]"
pytest
python -m snn_connectivity inspect-network networks/network_N100_p24_CC01_1.yaml
python -m snn_connectivity inspect-spikes examples/synthetic_fdata_N100_demo.txt --expected-n 100
```

GitHub Actions runs the same lint, test and smoke-check sequence on pushes and pull requests.

## Full activity-matching experiments

Install the training dependency:

```bash
python -m pip install -e ".[training,dev]"
```

The thesis training script uses explicit NumPy, Python and PyTorch seeds. GPU kernels and floating-point execution can still introduce small platform-dependent differences, so exact bitwise identity is not assumed across hardware.

The complete training workflows live in `experiments/`, while post-training evaluation lives in `analysis/`. These scripts have been reorganized and documented for review without changing the numerical definitions used in the thesis experiments. Reusable helpers are kept separately in `src/snn_connectivity/`, where they can be tested independently.

## Data leakage

For supervised edge classification, network instances should be treated as the grouping unit. Splitting individual edges from one network across train and test sets would overstate generalization because all of those edges share the same simulated network dynamics and learned-matrix history.

## Generated outputs

Large generated directories (`LIFoutput_files/`, `prob_matrices/`, threshold CSVs and training plots) are excluded by `.gitignore`. A small pair of deterministic example figures is tracked so a reviewer can understand the project without running the full experiment.
