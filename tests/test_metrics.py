import numpy as np

from snn_connectivity.metrics import evaluate_binary_predictions


def test_binary_metrics_are_well_defined():
    y_true = np.array([0, 0, 1, 1])
    y_probability = np.array([0.05, 0.2, 0.75, 0.95])

    metrics = evaluate_binary_predictions(y_true, y_probability)

    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.f1 == 1.0
