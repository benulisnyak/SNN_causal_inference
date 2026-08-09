import numpy as np

from snn_connectivity.features import build_directed_edge_features


def test_directed_edge_features_exclude_diagonal():
    matrices = np.array(
        [
            [[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]],
            [[0.0, 3.0, 4.0], [5.0, 0.0, 6.0], [7.0, 8.0, 0.0]],
        ]
    )

    features = build_directed_edge_features(matrices)

    assert len(features) == 6
    edge_01 = features[(features.source == 0) & (features.target == 1)].iloc[0]
    assert edge_01["mean"] == 2.0
    assert edge_01["median"] == 2.0
    assert edge_01["min"] == 1.0
    assert edge_01["max"] == 3.0
