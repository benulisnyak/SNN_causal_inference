import pandas as pd

from snn_connectivity.modeling import grouped_network_split


def test_grouped_split_has_no_network_overlap():
    frame = pd.DataFrame(
        {
            "network_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5,
            "mean": range(20),
        }
    )

    train_index, test_index = grouped_network_split(frame, test_size=0.5, random_state=7)
    train_groups = set(frame.iloc[train_index]["network_id"])
    test_groups = set(frame.iloc[test_index]["network_id"])

    assert train_groups.isdisjoint(test_groups)
    assert train_groups | test_groups == {"A", "B", "C", "D"}
