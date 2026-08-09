#!/usr/bin/env python3
"""
Create mean ROC and precision-recall curves from thresholding_stats_*.csv files.

For each unique network_statistics class, the script:
  * calculates ROC AUC and PR AUC separately for each network;
  * interpolates every network ROC curve onto a common FPR grid;
  * calculates mean TPR and a pointwise Student-t confidence interval at each
    common FPR coordinate;
  * interpolates every network PR curve onto a common recall grid and
    calculates the corresponding mean precision and confidence interval;
  * reports the mean individual-network AUC in each plot legend;
  * saves one ROC PDF, one PR PDF, and one consolidated summary CSV.

The interpolation means every valid network contributes exactly one value at
all coordinates of the common grid. Consequently, the number of observations
used for the confidence interval is constant across the grid for a given
statistics class and connection type.

AUC values remain trapezoidal areas calculated separately from each network's
original threshold curve. The legend reports the mean of those individual AUC
values, not the area under the interpolated mean curve.
"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


# =============================================================================
# Plot configuration
# =============================================================================

# Default input directory; the command-line argument can override it.
RESULTS_FOLDER = Path("stat_spec_prob_matrices/N100_p36_CC01")

INPUT_CSV_GLOB = "thresholding_stats_*.csv"
OUTPUT_SUBFOLDER = "metrics_plots"
SUMMARY_CSV_NAME = "mean_roc_pr_curve_data_FPR_binned_with_95CI.csv"

CONFIDENCE_LEVEL = 0.95

# Number of equally spaced coordinates from 0 to 1, inclusive. Each valid
# network is linearly interpolated onto these common grids before the mean and
# confidence interval are calculated. A value of 501 gives a grid spacing of
# 0.002.
NUM_FPR_BINS = 501
NUM_RECALL_BINS = 501

FIGURE_SIZE = (6.8, 5.3)
SAVE_DPI = 600
SHOW_FIGURE_TITLES = True
SHOW_GRID = True
CI_ALPHA = 0.14
AUC_DECIMALS = 3


# Excitatory source:
#     dashed blue, matching Total.
#
# Inhibitory source:
#     dashed orange, matching Undirected.
CONNECTION_TYPES = OrderedDict(
    [
        (
            "Total",
            {
                "prefix": "total",
                "color": "#1f77b4",
                "ls": "-",
            },
        ),
        (
            "Undirected",
            {
                "prefix": "undirected",
                "color": "#ff7f0e",
                "ls": "-",
            },
        ),
        (
            "Directed",
            {
                "prefix": "directed",
                "color": "#2ca02c",
                "ls": "-",
            },
        ),
        (
            "Excitatory",
            {
                "prefix": "excitatory_total",
                "color": "#1f77b4",
                "ls": "--",
            },
        ),
        (
            "Inhibitory",
            {
                "prefix": "inhibitory_total",
                "color": "#ff7f0e",
                "ls": "--",
            },
        ),
    ]
)


# =============================================================================
# General utilities
# =============================================================================


def natural_sort_key(value: object) -> list[object]:
    """
    Sort text naturally so that network 2 appears before network 10.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def set_plot_style() -> None:
    """
    Apply thesis-ready Matplotlib defaults.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9.5,
            "lines.linewidth": 1.9,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def required_metric_columns() -> list[str]:
    """
    Return the metric columns required from each threshold-analysis CSV.
    """
    columns: list[str] = []

    for style in CONNECTION_TYPES.values():
        prefix = style["prefix"]

        columns.extend(
            [
                f"{prefix}_false_positive_rate",
                f"{prefix}_true_positive_rate",
                f"{prefix}_precision",
            ]
        )

    return columns


# =============================================================================
# Input-file discovery and loading
# =============================================================================


def find_input_csvs(
    results_folder: Path,
    recursive: bool = True,
) -> list[Path]:
    """
    Find all thresholding_stats_*.csv files in the selected results folder.

    Files inside the metrics_plots output folder are excluded.
    """
    if not results_folder.is_dir():
        raise FileNotFoundError(
            f"Results folder does not exist: {results_folder}"
        )

    output_dir = (
        results_folder
        / OUTPUT_SUBFOLDER
    ).resolve()

    if recursive:
        iterator = results_folder.rglob(INPUT_CSV_GLOB)
    else:
        iterator = results_folder.glob(INPUT_CSV_GLOB)

    files: list[Path] = []

    for path in iterator:
        if not path.is_file():
            continue

        try:
            path.resolve().relative_to(output_dir)
            continue
        except ValueError:
            pass

        files.append(path)

    files.sort(
        key=lambda path: natural_sort_key(path)
    )

    if not files:
        raise FileNotFoundError(
            f"No files matching {INPUT_CSV_GLOB!r} were found in "
            f"{results_folder}."
        )

    return files


def read_threshold_csv(
    path: Path,
) -> pd.DataFrame:
    """
    Read and validate one threshold-analysis CSV.
    """
    dataframe = pd.read_csv(path)

    required_columns = [
        "network_statistics",
        "network_name",
        "threshold_index",
        "threshold",
        *required_metric_columns(),
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: "
            + ", ".join(missing_columns)
        )

    if dataframe.empty:
        raise ValueError(
            f"CSV contains no rows: {path}"
        )

    statistics_count = (
        dataframe["network_statistics"]
        .dropna()
        .nunique()
    )

    if statistics_count != 1:
        raise ValueError(
            f"Expected one network_statistics value in {path}."
        )

    network_count = (
        dataframe["network_name"]
        .dropna()
        .nunique()
    )

    if network_count != 1:
        raise ValueError(
            f"Expected one network_name value in {path}."
        )

    dataframe = dataframe.copy()

    dataframe["source_csv_file"] = str(path)

    numeric_columns = [
        "threshold_index",
        "threshold",
        *required_metric_columns(),
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    if dataframe["threshold_index"].isna().any():
        raise ValueError(
            f"Non-numeric threshold_index found in {path}."
        )

    if dataframe["threshold_index"].duplicated().any():
        raise ValueError(
            f"Duplicate threshold_index values found in {path}."
        )

    dataframe["threshold_index"] = (
        dataframe["threshold_index"]
        .astype(int)
    )

    dataframe = dataframe.sort_values(
        "threshold_index"
    ).reset_index(
        drop=True
    )

    return dataframe


def load_all_csvs(
    paths: list[Path],
) -> pd.DataFrame:
    """
    Load every threshold CSV into one combined dataframe.
    """
    dataframes: list[pd.DataFrame] = []

    for index, path in enumerate(
        paths,
        start=1,
    ):
        print(
            f"[{index}/{len(paths)}] Reading {path}"
        )

        dataframes.append(
            read_threshold_csv(path)
        )

    raw_dataframe = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    duplicate_files = (
        raw_dataframe[
            [
                "network_statistics",
                "network_name",
                "source_csv_file",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "network_statistics",
                "network_name",
            ]
        )
        .size()
    )

    duplicate_files = duplicate_files[
        duplicate_files > 1
    ]

    if not duplicate_files.empty:
        duplicate_description = "; ".join(
            f"{statistics}/{network}"
            for statistics, network
            in duplicate_files.index
        )

        raise ValueError(
            "Multiple CSV files were found for the same network "
            f"instance: {duplicate_description}"
        )

    return raw_dataframe


# =============================================================================
# Confidence intervals
# =============================================================================


def ci_stats(
    values: pd.Series,
    confidence_level: float,
) -> dict[str, float | int]:
    """
    Calculate the mean, standard deviation, standard error, and
    two-sided Student-t confidence interval.
    """
    array = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    number_of_values = int(
        array.size
    )

    if number_of_values == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n": 0,
        }

    mean_value = float(
        array.mean()
    )

    if number_of_values == 1:
        return {
            "mean": mean_value,
            "std": np.nan,
            "sem": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n": 1,
        }

    standard_deviation = float(
        array.std(ddof=1)
    )

    standard_error = float(
        standard_deviation
        / np.sqrt(number_of_values)
    )

    critical_value = float(
        student_t.ppf(
            0.5 + confidence_level / 2.0,
            df=number_of_values - 1,
        )
    )

    margin = (
        critical_value
        * standard_error
    )

    lower_bound = max(
        0.0,
        mean_value - margin,
    )

    upper_bound = min(
        1.0,
        mean_value + margin,
    )

    return {
        "mean": mean_value,
        "std": standard_deviation,
        "sem": standard_error,
        "ci_lower": lower_bound,
        "ci_upper": upper_bound,
        "n": number_of_values,
    }


# =============================================================================
# AUC calculations
# =============================================================================


def curve_auc(
    x_values: pd.Series,
    y_values: pd.Series,
) -> float:
    """
    Calculate a trapezoidal area under one threshold curve.

    The points are normally already in threshold order. If the x values
    are reversed or non-monotonic, they are corrected before integration.
    """
    x = pd.to_numeric(
        x_values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    y = pd.to_numeric(
        y_values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid_mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = np.clip(
        x[valid_mask],
        0.0,
        1.0,
    )

    y = np.clip(
        y[valid_mask],
        0.0,
        1.0,
    )

    if x.size < 2:
        return np.nan

    differences = np.diff(x)

    if np.all(
        differences <= 1e-12
    ):
        x = x[::-1]
        y = y[::-1]

    elif not np.all(
        differences >= -1e-12
    ):
        sorting_order = np.argsort(
            x,
            kind="mergesort",
        )

        x = x[sorting_order]
        y = y[sorting_order]

    if hasattr(
        np,
        "trapezoid",
    ):
        area = np.trapezoid(
            y,
            x,
        )
    else:
        area = np.trapz(
            y,
            x,
        )

    return float(
        np.clip(
            area,
            0.0,
            1.0,
        )
    )


def compute_individual_aucs(
    raw_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate ROC AUC and PR AUC separately for every network CSV.
    """
    rows: list[dict[str, object]] = []

    network_groups = raw_dataframe.groupby(
        [
            "network_statistics",
            "network_name",
        ],
        sort=True,
    )

    for (
        statistics_name,
        network_name,
    ), network_dataframe in network_groups:

        network_dataframe = network_dataframe.sort_values(
            "threshold_index"
        )

        source_file = str(
            network_dataframe["source_csv_file"].iloc[0]
        )

        for connection_label, style in CONNECTION_TYPES.items():
            prefix = style["prefix"]

            false_positive_rate = network_dataframe[
                f"{prefix}_false_positive_rate"
            ]

            true_positive_rate = network_dataframe[
                f"{prefix}_true_positive_rate"
            ]

            precision = network_dataframe[
                f"{prefix}_precision"
            ]

            roc_auc_value = curve_auc(
                false_positive_rate,
                true_positive_rate,
            )

            pr_auc_value = curve_auc(
                true_positive_rate,
                precision,
            )

            rows.append(
                {
                    "network_statistics": statistics_name,
                    "network_name": network_name,
                    "source_csv_file": source_file,
                    "connection_type": connection_label,
                    "connection_type_prefix": prefix,
                    "roc_auc": roc_auc_value,
                    "pr_auc": pr_auc_value,
                }
            )

    return pd.DataFrame(
        rows
    )


def summarize_aucs(
    individual_auc_dataframe: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    """
    Calculate the mean and 95% confidence interval of the individual-network
    AUC values within each network-statistics class.
    """
    rows: list[dict[str, object]] = []

    group_columns = [
        "network_statistics",
        "connection_type",
        "connection_type_prefix",
    ]

    grouped = individual_auc_dataframe.groupby(
        group_columns,
        sort=True,
    )

    for group_key, group_dataframe in grouped:
        (
            statistics_name,
            connection_label,
            connection_prefix,
        ) = group_key

        row: dict[str, object] = {
            "network_statistics": statistics_name,
            "connection_type": connection_label,
            "connection_type_prefix": connection_prefix,
        }

        for auc_name in [
            "roc_auc",
            "pr_auc",
        ]:
            auc_statistics = ci_stats(
                group_dataframe[auc_name],
                confidence_level,
            )

            for statistic_name, value in auc_statistics.items():
                row[
                    f"{auc_name}_{statistic_name}"
                ] = value

        rows.append(row)

    return pd.DataFrame(
        rows
    )



# =============================================================================
# Common-grid interpolation and curve aggregation
# =============================================================================


def first_value(
    dataframe: pd.DataFrame,
    column: str,
) -> object:
    """
    Return the first non-null value in a column.
    """
    if column not in dataframe.columns:
        return np.nan

    values = dataframe[column].dropna()

    if values.empty:
        return np.nan

    return values.iloc[0]


def prepare_curve_for_interpolation(
    x_values: pd.Series,
    y_values: pd.Series,
    curve_type: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Clean and order one network curve before interpolation.

    Duplicate x coordinates cannot be represented directly by a single-valued
    interpolation function. For duplicate FPR or recall values, the largest
    corresponding y value is retained. For ROC curves this preserves the upper
    ROC envelope. For PR curves it gives the usual interpolated upper precision
    envelope at repeated recall values.
    """
    x = pd.to_numeric(
        x_values,
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        y_values,
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_mask = np.isfinite(x) & np.isfinite(y)

    x = np.clip(x[valid_mask], 0.0, 1.0)
    y = np.clip(y[valid_mask], 0.0, 1.0)

    if x.size < 2:
        return None

    curve_dataframe = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    )

    curve_dataframe = (
        curve_dataframe
        .groupby("x", as_index=False, sort=True)["y"]
        .max()
        .sort_values("x")
        .reset_index(drop=True)
    )

    x = curve_dataframe["x"].to_numpy(dtype=float)
    y = curve_dataframe["y"].to_numpy(dtype=float)

    if x.size < 2:
        return None

    if curve_type == "roc":
        # ROC TPR should be non-decreasing as FPR increases. Small reversals can
        # occur because of numerical or input-order issues, so enforce the
        # monotonic upper envelope before interpolation.
        y = np.maximum.accumulate(y)

        if x[0] > 0.0:
            x = np.insert(x, 0, 0.0)
            y = np.insert(y, 0, 0.0)

        if x[-1] < 1.0:
            x = np.append(x, 1.0)
            y = np.append(y, 1.0)

    elif curve_type == "pr":
        # np.interp uses the nearest endpoint outside the observed range. This
        # fills the complete recall grid without inventing a new slope beyond
        # the network's observed PR curve.
        pass

    else:
        raise ValueError("curve_type must be 'roc' or 'pr'.")

    return x, y


def interpolate_curve(
    x_values: pd.Series,
    y_values: pd.Series,
    common_grid: np.ndarray,
    curve_type: str,
) -> np.ndarray | None:
    """
    Linearly interpolate one cleaned network curve onto a common grid.
    """
    prepared_curve = prepare_curve_for_interpolation(
        x_values,
        y_values,
        curve_type,
    )

    if prepared_curve is None:
        return None

    x, y = prepared_curve

    interpolated_y = np.interp(
        common_grid,
        x,
        y,
        left=float(y[0]),
        right=float(y[-1]),
    )

    return np.clip(interpolated_y, 0.0, 1.0)


def append_interpolated_summary_rows(
    rows: list[dict[str, object]],
    statistics_name: str,
    connection_label: str,
    connection_prefix: str,
    curve_type: str,
    common_grid: np.ndarray,
    interpolated_curves: list[np.ndarray],
    metadata: dict[str, object],
    confidence_level: float,
) -> None:
    """
    Add mean and confidence-interval rows for one interpolated curve family.
    """
    if not interpolated_curves:
        return

    curve_matrix = np.vstack(interpolated_curves)

    if curve_matrix.ndim != 2:
        raise ValueError("Interpolated curve matrix must be two-dimensional.")

    if curve_matrix.shape[1] != common_grid.size:
        raise ValueError("Interpolated curves do not match the common grid.")

    number_of_networks = int(curve_matrix.shape[0])

    for bin_index, x_value in enumerate(common_grid):
        y_statistics = ci_stats(
            pd.Series(curve_matrix[:, bin_index]),
            confidence_level,
        )

        row: dict[str, object] = {
            "network_statistics": statistics_name,
            "connection_type": connection_label,
            "connection_type_prefix": connection_prefix,
            "curve_type": curve_type,
            "bin_index": int(bin_index),
            "x_value": float(x_value),
            "confidence_level": confidence_level,
            "num_networks": number_of_networks,
            "interpolation_method": "linear",
            "duplicate_x_rule": "maximum_y",
        }

        for metadata_column, metadata_value in metadata.items():
            row[metadata_column] = metadata_value

        for statistic_name, value in y_statistics.items():
            row[f"y_{statistic_name}"] = value

        rows.append(row)


def build_curve_summary(
    raw_dataframe: pd.DataFrame,
    auc_summary_dataframe: pd.DataFrame,
    confidence_level: float,
    num_fpr_bins: int,
    num_recall_bins: int,
) -> pd.DataFrame:
    """
    Interpolate each network onto common FPR and recall grids, then calculate
    pointwise means and Student-t confidence intervals across networks.

    ROC rows contain TPR statistics at fixed FPR coordinates. PR rows contain
    precision statistics at fixed recall coordinates.
    """
    fpr_grid = np.linspace(0.0, 1.0, num_fpr_bins)
    recall_grid = np.linspace(0.0, 1.0, num_recall_bins)

    rows: list[dict[str, object]] = []

    metadata_columns = [
        "N",
        "p_code",
        "target_connection_probability",
        "cc_code",
        "target_clustering_coefficient",
        "threshold_mode",
        "threshold_rule",
    ]

    grouped_statistics = raw_dataframe.groupby(
        "network_statistics",
        sort=True,
    )

    for statistics_name, statistics_dataframe in grouped_statistics:
        metadata = {
            column: first_value(statistics_dataframe, column)
            for column in metadata_columns
        }

        network_groups = list(
            statistics_dataframe.groupby(
                "network_name",
                sort=True,
            )
        )

        for connection_label, style in CONNECTION_TYPES.items():
            prefix = style["prefix"]

            roc_curves: list[np.ndarray] = []
            pr_curves: list[np.ndarray] = []

            for network_name, network_dataframe in network_groups:
                network_dataframe = network_dataframe.sort_values(
                    "threshold_index"
                )

                false_positive_rate = network_dataframe[
                    f"{prefix}_false_positive_rate"
                ]

                true_positive_rate = network_dataframe[
                    f"{prefix}_true_positive_rate"
                ]

                precision = network_dataframe[
                    f"{prefix}_precision"
                ]

                interpolated_tpr = interpolate_curve(
                    false_positive_rate,
                    true_positive_rate,
                    fpr_grid,
                    "roc",
                )

                if interpolated_tpr is not None:
                    roc_curves.append(interpolated_tpr)
                else:
                    print(
                        "Warning: unable to interpolate ROC curve for "
                        f"{statistics_name}/{network_name}/{connection_label}."
                    )

                interpolated_precision = interpolate_curve(
                    true_positive_rate,
                    precision,
                    recall_grid,
                    "pr",
                )

                if interpolated_precision is not None:
                    pr_curves.append(interpolated_precision)
                else:
                    print(
                        "Warning: unable to interpolate PR curve for "
                        f"{statistics_name}/{network_name}/{connection_label}."
                    )

            append_interpolated_summary_rows(
                rows,
                str(statistics_name),
                connection_label,
                prefix,
                "roc",
                fpr_grid,
                roc_curves,
                metadata,
                confidence_level,
            )

            append_interpolated_summary_rows(
                rows,
                str(statistics_name),
                connection_label,
                prefix,
                "pr",
                recall_grid,
                pr_curves,
                metadata,
                confidence_level,
            )

    summary_dataframe = pd.DataFrame(rows)

    if summary_dataframe.empty:
        raise ValueError("No valid ROC or PR curves could be interpolated.")

    summary_dataframe = summary_dataframe.merge(
        auc_summary_dataframe,
        on=[
            "network_statistics",
            "connection_type",
            "connection_type_prefix",
        ],
        how="left",
        validate="many_to_one",
    )

    category_order = {
        style["prefix"]: index
        for index, style in enumerate(CONNECTION_TYPES.values())
    }

    curve_order = {
        "roc": 0,
        "pr": 1,
    }

    summary_dataframe["_category_order"] = (
        summary_dataframe["connection_type_prefix"]
        .map(category_order)
    )

    summary_dataframe["_curve_order"] = (
        summary_dataframe["curve_type"]
        .map(curve_order)
    )

    parsed_statistics = (
        summary_dataframe["network_statistics"]
        .astype(str)
        .str.extract(r"N(?P<_N>\d+)_p(?P<_p>\d+)_CC(?P<_CC>\d+)")
    )

    for sort_column in ["_N", "_p", "_CC"]:
        summary_dataframe[sort_column] = pd.to_numeric(
            parsed_statistics[sort_column],
            errors="coerce",
        )

    summary_dataframe = (
        summary_dataframe
        .sort_values(
            [
                "_N",
                "_p",
                "_CC",
                "network_statistics",
                "_category_order",
                "_curve_order",
                "bin_index",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_N",
                "_p",
                "_CC",
                "_category_order",
                "_curve_order",
            ]
        )
        .reset_index(drop=True)
    )

    return summary_dataframe

# =============================================================================
# Plot formatting
# =============================================================================


def format_axis(
    axis,
    x_label: str,
    y_label: str,
) -> None:
    """
    Apply common ROC and PR axis formatting.
    """
    axis.set_xlabel(
        x_label
    )

    axis.set_ylabel(
        y_label
    )

    axis.set_xlim(
        -0.02,
        1.02,
    )

    axis.set_ylim(
        -0.02,
        1.02,
    )

    axis.set_xticks(
        np.linspace(
            0.0,
            1.0,
            6,
        )
    )

    axis.set_yticks(
        np.linspace(
            0.0,
            1.0,
            6,
        )
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    axis.tick_params(
        direction="out"
    )

    if SHOW_GRID:
        axis.grid(
            True,
            alpha=0.22,
            linewidth=0.8,
        )


# =============================================================================
# Plot creation
# =============================================================================



def plot_curve_family(
    statistics_dataframe: pd.DataFrame,
    statistics_name: str,
    output_directory: Path,
    curve_type: str,
    confidence_level: float,
) -> Path:
    """
    Create either one FPR-binned ROC figure or one recall-binned PR figure.
    """
    if curve_type == "roc":
        auc_column = "roc_auc_mean"
        x_label = "False-positive rate"
        y_label = "True-positive rate"
        plot_token = "ROC"
        output_suffix = "FPR_binned"

    elif curve_type == "pr":
        auc_column = "pr_auc_mean"
        x_label = "Recall"
        y_label = "Precision"
        plot_token = "PR"
        output_suffix = "recall_binned"

    else:
        raise ValueError("curve_type must be 'roc' or 'pr'.")

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    for connection_label, style in CONNECTION_TYPES.items():
        prefix = style["prefix"]

        category_dataframe = (
            statistics_dataframe[
                (statistics_dataframe["connection_type_prefix"] == prefix)
                & (statistics_dataframe["curve_type"] == curve_type)
            ]
            .sort_values("bin_index")
        )

        if category_dataframe.empty:
            continue

        x_values = pd.to_numeric(
            category_dataframe["x_value"],
            errors="coerce",
        ).to_numpy(dtype=float)

        y_values = pd.to_numeric(
            category_dataframe["y_mean"],
            errors="coerce",
        ).to_numpy(dtype=float)

        lower_values = pd.to_numeric(
            category_dataframe["y_ci_lower"],
            errors="coerce",
        ).to_numpy(dtype=float)

        upper_values = pd.to_numeric(
            category_dataframe["y_ci_upper"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid_curve_mask = np.isfinite(x_values) & np.isfinite(y_values)

        x_values = x_values[valid_curve_mask]
        y_values = y_values[valid_curve_mask]
        lower_values = lower_values[valid_curve_mask]
        upper_values = upper_values[valid_curve_mask]

        if x_values.size == 0:
            continue

        auc_values = pd.to_numeric(
            category_dataframe[auc_column],
            errors="coerce",
        ).dropna()

        mean_auc = float(auc_values.iloc[0]) if not auc_values.empty else np.nan

        auc_text = (
            f"{mean_auc:.{AUC_DECIMALS}f}"
            if np.isfinite(mean_auc)
            else "N/A"
        )

        axis.plot(
            x_values,
            y_values,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=1.9,
            label=f"{connection_label}, AUC = {auc_text}",
        )

        valid_band_mask = (
            np.isfinite(lower_values)
            & np.isfinite(upper_values)
        )

        if np.any(valid_band_mask):
            axis.fill_between(
                x_values[valid_band_mask],
                lower_values[valid_band_mask],
                upper_values[valid_band_mask],
                color=style["color"],
                alpha=CI_ALPHA,
                linewidth=0.0,
            )

    format_axis(axis, x_label, y_label)

    if SHOW_FIGURE_TITLES:
        axis.set_title("Statistics Specific Classifier")

    axis.legend(loc="best")
    figure.tight_layout()

    output_path = (
        output_directory
        / f"{statistics_name}_mean_{plot_token}_curves_{output_suffix}_95CI.pdf"
    )

    figure.savefig(output_path, format="pdf")
    plt.close(figure)

    return output_path


def create_plots(
    summary_dataframe: pd.DataFrame,
    output_directory: Path,
    confidence_level: float,
) -> list[Path]:
    """
    Create ROC and PR figures for every unique network-statistics class.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_paths: list[Path] = []

    statistics_names = sorted(
        summary_dataframe[
            "network_statistics"
        ]
        .dropna()
        .unique(),
        key=natural_sort_key,
    )

    for statistics_name in statistics_names:
        statistics_dataframe = summary_dataframe[
            summary_dataframe[
                "network_statistics"
            ] == statistics_name
        ]

        roc_path = plot_curve_family(
            statistics_dataframe,
            str(statistics_name),
            output_directory,
            "roc",
            confidence_level,
        )

        pr_path = plot_curve_family(
            statistics_dataframe,
            str(statistics_name),
            output_directory,
            "pr",
            confidence_level,
        )

        saved_paths.extend(
            [
                roc_path,
                pr_path,
            ]
        )

    return saved_paths


# =============================================================================
# Command-line interface
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate ROC curves onto a common FPR grid and PR curves "
            "onto a common recall grid, calculate confidence intervals "
            "and mean AUCs, and save PDF "
            "figures."
        )
    )

    parser.add_argument(
        "--results-folder",
        type=Path,
        default=RESULTS_FOLDER,
        help=(
            "Folder containing threshold CSV files. "
            f"Default: {RESULTS_FOLDER}"
        ),
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=CONFIDENCE_LEVEL,
        help=(
            "Confidence level used for the Student-t intervals. "
            f"Default: {CONFIDENCE_LEVEL}"
        ),
    )

    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help=(
            "Search only the top level of the results folder."
        ),
    )

    parser.add_argument(
        "--num-fpr-bins",
        type=int,
        default=NUM_FPR_BINS,
        help=(
            "Number of equally spaced FPR grid coordinates, including 0 "
            f"and 1. Default: {NUM_FPR_BINS}"
        ),
    )

    parser.add_argument(
        "--num-recall-bins",
        type=int,
        default=NUM_RECALL_BINS,
        help=(
            "Number of equally spaced recall grid coordinates, including 0 "
            f"and 1. Default: {NUM_RECALL_BINS}"
        ),
    )

    return parser


# =============================================================================
# Main execution
# =============================================================================


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    if not (
        0.0
        < arguments.confidence_level
        < 1.0
    ):
        parser.error(
            "--confidence-level must be between 0 and 1."
        )

    if arguments.num_fpr_bins < 2:
        parser.error("--num-fpr-bins must be at least 2.")

    if arguments.num_recall_bins < 2:
        parser.error("--num-recall-bins must be at least 2.")

    set_plot_style()

    output_directory = (
        arguments.results_folder
        / OUTPUT_SUBFOLDER
    )

    csv_files = find_input_csvs(
        arguments.results_folder,
        recursive=(
            not arguments.non_recursive
        ),
    )

    print(
        f"Found {len(csv_files)} "
        "threshold-analysis CSV file(s)."
    )

    print(
        "Results folder: "
        f"{arguments.results_folder.resolve()}"
    )

    print(
        "Output folder: "
        f"{output_directory.resolve()}"
    )

    print(
        "Common FPR grid coordinates: "
        f"{arguments.num_fpr_bins}"
    )

    print(
        "Common recall grid coordinates: "
        f"{arguments.num_recall_bins}"
    )

    print()

    raw_dataframe = load_all_csvs(
        csv_files
    )


    individual_auc_dataframe = compute_individual_aucs(
        raw_dataframe
    )

    auc_summary_dataframe = summarize_aucs(
        individual_auc_dataframe,
        arguments.confidence_level,
    )

    summary_dataframe = build_curve_summary(
        raw_dataframe,
        auc_summary_dataframe,
        arguments.confidence_level,
        arguments.num_fpr_bins,
        arguments.num_recall_bins,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_directory
        / SUMMARY_CSV_NAME
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    saved_plot_paths = create_plots(
        summary_dataframe,
        output_directory,
        arguments.confidence_level,
    )

    number_of_statistics_classes = int(
        summary_dataframe[
            "network_statistics"
        ].nunique()
    )

    number_of_networks = int(
        raw_dataframe[
            [
                "network_statistics",
                "network_name",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print()
    print("Analysis complete.")

    print(
        "Unique statistics classes: "
        f"{number_of_statistics_classes}"
    )

    print(
        "Unique network instances: "
        f"{number_of_networks}"
    )

    print(
        f"Summary CSV: {summary_path}"
    )

    print(
        "Saved PDF figures:"
    )

    for path in saved_plot_paths:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()