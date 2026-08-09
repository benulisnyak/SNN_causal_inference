"""Train an activity-matching recurrent spiking neural network.

This script implements stage one of the connectivity-inference pipeline. For each
target spike-train dataset it detects population bursts, crops each burst from the
onset of continuous activity through its activity peak, and independently optimizes
a recurrent LIF network to reproduce that target sequence. The learned recurrent
weight matrix from each burst is saved for downstream edge-feature engineering.

The numerical training procedure and output conventions match the thesis experiments;
the surrounding structure and documentation are organized for reproducibility and
code review.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import multiprocessing as mp
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Dataset and reproducibility defaults
# -----------------------------------------------------------------------------
RANDOM_SEED = 15
PROJECT_ROOT = Path(".")
DATA_BIN_WIDTH_SECONDS = 0.004
MIN_NETWORK_ACTIVITY_FRACTION = 0.9
BURST_DETECTION_WINDOW_MS = 50.0


def configure_global_reproducibility(seed: int = 15):
    """Seed Python, NumPy, and PyTorch for repeatable experiment initialization."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Burst extraction
# -----------------------------
def extract_bursts_from_spiketrain(
    data: torch.Tensor,
    data_dtres: float,
    nest_window_ms: float = 50.0,
    min_network_activity_fraction: float = 0.9,
    spike_threshold: float = 0.5,
    min_burst_len: int | None = None,
    max_burst_len: int | None = None,
    return_device: torch.device | None = None,
):
    """
    Detect bursts using the same population-activity criterion as
    NEST_meta_routines.determine_burst_rate() and then crop each detected burst
    for SNN training.

    NEST detection criterion
    ------------------------
    1. Divide time into non-overlapping ``nest_window_ms`` windows (50 ms in
       NEST_smallNetworks.py).
    2. Within each window, count the number of unique neurons that spiked at
       least once.
    3. Mark the window as burst-active when

           unique_active_neurons / N > min_network_activity_fraction

       The strict ``>`` comparison intentionally matches the NEST code.
    4. A new burst is detected on each 0 -> 1 transition of this binary
       window-level signal. Consecutive burst-active windows belong to the same
       burst.

    Training-sample crop
    --------------------
    For every detected NEST burst, locate the 4 ms bin with maximum population
    activity inside its run of burst-active 50 ms windows. Starting at that
    peak, backtrack through consecutive 4 ms bins that each contain at least one
    active neuron. The returned sample therefore runs from the first bin of
    continuous activity through the population-activity peak, inclusive.

    Because the input has already been reduced to binary 4 ms occupancy, exact
    spike times within each 4 ms bin are unavailable. Each 4 ms bin is assigned
    to a NEST window using its bin-centre time. This is the closest deterministic
    reconstruction of the original 50 ms detector from the binned data.

    Parameters
    ----------
    data:
        Tensor of shape [T, N], containing binary or floating-point spike-bin
        occupancy.
    data_dtres:
        Width of each input time bin in seconds (normally 0.004).
    nest_window_ms:
        Width of the NEST population-activity window in milliseconds.
    min_network_activity_fraction:
        NEST burst threshold. A window is active only when the fraction is
        strictly greater than this value.
    min_burst_len, max_burst_len:
        Optional limits on the final cropped 4 ms training samples.

    Returns
    -------
    burst_samples:
        List of float32 tensors with shape [T_burst, N].
    burst_meta:
        Detection and crop metadata using absolute 4 ms-bin indices.
    """
    assert data.dim() == 2, "Expected data shape [T, N]"
    if data_dtres <= 0.0:
        raise ValueError("data_dtres must be > 0")
    if nest_window_ms <= 0.0:
        raise ValueError("nest_window_ms must be > 0")
    if not (0.0 <= min_network_activity_fraction <= 1.0):
        raise ValueError("min_network_activity_fraction must be between 0.0 and 1.0")
    if min_burst_len is not None and min_burst_len < 1:
        raise ValueError("min_burst_len must be >= 1 when provided")
    if max_burst_len is not None and max_burst_len < 1:
        raise ValueError("max_burst_len must be >= 1 when provided")
    if (
        min_burst_len is not None
        and max_burst_len is not None
        and min_burst_len > max_burst_len
    ):
        raise ValueError("min_burst_len cannot be greater than max_burst_len")

    T, N = data.shape
    if T == 0 or N == 0:
        return [], []

    x = data.detach().to("cpu")
    x = (x > spike_threshold).to(torch.uint8)
    global_active = x.any(dim=1)

    # NEST uses non-overlapping 50 ms intervals. Since 50 ms is not an integer
    # multiple of 4 ms, assign each 4 ms occupancy bin according to its centre
    # time. Boundary times belong to the earlier window, matching NEST's <= test.
    bin_dt_ms = float(data_dtres) * 1000.0
    total_duration_ms = float(T) * bin_dt_ms
    num_complete_windows = int(total_duration_ms // float(nest_window_ms))
    if num_complete_windows < 1:
        return [], []

    bin_centres_ms = (torch.arange(T, dtype=torch.float64) + 0.5) * bin_dt_ms
    window_edges_ms = torch.arange(
        num_complete_windows + 1,
        dtype=torch.float64,
    ) * float(nest_window_ms)

    # searchsorted(..., right=False) maps a centre exactly on an upper boundary
    # to the preceding window, consistent with xtimes <= ttExactMS + tauMS.
    window_for_bin = torch.searchsorted(
        window_edges_ms,
        bin_centres_ms,
        right=False,
    ) - 1

    valid_bin = (window_for_bin >= 0) & (window_for_bin < num_complete_windows)
    active_counts = torch.zeros(num_complete_windows, dtype=torch.int64)
    active_neuron_masks = []

    for window_idx in range(num_complete_windows):
        bins_in_window = valid_bin & (window_for_bin == window_idx)
        if bins_in_window.any():
            neurons_active = x[bins_in_window].any(dim=0)
        else:
            neurons_active = torch.zeros(N, dtype=torch.bool)
        active_neuron_masks.append(neurons_active)
        active_counts[window_idx] = int(neurons_active.sum().item())

    active_fractions = active_counts.to(torch.float64) / float(N)
    nest_burst_active = active_fractions > float(min_network_activity_fraction)

    # Match the NEST 0 -> 1 transition logic, while grouping consecutive active
    # 50 ms windows as one detected burst.
    burst_window_runs = []
    window_idx = 0
    while window_idx < num_complete_windows:
        if not bool(nest_burst_active[window_idx].item()):
            window_idx += 1
            continue

        run_start_window = window_idx
        while (
            window_idx + 1 < num_complete_windows
            and bool(nest_burst_active[window_idx + 1].item())
        ):
            window_idx += 1
        run_end_window = window_idx
        burst_window_runs.append((run_start_window, run_end_window))
        window_idx += 1

    burst_samples = []
    burst_meta = []

    for run_start_window, run_end_window in burst_window_runs:
        bins_in_run = (
            valid_bin
            & (window_for_bin >= run_start_window)
            & (window_for_bin <= run_end_window)
        )
        run_bin_indices = torch.nonzero(bins_in_run, as_tuple=False).flatten()
        if run_bin_indices.numel() == 0:
            continue

        run_activity = x[run_bin_indices].sum(dim=1)
        peak_relative_idx = int(torch.argmax(run_activity).item())
        peak_idx = int(run_bin_indices[peak_relative_idx].item())

        # Backtrack from the 4 ms peak to the first bin in the uninterrupted run
        # of network activity. Every retained bin has at least one active neuron.
        start_idx = peak_idx
        while start_idx > 0 and bool(global_active[start_idx - 1].item()):
            start_idx -= 1

        if max_burst_len is not None:
            start_idx = max(start_idx, peak_idx - int(max_burst_len) + 1)

        burst_len = peak_idx - start_idx + 1
        if min_burst_len is not None and burst_len < int(min_burst_len):
            continue

        sample = x[start_idx : peak_idx + 1].to(torch.float32)
        burst_samples.append(sample)

        run_counts = active_counts[run_start_window : run_end_window + 1]
        run_fractions = active_fractions[run_start_window : run_end_window + 1]
        trigger_neurons = torch.nonzero(
            active_neuron_masks[run_start_window],
            as_tuple=False,
        ).flatten().tolist()

        burst_meta.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(peak_idx),
                "peak_idx": int(peak_idx),
                "duration_steps": int(burst_len),
                "duration_seconds": float(burst_len * data_dtres),
                "nest_window_start_idx": int(run_start_window),
                "nest_window_end_idx": int(run_end_window),
                "nest_window_start_seconds": float(
                    run_start_window * nest_window_ms / 1000.0
                ),
                "nest_window_end_seconds": float(
                    (run_end_window + 1) * nest_window_ms / 1000.0
                ),
                "nest_trigger_active_neurons": int(
                    active_counts[run_start_window].item()
                ),
                "nest_trigger_active_fraction": float(
                    active_fractions[run_start_window].item()
                ),
                "nest_max_active_neurons": int(run_counts.max().item()),
                "nest_max_active_fraction": float(run_fractions.max().item()),
                "trigger_neurons": trigger_neurons,
                "nest_window_ms": float(nest_window_ms),
                "burst_threshold": float(min_network_activity_fraction),
            }
        )

    if return_device is not None:
        burst_samples = [sample.to(return_device) for sample in burst_samples]

    return burst_samples, burst_meta


def parse_target_dataset_filename(filename: str) -> dict:
    """Parse network statistics and dataset index from an fdata filename."""
    match = re.fullmatch(
        r"(?:fdata_)?N(?P<N>\d+)_p(?P<p>\d+)_CC(?P<cc>\d+)_(?P<dataset_idx>\d+)(?P<inh_off_suffix>_inhOff)?\.txt",
        filename,
    )
    if match is None:
        raise ValueError(
            "Filename must match one of these patterns: "
            "'N<int>_p<int>_CC<int>_<dataset_idx>.txt' or "
            "'N<int>_p<int>_CC<int>_<dataset_idx>_inhOff.txt', "
            "with an optional 'fdata_' prefix. "
            f"Got: {filename}"
        )
    parts = match.groupdict()
    return {
        "N": int(parts["N"]),
        "p": int(parts["p"]),
        "cc": int(parts["cc"]),
        "dataset_idx": int(parts["dataset_idx"]),
        "has_inh_off_suffix": parts.get("inh_off_suffix") is not None,
    }


def resolve_reference_dataset_folder(input_path: str, reference_folder: str) -> Path:
    """Resolve a requested statistics folder relative to the project data root."""
    reference_folder_path = Path(reference_folder)

    if reference_folder_path.is_absolute():
        dataset_dir = reference_folder_path
    elif reference_folder_path.parts and reference_folder_path.parts[0] == "fdata":
        dataset_dir = Path(input_path) / reference_folder_path
    else:
        dataset_dir = Path(input_path) / "fdata" / reference_folder_path

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            "Could not find target dataset folder. Expected either an absolute path, "
            f"'{Path(input_path) / 'fdata' / reference_folder}', or '{Path(input_path) / reference_folder}'. "
            f"Resolved path: {dataset_dir}"
        )

    return dataset_dir


def find_target_datasets_in_reference_folder(input_path: str, reference_folder: str) -> list[Path]:
    """Discover and naturally sort target spike-train files for one statistics class."""
    dataset_dir = resolve_reference_dataset_folder(input_path, reference_folder)

    matching_files = []
    for candidate in dataset_dir.glob("*.txt"):
        try:
            candidate_parts = parse_target_dataset_filename(candidate.name)
        except ValueError:
            continue
        matching_files.append((candidate_parts["dataset_idx"], candidate))

    if len(matching_files) == 0:
        raise FileNotFoundError(
            "No target datasets were found in the requested folder. "
            f"Folder checked: {dataset_dir}"
        )

    matching_files.sort(key=lambda x: (x[0], x[1].name))
    return [path for _, path in matching_files]


def make_dataset_output_suffix(data_file_path: str | Path) -> str:
    """Return the dataset suffix used in output filenames.

    The target data files may be named either:
      fdata_N100_p12_CC05_1.txt
      fdata_N100_p12_CC05_1_inhOff.txt

    Output files intentionally omit the legacy '_inhOff' suffix so both forms map
    to the same learned-connectivity output name.
    """
    data_file_path = Path(data_file_path)
    suffix = data_file_path.stem.removeprefix("fdata_")
    suffix = suffix.removesuffix("_inhOff")
    return suffix


def make_output_npy_path(save_dir: str | Path, data_file_path: str | Path) -> str:
    """Build the normalized path for saved burst-specific connectivity matrices."""
    suffix = make_dataset_output_suffix(data_file_path)
    return str(Path(save_dir) / f"connectivity_matrices_{suffix}.npy")


def make_legacy_output_npy_path(save_dir: str | Path, data_file_path: str | Path) -> str:
    """Return the old output path that may have kept/added '_inhOff' in the filename."""
    suffix = make_dataset_output_suffix(data_file_path)
    return str(Path(save_dir) / f"connectivity_matrices_{suffix}_inhOff.npy")


def make_hamming_loss_output_npy_path(save_dir: str | Path, data_file_path: str | Path) -> str:
    """Build the path for per-sample final Hamming-loss output."""
    suffix = make_dataset_output_suffix(data_file_path)
    return str(Path(save_dir) / f"final_hamming_losses_{suffix}.npy")


def make_legacy_hamming_loss_output_npy_path(save_dir: str | Path, data_file_path: str | Path) -> str:
    """Return the old Hamming-loss path that may have kept/added '_inhOff' in the filename."""
    suffix = make_dataset_output_suffix(data_file_path)
    return str(Path(save_dir) / f"final_hamming_losses_{suffix}_inhOff.npy")


def learned_connectivity_output_exists(save_dir: str | Path, data_file_path: str | Path) -> bool:
    """Return True if the normalized or legacy connectivity-matrix .npy output exists."""
    return bool(existing_learned_connectivity_output_paths(save_dir, data_file_path))


def existing_learned_connectivity_output_paths(save_dir: str | Path, data_file_path: str | Path) -> list[Path]:
    """Return existing normalized/legacy connectivity output paths without duplicates."""
    candidates = [
        Path(make_output_npy_path(save_dir, data_file_path)),
        Path(make_legacy_output_npy_path(save_dir, data_file_path)),
    ]
    existing_paths = []
    seen = set()
    for candidate in candidates:
        candidate_key = candidate.resolve() if candidate.exists() else candidate.absolute()
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.is_file():
            existing_paths.append(candidate)
    return existing_paths


def load_burst_samples(data_file_path: str | Path, data_dtres: float):
    """Load a binned spike train and return the cropped burst samples used for training."""
    data_file_path = Path(data_file_path)
    data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using data device:", data_device)
    print(f"Loading target dataset: {data_file_path}")

    data = torch.tensor(
        np.genfromtxt(str(data_file_path), delimiter=", "),
        dtype=torch.float32,
        device=data_device,
    )

    burst_samples, burst_meta = extract_bursts_from_spiketrain(
        data=data,
        data_dtres=data_dtres,
        nest_window_ms=BURST_DETECTION_WINDOW_MS,
        min_network_activity_fraction=MIN_NETWORK_ACTIVITY_FRACTION,
        return_device=torch.device("cpu"),
    )

    print(f"Found {len(burst_samples)} bursts.")
    if len(burst_samples) == 0:
        raise ValueError("No bursts found. Adjust extraction parameters or check input data.")

    lengths = torch.tensor([b.shape[0] for b in burst_samples])
    print(
        f"Burst length (steps): min={lengths.min().item()}, "
        f"median={lengths.median().item()}, max={lengths.max().item()}"
    )
    print(
        f"Burst length (sec):  min={lengths.min().item() * data_dtres:.3f}, "
        f"median={lengths.median().item() * data_dtres:.3f}, max={lengths.max().item() * data_dtres:.3f}"
    )
    return burst_samples, burst_meta


# -----------------------------
# Weight initialization
# -----------------------------
def make_random_W(
    N: int,
    p: float,
    G: float,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create a reproducible sparse recurrent weight initialization."""
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]. Use p=1.0 for all-to-all.")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    mask = (torch.rand((N, N), generator=gen, dtype=dtype) < p).to(dtype)
    W = G * torch.abs(torch.randn((N, N), generator=gen, dtype=dtype)) * mask / (math.sqrt(N) * p)
    W.fill_diagonal_(0.0)
    return W.to(device)


# -----------------------------
# Target utilities
# -----------------------------
def crop_target_to_peak(target_full: torch.Tensor) -> torch.Tensor:
    """Crop a target sequence through its maximum population-activity bin."""
    active = (target_full > 0).sum(dim=1)
    t_peak = int(torch.argmax(active).item())
    return target_full[: t_peak + 1].contiguous()


def find_first_spike_bin_and_neurons(target_bins: torch.Tensor):
    """Return the first active bin and the neuron indices active in that bin."""
    for tb in range(target_bins.shape[0]):
        spikers = torch.nonzero(target_bins[tb] > 0, as_tuple=False).squeeze(-1)
        if spikers.numel() > 0:
            return int(tb), spikers.to(torch.int64)
    raise ValueError("Target contains no spikes.")


# -----------------------------
# Surrogate spike function
# -----------------------------
class FastSigmoidSurrogate(torch.autograd.Function):
    """Hard forward spike with a fast-sigmoid surrogate derivative."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, beta: torch.Tensor):
        ctx.save_for_backward(input_, beta)
        return (input_ > 0).to(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_, beta = ctx.saved_tensors
        denom = beta * input_.abs() + 1.0
        grad_input = grad_output / (denom * denom)
        return grad_input, None


def surrogate_spike(input_: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Apply the hard-threshold spike function with a surrogate backward derivative."""
    beta_t = torch.as_tensor(beta, dtype=input_.dtype, device=input_.device)
    return FastSigmoidSurrogate.apply(input_, beta_t)


# -----------------------------
# Differentiable LIF SNN
# -----------------------------
class SurrogateLIFNetwork(nn.Module):
    """Recurrent LIF network with trainable weights and short-term synaptic depression."""
    def __init__(
        self,
        W_init: torch.Tensor,
        *,
        dt: float = 1e-4,
        delay_steps: int = 0,
        surrogate_beta: float = 10.0,
        seed: int = 1,
    ):
        super().__init__()
        if W_init.ndim != 2 or W_init.shape[0] != W_init.shape[1]:
            raise ValueError("W_init must be a square matrix.")
        if delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")

        self.N = int(W_init.shape[0])
        self.dt = float(dt)
        self.delay_steps = int(delay_steps)
        self.surrogate_beta = float(surrogate_beta)
        self.seed = int(seed)

        self.W = nn.Parameter(W_init.clone())

        device = W_init.device
        dtype = W_init.dtype

        self.register_buffer("offdiag_mask", (1.0 - torch.eye(self.N, device=device, dtype=dtype)))

        # LIF and synaptic parameters used throughout the activity-matching experiments.
        self.V_rest_mV = -65.0
        self.V_reset_mV = -65.0
        self.V_th_mV = -50.0
        self.tau_m = 20e-3
        self.R_m_Mohm = 100.0
        self.t_ref = 2e-3

        self.I_ext_mean_nA = 0.13
        self.I_ext_std_nA = 0.00

        self.w_nA = 0.15
        self.tau_syn = 50e-3

        self.U = 0.3
        self.tau_rec = 1200e-3

    def clamp_parameters_(self):
        with torch.no_grad():
            self.W.clamp_(min=-10.0, max=10.0)
            self.W.fill_diagonal_(0.0)

    def forward(
        self,
        *,
        steps: int,
        stim_step: int,
        stim_neurons,
        stim_value_nA: float,
        hard: bool = False,
    ) -> torch.Tensor:
        device = self.W.device
        dtype = self.W.dtype
        N = self.N

        W_eff = self.W * self.offdiag_mask

        V = torch.full((N,), self.V_rest_mV, device=device, dtype=dtype)
        I_syn = torch.zeros((N,), device=device, dtype=dtype)
        x = torch.ones((N,), device=device, dtype=dtype)
        ref_countdown = torch.zeros((N,), device=device, dtype=dtype)

        spikes = torch.zeros((steps, N), device=device, dtype=dtype)
        spike_buffer = [torch.zeros((N,), device=device, dtype=dtype) for _ in range(self.delay_steps + 1)]
        buf_idx = 0

        alpha_syn = math.exp(-self.dt / self.tau_syn)
        alpha_x = math.exp(-self.dt / self.tau_rec)
        ref_steps = float(round(self.t_ref / self.dt))

        v_reset_tensor = torch.tensor(self.V_reset_mV, device=device, dtype=dtype)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        for k in range(steps):
            if self.I_ext_std_nA > 0.0:
                noise = torch.randn((N,), generator=gen, dtype=dtype).to(device)
                I_ext = self.I_ext_mean_nA + self.I_ext_std_nA * noise
            else:
                I_ext = torch.full((N,), self.I_ext_mean_nA, device=device, dtype=dtype)

            if k == stim_step:
                I_ext = I_ext.clone()
                stim_idx = torch.as_tensor(stim_neurons, device=device, dtype=torch.int64).flatten()
                if stim_idx.numel() > 0:
                    I_ext[stim_idx] = float(stim_value_nA)

            x = 1.0 - (1.0 - x) * alpha_x
            I_syn = I_syn * alpha_syn

            spikes_to_deliver = spike_buffer[buf_idx]
            release = self.U * x * spikes_to_deliver
            x = x - release
            I_syn = I_syn + self.w_nA * (release @ W_eff)

            in_refr = ref_countdown > 0.0
            ref_countdown = torch.where(in_refr, ref_countdown - 1.0, ref_countdown)
            not_refr = ref_countdown <= 0.0

            dV = (-(V - self.V_rest_mV) + self.R_m_Mohm * (I_ext + I_syn)) * (self.dt / self.tau_m)
            V_candidate = torch.where(not_refr, V + dV, v_reset_tensor)

            if hard:
                spk = ((V_candidate - self.V_th_mV) >= 0.0).to(dtype) * not_refr.to(dtype)
            else:
                spk = surrogate_spike(V_candidate - self.V_th_mV, beta=self.surrogate_beta) * not_refr.to(dtype)

            V = torch.where(spk > 0.0, v_reset_tensor, V_candidate)
            ref_countdown = torch.where(
                spk > 0.0,
                torch.full_like(ref_countdown, ref_steps),
                ref_countdown,
            )

            spikes[k] = spk
            spike_buffer[buf_idx] = spk
            buf_idx = (buf_idx + 1) % (self.delay_steps + 1)

        return spikes


# -----------------------------
# Binning utilities
# -----------------------------
def bin_spikes_hard(
    spikes: torch.Tensor,
    dt: float,
    bin_dt: float = 4e-3,
    drop_remainder: bool = True,
):
    """Aggregate simulation steps into hard binary spike occupancy bins."""
    spikes = torch.as_tensor(spikes)
    if spikes.ndim != 2:
        raise ValueError("spikes must be shape (steps, N).")
    if dt <= 0 or bin_dt <= 0:
        raise ValueError("dt and bin_dt must be > 0.")

    device = spikes.device
    steps, N = spikes.shape

    bin_steps = int(round(bin_dt / dt))
    if bin_steps < 1:
        raise ValueError("bin_dt too small relative to dt")

    bin_dt_eff = bin_steps * dt
    n_full = steps // bin_steps
    rem = steps % bin_steps

    core = spikes[: n_full * bin_steps].reshape(n_full, bin_steps, N)
    binned = (core.sum(dim=1) > 0).to(torch.uint8)

    if rem != 0 and not drop_remainder:
        tail = spikes[n_full * bin_steps :]
        tail_binned = (tail.sum(dim=0, keepdim=True) > 0).to(torch.uint8)
        binned = torch.cat([binned, tail_binned], dim=0)

    t_bins = torch.arange(binned.shape[0], device=device, dtype=torch.float64) * bin_dt_eff
    return binned, t_bins, bin_steps


def bin_spikes_soft(
    spikes: torch.Tensor,
    dt: float,
    bin_dt: float = 4e-3,
    drop_remainder: bool = True,
):
    """
    Differentiable bin occupancy for training.
    We sum spikes within the bin and clamp to [0, 1], so the target remains
    binary occupancy while gradients still flow back through the surrogate spike.
    """
    spikes = torch.as_tensor(spikes)
    if spikes.ndim != 2:
        raise ValueError("spikes must be shape (steps, N).")
    if dt <= 0 or bin_dt <= 0:
        raise ValueError("dt and bin_dt must be > 0.")

    device = spikes.device
    steps, N = spikes.shape

    bin_steps = int(round(bin_dt / dt))
    if bin_steps < 1:
        raise ValueError("bin_dt too small relative to dt")

    bin_dt_eff = bin_steps * dt
    n_full = steps // bin_steps
    rem = steps % bin_steps

    core = spikes[: n_full * bin_steps].reshape(n_full, bin_steps, N)
    binned = core.sum(dim=1).clamp(0.0, 1.0)

    if rem != 0 and not drop_remainder:
        tail = spikes[n_full * bin_steps :]
        tail_binned = tail.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
        binned = torch.cat([binned, tail_binned], dim=0)

    t_bins = torch.arange(binned.shape[0], device=device, dtype=torch.float64) * bin_dt_eff
    return binned, t_bins, bin_steps


def hamming_loss_step(model_step01: torch.Tensor, target_step01: torch.Tensor) -> torch.Tensor:
    """Return the Hamming mismatch for one hard spike bin."""
    return torch.abs(model_step01.to(torch.int32) - target_step01.to(torch.int32)).sum()


def hamming_loss_per_timestep(model_stage01: torch.Tensor, target_stage01: torch.Tensor) -> torch.Tensor:
    """Return hard Hamming mismatch independently for each time bin."""
    return torch.abs(model_stage01.to(torch.int32) - target_stage01.to(torch.int32)).sum(dim=1)


def soft_hamming_loss_step(model_step_soft: torch.Tensor, target_step01: torch.Tensor) -> torch.Tensor:
    """Return the differentiable mismatch for one soft spike bin."""
    return torch.abs(model_step_soft - target_step01.to(dtype=model_step_soft.dtype)).sum()


def soft_hamming_loss_per_timestep(model_stage_soft: torch.Tensor, target_stage01: torch.Tensor) -> torch.Tensor:
    """Return differentiable mismatch independently for each time bin."""
    return torch.abs(model_stage_soft - target_stage01.to(dtype=model_stage_soft.dtype)).sum(dim=1)


def scalar_hard_error(model_val01: torch.Tensor, target_val01: torch.Tensor) -> int:
    """Convert a single hard prediction/target comparison to an integer error."""
    return int(torch.abs(model_val01.to(torch.int32) - target_val01.to(torch.int32)).item())


def scalar_soft_error(model_val_soft: torch.Tensor, target_val01: torch.Tensor) -> float:
    """Convert a single soft prediction/target comparison to a floating-point error."""
    return float(torch.abs(model_val_soft - target_val01.to(dtype=model_val_soft.dtype)).item())


# -----------------------------
# Prefix simulation / evaluation helpers
# -----------------------------
def simulate_prefix_soft(
    model: nn.Module,
    *,
    prefix_bins: int,
    stim_step: int,
    starter_neurons,
    stim_value_nA: float,
    dt: float,
    bin_dt_eff: float,
    init_delay_bins: int,
    bin_steps: int,
) -> torch.Tensor:
    """Simulate a target prefix and return differentiable binned activity."""
    total_bins_sim = init_delay_bins + prefix_bins
    steps = total_bins_sim * bin_steps
    spikes = model(
        steps=steps,
        stim_step=stim_step,
        stim_neurons=starter_neurons,
        stim_value_nA=float(stim_value_nA),
        hard=False,
    )
    model_stage_soft, _, _ = bin_spikes_soft(
        spikes,
        dt=dt,
        bin_dt=bin_dt_eff,
        drop_remainder=True,
    )
    return model_stage_soft[-prefix_bins:].contiguous()


def simulate_prefix_hard(
    model: nn.Module,
    *,
    prefix_bins: int,
    stim_step: int,
    starter_neurons,
    stim_value_nA: float,
    dt: float,
    bin_dt_eff: float,
    init_delay_bins: int,
    bin_steps: int,
) -> torch.Tensor:
    """Simulate a target prefix and return hard binned spike occupancy."""
    total_bins_sim = init_delay_bins + prefix_bins
    steps = total_bins_sim * bin_steps
    with torch.no_grad():
        spikes = model(
            steps=steps,
            stim_step=stim_step,
            stim_neurons=starter_neurons,
            stim_value_nA=float(stim_value_nA),
            hard=True,
        )
        model_stage_hard, _, _ = bin_spikes_hard(
            spikes,
            dt=dt,
            bin_dt=bin_dt_eff,
            drop_remainder=True,
        )
    return model_stage_hard[-prefix_bins:].contiguous()


def build_incoming_train_mask(
    N: int,
    post_idx: int,
    allowed_pre: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    learning_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the mask selecting trainable incoming weights for one postsynaptic neuron."""
    mask = torch.zeros((N, N), device=device, dtype=dtype)
    if allowed_pre.numel() > 0:
        if learning_scales is None:
            mask[allowed_pre, post_idx] = 1.0
        else:
            learning_scales = learning_scales.to(device=device, dtype=dtype)
            if learning_scales.shape != allowed_pre.shape:
                raise ValueError("learning_scales must have the same shape as allowed_pre")
            mask[allowed_pre, post_idx] = learning_scales
    mask[post_idx, post_idx] = 0.0
    return mask


def get_recent_previous_spikers_and_learning_scales(
    stage_hard_prefix: torch.Tensor,
    *,
    tb: int,
    post_idx: int,
    history_window_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Use the most recent hard spike time for each neuron in bins [0, tb-1].

    A presynaptic neuron is considered eligible only if its most recent spike is
    within `history_window_bins` bins of the current target bin `tb`.

    The learning scale decays linearly with recency:
      distance = tb - last_spike_bin
      scale = 1 - (distance - 1) / history_window_bins

    so distance=1 -> 1.0, distance=history_window_bins+1 -> 0.0.
    """
    if history_window_bins <= 0:
        raise ValueError("history_window_bins must be >= 1")

    device = stage_hard_prefix.device
    dtype = torch.float64
    N = stage_hard_prefix.shape[1]

    if tb <= 0:
        empty_idx = torch.empty((0,), device=device, dtype=torch.int64)
        empty_scale = torch.empty((0,), device=device, dtype=dtype)
        last_spike_bin = torch.full((N,), -1, device=device, dtype=torch.int64)
        return empty_idx, empty_scale, last_spike_bin

    prev_stage = stage_hard_prefix[:tb].to(torch.bool)
    last_spike_bin = torch.full((N,), -1, device=device, dtype=torch.int64)
    for prev_tb in range(tb):
        spikers = prev_stage[prev_tb]
        last_spike_bin = torch.where(
            spikers,
            torch.full_like(last_spike_bin, prev_tb),
            last_spike_bin,
        )

    distances = tb - last_spike_bin
    valid = (last_spike_bin >= 0) & (distances <= history_window_bins)
    valid[post_idx] = False

    allowed_pre = torch.nonzero(valid, as_tuple=False).squeeze(-1).to(torch.int64)
    if allowed_pre.numel() == 0:
        empty_scale = torch.empty((0,), device=device, dtype=dtype)
        return allowed_pre, empty_scale, last_spike_bin

    allowed_distances = distances[allowed_pre].to(dtype=dtype)
    learning_scales = 1.0 - (allowed_distances - 1.0) / float(history_window_bins)
    learning_scales = learning_scales.clamp_(min=0.0, max=1.0)
    return allowed_pre, learning_scales, last_spike_bin


def enforce_incoming_constraints_(
    model: SurrogateLIFNetwork,
    *,
    post_idx: int,
    allowed_pre: torch.Tensor,
):
    """Apply sign, diagonal, and mask constraints to an incoming weight column in place."""
    allowed_pre = allowed_pre.to(device=model.W.device, dtype=torch.int64)
    allowed_bool = torch.zeros((model.N,), device=model.W.device, dtype=torch.bool)
    if allowed_pre.numel() > 0:
        allowed_bool[allowed_pre] = True

    with torch.no_grad():
        disallowed = ~allowed_bool
        model.W[disallowed, post_idx] = 0.0
        model.W[post_idx, post_idx] = 0.0
        model.clamp_parameters_()


def ordered_neurons_for_bin(target_bin: torch.Tensor) -> list[int]:
    """Return active target neurons for a bin in deterministic index order."""
    active = torch.nonzero(target_bin > 0, as_tuple=False).squeeze(-1).tolist()
    silent = torch.nonzero(target_bin == 0, as_tuple=False).squeeze(-1).tolist()
    return active + silent


def grad_tensor_stats(x: torch.Tensor) -> dict:
    """Summarize finite values, norm, and magnitude statistics for a gradient tensor."""
    x = torch.as_tensor(x).detach()
    out = {
        "numel": int(x.numel()),
        "nonzero": int((x != 0).sum().item()) if x.numel() > 0 else 0,
        "finite": bool(torch.isfinite(x).all().item()) if x.numel() > 0 else True,
    }
    if x.numel() == 0:
        out.update(
            {
                "norm": 0.0,
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        )
        return out

    abs_x = x.abs()
    out.update(
        {
            "norm": float(torch.linalg.vector_norm(x).item()),
            "mean_abs": float(abs_x.mean().item()),
            "max_abs": float(abs_x.max().item()),
            "min": float(x.min().item()),
            "max": float(x.max().item()),
        }
    )
    return out


def should_print_gradient_diagnostics(
    *,
    enable: bool,
    tb: int,
    inner_it: int,
    suspect: bool,
    print_every: int,
    start_bin: int,
    only_when_suspect: bool,
) -> bool:
    """Decide whether the current update meets the configured diagnostic criteria."""
    if not enable or tb < start_bin:
        return False
    periodic = (print_every > 0) and (inner_it % print_every == 0)
    if only_when_suspect:
        return suspect
    return periodic or suspect


# -----------------------------
# Plotting / saving
# -----------------------------
def plot_connectivity_matrix_and_strengths(W_global, title_prefix="Global connectivity"):
    """Plot a learned connectivity matrix and its weight-strength distribution."""
    W = torch.as_tensor(W_global).detach().cpu()
    N = W.shape[0]

    plt.figure(figsize=(6, 5))
    plt.imshow(W.numpy(), aspect="auto")
    plt.colorbar(label="Weight")
    plt.title(f"{title_prefix}: matrix (N={N})")
    plt.xlabel("Post-syn neuron j")
    plt.ylabel("Pre-syn neuron i")
    plt.tight_layout()
    plt.show()

    mask_offdiag = ~torch.eye(N, dtype=torch.bool)
    w_off = W[mask_offdiag]
    w_nz = w_off[w_off != 0]

    if w_nz.numel() == 0:
        print("No nonzero off-diagonal weights to plot histogram for.")
        return

    plt.figure(figsize=(7, 3.5))
    plt.hist(w_nz.numpy(), bins=50)
    plt.title(f"{title_prefix}: off-diagonal weight strengths (nonzero)")
    plt.xlabel("Weight")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 3.5))
    plt.hist(w_nz.abs().numpy(), bins=50)
    plt.title(f"{title_prefix}: |weight| strengths (nonzero, off-diagonal)")
    plt.xlabel("|Weight|")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def save_sample_connectivity_npy(
    sample_connectivity_matrices: list[torch.Tensor | np.ndarray],
    path: str = "sample_connectivity_matrices.npy",
) -> np.ndarray:
    """
    Save learned connectivity matrices as a single stacked NumPy array.

    The output matches the reference connectivity-matrix .npy format:
      shape = (num_samples, N, N)
      dtype = float32

    Each sample matrix is stored at output[sample_position, :, :]. The caller sorts
    results by sample_idx before passing the matrices, so sample_position follows
    ascending sample index.
    """
    matrices_to_save = []
    for W in sample_connectivity_matrices:
        Wc = torch.as_tensor(W).detach().cpu().clone()
        Wc.fill_diagonal_(0.0)
        matrices_to_save.append(Wc.numpy().astype(np.float32, copy=False))

    if len(matrices_to_save) == 0:
        matrices_array = np.empty((0, 0, 0), dtype=np.float32)
    else:
        matrices_array = np.ascontiguousarray(np.stack(matrices_to_save, axis=0), dtype=np.float32)

    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    np.save(path, matrices_array)

    print(
        f"Saved {matrices_array.shape[0]} sample connectivity matrices to: {path} "
        f"(shape={matrices_array.shape}, dtype={matrices_array.dtype})"
    )
    return matrices_array


def save_final_hamming_losses_npy(
    results: list[dict],
    path: str = "final_hamming_losses.npy",
) -> np.ndarray:
    """
    Save per-sample final Hamming losses.

    Output columns:
      column 0 = sample_idx
      column 1 = final_total_hamming

    The rows are ordered by ascending sample_idx because the caller sorts results
    before passing them in.
    """
    hamming_loss_array = np.asarray(
        [
            [int(result["sample_idx"]), int(result["final_total_hamming"])]
            for result in results
        ],
        dtype=np.int64,
    )

    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    np.save(path, hamming_loss_array)

    print(
        f"Saved final Hamming losses to: {path} "
        f"(shape={hamming_loss_array.shape}, dtype={hamming_loss_array.dtype}; "
        "columns=[sample_idx, final_total_hamming])"
    )
    return hamming_loss_array


# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------
dt = 4e-3
bin_dt = 4e-3
train_seed = 1

sim_device = torch.device("cpu")
dtype = torch.float64

init_delay_bins = 1
stim_value_nA = 70.0

max_neuron_train_iters = 100
max_attempts_per_neuron_time = 2
learning_rate = 3e-1
attempt_lr_decay = 0.5
restart_noise_std = 0.05
surrogate_beta = 7.0
grad_clip_norm = 10.0

# Loss regularization and acceptance criteria
l1_lambda = 0  # 5e-5
previous_soft_barrier_scale = 0.5
current_soft_improve_tol = 1e-7

# Restrict candidate presynaptic neurons to recent model history; more recent
# spikes receive larger learning scales.
presynaptic_history_window_bins = 6

# Optional diagnostics for vanishing/exploding gradients during difficult samples.
enable_gradient_diagnostics = False
gradient_diag_print_every_inner_iters = 10
gradient_diag_only_when_suspect = False
gradient_diag_start_bin = 0
grad_vanish_threshold = 1e-10
grad_explode_threshold = 1e2
weight_large_threshold = 9.5


max_samples_to_use = None
selected_sample_idx = None  # Optional single-sample override for focused experiments.

# Skip a dataset when its normalized learned-connectivity output already exists.
enable_skip_trained_samples = True


plot_overlay_for_each_sample = False
plot_overlay_for_sample_indices = set()

# Parallel training across independently optimized burst samples
parallel_train_samples = True
max_parallel_workers = os.cpu_count()
worker_torch_threads = 1
verbose_sample_progress = False

bin_steps = int(round(bin_dt / dt))
bin_dt_eff = bin_steps * dt
t_stim = init_delay_bins * bin_steps

initial_connection_probability = 0.2
initial_weight_scale = 0.0


def build_sample_iter(burst_samples: list[torch.Tensor], *, selected_sample_idx, max_samples_to_use):
    """Select all burst samples or the configured subset for independent training."""
    num_total = len(burst_samples)
    if selected_sample_idx is not None:
        if not (0 <= selected_sample_idx < num_total):
            raise ValueError(f"selected_sample_idx={selected_sample_idx} out of range [0, {num_total - 1}]")
        return [(selected_sample_idx, burst_samples[selected_sample_idx])]
    if max_samples_to_use is None:
        return list(enumerate(burst_samples))
    return list(enumerate(burst_samples[: min(max_samples_to_use, num_total)]))


def train_single_sample(task: dict) -> dict:
    """Optimize one activity-matching SNN for a single cropped burst sample."""
    s_idx = int(task["sample_idx"])
    sample_array = np.asarray(task["sample_array"], dtype=np.float32)
    W_base_np = np.asarray(task["W_base"], dtype=np.float64)
    N = int(task["N"])
    dt = float(task["dt"])
    bin_dt_eff = float(task["bin_dt_eff"])
    bin_steps = int(task["bin_steps"])
    init_delay_bins = int(task["init_delay_bins"])
    t_stim = int(task["t_stim"])
    stim_value_nA = float(task["stim_value_nA"])
    surrogate_beta = float(task["surrogate_beta"])
    train_seed = int(task["train_seed"])
    max_neuron_train_iters = int(task["max_neuron_train_iters"])
    max_attempts_per_neuron_time = int(task["max_attempts_per_neuron_time"])
    learning_rate = float(task["learning_rate"])
    attempt_lr_decay = float(task["attempt_lr_decay"])
    restart_noise_std = float(task["restart_noise_std"])
    grad_clip_norm = float(task["grad_clip_norm"])
    l1_lambda = float(task["l1_lambda"])
    previous_soft_barrier_scale = float(task["previous_soft_barrier_scale"])
    current_soft_improve_tol = float(task["current_soft_improve_tol"])
    presynaptic_history_window_bins = int(task["presynaptic_history_window_bins"])
    enable_gradient_diagnostics = bool(task["enable_gradient_diagnostics"])
    gradient_diag_print_every_inner_iters = int(task["gradient_diag_print_every_inner_iters"])
    gradient_diag_only_when_suspect = bool(task["gradient_diag_only_when_suspect"])
    gradient_diag_start_bin = int(task["gradient_diag_start_bin"])
    grad_vanish_threshold = float(task["grad_vanish_threshold"])
    grad_explode_threshold = float(task["grad_explode_threshold"])
    weight_large_threshold = float(task["weight_large_threshold"])
    verbose_sample_progress = bool(task["verbose_sample_progress"])
    do_plot = bool(task["do_plot"])
    worker_torch_threads = max(1, int(task["worker_torch_threads"]))

    torch.set_num_threads(worker_torch_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    worker_seed = train_seed + s_idx
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    sim_device = torch.device("cpu")
    dtype = torch.float64

    model = SurrogateLIFNetwork(
        W_init=torch.as_tensor(W_base_np, dtype=dtype, device=sim_device),
        dt=dt,
        delay_steps=0,
        surrogate_beta=surrogate_beta,
        seed=train_seed,
    ).to(sim_device)
    model.clamp_parameters_()

    target_full = torch.as_tensor(sample_array, dtype=torch.float32, device=sim_device)
    target_full = (target_full > 0).to(torch.uint8)

    T_bins_full, N_from_data = target_full.shape
    if N_from_data != N:
        raise ValueError(f"Sample {s_idx}: dataset N={N_from_data} does not match expected N={N}.")

    target_peak = crop_target_to_peak(target_full)
    t0_bin_global, starter_neurons = find_first_spike_bin_and_neurons(target_peak)
    target = target_peak[t0_bin_global:].contiguous()
    B = int(target.shape[0])

    if verbose_sample_progress:
        starter_neurons_list = starter_neurons.detach().cpu().tolist()
        print(
            f"\n=== Sample {s_idx}: B={B}, N={N}, starter neurons={starter_neurons_list} "
            f"(from first active target bin) ==="
        )

    accepted_updates = 0
    rejected_updates = 0
    skipped_updates = 0

    prefix0_hard = simulate_prefix_hard(
        model,
        prefix_bins=1,
        stim_step=t_stim,
        starter_neurons=starter_neurons,
        stim_value_nA=stim_value_nA,
        dt=dt,
        bin_dt_eff=bin_dt_eff,
        init_delay_bins=init_delay_bins,
        bin_steps=bin_steps,
    )
    prefix0_loss = int(hamming_loss_per_timestep(prefix0_hard, target[:1]).sum().item())
    if verbose_sample_progress:
        print(f"Starter bin hard Hamming = {prefix0_loss}")

    for tb in range(1, B):
        target_prefix = target[: tb + 1]
        neuron_order = ordered_neurons_for_bin(target[tb])

        for post_idx in neuron_order:
            base_stage_hard = simulate_prefix_hard(
                model,
                prefix_bins=tb + 1,
                stim_step=t_stim,
                starter_neurons=starter_neurons,
                stim_value_nA=stim_value_nA,
                dt=dt,
                bin_dt_eff=bin_dt_eff,
                init_delay_bins=init_delay_bins,
                bin_steps=bin_steps,
            )
            base_stage_soft = simulate_prefix_soft(
                model,
                prefix_bins=tb + 1,
                stim_step=t_stim,
                starter_neurons=starter_neurons,
                stim_value_nA=stim_value_nA,
                dt=dt,
                bin_dt_eff=bin_dt_eff,
                init_delay_bins=init_delay_bins,
                bin_steps=bin_steps,
            ).detach()

            base_curr_hard = scalar_hard_error(base_stage_hard[tb, post_idx], target_prefix[tb, post_idx])
            base_curr_soft = scalar_soft_error(base_stage_soft[tb, post_idx], target_prefix[tb, post_idx])

            if base_curr_hard == 0:
                skipped_updates += 1
                continue

            base_prev_step_hard = hamming_loss_per_timestep(base_stage_hard[:tb], target_prefix[:tb])
            base_prev_step_soft = soft_hamming_loss_per_timestep(base_stage_soft[:tb], target_prefix[:tb])

            prev_spikers, prev_spiker_learning_scales, _ = get_recent_previous_spikers_and_learning_scales(
                base_stage_hard,
                tb=tb,
                post_idx=post_idx,
                history_window_bins=presynaptic_history_window_bins,
            )

            if prev_spikers.numel() == 0:
                rejected_updates += 1
                continue

            base_W = model.W.detach().clone()
            train_mask = build_incoming_train_mask(
                N,
                post_idx,
                prev_spikers,
                device=model.W.device,
                dtype=model.W.dtype,
                learning_scales=prev_spiker_learning_scales,
            )

            accepted = False

            for attempt in range(max_attempts_per_neuron_time):
                lr_try = learning_rate * (attempt_lr_decay ** attempt)

                with torch.no_grad():
                    model.W.copy_(base_W)
                    enforce_incoming_constraints_(model, post_idx=post_idx, allowed_pre=prev_spikers)
                    if attempt > 0 and restart_noise_std > 0.0:
                        noise_scale = restart_noise_std * (0.5 ** (attempt - 1))
                        model.W[prev_spikers, post_idx] += (
                            noise_scale
                            * prev_spiker_learning_scales.to(device=model.W.device, dtype=model.W.dtype)
                            * torch.randn(
                                (prev_spikers.numel(),),
                                device=model.W.device,
                                dtype=model.W.dtype,
                            )
                        )
                        enforce_incoming_constraints_(model, post_idx=post_idx, allowed_pre=prev_spikers)

                optimizer = torch.optim.Adam([model.W], lr=lr_try)

                for inner_it in range(max_neuron_train_iters):
                    optimizer.zero_grad(set_to_none=True)

                    stage_soft = simulate_prefix_soft(
                        model,
                        prefix_bins=tb + 1,
                        stim_step=t_stim,
                        starter_neurons=starter_neurons,
                        stim_value_nA=stim_value_nA,
                        dt=dt,
                        bin_dt_eff=bin_dt_eff,
                        init_delay_bins=init_delay_bins,
                        bin_steps=bin_steps,
                    )

                    curr_loss = torch.abs(
                        stage_soft[tb, post_idx] - target_prefix[tb, post_idx].to(dtype=stage_soft.dtype)
                    )

                    if tb > 0:
                        prev_step_soft = soft_hamming_loss_per_timestep(stage_soft[:tb], target_prefix[:tb])
                        prev_regression = torch.relu(prev_step_soft - base_prev_step_soft).sum() / max(1, tb)
                    else:
                        prev_regression = torch.zeros((), device=model.W.device, dtype=model.W.dtype)

                    #loss_reg = l1_lambda * model.W[:, post_idx].abs().sum()
                    loss = curr_loss * 2.0 + previous_soft_barrier_scale * prev_regression # + loss_reg)

                    loss.backward()

                    if model.W.grad is None:
                        raise RuntimeError("model.W.grad is None after backward(); cannot inspect or apply masked update.")

                    raw_grad_col = model.W.grad[:, post_idx].detach().clone()
                    raw_grad_allowed = raw_grad_col[prev_spikers]
                    raw_grad_stats = grad_tensor_stats(raw_grad_allowed)
                    weight_allowed_before_step = model.W.detach()[prev_spikers, post_idx]
                    weight_stats_before_step = grad_tensor_stats(weight_allowed_before_step)

                    with torch.no_grad():
                        model.W.grad.mul_(train_mask)

                    masked_grad_col = model.W.grad[:, post_idx].detach().clone()
                    masked_grad_allowed = masked_grad_col[prev_spikers]
                    masked_grad_stats = grad_tensor_stats(masked_grad_allowed)
                    total_grad_stats_preclip = grad_tensor_stats(model.W.grad.detach())
                    preclip_total_grad_norm = total_grad_stats_preclip["norm"]

                    nonfinite_grad = (
                        (not raw_grad_stats["finite"])
                        or (not masked_grad_stats["finite"])
                        or (not total_grad_stats_preclip["finite"])
                    )
                    vanishing_grad = (
                        masked_grad_stats["numel"] > 0
                        and masked_grad_stats["max_abs"] < grad_vanish_threshold
                    )
                    exploding_grad = (
                        masked_grad_stats["max_abs"] > grad_explode_threshold
                        or preclip_total_grad_norm > grad_explode_threshold
                    )
                    large_weight = weight_stats_before_step["max_abs"] > weight_large_threshold
                    suspect_grad = nonfinite_grad or vanishing_grad or exploding_grad or large_weight

                    if should_print_gradient_diagnostics(
                        enable=enable_gradient_diagnostics,
                        tb=tb,
                        inner_it=inner_it,
                        suspect=suspect_grad,
                        print_every=gradient_diag_print_every_inner_iters,
                        start_bin=gradient_diag_start_bin,
                        only_when_suspect=gradient_diag_only_when_suspect,
                    ):
                        recency_min = float(prev_spiker_learning_scales.min().item()) if prev_spiker_learning_scales.numel() > 0 else 0.0
                        recency_max = float(prev_spiker_learning_scales.max().item()) if prev_spiker_learning_scales.numel() > 0 else 0.0
                        print(
                            f"[grad-debug] sample={s_idx} bin={tb}/{B - 1} post={post_idx} "
                            f"attempt={attempt + 1}/{max_attempts_per_neuron_time} iter={inner_it + 1}/{max_neuron_train_iters} "
                            f"curr_loss={float(curr_loss.detach().item()):.6e} "
                            f"prev_reg={float(prev_regression.detach().item()):.6e} "
                            f"total_loss={float(loss.detach().item()):.6e} "
                            f"allowed_pre={prev_spikers.numel()} recency_scale=[{recency_min:.3f},{recency_max:.3f}] "
                            f"raw_allowed_grad_norm={raw_grad_stats['norm']:.6e} "
                            f"raw_allowed_grad_maxabs={raw_grad_stats['max_abs']:.6e} "
                            f"masked_allowed_grad_norm={masked_grad_stats['norm']:.6e} "
                            f"masked_allowed_grad_maxabs={masked_grad_stats['max_abs']:.6e} "
                            f"masked_allowed_grad_nonzero={masked_grad_stats['nonzero']} "
                            f"total_grad_norm_preclip={preclip_total_grad_norm:.6e} "
                            f"w_allowed_maxabs={weight_stats_before_step['max_abs']:.6e} "
                            f"flags=(nonfinite={nonfinite_grad},vanishing={vanishing_grad},exploding={exploding_grad},large_weight={large_weight})"
                        )

                    preclip_norm_return = torch.nn.utils.clip_grad_norm_([model.W], max_norm=grad_clip_norm)
                    postclip_total_grad_norm = grad_tensor_stats(model.W.grad.detach())["norm"]

                    if should_print_gradient_diagnostics(
                        enable=enable_gradient_diagnostics,
                        tb=tb,
                        inner_it=inner_it,
                        suspect=suspect_grad,
                        print_every=gradient_diag_print_every_inner_iters,
                        start_bin=gradient_diag_start_bin,
                        only_when_suspect=gradient_diag_only_when_suspect,
                    ):
                        print(
                            f"[grad-debug] sample={s_idx} bin={tb}/{B - 1} post={post_idx} "
                            f"attempt={attempt + 1}/{max_attempts_per_neuron_time} iter={inner_it + 1}/{max_neuron_train_iters} "
                            f"clip_return_norm={float(preclip_norm_return):.6e} "
                            f"total_grad_norm_postclip={postclip_total_grad_norm:.6e}"
                        )

                    optimizer.step()
                    enforce_incoming_constraints_(model, post_idx=post_idx, allowed_pre=prev_spikers)

                    if float(curr_loss.detach().item()) < 1e-5 and float(prev_regression.detach().item()) == 0.0:
                        break

                cand_stage_hard = simulate_prefix_hard(
                    model,
                    prefix_bins=tb + 1,
                    stim_step=t_stim,
                    starter_neurons=starter_neurons,
                    stim_value_nA=stim_value_nA,
                    dt=dt,
                    bin_dt_eff=bin_dt_eff,
                    init_delay_bins=init_delay_bins,
                    bin_steps=bin_steps,
                )
                cand_stage_soft = simulate_prefix_soft(
                    model,
                    prefix_bins=tb + 1,
                    stim_step=t_stim,
                    starter_neurons=starter_neurons,
                    stim_value_nA=stim_value_nA,
                    dt=dt,
                    bin_dt_eff=bin_dt_eff,
                    init_delay_bins=init_delay_bins,
                    bin_steps=bin_steps,
                ).detach()

                cand_prev_step_hard = hamming_loss_per_timestep(cand_stage_hard[:tb], target_prefix[:tb])
                cand_curr_hard = scalar_hard_error(cand_stage_hard[tb, post_idx], target_prefix[tb, post_idx])
                cand_curr_soft = scalar_soft_error(cand_stage_soft[tb, post_idx], target_prefix[tb, post_idx])

                no_previous_regression = bool(torch.all(cand_prev_step_hard <= base_prev_step_hard).item())
                improved_current = (
                    cand_curr_hard < base_curr_hard
                    or (
                        cand_curr_hard == base_curr_hard
                        and cand_curr_soft + current_soft_improve_tol < base_curr_soft
                    )
                )

                if no_previous_regression and improved_current:
                    accepted = True
                    accepted_updates += 1
                    break

            if not accepted:
                with torch.no_grad():
                    model.W.copy_(base_W)
                    model.clamp_parameters_()
                rejected_updates += 1

        prefix_hard = simulate_prefix_hard(
            model,
            prefix_bins=tb + 1,
            stim_step=t_stim,
            starter_neurons=starter_neurons,
            stim_value_nA=stim_value_nA,
            dt=dt,
            bin_dt_eff=bin_dt_eff,
            init_delay_bins=init_delay_bins,
            bin_steps=bin_steps,
        )
        prefix_step_hamming = hamming_loss_per_timestep(prefix_hard, target_prefix)
        prefix_total_hamming = int(prefix_step_hamming.sum().item())

        if verbose_sample_progress:
            print(
                f"Sample {s_idx} finished bin={tb}/{B - 1}, "
                f"prefix_total_hamming={prefix_total_hamming}, "
                f"accepted_updates={accepted_updates}, rejected_updates={rejected_updates}, skipped={skipped_updates}"
            )

    W_sample = model.W.detach().clone().cpu()
    W_sample.fill_diagonal_(0.0)

    plot_payload = None
    if do_plot:
        total_bins_sim = init_delay_bins + B
        steps = total_bins_sim * bin_steps

        with torch.no_grad():
            spikes_eval = model(
                steps=steps,
                stim_step=t_stim,
                stim_neurons=starter_neurons,
                stim_value_nA=float(stim_value_nA),
                hard=True,
            )
            binned_spikes01, _, _ = bin_spikes_hard(
                spikes_eval,
                dt=dt,
                bin_dt=bin_dt_eff,
                drop_remainder=True,
            )

        model_stage = binned_spikes01[-B:].detach().cpu().numpy()
        target_stage_cpu = target[:B].detach().cpu().numpy()
        plot_payload = {
            "model_stage": model_stage,
            "target_stage": target_stage_cpu,
            "bin_dt_eff": bin_dt_eff,
            "N": N,
        }

    final_stage_hard = simulate_prefix_hard(
        model,
        prefix_bins=B,
        stim_step=t_stim,
        starter_neurons=starter_neurons,
        stim_value_nA=stim_value_nA,
        dt=dt,
        bin_dt_eff=bin_dt_eff,
        init_delay_bins=init_delay_bins,
        bin_steps=bin_steps,
    )
    final_total_hamming = int(hamming_loss_per_timestep(final_stage_hard, target).sum().item())

    return {
        "sample_idx": s_idx,
        "W_sample": W_sample.numpy().astype(np.float32, copy=False),
        "accepted_updates": accepted_updates,
        "rejected_updates": rejected_updates,
        "skipped_updates": skipped_updates,
        "starter_bin_hamming": prefix0_loss,
        "final_total_hamming": final_total_hamming,
        "plot_payload": plot_payload,
    }


def plot_overlay_from_payload(sample_idx: int, payload: dict):
    """Plot target and model spike rasters for one trained burst sample."""
    if payload is None:
        return

    model_stage = torch.as_tensor(payload["model_stage"])
    target_stage_cpu = torch.as_tensor(payload["target_stage"])
    bin_dt_eff = float(payload["bin_dt_eff"])
    N = int(payload["N"])

    tt_idx, tn_idx = torch.nonzero(target_stage_cpu > 0, as_tuple=True)
    tt = (tt_idx.to(torch.float64) * bin_dt_eff).numpy()
    tn = tn_idx.numpy()

    mt_idx, mn_idx = torch.nonzero(model_stage > 0, as_tuple=True)
    mt = (mt_idx.to(torch.float64) * bin_dt_eff).numpy()
    mn = mn_idx.numpy()

    plt.figure(figsize=(10, 4))
    plt.scatter(tt, tn, s=12, marker="|", alpha=0.9, label="Target")
    plt.scatter(mt, mn, s=12, marker="|", alpha=0.6, label="Model")
    plt.xlabel("Time (s) [binned, aligned]")
    plt.ylabel("Neuron index")
    plt.title(f"Overlay raster: Target vs Model (sample {sample_idx})")
    plt.ylim(-1, N)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_one_dataset(data_file_path: str | Path):
    """Train independent activity-matching SNNs for every burst in one target dataset."""
    burst_samples, burst_meta = load_burst_samples(
        data_file_path=data_file_path,
        data_dtres=DATA_BIN_WIDTH_SECONDS,
    )

    first_sample = burst_samples[0]
    N = int(first_sample.shape[1])
    W_base = make_random_W(
        N=N,
        p=initial_connection_probability,
        G=initial_weight_scale,
        seed=train_seed,
        device=sim_device,
        dtype=dtype,
    )

    sample_iter = build_sample_iter(
        burst_samples,
        selected_sample_idx=selected_sample_idx,
        max_samples_to_use=max_samples_to_use,
    )
    if len(sample_iter) == 0:
        raise ValueError("No samples selected for training.")

    task_list = []
    for s_idx, sample in sample_iter:
        do_plot = plot_overlay_for_each_sample or (s_idx in plot_overlay_for_sample_indices)
        task_list.append(
            {
                "sample_idx": s_idx,
                "sample_array": torch.as_tensor(sample, device="cpu").numpy(),
                "W_base": W_base.detach().cpu().numpy(),
                "N": N,
                "dt": dt,
                "bin_dt_eff": bin_dt_eff,
                "bin_steps": bin_steps,
                "init_delay_bins": init_delay_bins,
                "t_stim": t_stim,
                "stim_value_nA": stim_value_nA,
                "surrogate_beta": surrogate_beta,
                "train_seed": train_seed,
                "max_neuron_train_iters": max_neuron_train_iters,
                "max_attempts_per_neuron_time": max_attempts_per_neuron_time,
                "learning_rate": learning_rate,
                "attempt_lr_decay": attempt_lr_decay,
                "restart_noise_std": restart_noise_std,
                "grad_clip_norm": grad_clip_norm,
                "l1_lambda": l1_lambda,
                "previous_soft_barrier_scale": previous_soft_barrier_scale,
                "current_soft_improve_tol": current_soft_improve_tol,
                "presynaptic_history_window_bins": presynaptic_history_window_bins,
                "enable_gradient_diagnostics": enable_gradient_diagnostics,
                "gradient_diag_print_every_inner_iters": gradient_diag_print_every_inner_iters,
                "gradient_diag_only_when_suspect": gradient_diag_only_when_suspect,
                "gradient_diag_start_bin": gradient_diag_start_bin,
                "grad_vanish_threshold": grad_vanish_threshold,
                "grad_explode_threshold": grad_explode_threshold,
                "weight_large_threshold": weight_large_threshold,
                "verbose_sample_progress": verbose_sample_progress,
                "do_plot": do_plot,
                "worker_torch_threads": worker_torch_threads,
            }
        )

    num_workers = min(max_parallel_workers, len(task_list)) if parallel_train_samples else 1
    num_workers = max(1, num_workers)

    print(
        f"Training {len(task_list)} samples independently for {Path(data_file_path).name} "
        f"with {'parallel' if num_workers > 1 else 'serial'} execution "
        f"(workers={num_workers}, worker_torch_threads={worker_torch_threads})."
    )

    results = []
    if num_workers == 1:
        for task in task_list:
            result = train_single_sample(task)
            results.append(result)
            print(
                f"Completed sample {result['sample_idx']} "
                f"(starter_bin_hamming={result['starter_bin_hamming']}, "
                f"accepted={result['accepted_updates']}, rejected={result['rejected_updates']}, skipped={result['skipped_updates']})."
            )
    else:
        mp_ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
            future_to_idx = {executor.submit(train_single_sample, task): task["sample_idx"] for task in task_list}
            for future in cf.as_completed(future_to_idx):
                result = future.result()
                results.append(result)
                print(
                    f"Completed sample {result['sample_idx']} "
                    f"(starter_bin_hamming={result['starter_bin_hamming']}, "
                    f"accepted={result['accepted_updates']}, rejected={result['rejected_updates']}, skipped={result['skipped_updates']})."
                )

    results.sort(key=lambda x: x["sample_idx"])
    sample_connectivity_matrices = []
    for result in results:
        W_sample = torch.as_tensor(result["W_sample"], dtype=torch.float32, device="cpu")
        W_sample.fill_diagonal_(0.0)
        sample_connectivity_matrices.append(W_sample)
        print(
            f"Saving sample {result['sample_idx']} "
            f"(final_total_hamming={result['final_total_hamming']})."
        )
        if result["plot_payload"] is not None:
            plot_overlay_from_payload(result["sample_idx"], result["plot_payload"])

    print(f"\nTrained on and saved {len(sample_connectivity_matrices)} samples from {Path(data_file_path).name}.")

    if len(sample_connectivity_matrices) == 1:
        plot_connectivity_matrix_and_strengths(
            sample_connectivity_matrices[0],
            title_prefix=f"Connectivity (independently trained sample, {Path(data_file_path).stem})",
        )
    elif len(sample_connectivity_matrices) > 1:
        print(
            "Learned one independent connectivity matrix per sample; "
            "skipping a final single-matrix plot because there is no shared global matrix."
        )

    save_dir = PROJECT_ROOT / "LIFoutput_files"
    output_npy_path = make_output_npy_path(save_dir, data_file_path)
    hamming_loss_npy_path = make_hamming_loss_output_npy_path(save_dir, data_file_path)
    save_sample_connectivity_npy(
        sample_connectivity_matrices,
        path=str(output_npy_path),
    )
    save_final_hamming_losses_npy(
        results,
        path=str(hamming_loss_npy_path),
    )


def parse_args():
    """Build the command-line interface for the activity-matching experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the SNN on all target dataset text files inside the requested reference folder. "
            "The folder is resolved relative to the project fdata/ directory unless you provide an absolute path "
            "or a path starting with fdata/."
        )
    )
    parser.add_argument(
        "reference_target_filename",
        help=(
            "Reference dataset folder name, for example 'N100_p24_CC03'. "
            "You can also pass an absolute path or a path starting with 'fdata/'."
        ),
    )
    parser.add_argument(
        "--skip-trained-samples",
        action="store_true",
        help=(
            "Skip each target dataset whose normalized learned connectivity matrix .npy "
            "already exists in LIFoutput_files. The normalized output name omits any "
            "legacy '_inhOff' suffix. This takes precedence over the module default."
        ),
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help=(
            "Train every target dataset even if its learned connectivity matrix .npy "
            "already exists. This takes precedence over the module default."
        ),
    )
    return parser.parse_args()


def main(reference_target_filename: str, *, skip_trained_samples: bool | None = None):
    """Run activity matching for every target dataset in one statistical-class folder."""
    configure_global_reproducibility(RANDOM_SEED)

    if skip_trained_samples is None:
        skip_trained_samples = enable_skip_trained_samples

    dataset_files = find_target_datasets_in_reference_folder(
        input_path=str(PROJECT_ROOT),
        reference_folder=reference_target_filename,
    )

    print(
        f"Found {len(dataset_files)} target dataset(s) in reference folder "
        f"'{reference_target_filename}':"
    )
    for dataset_file in dataset_files:
        print(f"  - {dataset_file.name}")

    save_dir = PROJECT_ROOT / "LIFoutput_files"
    skipped_dataset_count = 0
    trained_dataset_count = 0

    for dataset_file in dataset_files:
        output_npy_path = Path(make_output_npy_path(save_dir, dataset_file))
        existing_output_paths = existing_learned_connectivity_output_paths(save_dir, dataset_file)

        if skip_trained_samples and existing_output_paths:
            skipped_dataset_count += 1
            print(f"\n{'=' * 80}")
            print(f"Skipping dataset because learned connectivity output already exists: {dataset_file.name}")
            for existing_output_path in existing_output_paths:
                print(f"Existing output: {existing_output_path}")
            print(f"{'=' * 80}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Starting training for dataset: {dataset_file.name}")
        if skip_trained_samples:
            print(f"No existing learned connectivity output found at: {output_npy_path}")
        print(f"{'=' * 80}")
        train_one_dataset(dataset_file)
        trained_dataset_count += 1

    print(
        f"\nDataset loop complete: trained={trained_dataset_count}, "
        f"skipped_existing_outputs={skipped_dataset_count}, total={len(dataset_files)}."
    )


if __name__ == "__main__":
    args = parse_args()
    if args.skip_trained_samples and args.force_retrain:
        raise ValueError("Use either --skip-trained-samples or --force-retrain, not both.")

    skip_override = None
    if args.skip_trained_samples:
        skip_override = True
    elif args.force_retrain:
        skip_override = False

    main(args.reference_target_filename, skip_trained_samples=skip_override)
