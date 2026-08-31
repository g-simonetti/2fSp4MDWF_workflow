#!/usr/bin/env python3
"""
Reusable integrated autocorrelation time (tau_int) via Wolff's Gamma method
 with automatic windowing, while retaining the Berg binning curve for plot
 comparison.

Reporting:
  - Final reported values use Wolff's Gamma method.
  - The tau-vs-scale plot compares:
      * Wolff Gamma-method running tau_int(W)
      * Berg binning tau_int(Nb) = tau_BERG(Nb) / 2

I/O behavior:
  - Produces ONE JSON output file:
        {out_dir}/{base_name}_results.json

  - Produces plots:
        {out_dir}/{base_name}_vs_Nb.pdf
        {out_dir}/{base_name}_vs_n_therm.pdf
        {out_dir}/{base_name}_observable.pdf

No .txt output files are produced.

Public API:
  - compute_tau_from_file(input_file, out_dir, therm, plot_styles=None, base_name="tau_int")

CLI:
  python3 tau_int.py <input_file> <out_dir> <therm> [--plot_styles ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------

def apply_plot_styles(plot_styles_arg: str | None):
    if not plot_styles_arg:
        return
    parts = [p.strip() for p in str(plot_styles_arg).split(",") if p.strip()]
    if parts:
        plt.style.use(parts)
    if plt.rcParams.get("text.usetex", False) and shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def read_series_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a series from a text file.
      - If line has 1 token: use a 1-based running index as x, token as y
      - If line has >=2 tokens: interpret as (x, y, ...) and use the FIRST token as x
        and SECOND token as y
    Ignores blank lines and lines starting with '#'.
    """
    xs: list[float] = []
    ys: list[float] = []
    running_index = 0
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            running_index += 1
            try:
                x = float(running_index) if len(parts) == 1 else float(parts[0])
                y = float(parts[0]) if len(parts) == 1 else float(parts[1])
            except ValueError:
                continue
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def read_series_file(path: str) -> np.ndarray:
    return read_series_data(path)[1]


def resolve_therm_start_index(x_full: np.ndarray, therm: int) -> int:
    """
    Convert the selected thermalization cutoff into the first retained sample.

    When the input file has an explicit first-column coordinate, interpret
    `therm` in that same coordinate system and keep samples with x >= therm.
    """
    x_full = np.asarray(x_full, dtype=float)
    x_full = x_full[np.isfinite(x_full)]
    if x_full.size == 0:
        return 0

    therm = int(max(0, therm))
    idx = np.searchsorted(x_full, float(therm), side="left")
    return int(min(max(idx, 0), x_full.size))


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _sample_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.var(x, ddof=1))


def _autocovariance_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.asarray([], dtype=float)

    x = x - np.mean(x)
    gamma0 = np.var(x)
    if not np.isfinite(gamma0) or gamma0 <= 0.0:
        return np.asarray([], dtype=float)

    n = x.size
    gamma = np.correlate(x, x, mode="full")[n - 1 :]
    gamma /= np.arange(n, 0, -1, dtype=float)
    return gamma


# -----------------------------------------------------------------------------
# Berg binning tau_int (comparison curve only)
# -----------------------------------------------------------------------------

def berg_binning_tau_series_berg(x: np.ndarray, min_nbs: int = 4):
    """
    For Nb=1..Nb_max where Nbs=floor(N/Nb) >= min_nbs compute:
      tau_BERG(Nb) = (Var(binmeans)/Nbs) / (Var(raw)/N)
      err_BERG ~ tau_BERG * sqrt(2/(Nbs-1))
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return [], [], [], []

    var_f = _sample_var(x)
    if not np.isfinite(var_f) or var_f <= 0:
        return [], [], [], []

    s2_f = var_f / n

    nb_list: list[int] = []
    nbs_list: list[int] = []
    tau_list: list[float] = []
    err_list: list[float] = []

    nb_max = n // int(min_nbs)

    for nb in range(1, nb_max + 1):
        nbs = n // nb
        if nbs < min_nbs:
            break

        trimmed = x[: nbs * nb]
        bin_means = trimmed.reshape(nbs, nb).mean(axis=1)

        var_bin = _sample_var(bin_means)
        if not np.isfinite(var_bin) or var_bin <= 0:
            continue

        s2_f_nb = var_bin / nbs
        tau_berg = s2_f_nb / s2_f
        err_berg = abs(tau_berg) * np.sqrt(2.0 / (nbs - 1))

        nb_list.append(int(nb))
        nbs_list.append(int(nbs))
        tau_list.append(float(tau_berg))
        err_list.append(float(err_berg))

    return nb_list, nbs_list, tau_list, err_list


def find_first_nb_exceeding_c_tau_berg(
    nb_list: list[int],
    tau_berg_list: list[float],
    c_plateau: float = 4.0,
):
    nb = np.asarray(nb_list, dtype=float)
    tau = np.asarray(tau_berg_list, dtype=float)
    if nb.size == 0 or tau.size == 0 or nb.size != tau.size:
        return None
    for k in range(1, nb.size):  # skip Nb=1
        if np.isfinite(nb[k]) and np.isfinite(tau[k]) and tau[k] > 0:
            if nb[k] > c_plateau * tau[k]:
                return int(k)
    return None


def compute_tau_from_series_berg(
    x: np.ndarray,
    min_nbs: int = 4,
    c_plateau: float = 4.0,
):
    """
    Returns:
      tau_est, tau_err, Nb_est, Nbs_est, found, used_index,
      (Nb_list, Nbs_list, tauR_list, errR_list)

    where tauR = tau_BERG / 2 and errR = err_BERG / 2.
    """
    nb_list, nbs_list, tau_b_list, err_b_list = berg_binning_tau_series_berg(x, min_nbs=min_nbs)
    if not tau_b_list:
        return np.nan, np.nan, None, None, False, None, ([], [], [], [])

    used_index = find_first_nb_exceeding_c_tau_berg(nb_list, tau_b_list, c_plateau=c_plateau)
    if used_index is None:
        used_index = len(tau_b_list) - 1
        found = False
    else:
        found = True

    tau_r_list = [0.5 * t for t in tau_b_list]
    err_r_list = [0.5 * e for e in err_b_list]

    tau_est = float(tau_r_list[used_index])
    tau_err = float(err_r_list[used_index])
    nb_est = int(nb_list[used_index])
    nbs_est = int(nbs_list[used_index])

    return tau_est, tau_err, nb_est, nbs_est, found, used_index, (nb_list, nbs_list, tau_r_list, err_r_list)


# -----------------------------------------------------------------------------
# Wolff Gamma-method tau_int
# -----------------------------------------------------------------------------

def wolff_gamma_running_tau_series(
    x: np.ndarray,
    window_factor_s: float = 1.5,
    max_window: int | None = None,
    apply_bias_correction: bool = True,
):
    """
    Running Wolff Gamma-method estimate for W = 1..W_max.

    Returns:
      W_list, tau_list, err_list, g_list
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return [], [], [], []

    gamma = _autocovariance_series(x)
    if gamma.size == 0 or not np.isfinite(gamma[0]) or gamma[0] <= 0.0:
        return [], [], [], []

    if max_window is None:
        max_window = n // 2
    max_window = max(1, min(int(max_window), n - 1))

    w_arr = np.arange(1, max_window + 1, dtype=float)

    cbar_raw = gamma[0] + 2.0 * np.cumsum(gamma[1 : max_window + 1])
    tau_raw = cbar_raw / (2.0 * gamma[0])

    correction = 1.0 + (2.0 * w_arr + 1.0) / n if apply_bias_correction else 1.0
    cbar_report = cbar_raw * correction
    tau_report = cbar_report / (2.0 * gamma[0])

    # Use the uncorrected running estimate for windowing and for the error model,
    # then report the corrected tau_int values.
    delta_term = np.maximum(w_arr + 0.5 - tau_raw, 0.0)
    err_arr = 2.0 * np.abs(tau_report) * np.sqrt(delta_term / n)

    tau_decay = np.full_like(tau_raw, 1.0e-12, dtype=float)
    valid = tau_raw > 0.5
    if np.any(valid):
        ratio = (2.0 * tau_raw[valid] + 1.0) / (2.0 * tau_raw[valid] - 1.0)
        tau_decay[valid] = window_factor_s / np.log(ratio)

    g_arr = np.exp(-w_arr / tau_decay) - tau_decay / np.sqrt(w_arr * n)

    return (
        w_arr.astype(int).tolist(),
        tau_report.astype(float).tolist(),
        err_arr.astype(float).tolist(),
        g_arr.astype(float).tolist(),
    )


def compute_tau_from_series(
    x: np.ndarray,
    window_factor_s: float = 1.5,
    max_window_fraction: float = 0.5,
    apply_bias_correction: bool = True,
):
    """
    Returns:
      tau_est, tau_err, W_est, Nbs_est, found, used_index,
      (W_list, tau_list, err_list)

    The second returned integer is kept as None for compatibility with callers
    that expect a 5-tuple from the older implementation.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan, np.nan, None, None, False, None, ([], [], [])

    max_window = max(1, min(n - 1, int(np.floor(max_window_fraction * n))))
    w_list, tau_list, err_list, g_list = wolff_gamma_running_tau_series(
        x,
        window_factor_s=window_factor_s,
        max_window=max_window,
        apply_bias_correction=apply_bias_correction,
    )
    if not tau_list:
        return np.nan, np.nan, None, None, False, None, ([], [], [])

    used_index = next((i for i, g in enumerate(g_list) if np.isfinite(g) and g < 0.0), None)
    if used_index is None:
        used_index = len(w_list) - 1
        found = False
    else:
        found = True

    tau_est = float(tau_list[used_index])
    tau_err = float(err_list[used_index])
    w_est = int(w_list[used_index])

    return tau_est, tau_err, w_est, None, found, used_index, (w_list, tau_list, err_list)


# -----------------------------------------------------------------------------
# Plot outputs + JSON payload assembly
# -----------------------------------------------------------------------------

def plot_tau_comparison(
    w_list: list[int],
    tau_w_list: list[float],
    err_w_list: list[float],
    w_est: int | None,
    tau_est: float,
    nb_list: list[int],
    tau_b_list: list[float],
    err_b_list: list[float],
    nb_est: int | None,
    tau_b_est: float,
    plot_path: str,
):
    fig, (ax_w, ax_b) = plt.subplots(2, 1, figsize=(4.2, 4.8), layout="constrained")

    if w_list and tau_w_list and err_w_list:
        ax_w.errorbar(
            w_list,
            tau_w_list,
            yerr=err_w_list,
            fmt="o",
            markersize=2.8,
            markerfacecolor="none",
            capsize=2,
            elinewidth=0.8,
            linewidth=1.0,
            linestyle="-",
            label="Wolff $\\Gamma$-method",
        )

    if np.isfinite(tau_est):
        ax_w.axhline(tau_est, linewidth=1.0, linestyle="--", color="C0", label="Wolff selected")
    if np.isfinite(tau_b_est):
        ax_w.axhline(tau_b_est, linewidth=1.0, linestyle=":", color="C1", label="Berg selected")
    if np.isfinite(tau_est) and (w_est is not None):
        ax_w.plot([w_est], [tau_est], marker="o", markersize=4.0, linestyle="none", color="C0")

    ax_w.set_xlabel(r"Window $W$")
    ax_w.set_ylabel(r"$\tau_{\mathrm{int}}$")

    if nb_list and tau_b_list and err_b_list:
        ax_b.errorbar(
            nb_list,
            tau_b_list,
            yerr=err_b_list,
            fmt="s",
            markersize=2.8,
            markerfacecolor="none",
            capsize=2,
            elinewidth=0.8,
            linewidth=0.9,
            linestyle="-",
            label="Berg binning",
        )

    if np.isfinite(tau_est):
        ax_b.axhline(tau_est, linewidth=1.0, linestyle="--", color="C0", label="Wolff selected")
    if np.isfinite(tau_b_est):
        ax_b.axhline(tau_b_est, linewidth=1.0, linestyle=":", color="C1", label="Berg selected")
    if np.isfinite(tau_b_est) and (nb_est is not None):
        ax_b.plot([nb_est], [tau_b_est], marker="s", markersize=4.0, linestyle="none", color="C1")

    ax_b.set_xlabel(r"Bin size $N_b$")
    ax_b.set_ylabel(r"$\tau_{\mathrm{int}}$")

    fmt = ScalarFormatter(useMathText=True)
    for ax in (ax_w, ax_b):
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.legend(frameon=False, fontsize=8)

    fig.savefig(plot_path)
    plt.close(fig)


def plot_observable_series(
    x_full: np.ndarray,
    y_full: np.ndarray,
    therm: int,
    plot_path: str,
):
    """
    Plot the observable in its natural measurement order.
    The x-axis follows the first column of the input file when present, i.e.
    the associated configuration / trajectory number.
    A vertical dashed line marks the thermalization cut.
    """
    fig = plt.figure(figsize=(4.4, 2.8), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.15], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax)

    x_full = np.asarray(x_full, dtype=float)
    y_full = np.asarray(y_full, dtype=float)
    ax.plot(x_full, y_full, linewidth=0.8)

    if np.isfinite(float(therm)):
        ax.axvline(therm, linewidth=1.0, linestyle="--")

    ax.set_xlabel("Configuration / trajectory")
    ax.set_ylabel("Observable")

    y_finite = y_full[np.isfinite(y_full)]
    if y_finite.size > 0:
        n_bins = min(30, max(10, int(np.sqrt(y_finite.size))))
        ax_hist.hist(
            y_finite,
            bins=n_bins,
            orientation="horizontal",
            color="C0",
            alpha=0.35,
            edgecolor="C0",
            linewidth=0.6,
        )

    ax_hist.set_xlabel("Count")
    ax_hist.tick_params(axis="y", labelleft=False)

    fmt = ScalarFormatter(useMathText=True)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax_hist.xaxis.set_major_formatter(fmt)

    fig.savefig(plot_path)
    plt.close(fig)


def scan_tau_vs_n_therm(
    x_full: np.ndarray,
    y_full: np.ndarray,
    therm: int,
    window_factor_s: float = 1.5,
    max_window_fraction: float = 0.5,
    scan_step: int = 10,
    scan_min_points: int = 40,
):
    """
    Scan n_therm = 0, scan_step, 2*scan_step, ...
    up to max_therm = N - scan_min_points.

    Returns JSON-friendly dict, does not plot.
    """
    y_full = np.asarray(y_full, dtype=float)
    y_full = y_full[np.isfinite(y_full)]
    x_full = np.asarray(x_full, dtype=float)
    x_full = x_full[np.isfinite(x_full)]
    n = y_full.size
    if n < scan_min_points:
        return {
            "skipped": True,
            "reason": "not_enough_points",
            "therm": int(therm),
            "therm_display": float(max(0, therm)),
            "scan_step": int(scan_step),
            "scan_min_points": int(scan_min_points),
            "points": [],
            "berg_points": [],
        }

    max_therm = max(0, n - scan_min_points)

    points: list[dict[str, Any]] = []
    berg_points: list[dict[str, Any]] = []
    for n_therm in range(0, max_therm + 1, scan_step):
        y = y_full[n_therm:]
        if y.size < scan_min_points:
            continue
        x_display = float(x_full[n_therm]) if n_therm < len(x_full) else float(n_therm)
        tau_est, tau_err, *_ = compute_tau_from_series(
            y,
            window_factor_s=window_factor_s,
            max_window_fraction=max_window_fraction,
            apply_bias_correction=True,
        )
        if np.isfinite(tau_est) and np.isfinite(tau_err):
            points.append({
                "n_therm": int(n_therm),
                "x_display": float(x_display),
                "tau_int": float(tau_est),
                "err": float(tau_err),
            })
        berg_tau, berg_err, *_ = compute_tau_from_series_berg(y, min_nbs=4, c_plateau=4.0)
        if np.isfinite(berg_tau) and np.isfinite(berg_err):
            berg_points.append({
                "n_therm": int(n_therm),
                "x_display": float(x_display),
                "tau_int": float(berg_tau),
                "err": float(berg_err),
            })

    if len(points) < 2:
        return {
            "skipped": True,
            "reason": "not_enough_valid_scan_points",
            "therm": int(therm),
            "therm_display": float(max(0, therm)),
            "scan_step": int(scan_step),
            "scan_min_points": int(scan_min_points),
            "points": [],
            "berg_points": [],
        }

    return {
        "skipped": False,
        "therm": int(therm),
        "therm_display": float(max(0, therm)),
        "scan_step": int(scan_step),
        "scan_min_points": int(scan_min_points),
        "points": points,
        "berg_points": berg_points,
    }


def build_tau_history(
    x_used: np.ndarray,
    y_used: np.ndarray,
    min_points: int = 40,
):
    x_used = np.asarray(x_used, dtype=float)
    y_used = np.asarray(y_used, dtype=float)
    y_used = y_used[np.isfinite(y_used)]
    x_used = x_used[: y_used.size]

    wolff_points: list[dict[str, Any]] = []
    berg_points: list[dict[str, Any]] = []

    if y_used.size < min_points:
        return {
            "skipped": True,
            "wolff_points": [],
            "berg_points": [],
        }

    for end in range(min_points, y_used.size + 1):
        x_end = float(x_used[end - 1]) if end - 1 < x_used.size else float(end - 1)
        prefix = y_used[:end]

        tau_w, err_w, *_ = compute_tau_from_series(
            prefix,
            window_factor_s=1.5,
            max_window_fraction=0.5,
            apply_bias_correction=True,
        )
        if np.isfinite(tau_w):
            wolff_points.append({
                "end_index": int(end - 1),
                "x_end": x_end,
                "tau_int": float(tau_w),
            })

        tau_b, err_b, *_ = compute_tau_from_series_berg(prefix, min_nbs=4, c_plateau=4.0)
        if np.isfinite(tau_b):
            berg_points.append({
                "end_index": int(end - 1),
                "x_end": x_end,
                "tau_int": float(tau_b),
            })

    return {
        "skipped": False,
        "wolff_points": wolff_points,
        "berg_points": berg_points,
    }


def plot_scan(
    scan_payload: dict[str, Any],
    wolff_tau_est: float,
    berg_tau_est: float,
    plot_path: str,
):
    if scan_payload.get("skipped", True):
        return

    pts_w = scan_payload.get("points", [])
    pts_b = scan_payload.get("berg_points", [])
    if not pts_w and not pts_b:
        return

    therm_display = scan_payload.get("therm_display", None)

    fig, ax = plt.subplots(figsize=(3.7, 2.6), layout="constrained")

    if pts_w:
        therms_w = np.asarray([p["x_display"] for p in pts_w], dtype=float)
        taus_w = [p["tau_int"] for p in pts_w]
        errs_w = [p["err"] for p in pts_w]
        ax.errorbar(
            therms_w,
            taus_w,
            yerr=errs_w,
            fmt="o",
            alpha=0.4,
            markersize=3.0,
            markerfacecolor="none",
            capsize=2,
            elinewidth=0.9,
            linestyle="-",
            linewidth=0.9,
            label="Wolff $\\Gamma$-method",
        )

    if pts_b:
        therms_b = np.asarray([p["x_display"] for p in pts_b], dtype=float)
        taus_b = [p["tau_int"] for p in pts_b]
        errs_b = [p["err"] for p in pts_b]
        ax.errorbar(
            therms_b,
            taus_b,
            yerr=errs_b,
            fmt="s",
            alpha=0.4,
            markersize=3.0,
            markerfacecolor="none",
            capsize=2,
            elinewidth=0.9,
            linestyle="-",
            linewidth=0.9,
            label="Berg binning",
        )

    if np.isfinite(wolff_tau_est):
        ax.axhline(wolff_tau_est, linewidth=1.0, linestyle="--", color="C0", label="Wolff selected")
    if np.isfinite(berg_tau_est):
        ax.axhline(berg_tau_est, linewidth=1.0, linestyle=":", color="C1", label="Berg selected")

    x_candidates: list[float] = []
    if pts_w:
        x_candidates.extend([float(p["x_display"]) for p in pts_w])
    if pts_b:
        x_candidates.extend([float(p["x_display"]) for p in pts_b])
    if x_candidates:
        x_mid = 0.5 * max(x_candidates)
        ax.axvline(x_mid, linewidth=0.9, linestyle="-", color="0.8", zorder=0)

    if therm_display is not None and np.isfinite(therm_display):
        ax.axvline(therm_display, linewidth=1.0, linestyle="--", color="0.35", label="chosen therm")

    ax.set_xlabel(r"$n_{\mathrm{therm}}$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$")
    ax.set_xlim(left=0.0)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    fmt = ScalarFormatter(useMathText=True)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(plot_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def compute_tau_from_file(
    input_file: str,
    out_dir: str,
    therm: int,
    plot_styles: str | None = None,
    base_name: str = "tau_int",
):
    """
    Main reusable entry point.

    Reads series from input_file, applies therm cut, writes outputs into out_dir:
      - {base_name}_results.json          (contains running-window table, estimate, scan)
      - {base_name}_vs_Nb.pdf             (Wolff vs Berg comparison)
      - {base_name}_vs_n_therm.pdf
      - {base_name}_observable.pdf

    Returns:
      tau_est, tau_err, W_est, None, found
    """
    apply_plot_styles(plot_styles)

    x_full, y_full = read_series_data(input_file)
    if y_full.size == 0:
        raise ValueError(f"No finite data read from {input_file}")

    therm = int(max(0, therm))
    therm_start_idx = resolve_therm_start_index(x_full, therm)
    y = y_full[therm_start_idx:]

    os.makedirs(out_dir, exist_ok=True)

    tau_est, tau_err, nb_est, nbs_est, found, used_index, (w_list, tau_w_list, err_w_list) = compute_tau_from_series(
        y,
        window_factor_s=1.5,
        max_window_fraction=0.5,
        apply_bias_correction=True,
    )

    tau_b_est, err_b_est, nb_b_est, nbs_b_est, found_b, berg_used_index, (nb_list, nbs_list, tau_b_list, err_b_list) = \
        compute_tau_from_series_berg(y, min_nbs=4, c_plateau=4.0)

    plot_nb_path = os.path.join(out_dir, f"{base_name}_vs_Nb.pdf")
    plot_scan_path = os.path.join(out_dir, f"{base_name}_vs_n_therm.pdf")
    plot_obs_path = os.path.join(out_dir, f"{base_name}_observable.pdf")
    json_path = os.path.join(out_dir, f"{base_name}_results.json")

    plot_observable_series(x_full[therm_start_idx:], y_full[therm_start_idx:], therm, plot_obs_path)

    # Keep the old JSON array shape. For compatibility, the Nb field now stores
    # Wolff's summation window W and Nbs is left null.
    binning_table: list[dict[str, Any]] = []
    for i, (w, tau, err) in enumerate(zip(w_list, tau_w_list, err_w_list)):
        binning_table.append({
            "Nb": int(w),
            "Nbs": None,
            "tau_int": float(tau),
            "err": float(err),
            "used": bool(used_index is not None and i == used_index),
        })

    if used_index is None:
        used_reason = None
    else:
        used_reason = "first_negative_g_window" if found else "fallback_last_window"

    scan_payload = scan_tau_vs_n_therm(
        x_full,
        y_full,
        therm=therm,
        window_factor_s=1.5,
        max_window_fraction=0.5,
        scan_step=1,
        scan_min_points=40,
    )
    if (w_list and tau_w_list) or (nb_list and tau_b_list):
        plot_tau_comparison(
            w_list,
            tau_w_list,
            err_w_list,
            nb_est,
            tau_est,
            nb_list,
            tau_b_list,
            err_b_list,
            nb_b_est,
            tau_b_est,
            plot_nb_path,
        )

    plot_scan(
        scan_payload,
        tau_est,
        tau_b_est,
        plot_scan_path if not scan_payload.get("skipped", True) else plot_scan_path,
    )

    results: dict[str, Any] = {
        "ok": bool(np.isfinite(tau_est) and np.isfinite(tau_err)),
        "input": {
            "input_file": input_file,
            "out_dir": out_dir,
            "therm": int(therm),
            "therm_start_index": int(therm_start_idx),
            "N_full": int(y_full.size),
            "N_used": int(y.size),
        },
        "method": {
            "name": "wolff_gamma_method",
            "window_factor_S": 1.5,
            "max_window_fraction": 0.5,
            "apply_bias_correction": True,
            "comparison_curve": "berg_binning_plot_only",
            "schema_note": "Compatibility mode: fields named Nb/Nbs retain their old names; Nb stores Wolff's summation window W and Nbs is null.",
        },
        "estimate": {
            "tau_int": None if not np.isfinite(tau_est) else float(tau_est),
            "err": None if not np.isfinite(tau_err) else float(tau_err),
            "Nb": None if nb_est is None else int(nb_est),
            "Nbs": None if nbs_est is None else int(nbs_est),
            "found_plateau": bool(found),
            "used_index": None if used_index is None else int(used_index),
            "used_reason": used_reason,
        },
        "binning_table": binning_table,
        "n_therm_scan": scan_payload,
        "plots": {
            "tau_vs_nb_pdf": plot_nb_path,
            "tau_vs_n_therm_pdf": (None if scan_payload.get("skipped", True) else plot_scan_path),
            "observable_pdf": plot_obs_path,
        },
        "outputs": {
            "results_json": json_path,
        },
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    return tau_est, tau_err, nb_est, nbs_est, found


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _main_cli():
    ap = argparse.ArgumentParser(
        description="Generic tau_int (Wolff Gamma-method with Berg comparison curve). JSON-only text output."
    )
    ap.add_argument("input_file")
    ap.add_argument("out_dir")
    ap.add_argument("therm", type=int)
    ap.add_argument("--plot_styles", default=None)
    ap.add_argument("--base_name", default="tau_int")
    args = ap.parse_args()

    compute_tau_from_file(
        input_file=args.input_file,
        out_dir=args.out_dir,
        therm=args.therm,
        plot_styles=args.plot_styles,
        base_name=args.base_name,
    )


if __name__ == "__main__":
    _main_cli()
