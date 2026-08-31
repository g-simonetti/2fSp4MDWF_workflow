#!/usr/bin/env python3
"""
Residual-mass-specific tau_int helper using Wolff's derived-observable method.

Given matched numerator and denominator histories at a fixed timeslice t, this
constructs the projected observable for

    F = <numerator> / <denominator>

and applies the usual Wolff Gamma-method analysis to that projected history.

Outputs in `out_dir`:
  - projected_mres_series.txt
  - tau_int_mres_results.json
  - tau_int_mres_vs_W.pdf
  - tau_int_mres_vs_n_therm.pdf
  - tau_int_mres_observable.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from autocorr_time.tau_int import (
        apply_plot_styles,
        compute_tau_from_series,
        resolve_therm_start_index,
    )
except Exception as e:
    raise ImportError(
        "Failed to import helpers from src/autocorr_time/tau_int.py.\n"
        "Expected repo layout with 'src/autocorr_time/tau_int.py'."
    ) from e


def build_projected_mres_series(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
    traj_numbers: np.ndarray,
    therm: int,
):
    numerator_values = np.asarray(numerator_values, dtype=float)
    denominator_values = np.asarray(denominator_values, dtype=float)
    traj_numbers = np.asarray(traj_numbers, dtype=float)

    if numerator_values.ndim != 1 or denominator_values.ndim != 1:
        raise ValueError("Expected 1D numerator and denominator histories.")
    if numerator_values.size != denominator_values.size or numerator_values.size != traj_numbers.size:
        raise ValueError("Numerator, denominator, and trajectory histories must have matching lengths.")

    therm_start_idx = resolve_therm_start_index(traj_numbers, int(max(0, therm)))
    num_used = numerator_values[therm_start_idx:]
    den_used = denominator_values[therm_start_idx:]
    if num_used.size < 2 or den_used.size < 2:
        raise ValueError("Need at least 2 post-thermalized points for projected mres tau_int.")

    num_mean = float(np.mean(num_used))
    den_mean = float(np.mean(den_used))
    if not np.isfinite(den_mean) or den_mean == 0.0:
        raise ZeroDivisionError("Post-thermalized denominator mean is zero or non-finite.")

    ratio_mean = num_mean / den_mean
    # Wolff projection for the derived ratio F = A1 / A2:
    #   f1 = dF/dA1 = 1 / <A2>
    #   f2 = dF/dA2 = -<A1> / <A2>^2 = -F / <A2>
    # and the projected primary history is X_i = f1 A1_i + f2 A2_i.
    f1 = 1.0 / den_mean
    f2 = -num_mean / (den_mean * den_mean)
    projected = f1 * numerator_values + f2 * denominator_values

    return projected.astype(float), ratio_mean, num_mean, den_mean, f1, f2, therm_start_idx


def summarize_component_tau(values: np.ndarray, traj_numbers: np.ndarray, therm: int, label: str, t: int | None):
    therm_start_idx = resolve_therm_start_index(traj_numbers, int(max(0, therm)))
    used = np.asarray(values[therm_start_idx:], dtype=float)
    tau_est, tau_err, nb_est, nbs_est, found, *_ = compute_tau_from_series(
        used,
        window_factor_s=1.5,
        max_window_fraction=0.5,
        apply_bias_correction=True,
    )
    payload: dict[str, Any] = {
        "observable_label": str(label),
        "tau_int": None if not np.isfinite(tau_est) else float(tau_est),
        "tau_int_err": None if not np.isfinite(tau_err) else float(tau_err),
        "Nb_est": None if nb_est is None else int(nb_est),
        "Nbs_est": None if nbs_est is None else int(nbs_est),
        "found": bool(found),
    }
    if t is not None:
        payload["t"] = int(t)
    return payload


def write_projected_series_file(traj_numbers: np.ndarray, projected_values: np.ndarray, out_path: str):
    with open(out_path, "w") as f:
        f.write("# traj_number\tprojected_mres\n")
        for n, y in zip(traj_numbers, projected_values):
            f.write(f"{int(n)}\t{float(y):.16e}\n")


def scan_wolff_vs_n_therm(
    x_full: np.ndarray,
    y_full: np.ndarray,
    therm: int,
    scan_step: int = 1,
    scan_min_points: int = 40,
):
    x_full = np.asarray(x_full, dtype=float)
    y_full = np.asarray(y_full, dtype=float)
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
        }

    max_therm = max(0, n - scan_min_points)
    points: list[dict[str, Any]] = []
    for n_therm in range(0, max_therm + 1, scan_step):
        y = y_full[n_therm:]
        if y.size < scan_min_points:
            continue
        x_display = float(x_full[n_therm]) if n_therm < x_full.size else float(n_therm)
        tau_est, tau_err, *_ = compute_tau_from_series(
            y,
            window_factor_s=1.5,
            max_window_fraction=0.5,
            apply_bias_correction=True,
        )
        if np.isfinite(tau_est) and np.isfinite(tau_err):
            points.append(
                {
                    "n_therm": int(n_therm),
                    "x_display": float(x_display),
                    "tau_int": float(tau_est),
                    "err": float(tau_err),
                }
            )

    if len(points) < 2:
        return {
            "skipped": True,
            "reason": "not_enough_valid_scan_points",
            "therm": int(therm),
            "therm_display": float(max(0, therm)),
            "scan_step": int(scan_step),
            "scan_min_points": int(scan_min_points),
            "points": [],
        }

    return {
        "skipped": False,
        "therm": int(therm),
        "therm_display": float(max(0, therm)),
        "scan_step": int(scan_step),
        "scan_min_points": int(scan_min_points),
        "points": points,
    }


def scan_mres_vs_n_therm(
    x_full: np.ndarray,
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
    therm: int,
    scan_step: int = 1,
):
    x_full = np.asarray(x_full, dtype=float)
    numerator_values = np.asarray(numerator_values, dtype=float)
    denominator_values = np.asarray(denominator_values, dtype=float)
    n = numerator_values.size

    if n < 2:
        return {
            "skipped": True,
            "reason": "not_enough_points",
            "therm": int(therm),
            "therm_display": float(max(0, therm)),
            "scan_step": int(scan_step),
            "max_n_therm_index": 0,
            "points": [],
        }

    max_therm = n // 2
    points: list[dict[str, Any]] = []
    for n_therm in range(0, max_therm + 1, scan_step):
        num = numerator_values[n_therm:]
        den = denominator_values[n_therm:]
        if num.size < 2 or den.size < 2:
            continue
        den_mean = float(np.mean(den))
        if not np.isfinite(den_mean) or den_mean == 0.0:
            continue
        num_mean = float(np.mean(num))
        ratio_mean = num_mean / den_mean
        x_display = float(x_full[n_therm]) if n_therm < x_full.size else float(n_therm)
        points.append(
            {
                "n_therm": int(n_therm),
                "x_display": float(x_display),
                "mres": float(ratio_mean),
            }
        )

    if len(points) < 2:
        return {
            "skipped": True,
            "reason": "not_enough_valid_scan_points",
            "therm": int(therm),
            "therm_display": float(max(0, therm)),
            "scan_step": int(scan_step),
            "max_n_therm_index": int(max_therm),
            "points": [],
        }

    return {
        "skipped": False,
        "therm": int(therm),
        "therm_display": float(max(0, therm)),
        "scan_step": int(scan_step),
        "max_n_therm_index": int(max_therm),
        "points": points,
    }


def plot_projected_observable(
    x_used: np.ndarray,
    y_used: np.ndarray,
    therm: int,
    plot_path: str,
):
    fig = plt.figure(figsize=(4.4, 2.8), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.15], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax)

    ax.plot(x_used, y_used, linewidth=0.8, color="C0")
    if np.isfinite(float(therm)):
        ax.axvline(therm, linewidth=1.0, linestyle="--", color="0.35")

    ax.set_xlabel("Configuration / trajectory")
    ax.set_ylabel("Projected $m_{\\mathrm{res}}$")

    y_finite = y_used[np.isfinite(y_used)]
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


def plot_tau_vs_w(
    w_list: list[int],
    tau_list: list[float],
    err_list: list[float],
    w_est: int | None,
    tau_est: float,
    plot_path: str,
):
    fig, (ax, ax_zoom) = plt.subplots(2, 1, figsize=(3.9, 4.6), layout="constrained", sharey=False)

    if w_list and tau_list and err_list:
        for target_ax in (ax, ax_zoom):
            target_ax.errorbar(
                w_list,
                tau_list,
                yerr=err_list,
                fmt="o",
                markersize=2.8,
                markerfacecolor="none",
                capsize=2,
                elinewidth=0.8,
                linewidth=1.0,
                linestyle="-",
                color="C0",
            )

    if np.isfinite(tau_est):
        ax.axhline(tau_est, linewidth=1.0, linestyle="--", color="C0")
        ax_zoom.axhline(tau_est, linewidth=1.0, linestyle="--", color="C0")
    if np.isfinite(tau_est) and (w_est is not None):
        ax.plot([w_est], [tau_est], marker="o", markersize=4.0, linestyle="none", color="C0")
        ax_zoom.plot([w_est], [tau_est], marker="o", markersize=4.0, linestyle="none", color="C0")

    ax.set_xlabel(r"Window $W$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$")
    ax_zoom.set_xlabel(r"Window $W$ (zoom)")
    ax_zoom.set_ylabel(r"$\tau_{\mathrm{int}}$")
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_zoom.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_zoom.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    if w_list and tau_list:
        w_arr = np.asarray(w_list, dtype=float)
        tau_arr = np.asarray(tau_list, dtype=float)
        err_arr = np.asarray(err_list, dtype=float) if err_list else np.zeros_like(tau_arr)

        zoom_w_max = int(min(max(w_est * 2 if w_est is not None else 0, 25), w_arr[-1]))
        zoom_mask = w_arr <= float(max(zoom_w_max, 1))
        if np.count_nonzero(zoom_mask) >= 2:
            y_zoom = tau_arr[zoom_mask]
            yerr_zoom = err_arr[zoom_mask]
            y_min = float(np.nanmin(y_zoom - yerr_zoom))
            y_max = float(np.nanmax(y_zoom + yerr_zoom))
            y_pad = 0.08 * max(y_max - y_min, 1.0e-12)

            ax_zoom.set_xlim(0.0, float(zoom_w_max))
            ax_zoom.set_ylim(y_min - y_pad, y_max + y_pad)
            ax_zoom.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
        else:
            ax_zoom.set_visible(False)
    else:
        ax_zoom.set_visible(False)

    fig.savefig(plot_path)
    plt.close(fig)


def plot_tau_vs_n_therm(
    scan_payload: dict[str, Any],
    tau_est: float,
    plot_path: str,
):
    if scan_payload.get("skipped", True):
        return

    pts = scan_payload.get("points", [])
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(3.7, 2.6), layout="constrained")
    therms = np.asarray([p["x_display"] for p in pts], dtype=float)
    taus = [p["tau_int"] for p in pts]
    errs = [p["err"] for p in pts]
    ax.errorbar(
        therms,
        taus,
        yerr=errs,
        fmt="o",
        alpha=0.4,
        markersize=3.0,
        markerfacecolor="none",
        capsize=2,
        elinewidth=0.9,
        linestyle="-",
        linewidth=0.9,
        color="C0",
    )

    if np.isfinite(tau_est):
        ax.axhline(tau_est, linewidth=1.0, linestyle="--", color="C0")

    if therms.size > 0:
        ax.axvline(0.5 * float(np.max(therms)), linewidth=0.9, linestyle="-", color="0.8", zorder=0)

    therm_display = scan_payload.get("therm_display", None)
    if therm_display is not None and np.isfinite(therm_display):
        ax.axvline(therm_display, linewidth=1.0, linestyle="--", color="0.35")

    ax.set_xlabel(r"$n_{\mathrm{therm}}$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$")
    ax.set_xlim(left=0.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig(plot_path)
    plt.close(fig)


def plot_mres_vs_n_therm(
    scan_payload: dict[str, Any],
    selected_mres: float,
    plot_path: str,
):
    if scan_payload.get("skipped", True):
        return

    pts = scan_payload.get("points", [])
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(3.7, 2.6), layout="constrained")
    therms = np.asarray([p["x_display"] for p in pts], dtype=float)
    mres_vals = np.asarray([p["mres"] for p in pts], dtype=float)
    ax.plot(
        therms,
        mres_vals,
        marker="o",
        markersize=2.8,
        markerfacecolor="none",
        linewidth=0.9,
        linestyle="-",
        color="C0",
    )

    if np.isfinite(selected_mres):
        ax.axhline(selected_mres, linewidth=1.0, linestyle="--", color="C0")

    if therms.size > 0:
        ax.axvline(0.5 * float(np.max(therms)), linewidth=0.9, linestyle="-", color="0.8", zorder=0)

    therm_display = scan_payload.get("therm_display", None)
    if therm_display is not None and np.isfinite(therm_display):
        ax.axvline(therm_display, linewidth=1.0, linestyle="--", color="0.35")

    ax.set_xlabel(r"$n_{\mathrm{therm}}$")
    ax.set_ylabel(r"$\langle A\rangle / \langle B\rangle$")
    ax.set_xlim(left=0.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    fig.savefig(plot_path)
    plt.close(fig)


def compute_mres_tau_from_series(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
    traj_numbers: list[int] | np.ndarray,
    out_dir: str,
    therm: int,
    plot_styles: str | None = None,
    base_name: str = "tau_int_mres",
    numerator_label: str = "pj5q",
    denominator_label: str = "ptll",
    t: int | None = None,
    emit_json: bool = False,
):
    apply_plot_styles(plot_styles)

    numerator_values = np.asarray(numerator_values, dtype=float)
    denominator_values = np.asarray(denominator_values, dtype=float)
    traj_numbers = np.asarray(traj_numbers, dtype=float)

    projected_values, ratio_mean, num_mean, den_mean, f1, f2, therm_start_idx = build_projected_mres_series(
        numerator_values=numerator_values,
        denominator_values=denominator_values,
        traj_numbers=traj_numbers,
        therm=therm,
    )

    y_used = projected_values[therm_start_idx:]
    x_used = traj_numbers[therm_start_idx:]

    os.makedirs(out_dir, exist_ok=True)

    tau_est, tau_err, nb_est, nbs_est, found, used_index, running = compute_tau_from_series(
        y_used,
        window_factor_s=1.5,
        max_window_fraction=0.5,
        apply_bias_correction=True,
    )
    w_list, tau_w_list, err_w_list = running

    series_path = os.path.join(out_dir, "projected_mres_series.txt")
    json_path = os.path.join(out_dir, f"{base_name}_results.json")
    plot_w_path = os.path.join(out_dir, f"{base_name}_vs_W.pdf")
    plot_scan_path = os.path.join(out_dir, f"{base_name}_vs_n_therm.pdf")
    plot_obs_path = os.path.join(out_dir, f"{base_name}_observable.pdf")
    plot_mres_scan_path = os.path.join(out_dir, f"{base_name}_mres_vs_n_therm.pdf")

    write_projected_series_file(traj_numbers, projected_values, series_path)
    plot_projected_observable(x_used, y_used, therm, plot_obs_path)

    scan_payload = scan_wolff_vs_n_therm(
        x_full=traj_numbers,
        y_full=projected_values,
        therm=therm,
        scan_step=1,
        scan_min_points=40,
    )
    mres_scan_payload = scan_mres_vs_n_therm(
        x_full=traj_numbers,
        numerator_values=numerator_values,
        denominator_values=denominator_values,
        therm=therm,
        scan_step=1,
    )

    if w_list and tau_w_list:
        plot_tau_vs_w(w_list, tau_w_list, err_w_list, nb_est, tau_est, plot_w_path)
    plot_tau_vs_n_therm(scan_payload, tau_est, plot_scan_path)
    plot_mres_vs_n_therm(mres_scan_payload, ratio_mean, plot_mres_scan_path)

    component_num = summarize_component_tau(numerator_values, traj_numbers, therm, numerator_label, t)
    component_den = summarize_component_tau(denominator_values, traj_numbers, therm, denominator_label, t)

    if used_index is None:
        used_reason = None
    else:
        used_reason = "first_negative_g_window" if found else "fallback_last_window"

    results: dict[str, Any] = {
        "ok": bool(np.isfinite(tau_est) and np.isfinite(tau_err)),
        "input": {
            "out_dir": str(out_dir),
            "therm": int(max(0, therm)),
            "therm_start_index": int(therm_start_idx),
            "N_full": int(projected_values.size),
            "N_used": int(y_used.size),
            "numerator_label": str(numerator_label),
            "denominator_label": str(denominator_label),
            "trajectory_start": (None if traj_numbers.size == 0 else int(traj_numbers[0])),
            "trajectory_end": (None if traj_numbers.size == 0 else int(traj_numbers[-1])),
        },
        "method": {
            "name": "wolff_derived_ratio_method",
            "component_method": "wolff_gamma_method",
            "projection": "(A_i - (<A>/<B>) B_i) / <B>",
            "reference_means": "post_thermalized_subset",
            "window_factor_S": 1.5,
            "max_window_fraction": 0.5,
            "apply_bias_correction": True,
            "comparison_curve": None,
        },
        "estimate": {
            "tau_int": None if not np.isfinite(tau_est) else float(tau_est),
            "err": None if not np.isfinite(tau_err) else float(tau_err),
            "Nb": None if nb_est is None else int(nb_est),
            "Nbs": None if nbs_est is None else int(nbs_est),
            "found_plateau": bool(found),
            "used_index": None if used_index is None else int(used_index),
            "used_reason": used_reason,
            "source_component": "projected_mres",
        },
        "projected_observable": {
            "ratio_mean": float(ratio_mean),
            "numerator_mean": float(num_mean),
            "denominator_mean": float(den_mean),
            "f1": float(f1),
            "f2": float(f2),
            "series_file": str(series_path),
        },
        "components": {
            "numerator": component_num,
            "denominator": component_den,
        },
        "running_window_table": [
            {
                "W": int(w),
                "tau_int": float(tau),
                "err": float(err),
                "used": bool(used_index is not None and i == used_index),
            }
            for i, (w, tau, err) in enumerate(zip(w_list, tau_w_list, err_w_list))
        ],
        "n_therm_scan": scan_payload,
        "mres_n_therm_scan": mres_scan_payload,
        "plots": {
            "tau_vs_w_pdf": plot_w_path,
            "tau_vs_n_therm_pdf": (None if scan_payload.get("skipped", True) else plot_scan_path),
            "observable_pdf": plot_obs_path,
            "mres_vs_n_therm_pdf": (None if mres_scan_payload.get("skipped", True) else plot_mres_scan_path),
        },
        "outputs": {
            "results_json": json_path,
        },
    }
    if t is not None:
        results["input"]["t"] = int(t)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    if emit_json:
        print(json.dumps(results, indent=2, sort_keys=True))

    return results


def _main_cli():
    ap = argparse.ArgumentParser(
        description="Residual-mass tau_int helper using Wolff's derived-observable method."
    )
    ap.add_argument("numerator_file")
    ap.add_argument("denominator_file")
    ap.add_argument("out_dir")
    ap.add_argument("therm", type=int)
    ap.add_argument("--plot_styles", default=None)
    ap.add_argument("--base_name", default="tau_int_mres")
    ap.add_argument("--numerator_label", default="pj5q")
    ap.add_argument("--denominator_label", default="ptll")
    ap.add_argument("--t", type=int, default=None)
    args = ap.parse_args()

    def read_series(path: str):
        xs: list[int] = []
        ys: list[float] = []
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue
                xs.append(int(float(parts[0])))
                ys.append(float(parts[1]))
        return np.asarray(xs, dtype=int), np.asarray(ys, dtype=float)

    x_num, y_num = read_series(args.numerator_file)
    x_den, y_den = read_series(args.denominator_file)
    if x_num.size != x_den.size or not np.array_equal(x_num, x_den):
        raise ValueError("Numerator and denominator files must have the same matched trajectory numbers.")

    compute_mres_tau_from_series(
        numerator_values=y_num,
        denominator_values=y_den,
        traj_numbers=x_num,
        out_dir=args.out_dir,
        therm=args.therm,
        plot_styles=args.plot_styles,
        base_name=args.base_name,
        numerator_label=args.numerator_label,
        denominator_label=args.denominator_label,
        t=args.t,
        emit_json=True,
    )


if __name__ == "__main__":
    _main_cli()
