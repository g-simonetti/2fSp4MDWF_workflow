#!/usr/bin/env python3

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def _fold_correlator(data: np.ndarray) -> np.ndarray:
    """
    Fold correlator with periodic indexing:

        C_fold(t) = 1/2 * [ C(t) + C(Nt - t) ].
    """
    arr = np.asarray(data)
    t_extent = arr.shape[-1]
    idx = (-np.arange(t_extent)) % t_extent
    return 0.5 * (arr + arr[..., idx])


def _select_indices_by_delta(numbers_sorted: np.ndarray, delta_traj_conf: int) -> np.ndarray:
    """
    Mirror the measurement thinning used in the main residual-mass fit:
    after a thermal cut, keep start, start + delta, start + 2 delta, ...
    whenever those trajectory numbers are present.
    """
    numbers_sorted = np.asarray(numbers_sorted, dtype=int)
    if numbers_sorted.size == 0:
        return np.asarray([], dtype=int)

    delta_traj_conf = int(delta_traj_conf)
    if delta_traj_conf <= 0:
        return np.arange(numbers_sorted.size, dtype=int)

    first = int(numbers_sorted[0])
    last = int(numbers_sorted[-1])
    index_of = {int(n): i for i, n in enumerate(numbers_sorted)}

    keep: list[int] = []
    k = 0
    while True:
        n = first + k * delta_traj_conf
        if n in index_of:
            keep.append(index_of[n])
        if n > last:
            break
        k += 1

    return np.asarray(sorted(set(keep)), dtype=int)


def _bootstrap_ratio_of_means(
    data_num: np.ndarray,
    data_den: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
):
    """
    Bootstrap the folded ratio of ensemble means, matching the main fit script.
    """
    num = _fold_correlator(data_num)
    den = _fold_correlator(data_den)

    if num.shape != den.shape:
        raise ValueError(f"Shape mismatch: numerator {num.shape}, denominator {den.shape}")
    if num.ndim != 2:
        raise ValueError(f"Expected 2D arrays of shape (Ncfg, T), got {num.shape}")

    n_cfg, t_extent = num.shape
    if n_cfg < 2:
        raise ValueError(f"Need at least 2 configurations, got {n_cfg}")

    num_mean = num.mean(axis=0)
    den_mean = den.mean(axis=0)
    if np.any(den_mean == 0.0):
        raise ZeroDivisionError("Zero ensemble-mean denominator encountered in fitted mres scan.")

    ratio_mean = num_mean / den_mean

    ratios_boot = np.empty((int(n_boot), t_extent), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n_cfg, size=n_cfg)
        num_b = num[idx].mean(axis=0)
        den_b = den[idx].mean(axis=0)
        if np.any(den_b == 0.0):
            raise ZeroDivisionError("Zero bootstrap denominator encountered in fitted mres scan.")
        ratios_boot[b] = num_b / den_b

    return ratio_mean, ratios_boot


def _correlated_constant_fit(
    ratio_mean: np.ndarray,
    ratios_boot: np.ndarray,
    plateau_mask: np.ndarray,
):
    """
    Correlated constant fit using the bootstrap covariance, matching the main fit.
    """
    y = np.asarray(ratio_mean[plateau_mask], dtype=np.float64)
    y_boot = np.asarray(ratios_boot[:, plateau_mask], dtype=np.float64)

    if y_boot.ndim != 2:
        raise ValueError(f"Expected bootstrap array of shape (Nb, np), got {y_boot.shape}")

    n_boot, n_points = y_boot.shape
    if n_boot < 2:
        raise ValueError("Need at least 2 bootstrap replicas to estimate covariance.")
    if n_points < 1:
        raise ValueError("Plateau window is empty.")

    if n_points == 1:
        fit_value = float(y[0])
        fit_err = float(np.sqrt(np.cov(y_boot[:, 0], ddof=1)))
        return fit_value, fit_err

    cov = np.cov(y_boot, rowvar=False, ddof=1)
    cov_inv = np.linalg.pinv(cov, rcond=1e-12)
    one = np.ones(n_points, dtype=np.float64)
    denom = float(one @ cov_inv @ one)
    if denom <= 0.0:
        raise np.linalg.LinAlgError("Non-positive denominator in correlated fit for fitted mres scan.")

    fit_value = float((one @ cov_inv @ y) / denom)
    fit_err = float(np.sqrt(1.0 / denom))
    return fit_value, fit_err


def scan_fitted_mres_vs_n_therm(
    x_full: np.ndarray,
    num_full: np.ndarray,
    den_full: np.ndarray,
    therm: int,
    plateau_mask: np.ndarray,
    delta_traj_conf: int,
    scan_step: int = 1,
    n_boot: int = 2000,
    seed: int | None = None,
):
    """
    Scan the fitted residual mass versus the thermalization cut using the same
    folded ratio and correlated constant fit as the main residual-mass plot.
    """
    x_full = np.asarray(x_full, dtype=float)
    num_full = np.asarray(num_full, dtype=float)
    den_full = np.asarray(den_full, dtype=float)
    plateau_mask = np.asarray(plateau_mask, dtype=bool)

    n_cfg = num_full.shape[0]
    if n_cfg < 2:
        return {
            "skipped": True,
            "reason": "not_enough_points",
            "therm": int(therm),
            "therm_display": float(max(0, therm)),
            "scan_step": int(scan_step),
            "max_n_therm_index": 0,
            "points": [],
        }

    max_therm = n_cfg // 2
    points: list[dict[str, float | int]] = []
    for n_therm in range(0, max_therm + 1, scan_step):
        # Match the main script's convention: the cut value itself is excluded,
        # so the point at x_display = therm uses trajectories with n > therm.
        kept_numbers = x_full[n_therm + 1 :]
        if kept_numbers.size < 2:
            continue

        keep_rel = _select_indices_by_delta(kept_numbers, delta_traj_conf)
        if keep_rel.size < 2:
            continue

        num = num_full[n_therm + 1 :][keep_rel]
        den = den_full[n_therm + 1 :][keep_rel]
        if num.shape[0] < 2 or den.shape[0] < 2:
            continue

        point_seed = None
        if seed is not None:
            x_display_int = int(round(float(x_full[n_therm])))
            point_seed = int(seed) if x_display_int == int(therm) else int(seed) + int(n_therm) + 1

        try:
            rng = np.random.default_rng(point_seed)
            ratio_mean, ratios_boot = _bootstrap_ratio_of_means(
                data_num=num,
                data_den=den,
                n_boot=int(n_boot),
                rng=rng,
            )
            fit_value, fit_err = _correlated_constant_fit(
                ratio_mean=ratio_mean,
                ratios_boot=ratios_boot,
                plateau_mask=plateau_mask,
            )
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue

        x_display = float(x_full[n_therm])
        points.append(
            {
                "n_therm": int(n_therm),
                "x_display": float(x_display),
                "mres_fit": fit_value,
                "err": fit_err,
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


def plot_fitted_mres_vs_n_therm(
    scan_payload: dict[str, Any],
    selected_fit: float,
    plot_path: str,
):
    if scan_payload.get("skipped", True):
        return

    pts = scan_payload.get("points", [])
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(3.7, 2.6), layout="constrained")
    therms = np.asarray([p["x_display"] for p in pts], dtype=float)
    fit_vals = np.asarray([p["mres_fit"] for p in pts], dtype=float)
    fit_errs = np.asarray([p.get("err", np.nan) for p in pts], dtype=float)
    ax.errorbar(
        therms,
        fit_vals,
        yerr=fit_errs,
        marker="o",
        markersize=2.8,
        markerfacecolor="none",
        capsize=2,
        elinewidth=0.9,
        linewidth=0.9,
        linestyle="-",
        color="C0",
        alpha=0.45,
    )

    if np.isfinite(selected_fit):
        ax.axhline(selected_fit, linewidth=1.0, linestyle="--", color="C0")

    if therms.size > 0:
        ax.axvline(0.5 * float(np.max(therms)), linewidth=0.9, linestyle="-", color="0.8", zorder=0)

    therm_display = scan_payload.get("therm_display", None)
    if therm_display is not None and np.isfinite(therm_display):
        ax.axvline(therm_display, linewidth=1.0, linestyle="--", color="0.35")

    ax.set_xlabel(r"$n_{\mathrm{therm}}$")
    ax.set_ylabel(r"$am_{\mathrm{res}}^{\mathrm{fit}}$")
    ax.set_xlim(left=0.0)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.savefig(plot_path)
    plt.close(fig)


__all__ = [
    "plot_fitted_mres_vs_n_therm",
    "scan_fitted_mres_vs_n_therm",
]
