#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

plt.style.use("tableau-colorblind10")

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from bootstrap.bootstrap_seed import resolve_bootstrap_seed
except Exception as e:
    raise ImportError(
        "Failed to import resolve_bootstrap_seed from src/bootstrap/bootstrap_seed.py."
    ) from e


def read_pa0_file(filename: str) -> np.ndarray:
    """Read real part of wardIdentity/PA0 from an HDF5 file."""
    with h5py.File(filename, "r") as f:
        return f["wardIdentity/PA0"][:]["re"]


def read_ptll_file(filename: str, n_elems: int | None = None) -> np.ndarray:
    """Read real part of meson/meson_1/corr from an HDF5 file."""
    with h5py.File(filename, "r") as f:
        data = f["meson/meson_1/corr"][:]
        if n_elems is None or len(data) == n_elems:
            return data["re"]
    raise ValueError(f"corr dataset does not have {n_elems} entries in {filename}")


def central_time_derivative_oa4(data: np.ndarray) -> np.ndarray:
    """Fourth-order central derivative along the time direction."""
    return (
        -1.0 * np.roll(data, -2, axis=1)
        + 8.0 * np.roll(data, -1, axis=1)
        - 8.0 * np.roll(data, 1, axis=1)
        + 1.0 * np.roll(data, 2, axis=1)
    ) / 12.0


def fold_correlator(data: np.ndarray) -> np.ndarray:
    r"""
    Fold correlator according to

        C_fold(t) = 1/2 * [ C(t) + C(Nt - t) ]

    using periodic indexing, so Nt - t is interpreted mod Nt.
    """
    arr = np.asarray(data)
    t_full = arr.shape[-1]
    idx = (-np.arange(t_full)) % t_full
    return 0.5 * (arr + arr[..., idx])


def bootstrap_ratio_of_means(
    data_num: np.ndarray,
    data_den: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
):
    """
    Compute the ratio of ensemble means of folded correlators and estimate its
    bootstrap uncertainty.
    """
    num = fold_correlator(data_num)
    den = fold_correlator(data_den)

    if num.shape != den.shape:
        raise ValueError(f"Shape mismatch: numerator {num.shape}, denominator {den.shape}")
    if num.ndim != 2:
        raise ValueError(f"Expected 2D arrays of shape (Ncfg, T), got {num.shape}")

    n_cfg, n_t = num.shape
    if n_cfg < 2:
        raise ValueError(f"Need at least 2 configurations, got {n_cfg}")

    num_mean = num.mean(axis=0)
    den_mean = den.mean(axis=0)
    if np.any(den_mean == 0):
        bad = np.where(den_mean == 0)[0]
        raise ZeroDivisionError(
            f"Zero ensemble-mean denominator encountered at times {bad.tolist()}"
        )

    ratio_mean = num_mean / den_mean
    ratios_boot = np.empty((n_boot, n_t), dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n_cfg, size=n_cfg)
        num_b = num[idx].mean(axis=0)
        den_b = den[idx].mean(axis=0)
        if np.any(den_b == 0):
            bad = np.where(den_b == 0)[0]
            raise ZeroDivisionError(
                f"Zero bootstrap denominator encountered in replica {b} at times {bad.tolist()}"
            )
        ratios_boot[b] = num_b / den_b

    ratio_err = ratios_boot.std(axis=0, ddof=1)
    return ratio_mean, ratio_err, ratios_boot


def correlated_constant_fit(
    ratio_mean: np.ndarray,
    ratios_boot: np.ndarray,
    t_vals: np.ndarray,
    tmin: int,
    tmax: int,
    rcond: float = 1e-12,
):
    mask = (t_vals >= tmin) & (t_vals <= tmax)
    if not np.any(mask):
        raise ValueError(f"No points in plateau range [{tmin}, {tmax}]")

    y = np.asarray(ratio_mean[mask], dtype=np.float64)
    y_boot = np.asarray(ratios_boot[:, mask], dtype=np.float64)

    n_boot, n_pts = y_boot.shape
    if n_boot < 2:
        raise ValueError("Need at least 2 bootstrap replicas to estimate covariance.")

    if n_pts == 1:
        sigma = float(np.sqrt(np.cov(y_boot[:, 0], ddof=1)))
        cov = np.array([[sigma**2]], dtype=np.float64)
        return float(y[0]), sigma, None, cov, y, y_boot.mean(axis=0)

    cov = np.cov(y_boot, rowvar=False, ddof=1)
    cov_inv = np.linalg.pinv(cov, rcond=rcond)

    one = np.ones(n_pts, dtype=np.float64)
    denom = float(one @ cov_inv @ one)
    if denom <= 0:
        raise np.linalg.LinAlgError(
            "Non-positive denominator in correlated fit. Covariance may be singular."
        )

    value = float((one @ cov_inv @ y) / denom)
    error = float(np.sqrt(1.0 / denom))

    resid = y - value * one
    chi2 = float(resid @ cov_inv @ resid)
    dof = n_pts - 1
    red_chi2 = chi2 / dof if dof > 0 else None

    return value, error, red_chi2, cov, y, y_boot.mean(axis=0)


_TRAJ_RE = re.compile(r".*\.(\d+)\.h5$")


def traj_number_from_path(path: str) -> int:
    m = _TRAJ_RE.match(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot extract trajectory number from filename: {path}")
    return int(m.group(1))


def build_number_map(files: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for f in sorted(files):
        n = traj_number_from_path(f)
        if n not in out:
            out[n] = f
    return out


def select_numbers_by_delta(numbers_sorted: list[int], delta_traj_conf: int) -> list[int]:
    if not numbers_sorted:
        return []

    delta_traj_conf = int(delta_traj_conf)
    if delta_traj_conf <= 0:
        return list(numbers_sorted)

    start = numbers_sorted[0]
    num_set = set(numbers_sorted)
    last = numbers_sorted[-1]

    keep = []
    k = 0
    while True:
        n = start + k * delta_traj_conf
        if n in num_set:
            keep.append(n)
        if n > last:
            break
        k += 1

    return sorted(set(keep))


def main():
    parser = argparse.ArgumentParser(
        description="Compute the PCAC ratio from PA0 and pt_ll and write a JSON summary."
    )
    parser.add_argument("input_dir", help="mesons directory containing mres.*.h5 and pt_ll.*.h5")
    parser.add_argument("--label", default="", help="yes -> include beta, am0 label on plot")
    parser.add_argument("--mpcac_out", required=True, help="Output JSON file (m_pcac.json)")
    parser.add_argument("--plot_file", nargs="+", required=True, help="Output plot file(s)")
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--plateau_start", type=float, default=-1)
    parser.add_argument("--plateau_end", type=float, default=-1)
    parser.add_argument("--therm", type=int, required=True)
    parser.add_argument(
        "--delta_traj_conf",
        "--delta_traj_ps",
        dest="delta_traj_conf",
        type=int,
        required=True,
    )
    parser.add_argument("--beta", type=float, default=np.nan)
    parser.add_argument("--mass", type=float, default=np.nan)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--a5", type=float, required=True)
    parser.add_argument("--m5", type=float, required=True)
    parser.add_argument("--mpv", type=float, required=True)
    parser.add_argument("--Nt", type=int, required=True)
    parser.add_argument("--Ns", type=int, required=True)
    parser.add_argument("--Ls", type=int, required=True)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    have_plateau = (
        args.plateau_start is not None
        and args.plateau_end is not None
        and args.plateau_start >= 0
        and args.plateau_end >= 0
    )
    plateau_start = int(round(args.plateau_start)) if have_plateau else None
    plateau_end = int(round(args.plateau_end)) if have_plateau else None
    if have_plateau and plateau_end < plateau_start:
        raise ValueError(
            f"Invalid plateau range: plateau_start={plateau_start}, plateau_end={plateau_end}"
        )

    seed_info = resolve_bootstrap_seed(args.input_dir, cli_seed=args.seed)
    base_seed = int(seed_info["seed"])
    rng = np.random.default_rng(base_seed)

    pa0_files = glob.glob(os.path.join(args.input_dir, "mres.*.h5"))
    ptll_files = glob.glob(os.path.join(args.input_dir, "pt_ll.*.h5"))
    if not pa0_files or not ptll_files:
        raise FileNotFoundError(f"Missing mres.*.h5 or pt_ll.*.h5 files in {args.input_dir}")

    pa0_map = build_number_map(pa0_files)
    ptll_map = build_number_map(ptll_files)
    common_numbers = sorted(set(pa0_map.keys()) & set(ptll_map.keys()))
    if not common_numbers:
        raise FileNotFoundError("No matching trajectory numbers between mres.*.h5 and pt_ll.*.h5")

    therm = int(args.therm)
    full_numbers = [n for n in common_numbers if n > therm]
    if len(full_numbers) < 2:
        raise ValueError(
            f"Too few matched configurations after therm cut. "
            f"common={len(common_numbers)}, full(after therm)={len(full_numbers)}, therm={therm}"
        )

    meas_numbers = select_numbers_by_delta(full_numbers, delta_traj_conf=int(args.delta_traj_conf))
    if len(meas_numbers) < 2:
        raise ValueError(
            f"Too few configs after measurement thinning. "
            f"full(after therm)={len(full_numbers)}, meas(after delta)={len(meas_numbers)}, "
            f"delta_traj_conf={args.delta_traj_conf}"
        )

    pa0_meas_files = [pa0_map[n] for n in meas_numbers]
    ptll_meas_files = [ptll_map[n] for n in meas_numbers]

    pa0_meas = np.array([read_pa0_file(f) for f in pa0_meas_files])
    n_times_full = pa0_meas.shape[1]
    ptll_meas = np.array([read_ptll_file(f, n_elems=n_times_full) for f in ptll_meas_files])

    min_len_meas = min(len(pa0_meas), len(ptll_meas))
    pa0_meas = pa0_meas[:min_len_meas]
    ptll_meas = ptll_meas[:min_len_meas]
    used_numbers_meas = meas_numbers[:min_len_meas]

    if min_len_meas < 2:
        raise ValueError(f"Need at least 2 MEAS configurations after reading, got {min_len_meas}")

    if n_times_full != int(args.Nt):
        raise ValueError(f"Read Nt={n_times_full} from data, but CLI requested Nt={args.Nt}")
    if n_times_full % 2 != 0:
        raise ValueError(f"Nt must be even for the folded PCAC construction, got {n_times_full}")

    d_pa0_meas = central_time_derivative_oa4(pa0_meas)
    ratio_mean, ratio_err, ratios_boot = bootstrap_ratio_of_means(
        data_num=0.5 * d_pa0_meas,
        data_den=ptll_meas,
        n_boot=int(args.n_boot),
        rng=rng,
    )
    t_vals = np.arange(n_times_full)

    if have_plateau:
        if plateau_end > (n_times_full // 2):
            raise ValueError(
                f"plateau_end={plateau_end} exceeds max allowed {n_times_full // 2} for folded data"
            )

    avg = None
    err = None
    red_chi2 = None
    cov_plateau = None
    y_plateau = None
    y_boot_mean = None

    if have_plateau:
        avg, err, red_chi2, cov_plateau, y_plateau, y_boot_mean = correlated_constant_fit(
            ratio_mean=ratio_mean,
            ratios_boot=ratios_boot,
            t_vals=t_vals,
            tmin=plateau_start,
            tmax=plateau_end,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.mpcac_out)), exist_ok=True)

    payload = {
        "parameters": {
            "beta": float(args.beta) if np.isfinite(args.beta) else None,
            "mass": float(args.mass) if np.isfinite(args.mass) else None,
            "Nt": int(args.Nt),
            "Ns": int(args.Ns),
            "Ls": int(args.Ls),
            "alpha": float(args.alpha),
            "a5": float(args.a5),
            "m5": float(args.m5),
            "mpv": float(args.mpv),
        },
        "analysis_settings": {
            "plateau_start": int(plateau_start) if have_plateau else None,
            "plateau_end": int(plateau_end) if have_plateau else None,
            "therm": int(therm),
            "delta_traj_conf": int(args.delta_traj_conf),
            "n_boot": int(args.n_boot),
            "seed": int(base_seed),
            "seed_source": str(seed_info["source"]),
            "ratio_definition": "ratio_of_folded_ensemble_means",
            "derivative_scheme": "central_oa4",
            "bootstrap_unit": "configurations",
            "fit_method": "correlated_constant_fit" if have_plateau else None,
            "fit_error_estimator": "GLS_analytic" if have_plateau else None,
            "fit_performed": bool(have_plateau),
        },
        "ensembles": {
            "meas": {
                "n_cfg": int(min_len_meas),
                "traj_start": int(used_numbers_meas[0]),
                "traj_end": int(used_numbers_meas[-1]),
                "traj_numbers": [int(x) for x in used_numbers_meas],
            },
            "full": {
                "n_cfg": int(len(full_numbers)),
                "traj_start": int(full_numbers[0]),
                "traj_end": int(full_numbers[-1]),
                "traj_numbers": [int(x) for x in full_numbers],
            },
        },
        "mpcac_series": {
            "t": [int(x) for x in t_vals],
            "mpcac": [float(x) for x in ratio_mean],
            "mpcac_err": [float(x) for x in ratio_err],
            "folded": True,
        },
        "mpcac_extract": {},
    }

    if have_plateau:
        plateau_mask = (t_vals >= plateau_start) & (t_vals <= plateau_end)
        payload["mpcac_extract"].update(
            {
                "value": float(avg),
                "error": float(err),
                "reduced_chi2": float(red_chi2) if red_chi2 is not None else None,
                "plateau_start": int(plateau_start),
                "plateau_end": int(plateau_end),
                "n_plateau_points": int(np.sum(plateau_mask)),
                "plateau_t": [int(x) for x in t_vals[plateau_mask]],
                "plateau_y": [float(x) for x in y_plateau],
                "plateau_y_boot_mean": [float(x) for x in y_boot_mean],
                "covariance_matrix": cov_plateau.tolist(),
            }
        )

    with open(args.mpcac_out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    if args.plot_styles:
        parts = [p.strip() for p in str(args.plot_styles).split(",") if p.strip()]
        if parts:
            plt.style.use(parts)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    data_label = rf"$\beta={args.beta},\ am_0={args.mass}$" if args.label == "yes" else None
    title_str = (
        rf"$\alpha = {args.alpha},\ a_5/a = {args.a5},\ "
        rf"am_5 = {args.m5},\ am_{{\rm PV}} = {args.mpv}$"
    )
    ax.set_title(title_str, fontsize=10)

    t_plot_max = n_times_full // 2
    plot_mask = t_vals <= t_plot_max
    t_plot = t_vals[plot_mask]
    ratio_mean_plot = ratio_mean[plot_mask]
    ratio_err_plot = ratio_err[plot_mask]

    ax.errorbar(
        t_plot,
        ratio_mean_plot,
        yerr=ratio_err_plot,
        fmt="o",
        color="C4",
        label=data_label,
    )

    if have_plateau:
        fit_label = rf"$am_{{\rm PCAC}}^{{\rm fit}} = {avg:.5f}\,\pm\,{err:.5f}$"
        ax.axvspan(plateau_start, plateau_end, color="C2", alpha=0.2, label="Plateau range")
        ax.fill_between(
            [plateau_start, plateau_end],
            [avg - err, avg - err],
            [avg + err, avg + err],
            color="C1",
            alpha=0.25,
            linewidth=0,
        )
        ax.hlines(avg, plateau_start, plateau_end, color="C1", linestyle="--", label=fit_label)

    ax.set_xlim(-0.5, t_plot_max + 0.5)
    ax.set_xlabel(r"$t/a$")
    ax.set_ylabel(r"$am_{\rm PCAC}$")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if have_plateau or data_label:
        ax.legend()

    for plot_file in args.plot_file:
        os.makedirs(os.path.dirname(os.path.abspath(plot_file)) or ".", exist_ok=True)
        fig.savefig(plot_file, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
