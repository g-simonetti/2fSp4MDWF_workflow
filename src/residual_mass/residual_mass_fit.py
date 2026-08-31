#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# The workflow runs this file from the repository root, so add ``src/`` here to
# keep the analysis helpers importable in a release snapshot as well.
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from bootstrap.bootstrap_seed import resolve_bootstrap_seed
    from residual_mass.mres_fit_vs_therm import (
        plot_fitted_mres_vs_n_therm,
        scan_fitted_mres_vs_n_therm,
    )
    from residual_mass.residual_mass_plot import plot_residual_mass_fit
    from residual_mass.tau_int_mres import compute_mres_tau_from_series
except Exception as e:
    raise ImportError(
        "Failed to import bootstrap bootstrap-seed helpers or residual-mass "
        "fit-vs-therm helpers from src/.\n"
        "Expected repo layout with 'src/bootstrap/bootstrap_seed.py', "
        "'src/residual_mass/mres_fit_vs_therm.py', "
        "'src/residual_mass/residual_mass_plot.py' and "
        "compute_mres_tau_from_series from src/.\n"
        "Expected repo layout with 'src/residual_mass/tau_int_mres.py'."
    ) from e


def read_mres_file(filename: str) -> np.ndarray:
    """Read real part of wardIdentity/PJ5q from an mres HDF5 file."""
    with h5py.File(filename, "r") as f:
        return f["wardIdentity/PJ5q"][:]["re"]


def read_ptll_file(filename: str, n_elems: int | None = None) -> np.ndarray:
    """Read real part of meson/meson_1/corr from a pt_ll HDF5 file."""
    with h5py.File(filename, "r") as f:
        data = f["meson/meson_1/corr"][:]
        if n_elems is None or len(data) == n_elems:
            return data["re"]
    raise ValueError(f"corr dataset does not have {n_elems} entries in {filename}")


def fold_correlator(data: np.ndarray) -> np.ndarray:
    r"""
    Fold correlator according to

        C_fold(t) = 1/2 * [ C(t) + C(Nt - t) ]

    using periodic indexing, so Nt - t is interpreted mod Nt.
    """
    arr = np.asarray(data)
    T = arr.shape[-1]
    idx = (-np.arange(T)) % T
    return 0.5 * (arr + arr[..., idx])


# Build the residual-mass time series from the ratio of ensemble-averaged,
# folded correlators and estimate its uncertainty with bootstrap resampling.
def bootstrap_ratio_of_means(
    data_num: np.ndarray,
    data_den: np.ndarray,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
):
    r"""
    Compute the ratio of ensemble means of folded correlators:

        Rbar(t) = <num_fold(t)> / <den_fold(t)>

    and estimate its bootstrap uncertainty.
    """
    if rng is None:
        rng = np.random.default_rng()

    num = fold_correlator(data_num)
    den = fold_correlator(data_den)

    if num.shape != den.shape:
        raise ValueError(f"Shape mismatch: numerator {num.shape}, denominator {den.shape}")

    if num.ndim != 2:
        raise ValueError(f"Expected 2D arrays of shape (Ncfg, T), got {num.shape}")

    Ncfg, T = num.shape
    if Ncfg < 2:
        raise ValueError(f"Need at least 2 configurations, got {Ncfg}")

    num_mean = num.mean(axis=0)
    den_mean = den.mean(axis=0)

    if np.any(den_mean == 0):
        bad = np.where(den_mean == 0)[0]
        raise ZeroDivisionError(f"Zero ensemble-mean denominator encountered at times {bad.tolist()}")

    ratio_mean = num_mean / den_mean

    ratios_boot = np.empty((n_boot, T), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, Ncfg, size=Ncfg)
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


# Fit a constant over the chosen plateau window using the bootstrap covariance
# matrix so the release JSON stores a correlated extraction of ``am_res``.
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
    Yb = np.asarray(ratios_boot[:, mask], dtype=np.float64)

    if Yb.ndim != 2:
        raise ValueError(f"Expected bootstrap array of shape (Nb, np), got {Yb.shape}")

    Nb, np_ = Yb.shape
    if Nb < 2:
        raise ValueError("Need at least 2 bootstrap replicas to estimate covariance.")
    if np_ < 1:
        raise ValueError("Plateau window is empty.")

    y_boot_mean = Yb.mean(axis=0)

    if np_ == 1:
        m = float(y[0])
        sigma_m = float(np.sqrt(np.cov(Yb[:, 0], ddof=1)))
        red_chi2 = None
        cov = np.array([[sigma_m**2]], dtype=np.float64)
        return m, sigma_m, red_chi2, cov, y, y_boot_mean

    cov = np.cov(Yb, rowvar=False, ddof=1)
    cov_inv = np.linalg.pinv(cov, rcond=rcond)

    one = np.ones(np_, dtype=np.float64)
    denom = float(one @ cov_inv @ one)
    if denom <= 0:
        raise np.linalg.LinAlgError(
            "Non-positive denominator in correlated fit. Covariance may be singular or ill-conditioned."
        )

    m = float((one @ cov_inv @ y) / denom)
    sigma_m = float(np.sqrt(1.0 / denom))

    resid = y - m * one
    chi2 = float(resid @ cov_inv @ resid)
    dof = np_ - 1
    red_chi2 = chi2 / dof if dof > 0 else None

    return m, sigma_m, red_chi2, cov, y, y_boot_mean


_TRAJ_RE = re.compile(r".*\.(\d+)\.h5$")


def traj_number_from_path(path: str) -> int:
    base = os.path.basename(path)
    m = _TRAJ_RE.match(base)
    if not m:
        raise ValueError(f"Cannot extract trajectory number from filename: {path}")
    return int(m.group(1))


def build_number_map(files: list[str]) -> dict[int, str]:
    """Map trajectory-number -> filepath. If duplicates exist, keep the first in sorted order."""
    out: dict[int, str] = {}
    for f in sorted(files):
        n = traj_number_from_path(f)
        if n not in out:
            out[n] = f
    return out


def read_matched_series(
    mres_map: dict[int, str],
    ptll_map: dict[int, str],
    numbers: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the matched PJ5q and pt_ll correlator series for a fixed trajectory list.

    For the data release we keep this logic in one helper so the script has a
    single source of truth for how trajectories are paired and validated.
    """
    mres = np.array([read_mres_file(mres_map[n]) for n in numbers])
    n_times = mres.shape[1]
    ptll = np.array([read_ptll_file(ptll_map[n], n_elems=n_times) for n in numbers])

    if len(mres) != len(ptll):
        raise ValueError(
            f"Need matched series lengths after reading, got PJ5q={len(mres)} "
            f"and pt_ll={len(ptll)}"
        )

    return mres, ptll


def select_numbers_by_delta(numbers_sorted: list[int], delta_traj_conf: int) -> list[int]:
    """
    numbers_sorted must already be therm-cut and sorted.
    Keep arithmetic progression start=numbers_sorted[0], then start + k*delta_traj_conf (if present).

    If delta_traj_conf <= 0, return full list.
    """
    if not numbers_sorted:
        return []

    delta_traj_conf = int(delta_traj_conf)
    if delta_traj_conf <= 0:
        return list(numbers_sorted)

    start = numbers_sorted[0]
    num_set = set(numbers_sorted)
    last = numbers_sorted[-1]

    keep: list[int] = []
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
        description="Compute residual mass from HDF5 data using an optional correlated constant fit."
    )
    parser.add_argument("input_dir", help="mesons directory containing mres.*.h5 and pt_ll.*.h5")

    parser.add_argument("--label", default="", help="yes → include β, am0 label on plot")
    parser.add_argument("--mres_out", required=True, help="Output JSON file (m_res.json)")
    parser.add_argument("--plot_file", required=True, help="Output plot file")
    parser.add_argument("--plot_styles", default="")

    # plateau args may be passed as -1 by the workflow when missing
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

    parser.add_argument("--n_boot", type=int, default=2000, help="Number of bootstrap replicas")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed. If omitted, derive a deterministic seed from input_dir.",
    )

    args = parser.parse_args()

    # Metadata may encode missing plateau bounds as -1, so convert that to the
    # explicit "no fit window" case used below.
    have_plateau = (args.plateau_start is not None) and (args.plateau_end is not None)
    if have_plateau:
        have_plateau = (args.plateau_start >= 0) and (args.plateau_end >= 0)

    plateau_start = int(round(args.plateau_start)) if have_plateau else None
    plateau_end = int(round(args.plateau_end)) if have_plateau else None

    if have_plateau and plateau_end < plateau_start:
        raise ValueError(
            f"Invalid plateau range: plateau_start={plateau_start}, plateau_end={plateau_end}"
        )

    # Keep bootstrap replicas reproducible by default using the same path-based
    # seed convention as the shared helper. A CLI seed still overrides this.
    seed_info = resolve_bootstrap_seed(args.input_dir, cli_seed=args.seed)
    base_seed = int(seed_info["seed"])
    rng = np.random.default_rng(base_seed)

    # Pair the two correlator inputs through their trajectory numbers so the
    # release analysis always works with matched PJ5q and pt_ll histories.
    mres_files = glob.glob(os.path.join(args.input_dir, "mres.*.h5"))
    ptll_files = glob.glob(os.path.join(args.input_dir, "pt_ll.*.h5"))
    if not mres_files or not ptll_files:
        raise FileNotFoundError(f"Missing mres.*.h5 or pt_ll.*.h5 files in {args.input_dir}")

    mres_map = build_number_map(mres_files)
    ptll_map = build_number_map(ptll_files)

    common_numbers = sorted(set(mres_map.keys()) & set(ptll_map.keys()))
    if not common_numbers:
        raise FileNotFoundError("No matching trajectory numbers between mres.*.h5 and pt_ll.*.h5")

    # Read the matched trajectory history once, then derive the analysis views
    # below by slicing in memory. This keeps the release workflow easier to audit
    # and avoids rereading the same HDF5 content for each downstream product.
    mres_tau_all, ptll_tau_all = read_matched_series(mres_map, ptll_map, common_numbers)
    n_times = mres_tau_all.shape[1]
    used_numbers_tau = common_numbers

    # The "full" series applies only the thermalization cut and is used for the
    # fit inputs that should retain the complete post-thermalization history.
    therm = int(args.therm)
    full_numbers = [n for n in common_numbers if n > therm]
    if len(full_numbers) < 2:
        raise ValueError(
            f"Too few matched configurations after therm cut. "
            f"common={len(common_numbers)}, full(after therm)={len(full_numbers)}, therm={therm}"
        )

    index_by_number = {n: i for i, n in enumerate(common_numbers)}
    full_indices = [index_by_number[n] for n in full_numbers]
    mres_full = mres_tau_all[full_indices]
    ptll_full = ptll_tau_all[full_indices]
    used_numbers_full = full_numbers

    # The "meas" series applies the additional measurement thinning used for the
    # reported per-ensemble m_res series and plot.
    meas_numbers = select_numbers_by_delta(full_numbers, delta_traj_conf=int(args.delta_traj_conf))
    if len(meas_numbers) < 2:
        raise ValueError(
            f"Too few configs after measurement thinning. "
            f"full(after therm)={len(full_numbers)}, meas(after delta)={len(meas_numbers)}, "
            f"delta_traj_conf={args.delta_traj_conf}"
        )

    meas_indices = [index_by_number[n] for n in meas_numbers]
    mres_meas = mres_tau_all[meas_indices]
    ptll_meas = ptll_tau_all[meas_indices]
    used_numbers_meas = meas_numbers

    if len(used_numbers_meas) < 2:
        raise ValueError(
            f"Need at least 2 MEAS configurations after reading, got {len(used_numbers_meas)}"
        )

    if len(used_numbers_full) < 2:
        raise ValueError(
            f"Need at least 2 FULL configurations for tau_int after reading, got {len(used_numbers_full)}"
        )

    # Build the plotted residual-mass series from the measurement-thinned view.
    ratio_mean, ratio_err, ratios_boot = bootstrap_ratio_of_means(
        mres_meas,
        ptll_meas,
        n_boot=int(args.n_boot),
        rng=rng,
    )

    # If a plateau is defined in the metadata, extract a single am_res value
    # with a correlated constant fit and keep the covariance information.
    t_vals = np.arange(n_times)

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

    # For the autocorrelation analysis, use the plateau start when available and
    # fall back to Nt/2 otherwise so the reported timeslice tracks the fit setup.
    tau_t = int(plateau_start) if have_plateau else int(n_times // 2)

    if tau_t < 0 or tau_t >= n_times:
        raise ValueError(f"tau_int time slice t={tau_t} out of range [0, {n_times - 1}]")

    out_json_dir = os.path.dirname(os.path.abspath(args.mres_out)) or "."
    mres_tau_dir = os.path.join(out_json_dir, "tau_int_mres")
    mres_tau_results = compute_mres_tau_from_series(
        numerator_values=mres_tau_all[:, tau_t],
        denominator_values=ptll_tau_all[:, tau_t],
        traj_numbers=used_numbers_tau,
        out_dir=mres_tau_dir,
        therm=therm,
        plot_styles=args.plot_styles if args.plot_styles else None,
        base_name="tau_int_mres",
        numerator_label="pj5q",
        denominator_label="ptll",
        t=tau_t,
        emit_json=False,
    )

    pj5q_tau_info = dict(mres_tau_results["components"]["numerator"])
    pj5q_tau_info["folded"] = False
    ptll_tau_info = dict(mres_tau_results["components"]["denominator"])
    ptll_tau_info["folded"] = False
    mres_tau_info = {
        "t": int(tau_t),
        "tau_int": mres_tau_results["estimate"]["tau_int"],
        "tau_int_err": mres_tau_results["estimate"]["err"],
        "Nb_est": mres_tau_results["estimate"]["Nb"],
        "Nbs_est": mres_tau_results["estimate"]["Nbs"],
        "found": bool(mres_tau_results["estimate"]["found_plateau"]),
        "source_component": mres_tau_results["estimate"]["source_component"],
        "tau_int_dir": str(mres_tau_dir),
        "results_json": str(mres_tau_results["outputs"]["results_json"]),
    }

    # The release JSON is the compact machine-readable summary used by later
    # tables and plots, so keep the analysis settings explicit here.
    os.makedirs(os.path.dirname(os.path.abspath(args.mres_out)), exist_ok=True)

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
            "covariance_estimator": "bootstrap",
            "fit_method": "correlated_constant_fit" if have_plateau else None,
            "fit_error_estimator": "GLS_analytic" if have_plateau else None,
            "tau_int_series": "wolff_projected_mres",
            "tau_int_series_additional": ["unfolded_ptll", "unfolded_pj5q"],
            "fit_performed": bool(have_plateau),
        },
        "ensembles": {
            "meas": {
                "n_cfg": int(len(used_numbers_meas)),
                "traj_start": int(used_numbers_meas[0]),
                "traj_end": int(used_numbers_meas[-1]),
                "traj_numbers": [int(x) for x in used_numbers_meas],
            },
            "full": {
                "n_cfg": int(len(used_numbers_full)),
                "traj_start": int(used_numbers_full[0]),
                "traj_end": int(used_numbers_full[-1]),
                "traj_numbers": [int(x) for x in used_numbers_full],
            },
        },
        "mres_series": {
            "t": [int(t) for t in t_vals],
            "mres": [float(x) for x in ratio_mean],
            "mres_err": [float(x) for x in ratio_err],
            "folded": True,
        },
        "mres_extract": {
            "mres_tau_int": mres_tau_info,
            "ptll_tau_int": ptll_tau_info,
            "pj5q_tau_int": pj5q_tau_info,
        },
    }

    if have_plateau:
        plateau_mask = (t_vals >= plateau_start) & (t_vals <= plateau_end)
        payload["mres_extract"].update(
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

        fit_therm_scan = scan_fitted_mres_vs_n_therm(
            x_full=np.asarray(used_numbers_tau, dtype=float),
            num_full=mres_tau_all,
            den_full=ptll_tau_all,
            therm=therm,
            plateau_mask=plateau_mask,
            delta_traj_conf=int(args.delta_traj_conf),
            scan_step=1,
            n_boot=int(args.n_boot),
            seed=int(base_seed),
        )
        fit_therm_plot = os.path.join(mres_tau_dir, "tau_int_mres_fit_vs_n_therm.pdf")
        plot_fitted_mres_vs_n_therm(fit_therm_scan, float(avg), fit_therm_plot)
        payload["mres_extract"]["fit_n_therm_scan"] = fit_therm_scan
        payload["mres_extract"]["fit_n_therm_scan"]["plot_pdf"] = (
            None if fit_therm_scan.get("skipped", True) else fit_therm_plot
        )

    with open(args.mres_out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    t_plot_max = n_times // 2
    plot_mask = t_vals <= t_plot_max
    t_plot = t_vals[plot_mask]
    ratio_mean_plot = ratio_mean[plot_mask]
    ratio_err_plot = ratio_err[plot_mask]

    is_shamir_point = (
        np.isclose(args.alpha, 1.0)
        and np.isclose(args.a5, 1.0)
        and np.isclose(args.m5, 1.8)
        and np.isclose(args.mpv, 1.0)
    )
    if is_shamir_point:
        title_str = rf"$\beta = {args.beta},\ am_0 = {args.mass}$"
        data_label = None
    else:
        data_label = rf"$\beta={args.beta},\ am_0={args.mass}$" if args.label == "yes" else None
        title_str = (
            rf"$\alpha = {args.alpha},\ a_5/a = {args.a5},\ "
            rf"am_5 = {args.m5},\ am_{{\rm PV}} = {args.mpv}$"
        )
    plot_residual_mass_fit(
        args.plot_file,
        t_plot,
        ratio_mean_plot,
        ratio_err_plot,
        plot_styles=args.plot_styles if args.plot_styles else None,
        data_label=data_label,
        title=title_str,
        plateau_start=plateau_start if have_plateau else None,
        plateau_end=plateau_end if have_plateau else None,
        fit_value=avg if have_plateau else None,
        fit_error=err if have_plateau else None,
    )

if __name__ == "__main__":
    main()
