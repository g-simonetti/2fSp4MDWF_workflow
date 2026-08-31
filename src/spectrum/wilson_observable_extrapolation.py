#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from shared_continuum_models import (
    derive_wilson_start_parameters,
    fit_wilson_complete_model_linear,
    fit_wilson_complete_model_nonlinear,
    wilson_physical_model,
)
from shared_fit_serialization import (
    linear_fit_to_json_dict,
    physical_fit_to_json_dict,
    to_serializable,
)


DIR_RE = re.compile(
    r"^Sp4b(?P<beta>-?\d+(?:\.\d+)?)nF\d+mF(?P<m0>-?\d+(?:\.\d+)?)T(?P<Nt>\d+)L(?P<Ns>\d+)$"
)
DEFAULT_PAPER_SEED = (0.5, 2.0, -0.2)
DEFAULT_FIXED_PAPER_SEED = (0.3009, 2.993, 0.1308)
DEFAULT_WILSON_SHARED_NONLINEAR_P0 = [0.320, 2.9, -20.0, -0.183, 0.03, -1.0]

OBSERVABLE_CONFIG = {
    "mv": {
        "point_key": "mv_point",
        "sample_key": "mv_bootstrap_samples",
        "summary_sq_key": "mvw0_sq",
        "summary_key": "mvw0",
        "linear_label": (
            r"Wilson linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a m_{PS}^2 + E a + F a^2$"
        ),
        "fit_label": (
            r"Wilson physical: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "bootstrap_label": (
            r"Wilson bootstrap: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "central_linearized_uses_shared_p0": True,
    },
    "fps": {
        "point_key": "fps_point",
        "sample_key": "fps_bootstrap_samples",
        "summary_sq_key": "fpsw0_sq",
        "summary_key": "fpsw0",
        "linear_label": (
            r"Wilson linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a m_{PS}^2 + E a + F a^2$"
        ),
        "fit_label": (
            r"Wilson physical: $f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "bootstrap_label": (
            r"Wilson bootstrap: $f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "central_linearized_uses_shared_p0": False,
    },
}


def get_config(observable):
    try:
        return OBSERVABLE_CONFIG[observable]
    except KeyError as exc:
        valid = ", ".join(sorted(OBSERVABLE_CONFIG))
        raise ValueError(
            f"Unsupported observable '{observable}'. Expected one of: {valid}."
        ) from exc


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_dir_metadata(path):
    match = DIR_RE.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse Wilson directory name: {path}")
    return {
        "beta": float(match.group("beta")),
        "m0": float(match.group("m0")),
        "Nt": int(match.group("Nt")),
        "Ns": int(match.group("Ns")),
    }


def summary_stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    if arr.size == 1:
        sdev = 0.0
    else:
        sdev = float(np.std(arr, ddof=1))
    return {
        "mean": float(np.mean(arr)),
        "sdev": sdev,
        "n": int(arr.size),
    }


def _std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _robust_keep_mask(params):
    params = np.asarray(params, dtype=float)
    n_rows = params.shape[0]
    keep = np.all(np.isfinite(params), axis=1)

    if n_rows < 5 or np.sum(keep) < 5:
        return keep, []

    med = np.median(params[keep], axis=0)
    mad = np.median(np.abs(params[keep] - med), axis=0)
    robust_sigma = 1.4826 * mad

    rejected = []
    for i in range(n_rows):
        if not keep[i]:
            rejected.append({"row_index": int(i), "reason": "non_finite_parameters"})
            continue

        deviations = np.abs(params[i] - med)
        flagged_columns = []
        for j, sigma_j in enumerate(robust_sigma):
            if sigma_j <= 0.0:
                continue
            if deviations[j] > 10.0 * sigma_j:
                flagged_columns.append(int(j))

        if flagged_columns:
            keep[i] = False
            rejected.append(
                {
                    "row_index": int(i),
                    "reason": "robust_parameter_outlier",
                    "flagged_columns": flagged_columns,
                }
            )

    return keep, rejected


def _extract_fit_arrays(points):
    x = np.asarray([point["x"] for point in points], dtype=float)
    a = np.asarray([point["a_over_w0"] for point in points], dtype=float)
    y = np.asarray([point["y"] for point in points], dtype=float)
    ye = np.asarray([point["yerr"] for point in points], dtype=float)
    return x, a, y, ye


def _build_wilson_fit_result(params, cov, chi2, dof, label):
    errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {
        "model_key": "wilson_physical",
        "stage": "nonlinear",
        "label": label,
        "m_M_chi_sq": float(params[0]),
        "m_M_chi_sq_err": float(errs[0]),
        "L_m_M": float(params[1]),
        "L_m_M_err": float(errs[1]),
        "Q_m_M": float(params[2]),
        "Q_m_M_err": float(errs[2]),
        "W_m_M": float(params[3]),
        "W_m_M_err": float(errs[3]),
        "R_m_M": float(params[4]),
        "R_m_M_err": float(errs[4]),
        "C_m_M": float(params[5]),
        "C_m_M_err": float(errs[5]),
        "cov": cov,
        "chi2": float(chi2),
        "dof": int(dof),
    }


def _finite_difference_jacobian(model, inputs, params, rel_step=1e-6):
    params = np.asarray(params, dtype=float)
    f0 = np.asarray(model(inputs, *params), dtype=float)
    jac = np.empty((f0.size, params.size), dtype=float)
    for i in range(params.size):
        step = rel_step * max(1.0, abs(params[i]))
        plus = params.copy()
        minus = params.copy()
        plus[i] += step
        minus[i] -= step
        f_plus = np.asarray(model(inputs, *plus), dtype=float)
        f_minus = np.asarray(model(inputs, *minus), dtype=float)
        jac[:, i] = (f_plus - f_minus) / (2.0 * step)
    return jac


def _weighted_covariance_from_jacobian(jac, sigma):
    sigma = np.asarray(sigma, dtype=float)
    weighted = jac / sigma[:, None]
    fisher = weighted.T @ weighted
    return np.linalg.pinv(fisher)


def wilson_leading_model(inputs, m_M_chi_sq, L_m_M, W_m_M):
    m_ps_sq, a = inputs
    return m_M_chi_sq * (1.0 + L_m_M * m_ps_sq) + W_m_M * a


def minimize_like_original_repo(objective, x0, solver_method="TNC"):
    result = minimize(
        objective,
        x0=np.asarray(x0, dtype=float),
        method=solver_method,
        tol=10**-16,
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Wilson fit produced non-finite parameters with {solver_method}: {result.message}"
        )
    return result


def fit_wilson_leading_model_tnc(points, seed=DEFAULT_PAPER_SEED, solver_method="TNC"):
    x, a, y, ye = _extract_fit_arrays(points)

    def objective(params):
        residuals = (y - wilson_leading_model((x, a), *params)) / ye
        return float(np.sum(residuals**2))

    result = minimize_like_original_repo(objective, seed, solver_method=solver_method)
    return {
        "m_M_chi_sq": float(result.x[0]),
        "L_m_M": float(result.x[1]),
        "W_m_M": float(result.x[2]),
        "chi2": float(result.fun),
        "solver_method": solver_method,
    }


def fit_wilson_complete_model_nonlinear_tnc(
    points,
    p0,
    fit_label,
    solver_method="TNC",
):
    x, a, y, ye = _extract_fit_arrays(points)

    def objective(params):
        residuals = (y - wilson_physical_model((x, a), *params)) / ye
        return float(np.sum(residuals**2))

    result = minimize_like_original_repo(objective, p0, solver_method=solver_method)

    params = np.asarray(result.x, dtype=float)
    jac = _finite_difference_jacobian(wilson_physical_model, (x, a), params)
    cov = _weighted_covariance_from_jacobian(jac, ye)
    dof = int(len(y) - len(params) - 1)
    return _build_wilson_fit_result(params, cov, float(result.fun), dof, fit_label)


def build_paper_strategy_start(points, seed=DEFAULT_PAPER_SEED, solver_method="TNC"):
    leading_fit = fit_wilson_leading_model_tnc(points, seed=seed, solver_method=solver_method)
    start = {
        "m_M_chi_sq": leading_fit["m_M_chi_sq"],
        "L_m_M": leading_fit["L_m_M"],
        "Q_m_M": 0.0,
        "W_m_M": leading_fit["W_m_M"],
        "R_m_M": 0.0,
        "C_m_M": 0.0,
    }
    return start, leading_fit


def build_fixed_paper_seed_start(seed=DEFAULT_PAPER_SEED):
    return {
        "m_M_chi_sq": float(seed[0]),
        "L_m_M": float(seed[1]),
        "Q_m_M": 0.0,
        "W_m_M": float(seed[2]),
        "R_m_M": 0.0,
        "C_m_M": 0.0,
    }


def _read_mps_source(path, ps_source):
    if ps_source == "gevp":
        mps_path = path / "meson_gevp_f_ps_samples.json"
        mps_key = "gevp_f_ps_E0_mass_samples"
        mps_label = "gevp_f_ps_E0_mass"
    else:
        mps_path = path / "meson_extraction_f_ps_samples.json"
        mps_key = "f_ps_mass_samples"
        mps_label = "f_ps_mass"

    if not mps_path.exists():
        raise FileNotFoundError(
            f"Requested Wilson mPS source '{ps_source}' is missing in {path}: {mps_path.name}"
        )

    mps_data = read_json(mps_path)
    if mps_key not in mps_data:
        raise KeyError(
            f"Requested Wilson mPS source '{ps_source}' in {mps_path} does not contain key '{mps_key}'."
        )

    return mps_data, mps_key, mps_label, mps_path.name


def _read_mv_source(path, mv_source):
    if mv_source == "gevp":
        mv_path = path / "meson_gevp_f_v_samples.json"
        mv_key = "gevp_f_v_E0_mass_samples"
        mv_label = "gevp_f_v_E0_mass"
    else:
        mv_path = path / "meson_extraction_f_v_samples.json"
        mv_key = "f_v_mass_samples"
        mv_label = "f_v_mass"

    if not mv_path.exists():
        raise FileNotFoundError(
            f"Requested Wilson mV source '{mv_source}' is missing in {path}: {mv_path.name}"
        )

    mv_data = read_json(mv_path)
    if mv_key not in mv_data:
        raise KeyError(
            f"Requested Wilson mV source '{mv_source}' in {mv_path} does not contain key '{mv_key}'."
        )

    return mv_data, mv_key, mv_label, mv_path.name


def read_ensemble_dir(path, mv_source="extraction", ps_source="extraction"):
    meta = parse_dir_metadata(path)

    w0_data = read_json(path / "w0_samples.json")
    mps_data, mps_key, mps_label, mps_filename = _read_mps_source(path, ps_source)
    mv_data, mv_key, mv_label, mv_filename = _read_mv_source(path, mv_source)
    fps_data = read_json(path / "decay_constant_f_ps_samples.json")

    ensemble_names = {
        str(w0_data.get("ensemble_name", "")),
        str(mps_data.get("ensemble_name", "")),
        str(mv_data.get("ensemble_name", "")),
        str(fps_data.get("ensemble_name", "")),
    }
    if len(ensemble_names) != 1:
        raise ValueError(f"Mismatched ensemble names in {path}")
    ensemble = ensemble_names.pop()

    local_beta = {
        float(w0_data.get("beta")),
        float(mps_data.get("beta")),
        float(mv_data.get("beta")),
        float(fps_data.get("beta")),
    }
    local_m0 = {
        float(w0_data.get("mF")),
        float(mps_data.get("mF")),
        float(mv_data.get("mF")),
        float(fps_data.get("mF")),
    }
    if len(local_beta) != 1 or len(local_m0) != 1:
        raise ValueError(f"Mismatched beta/m0 metadata in {path}")

    beta = local_beta.pop()
    m0 = local_m0.pop()
    if abs(beta - meta["beta"]) > 1e-12 or abs(m0 - meta["m0"]) > 1e-12:
        raise ValueError(f"Directory metadata mismatch in {path}")

    w0 = np.asarray(w0_data["w0_samples"], dtype=float)
    mps = np.asarray(mps_data[mps_key], dtype=float)
    mv = np.asarray(mv_data[mv_key], dtype=float)
    fps = np.asarray(fps_data["f_ps_decay_constant_samples"], dtype=float)

    n_local = int(min(len(w0), len(mps), len(mv), len(fps)))
    if n_local <= 0:
        raise ValueError(f"No bootstrap samples found in {path}")

    w0 = w0[:n_local]
    mps = mps[:n_local]
    mv = mv[:n_local]
    fps = fps[:n_local]

    valid = np.isfinite(w0) & np.isfinite(mps) & np.isfinite(mv) & np.isfinite(fps) & (w0 > 0.0)
    if not np.any(valid):
        raise ValueError(f"No valid bootstrap samples found in {path}")

    w0 = w0[valid]
    mps = mps[valid]
    mv = mv[valid]
    fps = fps[valid]

    a = 1.0 / w0
    a_sq = a**2
    x = (mps * w0) ** 2
    x_sq = x**2
    x_a = x * a
    y_mv = (mv * w0) ** 2
    y_fps = (fps * w0) ** 2

    mv_point = {
        "Ensemble": ensemble,
        "beta": beta,
        "m0": m0,
        "x": float(np.mean(x)),
        "xerr": _std(x),
        "y": float(np.mean(y_mv)),
        "yerr": _std(y_mv),
        "a_over_w0": float(np.mean(a)),
        "a_over_w0_err": _std(a),
        "a_over_w0_sq": float(np.mean(a_sq)),
        "a_over_w0_sq_err": _std(a_sq),
    }

    fps_point = {
        "Ensemble": ensemble,
        "beta": beta,
        "m0": m0,
        "x": float(np.mean(x)),
        "xerr": _std(x),
        "y": float(np.mean(y_fps)),
        "yerr": _std(y_fps),
        "a_over_w0": float(np.mean(a)),
        "a_over_w0_err": _std(a),
        "a_over_w0_sq": float(np.mean(a_sq)),
        "a_over_w0_sq_err": _std(a_sq),
    }

    replicas_mv = []
    replicas_fps = []
    for i in range(len(x)):
        replicas_mv.append(
            {
                "index": int(i),
                "Ensemble": ensemble,
                "beta": beta,
                "m0": m0,
                "x": float(x[i]),
                "x_sq": float(x_sq[i]),
                "y": float(y_mv[i]),
                "a_over_w0": float(a[i]),
                "a_over_w0_sq": float(a_sq[i]),
                "x_a_over_w0": float(x_a[i]),
                "w0": float(w0[i]),
                "m_ps": float(mps[i]),
                "m_v": float(mv[i]),
                "f_ps": float(fps[i]),
            }
        )
        replicas_fps.append(
            {
                "index": int(i),
                "Ensemble": ensemble,
                "beta": beta,
                "m0": m0,
                "x": float(x[i]),
                "x_sq": float(x_sq[i]),
                "y": float(y_fps[i]),
                "a_over_w0": float(a[i]),
                "a_over_w0_sq": float(a_sq[i]),
                "x_a_over_w0": float(x_a[i]),
                "w0": float(w0[i]),
                "m_ps": float(mps[i]),
                "f_ps": float(fps[i]),
            }
        )

    return {
        "directory": str(path),
        "Ensemble": ensemble,
        "beta": beta,
        "m0": m0,
        "Nt": meta["Nt"],
        "Ns": meta["Ns"],
        "mps_source": ps_source,
        "mps_source_file": mps_filename,
        "mps_source_observable": mps_label,
        "mv_source": mv_source,
        "mv_source_file": mv_filename,
        "mv_source_observable": mv_label,
        "n_boot_available": int(len(x)),
        "mv_point": mv_point,
        "fps_point": fps_point,
        "mv_bootstrap_samples": replicas_mv,
        "fps_bootstrap_samples": replicas_fps,
        "derived_statistics": {
            "w0": summary_stats(w0),
            "mps": summary_stats(mps),
            "mv": summary_stats(mv),
            "fps": summary_stats(fps),
            "mpsw0_sq": summary_stats(x),
            "mpsw0_4": summary_stats(x_sq),
            "a_over_w0": summary_stats(a),
            "a_over_w0_sq": summary_stats(a_sq),
            "mpsw0_sq_times_a_over_w0": summary_stats(x_a),
            "mvw0_sq": summary_stats(y_mv),
            "fpsw0_sq": summary_stats(y_fps),
        },
    }


def collect_ensemble_dirs(root, mv_source="extraction", ps_source="extraction"):
    dirs = [path for path in Path(root).iterdir() if path.is_dir()]
    if not dirs:
        raise ValueError(f"No Wilson ensemble directories found in {root}")

    rows = [read_ensemble_dir(path, mv_source=mv_source, ps_source=ps_source) for path in dirs]
    rows.sort(key=lambda item: (item["beta"], item["m0"], item["Ensemble"]))
    return rows


def build_wilson_bootstrap_point_sets(ensemble_rows, observable):
    cfg = get_config(observable)
    n_boot = min(row["n_boot_available"] for row in ensemble_rows)
    if n_boot <= 0:
        raise ValueError("No shared Wilson bootstrap replicas available.")

    points = [row[cfg["point_key"]] for row in ensemble_rows]
    point_sets = []

    for iboot in range(n_boot):
        point_set = []
        for row in ensemble_rows:
            sample = row[cfg["sample_key"]][iboot]
            point_set.append(
                {
                    "beta": sample["beta"],
                    "m0": sample["m0"],
                    "x": sample["x"],
                    "y": sample["y"],
                    "a_over_w0": sample["a_over_w0"],
                    "a_over_w0_sq": sample["a_over_w0_sq"],
                    "yerr": row[cfg["point_key"]]["yerr"],
                }
            )
        point_sets.append(point_set)

    return points, point_sets, n_boot


def fit_wilson_bootstrap_replica(points, p0, fit_label, *, solver_mode="curve_fit"):
    if solver_mode == "paper_tnc":
        fit = fit_wilson_complete_model_nonlinear_tnc(
            points,
            p0=p0,
            fit_label=fit_label,
            solver_method="TNC",
        )
    else:
        fit = fit_wilson_complete_model_nonlinear(points, p0=p0, fit_label=fit_label)
    params = np.array(
        [
            fit["m_M_chi_sq"],
            fit["L_m_M"],
            fit["Q_m_M"],
            fit["W_m_M"],
            fit["R_m_M"],
            fit["C_m_M"],
        ],
        dtype=float,
    )
    return params, float(fit["chi2"])


def fit_wilson_bootstrap_summary(
    bootstrap_point_sets,
    mean_points,
    start_params,
    observable,
    *,
    init_strategy="linearized",
):
    cfg = get_config(observable)
    default_p0 = [
        start_params["m_M_chi_sq"],
        start_params["L_m_M"],
        start_params["Q_m_M"],
        start_params["W_m_M"],
        start_params["R_m_M"],
        start_params["C_m_M"],
    ]

    samples = []
    failures = []
    success_rows = []
    shared_leading_fit = None
    shared_solver_mode = "curve_fit"

    if init_strategy in {"paper_tnc", "fixed_paper_seed"}:
        shared_solver_mode = "paper_tnc"
        if init_strategy == "paper_tnc":
            shared_leading_fit = {
                "m_M_chi_sq": float(start_params["m_M_chi_sq"]),
                "L_m_M": float(start_params["L_m_M"]),
                "W_m_M": float(start_params["W_m_M"]),
            }

    for b, point_set in enumerate(bootstrap_point_sets):
        try:
            popt, chi2 = fit_wilson_bootstrap_replica(
                point_set,
                default_p0,
                cfg["fit_label"],
                solver_mode=shared_solver_mode,
            )
            sample = {
                "index": int(b),
                "m_M_chi_sq": float(popt[0]),
                "L_m_M": float(popt[1]),
                "Q_m_M": float(popt[2]),
                "W_m_M": float(popt[3]),
                "R_m_M": float(popt[4]),
                "C_m_M": float(popt[5]),
                "chi2": chi2,
            }
            if shared_leading_fit is not None:
                sample["leading_fit_start"] = {
                    "m_M_chi_sq": shared_leading_fit["m_M_chi_sq"],
                    "L_m_M": shared_leading_fit["L_m_M"],
                    "W_m_M": shared_leading_fit["W_m_M"],
                }
            success_rows.append((int(b), popt, chi2, sample))
            samples.append(sample)
        except Exception as exc:
            failures.append({"index": int(b), "error": str(exc)})
            samples.append(None)

    if not success_rows:
        raise RuntimeError("All bootstrap Wilson continuum fits failed.")

    raw_params = np.asarray([row[1] for row in success_rows], dtype=float)
    keep_mask, rejected = _robust_keep_mask(raw_params)

    param_rows = []
    chi2_values = []
    for row_idx, (boot_index, popt, chi2, _sample) in enumerate(success_rows):
        if keep_mask[row_idx]:
            param_rows.append(popt)
            chi2_values.append(chi2)
            continue

        samples[boot_index] = None
        reject_info = rejected.pop(0) if rejected else {"reason": "rejected_bootstrap_replica"}
        failures.append(
            {
                "index": int(boot_index),
                "error": reject_info["reason"],
                **{k: v for k, v in reject_info.items() if k != "row_index"},
            }
        )

    if not param_rows:
        raise RuntimeError("All bootstrap Wilson continuum fits were rejected by the outlier filter.")

    params = np.asarray(param_rows, dtype=float)
    mean_params = np.mean(params, axis=0)
    cov = np.cov(params, rowvar=False, ddof=1) if params.shape[0] > 1 else np.zeros((6, 6), dtype=float)

    mean_x = np.asarray([point["x"] for point in mean_points], dtype=float)
    mean_a = np.asarray([point["a_over_w0"] for point in mean_points], dtype=float)
    mean_y = np.asarray([point["y"] for point in mean_points], dtype=float)
    mean_ye = np.asarray([point["yerr"] for point in mean_points], dtype=float)
    central_residuals = mean_y - wilson_physical_model((mean_x, mean_a), *mean_params)
    final_chi2 = float(np.sum((central_residuals / mean_ye) ** 2))
    final_dof = int(len(mean_y) - len(mean_params) - 1)

    errs = np.sqrt(np.diag(cov))
    fit = {
        "model_key": "wilson_physical_bootstrap",
        "stage": "bootstrap_summary",
        "label": cfg["bootstrap_label"],
        "m_M_chi_sq": float(mean_params[0]),
        "m_M_chi_sq_err": float(errs[0]),
        "L_m_M": float(mean_params[1]),
        "L_m_M_err": float(errs[1]),
        "Q_m_M": float(mean_params[2]),
        "Q_m_M_err": float(errs[2]),
        "W_m_M": float(mean_params[3]),
        "W_m_M_err": float(errs[3]),
        "R_m_M": float(mean_params[4]),
        "R_m_M_err": float(errs[4]),
        "C_m_M": float(mean_params[5]),
        "C_m_M_err": float(errs[5]),
        "cov": cov,
        "chi2": final_chi2,
        "dof": final_dof,
        "bootstrap_meta": {
            "n_requested": int(len(bootstrap_point_sets)),
            "n_success": int(params.shape[0]),
            "n_failed": int(len(bootstrap_point_sets) - params.shape[0]),
            "n_rejected_outliers": int(np.sum(~keep_mask)),
            "mean_chi2": float(np.mean(chi2_values)),
            "sdev_chi2": float(np.std(chi2_values, ddof=1)) if len(chi2_values) > 1 else 0.0,
        },
        "bootstrap_samples": samples,
        "bootstrap_failures": failures,
    }
    return fit


def bootstrap_fit_to_json_dict(fit, observable):
    cfg = get_config(observable)
    out = physical_fit_to_json_dict(fit)
    out["bootstrap_meta"] = fit["bootstrap_meta"]
    out["bootstrap_samples"] = fit["bootstrap_samples"]
    out["bootstrap_failures"] = fit["bootstrap_failures"]
    out["continuum_limit"] = {
        cfg["summary_sq_key"]: {
            "mean": fit["m_M_chi_sq"],
            "sdev": fit["m_M_chi_sq_err"],
        },
        cfg["summary_key"]: {
            "mean": float(np.sqrt(fit["m_M_chi_sq"])) if fit["m_M_chi_sq"] >= 0.0 else None,
            "sdev": (
                float(0.5 * fit["m_M_chi_sq_err"] / np.sqrt(fit["m_M_chi_sq"]))
                if fit["m_M_chi_sq"] > 0.0
                else None
            ),
        },
    }
    return to_serializable(out)


def save_results(
    output_data,
    observable,
    ensemble_rows,
    wilson_points,
    wilson_fit_linear,
    wilson_fit_nonlinear,
    wilson_fit_bootstrap,
    n_boot,
    *,
    init_strategy,
    paper_seed,
    starting_parameters,
    mv_source,
    ps_source,
):
    output_path = Path(output_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "observable": observable,
        "source_directory": str(Path(ensemble_rows[0]["directory"]).parent),
        "n_wilson_points_used": len(wilson_points),
        "n_wilson_bootstrap_requested": int(n_boot),
        "n_wilson_bootstrap_success": wilson_fit_bootstrap["bootstrap_meta"]["n_success"],
        "n_wilson_bootstrap_failed": wilson_fit_bootstrap["bootstrap_meta"]["n_failed"],
        "ensembles": to_serializable(ensemble_rows),
        "points": {
            "wilson": to_serializable(wilson_points),
            "wilson_mv": to_serializable([row["mv_point"] for row in ensemble_rows]),
            "wilson_fps": to_serializable([row["fps_point"] for row in ensemble_rows]),
        },
        "fit_options": {
            "observable": observable,
            "init_strategy": init_strategy,
            "ps_source": ps_source,
            "mv_source": mv_source,
            "paper_seed": {
                "m_M_chi_sq": float(paper_seed[0]),
                "L_m_M": float(paper_seed[1]),
                "W_m_M": float(paper_seed[2]),
            },
        },
        "fits": {
            "wilson_physical": {
                "linearized": linear_fit_to_json_dict(wilson_fit_linear),
                "starting_parameters": to_serializable(starting_parameters),
                "central_nonlinear": physical_fit_to_json_dict(wilson_fit_nonlinear),
                "bootstrap_summary": bootstrap_fit_to_json_dict(wilson_fit_bootstrap, observable),
            }
        },
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"Saved Wilson {observable} extrapolation data -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build Wilson bootstrap chiral-continuum fit data from per-ensemble "
            "bootstrap sample files stored under a root directory."
        )
    )
    parser.add_argument(
        "--observable",
        required=True,
        help="Which observable to fit: 'mv' or 'fps'.",
    )
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--dir", required=True, help="Directory containing Wilson ensemble subdirectories")
    parser.add_argument("--output_data", required=True, help="Output JSON file")
    parser.add_argument(
        "--wilson-init-strategy",
        choices=["linearized", "paper_tnc", "fixed_paper_seed"],
        default="linearized",
        help="How to initialize the Wilson nonlinear fit.",
    )
    parser.add_argument(
        "--wilson-v-source",
        choices=["extraction", "gevp"],
        default="extraction",
        help="Which Wilson vector-mass input to use.",
    )
    parser.add_argument(
        "--wilson-ps-source",
        choices=["extraction", "gevp"],
        default="extraction",
        help="Which Wilson pseudoscalar-mass input to use.",
    )
    parser.add_argument(
        "--paper-init-mchi-sq",
        type=float,
        default=None,
        help="Initial m_M_chi_sq for the Wilson seed fit.",
    )
    parser.add_argument(
        "--paper-init-L",
        type=float,
        default=None,
        help="Initial L_m_M for the Wilson seed fit.",
    )
    parser.add_argument(
        "--paper-init-W",
        type=float,
        default=None,
        help="Initial W_m_M for the Wilson seed fit.",
    )
    args = parser.parse_args()
    cfg = get_config(args.observable)

    default_seed = (
        DEFAULT_FIXED_PAPER_SEED
        if args.wilson_init_strategy == "fixed_paper_seed"
        else DEFAULT_PAPER_SEED
    )
    paper_seed = (
        float(args.paper_init_mchi_sq) if args.paper_init_mchi_sq is not None else float(default_seed[0]),
        float(args.paper_init_L) if args.paper_init_L is not None else float(default_seed[1]),
        float(args.paper_init_W) if args.paper_init_W is not None else float(default_seed[2]),
    )

    ensemble_rows = collect_ensemble_dirs(
        args.dir,
        mv_source=args.wilson_v_source,
        ps_source=args.wilson_ps_source,
    )
    wilson_points, bootstrap_point_sets, n_boot = build_wilson_bootstrap_point_sets(
        ensemble_rows,
        args.observable,
    )

    wilson_fit_linear = fit_wilson_complete_model_linear(
        wilson_points,
        fit_label=cfg["linear_label"],
    )
    if args.wilson_init_strategy == "paper_tnc":
        start_params, leading_fit = build_paper_strategy_start(
            wilson_points,
            seed=paper_seed,
            solver_method="TNC",
        )
        start_p0 = [
            start_params["m_M_chi_sq"],
            start_params["L_m_M"],
            start_params["Q_m_M"],
            start_params["W_m_M"],
            start_params["R_m_M"],
            start_params["C_m_M"],
        ]
        wilson_fit_nonlinear = fit_wilson_complete_model_nonlinear_tnc(
            wilson_points,
            start_p0,
            fit_label=cfg["fit_label"],
            solver_method="TNC",
        )
        wilson_fit_nonlinear["initialization_strategy"] = "paper_tnc"
        wilson_fit_nonlinear["leading_fit_start"] = leading_fit
    elif args.wilson_init_strategy == "fixed_paper_seed":
        start_params = build_fixed_paper_seed_start(seed=paper_seed)
        start_p0 = [
            start_params["m_M_chi_sq"],
            start_params["L_m_M"],
            start_params["Q_m_M"],
            start_params["W_m_M"],
            start_params["R_m_M"],
            start_params["C_m_M"],
        ]
        wilson_fit_nonlinear = fit_wilson_complete_model_nonlinear_tnc(
            wilson_points,
            start_p0,
            fit_label=cfg["fit_label"],
            solver_method="TNC",
        )
        wilson_fit_nonlinear["initialization_strategy"] = "fixed_paper_seed"
        wilson_fit_nonlinear["fixed_seed_start"] = {
            "m_M_chi_sq": float(start_params["m_M_chi_sq"]),
            "L_m_M": float(start_params["L_m_M"]),
            "Q_m_M": float(start_params["Q_m_M"]),
            "W_m_M": float(start_params["W_m_M"]),
            "R_m_M": float(start_params["R_m_M"]),
            "C_m_M": float(start_params["C_m_M"]),
        }
    else:
        start_params = derive_wilson_start_parameters(wilson_fit_linear)
        if cfg["central_linearized_uses_shared_p0"]:
            wilson_fit_nonlinear = fit_wilson_complete_model_nonlinear(
                wilson_points,
                p0=DEFAULT_WILSON_SHARED_NONLINEAR_P0,
                fit_label=cfg["fit_label"],
            )
        else:
            wilson_fit_nonlinear = fit_wilson_complete_model_nonlinear(
                wilson_points,
                wilson_fit_linear,
                fit_label=cfg["fit_label"],
            )

    wilson_fit_bootstrap = fit_wilson_bootstrap_summary(
        bootstrap_point_sets,
        wilson_points,
        start_params,
        args.observable,
        init_strategy=args.wilson_init_strategy,
    )

    save_results(
        output_data=args.output_data,
        observable=args.observable,
        ensemble_rows=ensemble_rows,
        wilson_points=wilson_points,
        wilson_fit_linear=wilson_fit_linear,
        wilson_fit_nonlinear=wilson_fit_nonlinear,
        wilson_fit_bootstrap=wilson_fit_bootstrap,
        n_boot=n_boot,
        init_strategy=args.wilson_init_strategy,
        paper_seed=paper_seed,
        starting_parameters=start_params,
        mv_source=args.wilson_v_source,
        ps_source=args.wilson_ps_source,
    )


if __name__ == "__main__":
    main()
