#!/usr/bin/env python3
"""
Bootstrap chiral-continuum fit for the vector meson mass.

This release script reconstructs per-ensemble MDWF bootstrap points from the
stored spectrum and Wilson-flow summaries, performs the MDWF bootstrap fit,
optionally combines in Wilson reference data, and writes both the plot and a
compact JSON record of the fit inputs and outputs.
"""
import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.ticker import ScalarFormatter

from shared_continuum_models import (
    derive_dw2_start_parameters,
    derive_wilson_start_parameters,
    dw2_physical_continuum_line_and_band as continuum_line_and_band_dw2,
    dw2_physical_model,
    fit_dw2_continuum_linear,
    fit_dw2_continuum_nonlinear,
    fit_wilson_complete_model_linear,
    fit_wilson_complete_model_nonlinear,
    wilson_physical_continuum_line_and_band as wilson_continuum_line_and_band,
)
from shared_fit_serialization import (
    linear_fit_to_json_dict,
    physical_fit_to_json_dict,
    to_serializable,
)

plt.style.use("tableau-colorblind10")
DEFAULT_WILSON_SHARED_NONLINEAR_P0 = [0.320, 2.9, -20.0, -0.183, 0.03, -1.0]
MDWF_FIT_COLOR = "#0072B2"
WILSON_FIT_COLOR = "#D55E00"


def mdwf_physical_formula(fix_q_to_zero=False):
    if fix_q_to_zero:
        return (
            r"$(w_0 m_{\rm V})^2 = (w_0 m^\chi_{\rm V})^2"
            r"(1 + L_{m,\rm V}(w_0 m_{\rm PS})^2)$"
            "\n"
            r"$\qquad\qquad + W_{m,\rm V}(a/w_0)^2$"
        )
    return (
        r"$(w_0 m_{\rm V})^2 = (w_0 m^\chi_{\rm V})^2"
        r"(1 + L_{m,\rm V}(w_0 m_{\rm PS})^2 + Q_{m,\rm V}(w_0 m_{\rm PS})^4)$"
        "\n"
        r"$\qquad\qquad + W_{m,\rm V}(a/w_0)^2$"
    )


def wilson_physical_formula(fix_q_to_zero=False):
    if fix_q_to_zero:
        return (
            r"$(w_0 m_{\rm V})^2 = (w_0 m^\chi_{\rm V})^2"
            r"(1 + L_{m,\rm V}(w_0 m_{\rm PS})^2)$"
            "\n"
            r"$\qquad\qquad + W_{m,\rm V}(a/w_0) + R_{m,\rm V}(a/w_0)^2"
            r" + C_{m,\rm V}(a/w_0)(w_0 m_{\rm PS})^4$"
        )
    return (
        r"$(w_0 m_{\rm V})^2 = (w_0 m^\chi_{\rm V})^2"
        r"(1 + L_{m,\rm V}(w_0 m_{\rm PS})^2 + Q_{m,\rm V}(w_0 m_{\rm PS})^4)$"
        "\n"
        r"$\qquad\qquad + W_{m,\rm V}(a/w_0) + R_{m,\rm V}(a/w_0)^2"
        r" + C_{m,\rm V}(a/w_0)(w_0 m_{\rm PS})^4$"
    )


def mdwf_bootstrap_label(fix_q_to_zero=False):
    if fix_q_to_zero:
        return (
            r"MDWF bootstrap: $(w_0 m_{\rm V})^2 = (w_0 m^\chi_{\rm V})^2"
            r"(1 + L_{m,\rm V}(w_0 m_{\rm PS})^2) + W_{m,\rm V}(a/w_0)^2$"
        )
    return (
        r"MDWF bootstrap: $(w_0 m_{\rm V})^2 = (w_0 m^\chi_{\rm V})^2"
        r"(1 + L_{m,\rm V}(w_0 m_{\rm PS})^2 + Q_{m,\rm V}(w_0 m_{\rm PS})^4)"
        r" + W_{m,\rm V}(a/w_0)^2$"
    )


def mdwf_fit_label(fix_q_to_zero=False):
    prefix = "MDWF"
    formula = mdwf_physical_formula(fix_q_to_zero).replace("\n", " ")
    return f"{prefix}: {formula}"


def wilson_fit_label(fix_q_to_zero=False):
    if fix_q_to_zero:
        return (
            r"Wilson: $(w_0 m_{\rm V})^2 = (w_0 m^\chi_{\rm V})^2"
            r"(1 + L_{m,\rm V}(w_0 m_{\rm PS})^2)$"
            r" $+ W_{m,\rm V}(a/w_0) + R_{m,\rm V}(a/w_0)^2"
            r" + C_{m,\rm V}(a/w_0)(w_0 m_{\rm PS})^4$"
        )
    prefix = "Wilson"
    formula = wilson_physical_formula(fix_q_to_zero).replace("\n", " ")
    return f"{prefix}: {formula}"


def resolve_fit_mode(mode_arg, legacy_fix_flag):
    if mode_arg is not None:
        return mode_arg
    return "q0" if legacy_fix_flag else "full"


def read_json_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as handle:
            text = handle.read()
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r"(?<=\})\s*(?=\{)", ",\n", text)
        try:
            return json.loads(repaired)
        except Exception as exc:
            raise ValueError(f"Could not read JSON file: {filename}\n{exc}") from exc
    except Exception as exc:
        raise ValueError(f"Could not read JSON file: {filename}\n{exc}") from exc


def parse_pair(obj, key, filename):
    if key not in obj:
        raise ValueError(f"Missing key '{key}' in '{filename}'")

    val = obj[key]
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(val[0]), float(val[1])
    if isinstance(val, dict) and "mean" in val and "sdev" in val:
        return float(val["mean"]), float(val["sdev"])

    raise ValueError(
        f"Key '{key}' in '{filename}' must be either [value, error] "
        f"or {{'mean': ..., 'sdev': ...}}, got: {val}"
    )


def read_wilson_spectrum_json(filename):
    data = read_json_file(filename)
    if not isinstance(data, list):
        raise ValueError(
            f"Wilson spectrum JSON '{filename}' must contain a list of ensembles."
        )

    out = {}
    for row in data:
        if "Ensemble" not in row:
            raise ValueError(f"Missing 'Ensemble' key in '{filename}'")

        ens = row["Ensemble"]
        am_ps, am_ps_err = parse_pair(row, "amps", filename)
        am_v, am_v_err = parse_pair(row, "amv", filename)

        out[ens] = {
            "beta": float(row["beta"]),
            "m0": float(row["m0"]),
            "am_ps": am_ps,
            "am_ps_err": am_ps_err,
            "am_v": am_v,
            "am_v_err": am_v_err,
        }

    return out


def read_wilson_wflow_json(filename):
    data = read_json_file(filename)
    if not isinstance(data, list):
        raise ValueError(
            f"Wilson wflow JSON '{filename}' must contain a list of ensembles."
        )

    out = {}
    for row in data:
        if "Ensemble" not in row:
            raise ValueError(f"Missing 'Ensemble' key in '{filename}'")

        ens = row["Ensemble"]
        w0a, w0a_err = parse_pair(row, "w0a", filename)
        out[ens] = {
            "beta": float(row["beta"]),
            "m0": float(row["m0"]),
            "w0a": w0a,
            "w0a_err": w0a_err,
        }

    return out


def extract_beta_mass_from_path(path):
    parts = Path(path).parts
    beta = None
    mass = None

    for part in parts:
        if part.startswith("B"):
            try:
                beta = float(part[1:])
            except ValueError:
                pass
        elif part.startswith("M") and mass is None:
            try:
                mass = float(part[1:])
            except ValueError:
                pass

    return beta, mass


def mw0_sq_and_error_from_w0a(am, am_err, w0a, w0a_err):
    x = (am * w0a) ** 2
    dx_dam = 2.0 * am * (w0a**2)
    dx_dw0a = 2.0 * (am**2) * w0a
    var = (dx_dam**2) * (am_err**2) + (dx_dw0a**2) * (w0a_err**2)
    return x, np.sqrt(var)


def a_over_w0_and_error(w0a, w0a_err):
    z = 1.0 / w0a
    z_err = w0a_err / (w0a**2)
    return z, z_err


def square_with_error(z, z_err):
    q = z**2
    q_err = 2.0 * abs(z) * z_err
    return q, q_err


def collect_wilson_points(spectrum_file, wflow_file):
    wilson_spec = read_wilson_spectrum_json(spectrum_file)
    wilson_wflow = read_wilson_wflow_json(wflow_file)

    common_ensembles = sorted(set(wilson_spec) & set(wilson_wflow))
    if not common_ensembles:
        raise ValueError(
            "No common ensembles found between Wilson spectrum and wflow JSON files."
        )

    wilson_points = []
    for ens in common_ensembles:
        srow = wilson_spec[ens]
        wrow = wilson_wflow[ens]

        if abs(srow["beta"] - wrow["beta"]) > 1e-12:
            raise ValueError(
                f"Beta mismatch for Wilson ensemble '{ens}': "
                f"{srow['beta']} vs {wrow['beta']}"
            )
        if abs(srow["m0"] - wrow["m0"]) > 1e-12:
            raise ValueError(
                f"m0 mismatch for Wilson ensemble '{ens}': "
                f"{srow['m0']} vs {wrow['m0']}"
            )

        x, xerr = mw0_sq_and_error_from_w0a(
            srow["am_ps"], srow["am_ps_err"], wrow["w0a"], wrow["w0a_err"]
        )
        y, yerr = mw0_sq_and_error_from_w0a(
            srow["am_v"], srow["am_v_err"], wrow["w0a"], wrow["w0a_err"]
        )

        z_lin, z_lin_err = a_over_w0_and_error(wrow["w0a"], wrow["w0a_err"])
        z_quad, z_quad_err = square_with_error(z_lin, z_lin_err)

        wilson_points.append(
            {
                "Ensemble": ens,
                "beta": srow["beta"],
                "m0": srow["m0"],
                "x": x,
                "xerr": xerr,
                "y": y,
                "yerr": yerr,
                "a_over_w0": z_lin,
                "a_over_w0_err": z_lin_err,
                "a_over_w0_sq": z_quad,
                "a_over_w0_sq_err": z_quad_err,
            }
        )

    return wilson_points


def make_wilson_fit_text(fit):
    text = (
        r"$\mathrm{Wilson\ (nonlinear)}:$" "\n"
        r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$" "\n"
        r"$\qquad\qquad + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
    )

    text += "\n" + rf"$m_{{M,\chi}}^2 = {fit['m_M_chi_sq']:.4f} \pm {fit['m_M_chi_sq_err']:.4f}$"
    text += "\n" + rf"$L_{{m_M}} = {fit['L_m_M']:.4f} \pm {fit['L_m_M_err']:.4f}$"
    text += "\n" + rf"$Q_{{m_M}} = {fit['Q_m_M']:.4f} \pm {fit['Q_m_M_err']:.4f}$"
    text += "\n" + rf"$W_{{m_M}} = {fit['W_m_M']:.4f} \pm {fit['W_m_M_err']:.4f}$"
    text += "\n" + rf"$R_{{m_M}} = {fit['R_m_M']:.4f} \pm {fit['R_m_M_err']:.4f}$"
    text += "\n" + rf"$C_{{m_M}} = {fit['C_m_M']:.4f} \pm {fit['C_m_M_err']:.4f}$"

    if fit["dof"] > 0:
        text += "\n" + rf"$\chi^2/\mathrm{{dof}} = {fit['chi2']:.2f}/{fit['dof']}$"

    return text


def print_wilson_fit_summary(fit, title):
    print(f"{title}:")
    if "m_M_chi_sq" in fit:
        print(f"  m_M_chi_sq = {fit['m_M_chi_sq']:.8g} ± {fit['m_M_chi_sq_err']:.3g}")
        print(f"  L_m_M = {fit['L_m_M']:.8g} ± {fit['L_m_M_err']:.3g}")
        print(f"  Q_m_M = {fit['Q_m_M']:.8g} ± {fit['Q_m_M_err']:.3g}")
        print(f"  W_m_M = {fit['W_m_M']:.8g} ± {fit['W_m_M_err']:.3g}")
        print(f"  R_m_M = {fit['R_m_M']:.8g} ± {fit['R_m_M_err']:.3g}")
        print(f"  C_m_M = {fit['C_m_M']:.8g} ± {fit['C_m_M_err']:.3g}")
    else:
        for i, (term, coeff, err) in enumerate(
            zip(fit["basis_terms"], fit["coeffs"], fit["coeff_errs"])
        ):
            name = chr(ord("A") + i)
            print(f"  {name} = {coeff:.8g} ± {err:.3g}   [{term}]")

    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


def print_starting_parameters(title, params):
    print(f"{title}:")
    for key, value in params.items():
        print(f"  {key} = {value:.8g}")


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


def _require_key(obj, keys, filename):
    cur = obj
    for key in keys:
        if key not in cur:
            joined = ".".join(keys)
            raise ValueError(f"Missing key '{joined}' in '{filename}'")
        cur = cur[key]
    return cur


def read_spectrum_bootstrap_json(filename):
    data = read_json_file(filename)
    bootstrap = _require_key(data, ["bootstrap"], filename)
    pp = _require_key(data, ["results", "bootstrap_fit", "PP", "samples"], filename)
    vv = _require_key(data, ["results", "bootstrap_fit", "VV", "samples"], filename)

    return {
        "bootstrap": bootstrap,
        "pp_samples": pp,
        "vv_samples": vv,
    }


def read_wflow_bootstrap_json(filename):
    data = read_json_file(filename)
    bootstrap = _require_key(data, ["bootstrap", "w0"], filename)
    samples = _require_key(data, ["bootstrap", "w0", "samples"], filename)
    summary = _require_key(data, ["summary"], filename)
    return {
        "bootstrap": bootstrap,
        "w0_samples": samples,
        "summary": summary,
    }


def read_precomputed_wilson_bootstrap_json(filename):
    # Reuse the Wilson bootstrap summary when it is already available so the
    # release workflow does not need to refit Wilson data unnecessarily.
    data = read_json_file(filename)
    wilson_points = _require_key(data, ["points", "wilson"], filename)
    fit_block = _require_key(data, ["fits", "wilson_physical"], filename)
    return {
        "ensembles": data.get("ensembles", []),
        "wilson_points": wilson_points,
        "linearized": _require_key(fit_block, ["linearized"], filename),
        "starting_parameters": _require_key(fit_block, ["starting_parameters"], filename),
        "central_nonlinear": fit_block.get("central_nonlinear"),
        "bootstrap_summary": _require_key(fit_block, ["bootstrap_summary"], filename),
    }


def ensure_bootstrap_alignment(spec_bootstrap, flow_bootstrap, spec_path, flow_path):
    # The MDWF observable point is built from spectrum and wflow bootstrap
    # replicas, so both inputs must refer to the same bootstrap ensemble.
    checks = [
        ("path_key", spec_bootstrap.get("path_key"), flow_bootstrap.get("path_key")),
        ("seed", spec_bootstrap.get("seed"), flow_bootstrap.get("seed")),
        ("n_boot", spec_bootstrap.get("n_boot"), flow_bootstrap.get("n_boot")),
        ("cfg_numbers", spec_bootstrap.get("cfg_numbers"), flow_bootstrap.get("cfg_numbers")),
        ("boot_idx", spec_bootstrap.get("boot_idx"), flow_bootstrap.get("boot_idx")),
    ]
    for name, lhs, rhs in checks:
        if lhs != rhs:
            raise ValueError(
                "Bootstrap mismatch between spectrum and wflow for\n"
                f"  spectrum: {spec_path}\n"
                f"  wflow:    {flow_path}\n"
                f"Field '{name}' differs. This usually means the selected "
                "configurations or bootstrap ensemble are not aligned."
            )


def build_dw_bootstrap_ensemble(spec_path, wflow_path):
    # Construct one MDWF ensemble point together with all of its bootstrap
    # replicas so later fitting stages can work from a single validated object.
    beta, mass = extract_beta_mass_from_path(spec_path)
    if beta is None or mass is None:
        raise ValueError(f"Could not extract beta/mass from path: {spec_path}")

    spec = read_spectrum_bootstrap_json(spec_path)
    wflow = read_wflow_bootstrap_json(wflow_path)
    ensure_bootstrap_alignment(spec["bootstrap"], wflow["bootstrap"], spec_path, wflow_path)

    pp_samples = spec["pp_samples"]
    vv_samples = spec["vv_samples"]
    w0_samples = wflow["w0_samples"]
    n_boot = int(spec["bootstrap"]["n_boot"])

    if len(pp_samples) != n_boot or len(vv_samples) != n_boot or len(w0_samples) != n_boot:
        raise ValueError(
            f"Inconsistent bootstrap sample count for ensemble:\n"
            f"  spectrum: {spec_path}\n"
            f"  wflow:    {wflow_path}"
        )

    replica_points = []
    for b in range(n_boot):
        pp_b = pp_samples[b]
        vv_b = vv_samples[b]
        w0_b = w0_samples[b]

        if pp_b is None or vv_b is None or w0_b is None:
            replica_points.append(None)
            continue

        m_ps = pp_b.get("m_ps")
        m_v = vv_b.get("m_v")
        w0 = w0_b.get("w0")
        if m_ps is None or m_v is None or w0 is None:
            replica_points.append(None)
            continue

        if w0 == 0.0:
            replica_points.append(None)
            continue

        x = float((m_ps * w0) ** 2)
        y = float((m_v * w0) ** 2)
        a_over_w0 = float(1.0 / w0)
        replica_points.append(
            {
                "index": int(b),
                "beta": beta,
                "m0": mass,
                "x": x,
                "y": y,
                "a_over_w0": a_over_w0,
                "w0": float(w0),
                "m_ps": float(m_ps),
                "m_v": float(m_v),
            }
        )

    valid = [sample for sample in replica_points if sample is not None]
    if not valid:
        raise RuntimeError(f"No valid bootstrap replicas for ensemble: {spec_path}")

    x_stats = summary_stats([sample["x"] for sample in valid])
    y_stats = summary_stats([sample["y"] for sample in valid])
    a_stats = summary_stats([sample["a_over_w0"] for sample in valid])

    point = {
        "beta": beta,
        "m0": mass,
        "x": x_stats["mean"],
        "xerr": x_stats["sdev"],
        "y": y_stats["mean"],
        "yerr": y_stats["sdev"],
        "a_over_w0": a_stats["mean"],
        "a_over_w0_err": a_stats["sdev"],
        "a_over_w0_sq": a_stats["mean"] ** 2,
        "a_over_w0_sq_err": 2.0 * abs(a_stats["mean"]) * a_stats["sdev"],
    }

    n_failed = sum(sample is None for sample in replica_points)

    return {
        "point": point,
        "bootstrap_meta": {
            "n_boot": int(n_boot),
            "n_success": int(len(valid)),
            "n_failed": int(n_failed),
        },
        "bootstrap_samples": replica_points,
        "paths": {"spectrum": spec_path, "wflow": wflow_path},
    }


def collect_dw_bootstrap_ensembles(spectrum_files, wflow_files):
    # Align the per-ensemble bootstrap replicas across all selected MDWF inputs
    # so replica ``b`` always corresponds to the same bootstrap draw everywhere.
    if len(spectrum_files) != len(wflow_files):
        raise ValueError("Number of --spectrum files must equal number of --wflow files.")

    ensembles = [
        build_dw_bootstrap_ensemble(spec_path, wflow_path)
        for spec_path, wflow_path in zip(spectrum_files, wflow_files)
    ]

    points = [entry["point"] for entry in ensembles]
    n_boot_values = {int(entry["bootstrap_meta"]["n_boot"]) for entry in ensembles}
    if len(n_boot_values) != 1:
        raise ValueError("All MDWF ensembles must have the same number of bootstrap replicas.")

    n_boot = n_boot_values.pop()
    bootstrap_point_sets = []
    failures = []

    for b in range(n_boot):
        point_set = []
        missing = []
        for entry in ensembles:
            sample = entry["bootstrap_samples"][b]
            if sample is None:
                missing.append(entry["paths"]["spectrum"])
                continue
            point_set.append(
                {
                    "beta": sample["beta"],
                    "m0": sample["m0"],
                    "x": sample["x"],
                    "y": sample["y"],
                    "a_over_w0": sample["a_over_w0"],
                    "yerr": entry["point"]["yerr"],
                }
            )

        if missing:
            failures.append(
                {
                    "index": int(b),
                    "reason": "missing ensemble sample",
                    "paths": missing,
                }
            )
            bootstrap_point_sets.append(None)
            continue

        bootstrap_point_sets.append(point_set)

    return points, bootstrap_point_sets, failures


def build_wilson_bootstrap_point_sets_from_precomputed(
    ensemble_rows,
    *,
    sample_key="mv_bootstrap_samples",
    point_key="mv_point",
):
    if not ensemble_rows:
        raise ValueError("No Wilson ensemble rows available in precomputed bootstrap JSON.")

    n_boot = min(int(row.get("n_boot_available", 0)) for row in ensemble_rows)
    if n_boot <= 0:
        raise ValueError("No shared Wilson bootstrap replicas available.")

    points = [row[point_key] for row in ensemble_rows]
    point_sets = []
    for iboot in range(n_boot):
        point_set = []
        for row in ensemble_rows:
            sample = row[sample_key][iboot]
            point_set.append(
                {
                    "beta": sample["beta"],
                    "m0": sample["m0"],
                    "x": sample["x"],
                    "y": sample["y"],
                    "a_over_w0": sample["a_over_w0"],
                    "a_over_w0_sq": sample["a_over_w0_sq"],
                    "yerr": row[point_key]["yerr"],
                }
            )
        point_sets.append(point_set)
    return points, point_sets, n_boot


def fit_dw2_bootstrap_replica(points, p0, fix_q_to_zero=False):
    fit = fit_dw2_continuum_nonlinear(
        points,
        p0=p0,
        fit_label=mdwf_fit_label(fix_q_to_zero),
        fix_Q_to_zero=fix_q_to_zero,
    )
    params = np.array(
        [
            fit["m_M_chi_sq"],
            fit["L_m_M"],
            fit["Q_m_M"],
            fit["R_m_M"],
        ],
        dtype=float,
    )
    return params, np.asarray(fit["cov"], dtype=float), float(fit["chi2"])


def bootstrap_param_rows(samples):
    rows = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        try:
            rows.append(
                [
                    float(sample["m_M_chi_sq"]),
                    float(sample["L_m_M"]),
                    float(sample["Q_m_M"]),
                    float(sample["R_m_M"]),
                ]
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return np.empty((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def bootstrap_summary_predictions(dw_points, mean_params):
    # Evaluate the fitted bootstrap-summary model on the central MDWF points and
    # keep the resulting residuals in the JSON for release-time inspection.
    mean_x = np.array([p["x"] for p in dw_points], dtype=float)
    mean_a = np.array([p["a_over_w0"] for p in dw_points], dtype=float)
    mean_y = np.array([p["y"] for p in dw_points], dtype=float)
    mean_ye = np.array([p["yerr"] for p in dw_points], dtype=float)
    dof = int(len(mean_y) - len(mean_params))

    parameter_average_model_y = dw2_physical_model((mean_x, mean_a), *mean_params)
    parameter_average_residuals = mean_y - parameter_average_model_y
    parameter_average_chi2 = float(
        np.sum((parameter_average_residuals / mean_ye) ** 2)
    )

    return {
        "x": mean_x,
        "a_over_w0": mean_a,
        "observed_y": mean_y,
        "observed_yerr": mean_ye,
        "parameter_average_model_y": parameter_average_model_y,
        "parameter_average_residuals": parameter_average_residuals,
        "parameter_average_chi2": parameter_average_chi2,
        "dof": dof,
    }


def fit_wilson_bootstrap_replica(points, p0, *, fix_q_to_zero=False):
    fit = fit_wilson_complete_model_nonlinear(
        points,
        p0=p0,
        fit_label=wilson_fit_label(fix_q_to_zero),
        fix_Q_to_zero=fix_q_to_zero,
    )
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
    *,
    fix_q_to_zero=False,
):
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

    for b, point_set in enumerate(bootstrap_point_sets):
        try:
            popt, chi2 = fit_wilson_bootstrap_replica(
                point_set,
                default_p0,
                fix_q_to_zero=fix_q_to_zero,
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
            success_rows.append((int(b), popt, chi2, sample))
            samples.append(sample)
        except Exception as exc:
            failures.append({"index": int(b), "error": str(exc)})
            samples.append(None)

    if not success_rows:
        raise RuntimeError("All bootstrap Wilson continuum fits failed.")

    params = np.asarray([row[1] for row in success_rows], dtype=float)
    chi2_values = [row[2] for row in success_rows]
    mean_params = np.mean(params, axis=0)
    cov = np.cov(params, rowvar=False, ddof=1) if params.shape[0] > 1 else np.zeros((6, 6), dtype=float)

    mean_x = np.asarray([point["x"] for point in mean_points], dtype=float)
    mean_a = np.asarray([point["a_over_w0"] for point in mean_points], dtype=float)
    mean_y = np.asarray([point["y"] for point in mean_points], dtype=float)
    mean_ye = np.asarray([point["yerr"] for point in mean_points], dtype=float)

    if fix_q_to_zero:
        model_y = (
            mean_params[0] * (1.0 + mean_params[1] * mean_x)
            + mean_params[3] * mean_a
            + mean_params[4] * mean_a**2
            + mean_params[5] * mean_a * mean_x
        )
        final_dof = int(len(mean_y) - 5 - 1)
    else:
        model_y = (
            mean_params[0] * (1.0 + mean_params[1] * mean_x + mean_params[2] * mean_x**2)
            + mean_params[3] * mean_a
            + mean_params[4] * mean_a**2
            + mean_params[5] * mean_a * mean_x
        )
        final_dof = int(len(mean_y) - 6 - 1)

    central_residuals = mean_y - model_y
    final_chi2 = float(np.sum((central_residuals / mean_ye) ** 2))

    errs = np.sqrt(np.diag(cov))
    return {
        "model_key": "wilson_physical_bootstrap",
        "stage": "bootstrap_summary",
        "label": wilson_fit_label(fix_q_to_zero).replace("Wilson:", "Wilson bootstrap:"),
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
        "fix_Q_to_zero": bool(fix_q_to_zero),
        "bootstrap_meta": {
            "n_requested": int(len(bootstrap_point_sets)),
            "n_success": int(params.shape[0]),
            "n_failed": int(len(bootstrap_point_sets) - params.shape[0]),
            "n_rejected_outliers": 0,
            "mean_chi2": float(np.mean(chi2_values)),
            "sdev_chi2": float(np.std(chi2_values, ddof=1)) if len(chi2_values) > 1 else 0.0,
        },
        "bootstrap_samples": samples,
        "bootstrap_failures": failures,
    }


def fit_dw2_bootstrap_summary(
    bootstrap_point_sets,
    dw_points,
    central_fit,
    start_params,
    *,
    fix_q_to_zero=False,
):
    # Fit every bootstrap replica and summarize the successful MDWF fits.
    p0 = [
        start_params["m_M_chi_sq"],
        start_params["L_m_M"],
        start_params["Q_m_M"],
        start_params["R_m_M"],
    ]

    samples = []
    failures = []
    success_rows = []

    for b, point_set in enumerate(bootstrap_point_sets):
        if point_set is None:
            samples.append(None)
            continue
        try:
            popt, cov_replica, chi2 = fit_dw2_bootstrap_replica(
                point_set,
                p0,
                fix_q_to_zero=fix_q_to_zero,
            )
            sample = {
                "index": int(b),
                "m_M_chi_sq": float(popt[0]),
                "L_m_M": float(popt[1]),
                "Q_m_M": float(popt[2]),
                "R_m_M": float(popt[3]),
                "parameter_order": ["m_M_chi_sq", "L_m_M", "Q_m_M", "R_m_M"],
                "cov": cov_replica,
                "chi2": chi2,
            }
            success_rows.append((int(b), popt, chi2, sample))
            samples.append(sample)
        except Exception as exc:
            failures.append({"index": int(b), "error": str(exc)})
            samples.append(None)

    if not success_rows:
        raise RuntimeError("All bootstrap MDWF continuum fits failed.")

    params = np.asarray([row[1] for row in success_rows], dtype=float)
    chi2_values = [row[2] for row in success_rows]
    mean_params = np.mean(params, axis=0)
    if params.shape[0] > 1:
        cov = np.cov(params, rowvar=False, ddof=1)
    else:
        cov = np.zeros((4, 4), dtype=float)

    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)

    summary_predictions = bootstrap_summary_predictions(dw_points, mean_params)
    summary_predictions["dof"] = int(len(dw_points) - (3 if fix_q_to_zero else 4))
    mean_model_y = np.asarray(
        summary_predictions["parameter_average_model_y"],
        dtype=float,
    )

    mean_y = np.asarray(summary_predictions["observed_y"], dtype=float)
    mean_ye = np.asarray(summary_predictions["observed_yerr"], dtype=float)
    central_residuals = mean_y - mean_model_y
    final_chi2 = float(np.sum((central_residuals / mean_ye) ** 2))
    final_dof = int(summary_predictions["dof"])

    errs = np.sqrt(np.diag(cov))
    fit = {
        "model_key": "dw2_physical_bootstrap",
        "stage": "bootstrap_summary",
        "label": (
            mdwf_bootstrap_label(fix_q_to_zero)
        ),
        "m_M_chi_sq": float(mean_params[0]),
        "m_M_chi_sq_err": float(errs[0]),
        "L_m_M": float(mean_params[1]),
        "L_m_M_err": float(errs[1]),
        "Q_m_M": float(mean_params[2]),
        "Q_m_M_err": float(errs[2]),
        "R_m_M": float(mean_params[3]),
        "R_m_M_err": float(errs[3]),
        "cov": cov,
        "chi2": final_chi2,
        "dof": final_dof,
        "fix_Q_to_zero": bool(fix_q_to_zero),
        "summary_predictions": summary_predictions,
        "bootstrap_meta": {
            "n_requested": int(len(bootstrap_point_sets)),
            "n_success": int(params.shape[0]),
            "n_failed": int(len(bootstrap_point_sets) - params.shape[0]),
            "n_rejected_outliers": 0,
            "mean_chi2": float(np.mean(chi2_values)),
            "sdev_chi2": float(np.std(chi2_values, ddof=1)) if len(chi2_values) > 1 else 0.0,
        },
        "bootstrap_samples": samples,
        "bootstrap_failures": failures,
    }
    return fit


def make_dw2_bootstrap_fit_text(fit):
    text = (
        r"$\mathrm{MDWF\ (bootstrap)}:$" "\n"
        r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4) + R_{m_M} a^2$" "\n"
        rf"$m_{{M,\chi}}^2 = {fit['m_M_chi_sq']:.4f} \pm {fit['m_M_chi_sq_err']:.4f}$" "\n"
        rf"$L_{{m_M}} = {fit['L_m_M']:.4f} \pm {fit['L_m_M_err']:.4f}$" "\n"
        rf"$Q_{{m_M}} = {fit['Q_m_M']:.4f} \pm {fit['Q_m_M_err']:.4f}$" "\n"
        rf"$R_{{m_M}} = {fit['R_m_M']:.4f} \pm {fit['R_m_M_err']:.4f}$"
    )
    if fit["dof"] > 0:
        text += "\n" + rf"$\chi^2/\mathrm{{dof}} = {fit['chi2']:.2f}/{fit['dof']}$"
    return text


def plot_points_and_fits(
    dw_points,
    dw_fit,
    output_file,
    wilson_points=None,
    wilson_fit=None,
    dw_fit_central=None,
    wilson_fit_central=None,
    set_ylim=None,
):
    # Plot the ensemble points together with the central fit curves that are
    # meant to appear in the release figure. The bootstrap summary itself is
    # written to JSON rather than drawn as a separate curve choice.
    wilson_points = wilson_points or []
    all_betas = sorted(
        {p["beta"] for p in dw_points}.union({p["beta"] for p in wilson_points})
    )
    beta_colors = {b: f"C{i % 10}" for i, b in enumerate(all_betas)}

    fig, ax = plt.subplots(figsize=(8.4, 5.0), layout="constrained")

    for p in dw_points:
        ax.errorbar(
            p["x"],
            p["y"],
            xerr=p["xerr"],
            yerr=p["yerr"],
            fmt="o",
            linestyle="none",
            color=beta_colors[p["beta"]],
            markerfacecolor=beta_colors[p["beta"]],
            markeredgecolor=beta_colors[p["beta"]],
            markersize=5,
            capsize=2,
        )

    for p in wilson_points:
        ax.errorbar(
            p["x"],
            p["y"],
            xerr=p["xerr"],
            yerr=p["yerr"],
            fmt="s",
            linestyle="none",
            color=beta_colors[p["beta"]],
            markerfacecolor="none",
            markeredgecolor=beta_colors[p["beta"]],
            markersize=5,
            capsize=2,
        )

    x_all = [p["x"] for p in dw_points]
    x_all.extend(p["x"] for p in wilson_points)
    x_grid = np.linspace(0.0, 1.05 * max(x_all), 400)

    if wilson_fit is not None and wilson_points:
        central_wilson_fit = wilson_fit_central if wilson_fit_central is not None else wilson_fit
        if central_wilson_fit is not None:
            y_wilson_central, y_wilson_central_err = wilson_continuum_line_and_band(
                x_grid,
                central_wilson_fit,
            )
            ax.plot(
                x_grid,
                y_wilson_central,
                linestyle="-",
                color=WILSON_FIT_COLOR,
                linewidth=0.8,
                alpha=0.9,
            )
            ax.fill_between(
                x_grid,
                y_wilson_central - y_wilson_central_err,
                y_wilson_central + y_wilson_central_err,
                color=WILSON_FIT_COLOR,
                alpha=0.12,
                linewidth=0,
            )

    central_dw_fit = dw_fit_central if dw_fit_central is not None else dw_fit
    if central_dw_fit is not None:
        y_dw_central, y_dw_central_err = continuum_line_and_band_dw2(x_grid, central_dw_fit)
        ax.plot(
            x_grid,
            y_dw_central,
            linestyle="-",
            color=MDWF_FIT_COLOR,
            linewidth=0.8,
            alpha=0.9,
        )
        ax.fill_between(
            x_grid,
            y_dw_central - y_dw_central_err,
            y_dw_central + y_dw_central_err,
            color=MDWF_FIT_COLOR,
            alpha=0.14,
            linewidth=0,
        )

    ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")
    ax.set_ylabel(r"$(m_{\rm V} w_0)^2$")
    ax.set_xlim(0.0, float(x_grid[-1]))
    if set_ylim is None:
        ax.set_ylim(bottom=0.0)
    else:
        ymin, ymax = set_ylim
        kwargs = {}
        if ymin is not None:
            kwargs["bottom"] = ymin
        if ymax is not None:
            kwargs["top"] = ymax
        ax.set_ylim(**kwargs)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    wilson_betas = sorted({p["beta"] for p in wilson_points})
    dw_betas = sorted({p["beta"] for p in dw_points})

    mdwf_formula = mdwf_physical_formula(dw_fit.get("fix_Q_to_zero", False))
    wilson_formula = wilson_physical_formula(
        wilson_fit.get("fix_Q_to_zero", False) if wilson_fit is not None else False
    )

    def _legend_errorbar(color, marker, filled):
        face = color if filled else "none"
        return ax.errorbar(
            [np.nan],
            [np.nan],
            xerr=[1.0],
            yerr=[1.0],
            fmt=marker,
            linestyle="none",
            color=color,
            markerfacecolor=face,
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=4,
            elinewidth=0.6,
            capsize=1.5,
        )

    mdwf_handles = [
        Line2D([], [], linestyle="-", color=MDWF_FIT_COLOR, linewidth=0.8, alpha=0.9),
    ]
    mdwf_labels = ["Central-values fit"]
    for beta in dw_betas:
        mdwf_handles.append(_legend_errorbar(beta_colors[beta], "o", filled=True))
        mdwf_labels.append(rf"$\beta={beta}$")

    mdwf_legend = ax.legend(
        mdwf_handles,
        mdwf_labels,
        title="DWF fitting model:\n" + mdwf_formula,
        loc="upper left",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.9,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=1.5,
        handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.35, yerr_size=0.35)},
    )
    mdwf_legend.get_frame().set_edgecolor("0.8")
    mdwf_legend.get_title().set_multialignment("center")
    ax.add_artist(mdwf_legend)

    if wilson_fit is not None and wilson_points:
        wilson_handles = [
            Line2D([], [], linestyle="-", color=WILSON_FIT_COLOR, linewidth=0.8, alpha=0.9),
        ]
        wilson_labels = ["Central-values fit"]
        for beta in wilson_betas:
            wilson_handles.append(_legend_errorbar(beta_colors[beta], "s", filled=False))
            wilson_labels.append(rf"$\beta={beta}$")
        wilson_legend = ax.legend(
            wilson_handles,
            wilson_labels,
            title="Wilson fitting model:\n" + wilson_formula,
            loc="lower right",
            fontsize=9,
            title_fontsize=9,
            framealpha=0.9,
            borderpad=0.55,
            labelspacing=0.45,
            handlelength=1.5,
            handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.35, yerr_size=0.35)},
        )
        wilson_legend.get_frame().set_edgecolor("0.8")
        wilson_legend.get_title().set_multialignment("center")
        ax.add_artist(wilson_legend)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_file, dpi=300)
    plt.close()


def bootstrap_fit_to_json_dict(fit):
    # Keep the bootstrap bookkeeping alongside the usual fit parameters so the
    # release JSON remains self-contained.
    out = physical_fit_to_json_dict(fit)
    out["summary_predictions"] = fit.get("summary_predictions")
    out["bootstrap_meta"] = fit["bootstrap_meta"]
    out["bootstrap_samples"] = fit["bootstrap_samples"]
    out["bootstrap_failures"] = fit["bootstrap_failures"]
    out["chiral_continuum_limit"] = {
        "mvw0_sq": {
            "mean": fit["m_M_chi_sq"],
            "sdev": fit["m_M_chi_sq_err"],
        },
        "mvw0": {
            "mean": float(np.sqrt(fit["m_M_chi_sq"])) if fit["m_M_chi_sq"] >= 0.0 else None,
            "sdev": (
                float(0.5 * fit["m_M_chi_sq_err"] / np.sqrt(fit["m_M_chi_sq"]))
                if fit["m_M_chi_sq"] > 0.0
                else None
            ),
        },
    }
    return to_serializable(out)


def save_fit_results_json(
    output_data,
    dw_points,
    bootstrap_point_sets,
    dw_fit_linear,
    dw_fit_central,
    dw_fit_bootstrap,
    wilson_points=None,
    wilson_fit_linear=None,
    wilson_fit_nonlinear=None,
    wilson_fit_bootstrap=None,
    wilson_fit_starting_parameters=None,
):
    # Store the selected points, fit inputs, and final summaries in one JSON
    # payload so downstream tables and plots can reuse the same source of truth.
    output_dir = os.path.dirname(output_data)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "n_dw_points_used": len(dw_points),
        "n_dw_bootstrap_requested": len(bootstrap_point_sets),
        "n_dw_bootstrap_success": dw_fit_bootstrap["bootstrap_meta"]["n_success"],
        "n_dw_bootstrap_failed": dw_fit_bootstrap["bootstrap_meta"]["n_failed"],
        "points": {
            "dw": to_serializable(dw_points),
            "wilson": to_serializable(wilson_points or []),
        },
        "fits": {
            "dw2": {
                "linearized": linear_fit_to_json_dict(dw_fit_linear),
                "starting_parameters": to_serializable(
                    derive_dw2_start_parameters(dw_fit_linear)
                ),
                "central_nonlinear": physical_fit_to_json_dict(dw_fit_central),
                "bootstrap_summary": bootstrap_fit_to_json_dict(dw_fit_bootstrap),
            },
        },
    }

    if wilson_points is not None and wilson_fit_linear is not None:
        payload["n_wilson_points_used"] = len(wilson_points)
        wilson_payload = {
            "linearized": linear_fit_to_json_dict(wilson_fit_linear),
            "starting_parameters": to_serializable(
                wilson_fit_starting_parameters
                if wilson_fit_starting_parameters is not None
                else derive_wilson_start_parameters(wilson_fit_linear)
            ),
        }
        if wilson_fit_nonlinear is not None:
            wilson_payload["central_nonlinear"] = physical_fit_to_json_dict(
                wilson_fit_nonlinear
            )
        if wilson_fit_bootstrap is not None:
            wilson_payload["bootstrap_summary"] = to_serializable(wilson_fit_bootstrap)
        payload["fits"]["wilson_physical"] = wilson_payload

    with open(output_data, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"✓ Saved fit data → {output_data}")


def main():
    # The CLI mirrors the Snakemake rule inputs closely so the release workflow
    # can call this script without extra translation layers.
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap MDWF chiral-continuum fit for (m_V w0)^2 vs (m_PS w0)^2. "
            "The displayed MDWF points are bootstrap means with bootstrap standard "
            "deviations, and the MDWF continuum fit is obtained by fitting each "
            "bootstrap replica and summarising the fitted parameters."
        )
    )
    parser.add_argument("--plot_styles", default="")
    parser.add_argument(
        "--spectrum",
        nargs="+",
        required=True,
        help="List of DWF/MDWF spectrum.json files",
    )
    parser.add_argument(
        "--wflow",
        nargs="+",
        required=True,
        help="List of DWF/MDWF wflow_extract.json files",
    )
    parser.add_argument(
        "--spectrum_w",
        default="",
        help="Optional Wilson spectrum table JSON",
    )
    parser.add_argument(
        "--wflow_w",
        default="",
        help="Optional Wilson ensemble/wflow table JSON",
    )
    parser.add_argument(
        "--wilsons_data",
        default="intermediary_data/NF2/spectrum/wilson/wilson_extrapolation.json",
        help="Optional precomputed Wilson bootstrap JSON",
    )
    parser.add_argument(
        "--output_plot",
        "--output_file",
        dest="output_plot",
        required=True,
        help="Output plot file",
    )
    parser.add_argument(
        "--output_data",
        required=True,
        help="Output JSON file storing fit results",
    )
    parser.add_argument(
        "--dw-fit-mode",
        choices=["full", "q0"],
        default=None,
        help="MDWF fit mode: full fit or Q_m_M fixed to zero.",
    )
    parser.add_argument(
        "--wilson-fit-mode",
        choices=["full", "q0"],
        default=None,
        help="Wilson fit mode: full fit or Q_m_M fixed to zero.",
    )
    parser.add_argument(
        "--fix_dw_q_zero",
        action="store_true",
        help="Constrain the MDWF nonlinear fit by fixing Q_m_M = 0.",
    )
    parser.add_argument(
        "--fix_wilson_q_zero",
        action="store_true",
        help=(
            "Constrain the Wilson nonlinear fit by fixing Q_m_M = 0. "
            "If set, the script refits Wilson from raw Wilson inputs instead of "
            "using a precomputed Wilson JSON."
        ),
    )
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    dw_fit_mode = resolve_fit_mode(args.dw_fit_mode, args.fix_dw_q_zero)
    wilson_fit_mode = resolve_fit_mode(args.wilson_fit_mode, args.fix_wilson_q_zero)
    fix_dw_q_zero = dw_fit_mode == "q0"
    fix_wilson_q_zero = wilson_fit_mode == "q0"

    dw_points, bootstrap_point_sets, bootstrap_input_failures = collect_dw_bootstrap_ensembles(
        args.spectrum,
        args.wflow,
    )

    dw_fit_linear = fit_dw2_continuum_linear(dw_points)
    start_params = derive_dw2_start_parameters(dw_fit_linear)
    dw_fit_central = fit_dw2_continuum_nonlinear(
        dw_points,
        dw_fit_linear,
        fit_label=mdwf_fit_label(fix_dw_q_zero),
        fix_Q_to_zero=fix_dw_q_zero,
    )
    dw_fit_bootstrap = fit_dw2_bootstrap_summary(
        bootstrap_point_sets,
        dw_points,
        dw_fit_central,
        start_params,
        fix_q_to_zero=fix_dw_q_zero,
    )
    dw_fit_bootstrap["bootstrap_failures"] = (
        bootstrap_input_failures + dw_fit_bootstrap["bootstrap_failures"]
    )
    dw_fit_bootstrap["bootstrap_meta"]["n_failed"] = len(dw_fit_bootstrap["bootstrap_failures"])

    wilson_points = []
    wilson_fit_linear = None
    wilson_fit = None
    wilson_fit_bootstrap = None
    wilson_fit_starting_parameters = None
    wilson_fit_central = None
    wilsons_json_path = Path(args.wilsons_data) if args.wilsons_data else None
    if fix_wilson_q_zero and not (args.spectrum_w and args.wflow_w):
        raise ValueError(
            "Wilson q0 mode requires raw Wilson inputs via --spectrum_w and --wflow_w."
        )

    if wilsons_json_path and wilsons_json_path.exists() and not fix_wilson_q_zero:
        wilson_data = read_precomputed_wilson_bootstrap_json(str(wilsons_json_path))
        wilson_points = wilson_data["wilson_points"]
        wilson_fit_linear = wilson_data["linearized"]
        wilson_fit = wilson_data["bootstrap_summary"]
        wilson_fit_bootstrap = wilson_data["bootstrap_summary"]
        wilson_fit_starting_parameters = wilson_data["starting_parameters"]
        wilson_fit_central = wilson_data["central_nonlinear"]
    elif wilsons_json_path and wilsons_json_path.exists() and fix_wilson_q_zero:
        wilson_data = read_precomputed_wilson_bootstrap_json(str(wilsons_json_path))
        wilson_points = wilson_data["wilson_points"]
        wilson_fit_linear = fit_wilson_complete_model_linear(wilson_points)
        wilson_fit_starting_parameters = derive_wilson_start_parameters(wilson_fit_linear)
        wilson_fit_central = fit_wilson_complete_model_nonlinear(
            wilson_points,
            p0=DEFAULT_WILSON_SHARED_NONLINEAR_P0,
            fit_label=wilson_fit_label(True),
            fix_Q_to_zero=True,
        )
        _mean_points, wilson_bootstrap_point_sets, _n_boot = (
            build_wilson_bootstrap_point_sets_from_precomputed(
                wilson_data["ensembles"],
                sample_key="mv_bootstrap_samples",
                point_key="mv_point",
            )
        )
        wilson_fit_bootstrap = fit_wilson_bootstrap_summary(
            wilson_bootstrap_point_sets,
            wilson_points,
            wilson_fit_starting_parameters,
            fix_q_to_zero=True,
        )
        wilson_fit = wilson_fit_bootstrap
    elif args.spectrum_w and args.wflow_w:
        wilson_points = collect_wilson_points(args.spectrum_w, args.wflow_w)
        wilson_fit_linear = fit_wilson_complete_model_linear(wilson_points)
        wilson_fit = fit_wilson_complete_model_nonlinear(
            wilson_points,
            p0=DEFAULT_WILSON_SHARED_NONLINEAR_P0,
            fit_label=wilson_fit_label(fix_wilson_q_zero),
            fix_Q_to_zero=fix_wilson_q_zero,
        )
        wilson_fit_starting_parameters = derive_wilson_start_parameters(wilson_fit_linear)
        wilson_fit_central = wilson_fit

    plot_points_and_fits(
        dw_points=dw_points,
        dw_fit=dw_fit_bootstrap,
        wilson_points=wilson_points,
        wilson_fit=wilson_fit,
        dw_fit_central=dw_fit_central,
        wilson_fit_central=wilson_fit_central,
        output_file=args.output_plot,
    )

    save_fit_results_json(
        output_data=args.output_data,
        dw_points=dw_points,
        bootstrap_point_sets=bootstrap_point_sets,
        dw_fit_linear=dw_fit_linear,
        dw_fit_central=dw_fit_central,
        dw_fit_bootstrap=dw_fit_bootstrap,
        wilson_points=wilson_points,
        wilson_fit_linear=wilson_fit_linear,
        wilson_fit_nonlinear=wilson_fit_central,
        wilson_fit_bootstrap=wilson_fit_bootstrap,
        wilson_fit_starting_parameters=wilson_fit_starting_parameters,
    )

    print(f"✓ Saved plot → {args.output_plot}")
    print_starting_parameters(
        "DWF/MDWF starting parameters from linearized fit",
        start_params,
    )
    if wilson_fit_linear is not None and wilson_fit_starting_parameters is not None:
        print_starting_parameters(
            "Wilson initial fit parameters",
            wilson_fit_starting_parameters,
        )
        if wilson_fit_central is not None:
            print_wilson_fit_summary(
                wilson_fit_central,
                "Wilson complete model [central-value fit]",
            )
        if wilson_fit_bootstrap is not None:
            print_wilson_fit_summary(
                wilson_fit_bootstrap,
                "Wilson complete model [bootstrap mean ± std]",
            )
        elif wilson_fit is not None:
            print_wilson_fit_summary(wilson_fit, "Wilson complete model")


if __name__ == "__main__":
    main()
