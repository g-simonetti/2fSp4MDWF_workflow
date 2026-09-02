#!/usr/bin/env python3
"""
Bootstrap chiral-continuum fit for the renormalized pseudoscalar decay constant.

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
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from shared_continuum_models import (
    derive_dw2_start_parameters as _shared_derive_dw2_start_parameters,
    derive_wilson_start_parameters as _shared_derive_wilson_start_parameters,
    dw2_physical_model,
    fit_dw2_continuum_linear as _shared_fit_dw2_continuum_linear,
    fit_dw2_continuum_nonlinear as _shared_fit_dw2_continuum_nonlinear,
    fit_wilson_complete_model_linear as _shared_fit_wilson_complete_model_linear,
    fit_wilson_complete_model_nonlinear as _shared_fit_wilson_complete_model_nonlinear,
    solve_weighted_least_squares,
    wilson_physical_continuum_line_and_band as _shared_wilson_continuum_line_and_band,
)
from shared_fit_serialization import to_serializable

plt.style.use("tableau-colorblind10")
MDWF_FIT_COLOR = "#0072B2"
WILSON_FIT_COLOR = "#D55E00"
PLOT_FITS = [
    "wilson_4",
    "dw2",
]

FPS_WILSON_PHYSICAL_LABEL = (
    r"Wilson: $(w_0 f^{\rm ren}_{\rm PS})^2 = (w_0 f^{\chi}_{\rm PS})^2"
    r"(1 + L_{f,\rm PS}(w_0 m_{\rm PS})^2 + Q_{f,\rm PS}(w_0 m_{\rm PS})^4)$"
    r" $+ W_{f,\rm PS}(a/w_0) + R_{f,\rm PS}(a/w_0)^2"
    r" + C_{f,\rm PS}(a/w_0)(w_0 m_{\rm PS})^4$"
)


def fps_mdwf_physical_formula(fix_q_to_zero=False):
    if fix_q_to_zero:
        return (
            r"$(w_0 f^{\rm ren}_{\rm PS})^2 = (w_0 f^{\chi}_{\rm PS})^2"
            r"(1 + L_{f,\rm PS}(w_0 m_{\rm PS})^2)$"
            "\n"
            r"$\qquad\qquad + W_{f,\rm PS}(a/w_0)^2$"
        )
    return (
        r"$(w_0 f^{\rm ren}_{\rm PS})^2 = (w_0 f^{\chi}_{\rm PS})^2"
        r"(1 + L_{f,\rm PS}(w_0 m_{\rm PS})^2 + Q_{f,\rm PS}(w_0 m_{\rm PS})^4)$"
        "\n"
        r"$\qquad\qquad + W_{f,\rm PS}(a/w_0)^2$"
    )


def fps_wilson_physical_formula(fix_q_to_zero=False):
    if fix_q_to_zero:
        return (
            r"$(w_0 f^{\rm ren}_{\rm PS})^2 = (w_0 f^{\chi}_{\rm PS})^2"
            r"(1 + L_{f,\rm PS}(w_0 m_{\rm PS})^2)$"
            "\n"
            r"$\qquad\qquad + W_{f,\rm PS}(a/w_0) + R_{f,\rm PS}(a/w_0)^2"
            r" + C_{f,\rm PS}(a/w_0)(w_0 m_{\rm PS})^4$"
        )
    return (
        r"$(w_0 f^{\rm ren}_{\rm PS})^2 = (w_0 f^{\chi}_{\rm PS})^2"
        r"(1 + L_{f,\rm PS}(w_0 m_{\rm PS})^2 + Q_{f,\rm PS}(w_0 m_{\rm PS})^4)$"
        "\n"
        r"$\qquad\qquad + W_{f,\rm PS}(a/w_0) + R_{f,\rm PS}(a/w_0)^2"
        r" + C_{f,\rm PS}(a/w_0)(w_0 m_{\rm PS})^4$"
    )


def fps_mdwf_bootstrap_label(fix_q_to_zero=False):
    if fix_q_to_zero:
        return (
            r"MDWF bootstrap: $(w_0 f^{\rm ren}_{\rm PS})^2 = (w_0 f^{\chi}_{\rm PS})^2"
            r"(1 + L_{f,\rm PS}(w_0 m_{\rm PS})^2) + W_{f,\rm PS}(a/w_0)^2$"
        )
    return (
        r"MDWF bootstrap: $(w_0 f^{\rm ren}_{\rm PS})^2 = (w_0 f^{\chi}_{\rm PS})^2"
        r"(1 + L_{f,\rm PS}(w_0 m_{\rm PS})^2 + Q_{f,\rm PS}(w_0 m_{\rm PS})^4)"
        r" + W_{f,\rm PS}(a/w_0)^2$"
    )


def fps_wilson_fit_label(fix_q_to_zero=False):
    if fix_q_to_zero:
        return (
            r"Wilson: $(w_0 f^{\rm ren}_{\rm PS})^2 = (w_0 f^{\chi}_{\rm PS})^2"
            r"(1 + L_{f,\rm PS}(w_0 m_{\rm PS})^2)$"
            r" $+ W_{f,\rm PS}(a/w_0) + R_{f,\rm PS}(a/w_0)^2"
            r" + C_{f,\rm PS}(a/w_0)(w_0 m_{\rm PS})^4$"
        )
    return FPS_WILSON_PHYSICAL_LABEL


def fps_mdwf_fit_label(fix_q_to_zero=False):
    prefix = "MDWF"
    formula = fps_mdwf_physical_formula(fix_q_to_zero).replace("\n", " ")
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
        af_ps, af_ps_err = parse_pair(row, "afps", filename)
        out[ens] = {
            "beta": float(row["beta"]),
            "m0": float(row["m0"]),
            "am_ps": am_ps,
            "am_ps_err": am_ps_err,
            "af_ps": af_ps,
            "af_ps_err": af_ps_err,
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


def fw0_sq_and_error_from_w0a(af, af_err, w0a, w0a_err):
    y = (af * w0a) ** 2
    dy_daf = 2.0 * af * (w0a**2)
    dy_dw0a = 2.0 * (af**2) * w0a
    var = (dy_daf**2) * (af_err**2) + (dy_dw0a**2) * (w0a_err**2)
    return y, np.sqrt(var)


def a_over_w0_and_error(w0a, w0a_err):
    z = 1.0 / w0a
    z_err = w0a_err / (w0a**2)
    return z, z_err


def square_with_error(z, z_err):
    q = z**2
    q_err = 2.0 * abs(z) * z_err
    return q, q_err


def exclude_wilson_endpoints(points, n_first=0, m_last=0, x_key="x"):
    if n_first < 0 or m_last < 0:
        raise ValueError("n_first and m_last must be non-negative.")

    if n_first + m_last >= len(points) and len(points) > 0:
        raise ValueError(
            f"Cannot exclude first {n_first} and last {m_last} points: "
            f"only {len(points)} Wilson points available."
        )

    sorted_points = sorted(points, key=lambda p: p[x_key])
    removed_first = sorted_points[:n_first] if n_first > 0 else []
    removed_last = sorted_points[-m_last:] if m_last > 0 else []
    kept_start = n_first
    kept_stop = len(sorted_points) - m_last if m_last > 0 else len(sorted_points)
    kept = sorted_points[kept_start:kept_stop]
    return kept, removed_first, removed_last


def print_removed_wilson_points(removed_points, which):
    if not removed_points:
        return

    print(f"Excluded Wilson points from the {which} in x:")
    for p in removed_points:
        print(
            f"  Ensemble={p['Ensemble']}, beta={p['beta']}, m0={p['m0']}, "
            f"x=(m_PS w0)^2={p['x']:.8g}"
        )


def fit_dw_continuum(points):
    n_params = 3

    if len(points) < n_params:
        raise ValueError(
            f"Need at least {n_params} DWF/MDWF points for continuum fit."
        )

    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    ye = np.array([p["yerr"] for p in points], dtype=float)
    z = np.array([p["a_over_w0_sq"] for p in points], dtype=float)
    M = np.column_stack([np.ones_like(x), x, z])

    coeffs, errs, cov, chi2, dof = solve_weighted_least_squares(
        M, y, ye, "DWF/MDWF"
    )

    A, B, C = coeffs
    A_err, B_err, C_err = errs

    if A <= 0:
        L = np.nan
        L_err = np.nan
    else:
        L = B / A
        dL_dA = -B / (A**2)
        dL_dB = 1.0 / A
        var_L = (
            dL_dA**2 * cov[0, 0]
            + dL_dB**2 * cov[1, 1]
            + 2.0 * dL_dA * dL_dB * cov[0, 1]
        )
        L_err = np.sqrt(max(var_L, 0.0))

    return {
        "A": A,
        "A_err": A_err,
        "B": B,
        "B_err": B_err,
        "C": C,
        "C_err": C_err,
        "cov": cov,
        "chi2": chi2,
        "dof": dof,
        "L": L,
        "L_err": L_err,
        "model_key": "dw",
        "label": r"MDWF: $A + Bx + C(a/w_0)^2$",
        "label_plain": "MDWF: A + Bx + C(a/w0)^2",
        "stage": "linearized",
    }


def continuum_line_and_band_dw(x, fit):
    A = fit["A"]
    B = fit["B"]
    cov = fit["cov"]

    y = A + B * x
    var = cov[0, 0] + x**2 * cov[1, 1] + 2.0 * x * cov[0, 1]
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)
    return y, err


def continuum_line_and_band_dw2(x, fit):
    A = fit["A"]
    B = fit["B"]
    C = fit["C"]
    cov = fit["cov"]

    y = A + B * x + C * x**2
    M_cont = np.column_stack([np.ones_like(x), x, x**2])
    cov_cont = cov[:3, :3]
    var = np.einsum("ij,jk,ik->i", M_cont, cov_cont, M_cont)
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)
    return y, err


def wilson_continuum_line_and_band(x_grid, fit):
    basis_terms = fit["basis_terms"]
    coeffs = fit["coeffs"]
    cov = fit["cov"]

    survivors = []
    survivor_indices = []
    for i, term in enumerate(basis_terms):
        if term in {"1", "x", "x2"}:
            survivor_indices.append(i)
            if term == "1":
                survivors.append(np.ones_like(x_grid))
            elif term == "x":
                survivors.append(x_grid)
            elif term == "x2":
                survivors.append(x_grid**2)

    if not survivors:
        raise ValueError("Wilson continuum model has no surviving continuum terms.")

    M_cont = np.column_stack(survivors)
    c_cont = coeffs[survivor_indices]
    cov_cont = cov[np.ix_(survivor_indices, survivor_indices)]
    y = M_cont @ c_cont
    var = np.einsum("ij,jk,ik->i", M_cont, cov_cont, M_cont)
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)
    return y, err


def print_dw_fit_summary(fit):
    print("DWF/MDWF continuum fit:")
    print(f"  A = {fit['A']:.8g} ± {fit['A_err']:.3g}")
    print(f"  B = {fit['B']:.8g} ± {fit['B_err']:.3g}")
    print(f"  C = {fit['C']:.8g} ± {fit['C_err']:.3g}")
    print(f"  L = {fit['L']:.8g} ± {fit['L_err']:.3g}")
    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


def print_dw2_fit_summary(fit):
    print("DWF/MDWF continuum fit [with x^2]:")
    print(f"  A = {fit['A']:.8g} ± {fit['A_err']:.3g}")
    print(f"  B = {fit['B']:.8g} ± {fit['B_err']:.3g}")
    print(f"  C = {fit['C']:.8g} ± {fit['C_err']:.3g}")
    print(f"  D = {fit['D']:.8g} ± {fit['D_err']:.3g}")
    print(f"  L = {fit['L']:.8g} ± {fit['L_err']:.3g}")
    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


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
        y, yerr = fw0_sq_and_error_from_w0a(
            srow["af_ps"], srow["af_ps_err"], wrow["w0a"], wrow["w0a_err"]
        )
        a_over_w0, a_over_w0_err = a_over_w0_and_error(
            wrow["w0a"], wrow["w0a_err"]
        )
        a_over_w0_sq, a_over_w0_sq_err = square_with_error(
            a_over_w0, a_over_w0_err
        )

        wilson_points.append(
            {
                "Ensemble": ens,
                "beta": srow["beta"],
                "m0": srow["m0"],
                "x": x,
                "xerr": xerr,
                "y": y,
                "yerr": yerr,
                "a_over_w0": a_over_w0,
                "a_over_w0_err": a_over_w0_err,
                "a_over_w0_sq": a_over_w0_sq,
                "a_over_w0_sq_err": a_over_w0_sq_err,
            }
        )

    return wilson_points


def validate_plot_fit_keys(plot_fit_keys, available_fit_keys):
    invalid = sorted(set(plot_fit_keys) - set(available_fit_keys))
    if invalid:
        raise ValueError(
            f"Unknown fit key(s) in PLOT_FITS: {invalid}\n"
            f"Allowed keys: {sorted(available_fit_keys)}"
        )


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


def select_bootstrap_plot_fit_keys(has_wilson):
    plot_fit_keys = []
    for key in PLOT_FITS:
        mapped = "wilson_physical" if key.startswith("wilson_") else key
        if mapped == "wilson_physical" and not has_wilson:
            continue
        if mapped not in plot_fit_keys:
            plot_fit_keys.append(mapped)

    if "dw2" not in plot_fit_keys:
        plot_fit_keys.append("dw2")

    return plot_fit_keys


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
    return {
        "bootstrap": _require_key(data, ["bootstrap"], filename),
        "pp_samples": _require_key(
            data, ["results", "bootstrap_fit", "PP", "samples"], filename
        ),
        "sim_samples": _require_key(
            data,
            ["results", "bootstrap_fit", "simultaneous_PP_A0P", "samples"],
            filename,
        ),
        "za_samples": _require_key(
            data, ["results", "bootstrap_fit", "Z_A", "samples"], filename
        ),
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
    sim_samples = spec["sim_samples"]
    za_samples = spec["za_samples"]
    w0_samples = wflow["w0_samples"]
    n_boot = int(spec["bootstrap"]["n_boot"])

    if (
        len(pp_samples) != n_boot
        or len(sim_samples) != n_boot
        or len(za_samples) != n_boot
        or len(w0_samples) != n_boot
    ):
        raise ValueError(
            f"Inconsistent bootstrap sample count for ensemble:\n"
            f"  spectrum: {spec_path}\n"
            f"  wflow:    {wflow_path}"
        )

    replica_points = []
    for b in range(n_boot):
        pp_b = pp_samples[b]
        sim_b = sim_samples[b]
        za_b = za_samples[b]
        w0_b = w0_samples[b]

        if pp_b is None or sim_b is None or za_b is None or w0_b is None:
            replica_points.append(None)
            continue

        m_ps = pp_b.get("m_ps")
        f_ps = sim_b.get("f_ps")
        z_a = za_b.get("Z_A")
        w0 = w0_b.get("w0")
        if m_ps is None or f_ps is None or z_a is None or w0 in (None, 0.0):
            replica_points.append(None)
            continue

        fps = float(z_a) * float(f_ps)
        x = float((float(m_ps) * float(w0)) ** 2)
        y = float((fps * float(w0)) ** 2)
        a_over_w0 = float(1.0 / float(w0))
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
                "f_ps": float(f_ps),
                "Z_A": float(z_a),
                "fps": fps,
            }
        )

    valid = [sample for sample in replica_points if sample is not None]
    if not valid:
        raise RuntimeError(f"No valid bootstrap replicas for ensemble: {spec_path}")

    x_stats = summary_stats([sample["x"] for sample in valid])
    y_stats = summary_stats([sample["y"] for sample in valid])
    a_stats = summary_stats([sample["a_over_w0"] for sample in valid])
    z_stats = summary_stats([sample["Z_A"] for sample in valid])
    fps_stats = summary_stats([sample["fps"] for sample in valid])

    point = {
        "beta": beta,
        "m0": mass,
        "x": x_stats["mean"],
        "xerr": x_stats["sdev"],
        "y": y_stats["mean"],
        "yerr": y_stats["sdev"],
        "Z_A": z_stats["mean"],
        "Z_A_err": z_stats["sdev"],
        "fps": fps_stats["mean"],
        "fps_err": fps_stats["sdev"],
        "a_over_w0": a_stats["mean"],
        "a_over_w0_err": a_stats["sdev"],
        "a_over_w0_sq": a_stats["mean"] ** 2,
        "a_over_w0_sq_err": 2.0 * abs(a_stats["mean"]) * a_stats["sdev"],
    }

    return {
        "point": point,
        "bootstrap_samples": replica_points,
        "bootstrap_meta": spec["bootstrap"],
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
                    "a_over_w0_sq": sample["a_over_w0"] ** 2,
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


def derive_dw2_start_parameters(linear_fit):
    return _shared_derive_dw2_start_parameters(linear_fit)


def fit_dw2_continuum_linear(points):
    return _shared_fit_dw2_continuum_linear(points)


def fit_dw2_continuum_nonlinear(points, initial_fit, *, fix_q_to_zero=False):
    return _shared_fit_dw2_continuum_nonlinear(
        points,
        initial_fit,
        fit_label=fps_mdwf_fit_label(fix_q_to_zero),
        fix_Q_to_zero=fix_q_to_zero,
    )


def fit_dw2_bootstrap_replica(points, *, fix_q_to_zero=False):
    linear_fit = fit_dw2_continuum_linear(points)
    nonlinear_fit = fit_dw2_continuum_nonlinear(
        points,
        linear_fit,
        fix_q_to_zero=fix_q_to_zero,
    )
    params = np.array(
        [
            nonlinear_fit["m_M_chi_sq"],
            nonlinear_fit["L_m_M"],
            nonlinear_fit["Q_m_M"],
            nonlinear_fit["R_m_M"],
        ],
        dtype=float,
    )
    return linear_fit, nonlinear_fit, params, float(nonlinear_fit["chi2"])


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
            replica_linear_fit = fit_dw2_continuum_linear(point_set)
            replica_start_params = derive_dw2_start_parameters(replica_linear_fit)
            replica_non_linear_fit = fit_dw2_continuum_nonlinear(
                point_set,
                replica_linear_fit,
                fix_q_to_zero=fix_q_to_zero,
            )
            params = np.array(
                [
                    replica_non_linear_fit["m_M_chi_sq"],
                    replica_non_linear_fit["L_m_M"],
                    replica_non_linear_fit["Q_m_M"],
                    replica_non_linear_fit["R_m_M"],
                ],
                dtype=float,
            )
            chi2 = float(replica_non_linear_fit["chi2"])
            sample = {
                "index": int(b),
                "m_M_chi_sq": float(params[0]),
                "L_m_M": float(params[1]),
                "Q_m_M": float(params[2]),
                "R_m_M": float(params[3]),
                "parameter_order": ["m_M_chi_sq", "L_m_M", "Q_m_M", "R_m_M"],
                "cov": np.asarray(replica_non_linear_fit["cov"], dtype=float),
                "chi2": chi2,
                "linearized": linear_fit_to_json_dict(replica_linear_fit),
                "starting_parameters": to_serializable(replica_start_params),
            }
            success_rows.append((int(b), params, chi2, sample))
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

    summary_predictions = bootstrap_summary_predictions(dw_points, mean_params)
    summary_predictions["dof"] = int(len(dw_points) - (3 if fix_q_to_zero else 4))
    mean_model_y = np.asarray(
        summary_predictions["parameter_average_model_y"],
        dtype=float,
    )

    mean_y = np.asarray(summary_predictions["observed_y"], dtype=float)
    mean_ye = np.asarray(summary_predictions["observed_yerr"], dtype=float)
    residuals = mean_y - mean_model_y
    final_chi2 = float(np.sum((residuals / mean_ye) ** 2))
    final_dof = int(summary_predictions["dof"])

    errs = np.sqrt(np.diag(cov))
    fit = {
        "model_key": "dw2_physical_bootstrap",
        "stage": "bootstrap_summary",
        "label": (
            fps_mdwf_bootstrap_label(fix_q_to_zero)
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


def derive_wilson_start_parameters(linear_fit):
    return _shared_derive_wilson_start_parameters(linear_fit)


def fit_wilson_complete_model_linear(points):
    return _shared_fit_wilson_complete_model_linear(points)


def fit_wilson_complete_model_nonlinear(
    points,
    initial_fit=None,
    *,
    p0=None,
    fix_q_to_zero=False,
):
    return _shared_fit_wilson_complete_model_nonlinear(
        points,
        initial_fit,
        p0=p0,
        fit_label=fps_wilson_fit_label(fix_q_to_zero),
        fix_Q_to_zero=fix_q_to_zero,
    )


def wilson_physical_continuum_line_and_band(m_ps_sq, fit):
    return _shared_wilson_continuum_line_and_band(m_ps_sq, fit)


def make_wilson_fit_text(fit):
    text = (
        r"$\mathrm{Wilson\ (nonlinear)}:$" "\n"
        r"$f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$" "\n"
        r"$\qquad\qquad + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
    )
    text += "\n" + rf"$f_{{{{\rm PS}},\chi}}^2 = {fit['m_M_chi_sq']:.4f} \pm {fit['m_M_chi_sq_err']:.4f}$"
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
    print(f"  m_M_chi_sq = {fit['m_M_chi_sq']:.8g} ± {fit['m_M_chi_sq_err']:.3g}")
    print(f"  L_m_M = {fit['L_m_M']:.8g} ± {fit['L_m_M_err']:.3g}")
    print(f"  Q_m_M = {fit['Q_m_M']:.8g} ± {fit['Q_m_M_err']:.3g}")
    print(f"  W_m_M = {fit['W_m_M']:.8g} ± {fit['W_m_M_err']:.3g}")
    print(f"  R_m_M = {fit['R_m_M']:.8g} ± {fit['R_m_M_err']:.3g}")
    print(f"  C_m_M = {fit['C_m_M']:.8g} ± {fit['C_m_M_err']:.3g}")
    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


def print_starting_parameters(title, params):
    print(f"{title}:")
    for key, value in params.items():
        print(f"  {key} = {value:.8g}")


def physical_dw2_to_plot_fit(fit):
    a = float(fit["m_M_chi_sq"])
    l_val = float(fit["L_m_M"])
    q_val = float(fit["Q_m_M"])
    r_val = float(fit["R_m_M"])
    cov_phys = np.asarray(fit["cov"], dtype=float)

    coeff_b = a * l_val
    coeff_c = a * q_val
    jac = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [l_val, a, 0.0, 0.0],
            [q_val, 0.0, a, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cov_coeff = jac @ cov_phys @ jac.T
    errs = np.sqrt(np.diag(cov_coeff))

    return {
        "A": a,
        "A_err": float(errs[0]),
        "B": coeff_b,
        "B_err": float(errs[1]),
        "C": coeff_c,
        "C_err": float(errs[2]),
        "D": r_val,
        "D_err": float(errs[3]),
        "cov": cov_coeff,
        "chi2": float(fit["chi2"]),
        "dof": int(fit["dof"]),
        "L": float(l_val),
        "L_err": float(fit["L_m_M_err"]),
        "model_key": "dw2",
        "label": fit["label"],
        "label_plain": fit.get("label", ""),
        "fix_Q_to_zero": bool(fit.get("fix_Q_to_zero", False)),
    }


def plot_points_and_fits_bootstrap(
    dw_points,
    wilson_points,
    all_fits,
    plot_fit_keys,
    output_plot,
    dw2_fit_central=None,
    wilson_fit_central=None,
):
    # Plot the ensemble points together with the central fit curves that are
    # meant to appear in the release figure. The bootstrap summary itself is
    # written to JSON rather than drawn as a separate curve choice.
    validate_plot_fit_keys(plot_fit_keys, all_fits.keys())

    all_betas = sorted(
        {p["beta"] for p in dw_points}.union({p["beta"] for p in wilson_points})
    )
    beta_colors = {b: f"C{i % 10}" for i, b in enumerate(all_betas)}

    fig, ax = plt.subplots(figsize=(9.2, 5.8), layout="constrained")

    dw_label_used = False
    for p in dw_points:
        label = "DWF/MDWF data" if not dw_label_used else None
        dw_label_used = True
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
            alpha=0.95,
            label=label,
        )

    wilson_label_used = False
    for p in wilson_points:
        label = "Wilson data" if not wilson_label_used else None
        wilson_label_used = True
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
            alpha=0.95,
            label=label,
        )

    x_all = np.array([p["x"] for p in dw_points] + [p["x"] for p in wilson_points])
    x_max = 1.05 * np.max(x_all)
    x_grid = np.linspace(0.0, x_max, 500)

    style_map = {
        "dw": {"color": "tab:green", "linestyle": (0, (5, 2, 1, 2)), "alpha_band": 0.12},
        "dw2": {"color": MDWF_FIT_COLOR, "linestyle": "-", "alpha_band": 0.14},
        "wilson_physical": {"color": WILSON_FIT_COLOR, "linestyle": ":", "alpha_band": 0.12},
    }

    for fit_key in plot_fit_keys:
        fit = all_fits[fit_key]
        style = style_map[fit_key]
        if fit_key == "dw":
            y_fit, y_err = continuum_line_and_band_dw(x_grid, fit)
            ax.plot(
                x_grid,
                y_fit,
                color=style["color"],
                linestyle=style["linestyle"],
                label=fit["label"],
            )
            ax.fill_between(
                x_grid,
                y_fit - y_err,
                y_fit + y_err,
                color=style["color"],
                alpha=style["alpha_band"],
                linewidth=0,
            )
            continue
        elif fit_key == "dw2":
            central_fit = dw2_fit_central if dw2_fit_central is not None else fit
            y_central, y_central_err = continuum_line_and_band_dw2(x_grid, central_fit)
            ax.plot(
                x_grid,
                y_central,
                linestyle="-",
                color=style["color"],
                linewidth=0.8,
                alpha=0.9,
                label="MDWF central fit",
            )
            ax.fill_between(
                x_grid,
                y_central - y_central_err,
                y_central + y_central_err,
                color=style["color"],
                alpha=style["alpha_band"],
                linewidth=0,
            )
            continue
        elif fit_key == "wilson_physical":
            central_fit = wilson_fit_central if wilson_fit_central is not None else fit
            y_central, y_central_err = wilson_physical_continuum_line_and_band(
                x_grid,
                central_fit,
            )
            ax.plot(
                x_grid,
                y_central,
                linestyle="-",
                color=style["color"],
                linewidth=0.8,
                alpha=0.9,
            )
            ax.fill_between(
                x_grid,
                y_central - y_central_err,
                y_central + y_central_err,
                color=style["color"],
                alpha=style["alpha_band"],
                linewidth=0,
            )
            continue
        else:
            y_fit, y_err = wilson_continuum_line_and_band(x_grid, fit)

        ax.plot(
            x_grid,
            y_fit,
            color=style["color"],
            linestyle=style["linestyle"],
            label=fit["label"],
        )
        ax.fill_between(
            x_grid,
            y_fit - y_err,
            y_fit + y_err,
            color=style["color"],
            alpha=style["alpha_band"],
            linewidth=0,
        )

    ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")
    ax.set_ylabel(r"$(f^{\rm ren}_{\rm PS} w_0)^2$")
    ax.set_xlim(0.0, float(x_grid[-1]))
    ax.set_ylim(0.0, 0.0200)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    wilson_betas = sorted({p["beta"] for p in wilson_points})
    dw_betas = sorted({p["beta"] for p in dw_points})

    mdwf_formula = fps_mdwf_physical_formula(all_fits["dw2"].get("fix_Q_to_zero", False))
    wilson_formula = fps_wilson_physical_formula(
        all_fits["wilson_physical"].get("fix_Q_to_zero", False)
        if "wilson_physical" in all_fits
        else False
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

    if "wilson_physical" in plot_fit_keys and wilson_points:
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

    output_dir = os.path.dirname(output_plot)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_plot, dpi=300)
    plt.close()


def linear_fit_to_json_dict(fit):
    out = {
        "model_key": fit["model_key"],
        "stage": fit["stage"],
        "label": fit["label"],
        "chi2": fit["chi2"],
        "dof": fit["dof"],
        "cov": fit["cov"],
    }
    if "basis_terms" in fit:
        out["basis_terms"] = fit["basis_terms"]
        out["coeffs"] = fit["coeffs"]
        out["coeff_errs"] = fit["coeff_errs"]
    return to_serializable(out)


def physical_fit_to_json_dict(fit):
    out = {
        "model_key": fit["model_key"],
        "stage": fit["stage"],
        "label": fit["label"],
        "chi2": fit["chi2"],
        "dof": fit["dof"],
        "cov": fit["cov"],
    }
    for key in [
        "m_M_chi_sq",
        "m_M_chi_sq_err",
        "L_m_M",
        "L_m_M_err",
        "Q_m_M",
        "Q_m_M_err",
        "R_m_M",
        "R_m_M_err",
    ]:
        if key in fit:
            out[key] = fit[key]
    for key in ["W_m_M", "W_m_M_err", "C_m_M", "C_m_M_err"]:
        if key in fit:
            out[key] = fit[key]
    if "fix_Q_to_zero" in fit:
        out["fix_Q_to_zero"] = fit["fix_Q_to_zero"]
    return to_serializable(out)


def bootstrap_fit_to_json_dict(fit):
    # Keep the bootstrap bookkeeping alongside the usual fit parameters so the
    # release JSON remains self-contained.
    out = physical_fit_to_json_dict(fit)
    out["summary_predictions"] = fit.get("summary_predictions")
    out["bootstrap_meta"] = fit["bootstrap_meta"]
    out["bootstrap_samples"] = fit["bootstrap_samples"]
    out["bootstrap_failures"] = fit["bootstrap_failures"]
    out["continuum_limit"] = {
        "fpsw0_sq": {
            "mean": fit["m_M_chi_sq"],
            "sdev": fit["m_M_chi_sq_err"],
        },
        "fpsw0": {
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
    plot_fit_keys,
    dw_points,
    bootstrap_point_sets,
    dw_fit_linear,
    dw_fit_central,
    dw2_fit_bootstrap,
    wilson_points,
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
        "plot_fits": plot_fit_keys,
        "n_dw_points_used": len(dw_points),
        "n_dw_bootstrap_requested": len(bootstrap_point_sets),
        "n_dw_bootstrap_success": dw2_fit_bootstrap["bootstrap_meta"]["n_success"],
        "n_dw_bootstrap_failed": dw2_fit_bootstrap["bootstrap_meta"]["n_failed"],
        "points": {
            "dw": to_serializable(dw_points),
            "wilson": to_serializable(wilson_points),
        },
        "fits": {
            "dw2": {
                "linearized": linear_fit_to_json_dict(dw_fit_linear),
                "starting_parameters": to_serializable(
                    derive_dw2_start_parameters(dw_fit_linear)
                ),
                "central_nonlinear": physical_fit_to_json_dict(dw_fit_central),
                "bootstrap_summary": bootstrap_fit_to_json_dict(dw2_fit_bootstrap),
            },
        },
    }

    if (
        wilson_points is not None
        and wilson_fit_linear is not None
        and wilson_fit_nonlinear is not None
    ):
        payload["n_wilson_points_used"] = len(wilson_points)
        wilson_payload = {
            "linearized": linear_fit_to_json_dict(wilson_fit_linear),
            "starting_parameters": to_serializable(
                wilson_fit_starting_parameters
                if wilson_fit_starting_parameters is not None
                else derive_wilson_start_parameters(wilson_fit_linear)
            ),
            "nonlinear": physical_fit_to_json_dict(wilson_fit_nonlinear),
        }
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
            "Bootstrap MDWF continuum fit for (f_PS w0)^2 vs (m_PS w0)^2. "
            "The MDWF points are built from bootstrap summaries of PP, Z_A, "
            "simultaneous PP+A0P, and w0, then each bootstrap replica is refit "
            "to the MDWF dw2 ansatz."
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
        default="intermediary_data/NF2/spectrum/wilson/wilson_extrapolation_f_ps.json",
        help="Optional precomputed Wilson bootstrap JSON",
    )
    parser.add_argument(
        "--exclude_first_wilson",
        type=int,
        default=4,
        help=(
            "Exclude this many Wilson points with the smallest x=(m_PS w0)^2 "
            "before fitting and plotting."
        ),
    )
    parser.add_argument(
        "--exclude_last_wilson",
        type=int,
        default=0,
        help=(
            "Exclude this many Wilson points with the largest x=(m_PS w0)^2 "
            "before fitting and plotting."
        ),
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

    dw_fit = fit_dw_continuum(dw_points)
    dw_fit_linear = fit_dw2_continuum_linear(dw_points)
    start_params = derive_dw2_start_parameters(dw_fit_linear)
    dw2_fit_central = fit_dw2_continuum_nonlinear(
        dw_points,
        dw_fit_linear,
        fix_q_to_zero=fix_dw_q_zero,
    )
    dw2_fit_bootstrap = fit_dw2_bootstrap_summary(
        bootstrap_point_sets,
        dw_points,
        dw2_fit_central,
        start_params,
        fix_q_to_zero=fix_dw_q_zero,
    )
    dw2_fit_bootstrap["bootstrap_failures"] = (
        bootstrap_input_failures + dw2_fit_bootstrap["bootstrap_failures"]
    )
    dw2_fit_bootstrap["bootstrap_meta"]["n_failed"] = len(
        dw2_fit_bootstrap["bootstrap_failures"]
    )

    wilson_points = []
    removed_first = []
    removed_last = []
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
        wilson_fit["label"] = fps_wilson_fit_label(False)
        wilson_fit_bootstrap["label"] = fps_wilson_fit_label(False)
        if wilson_fit_central is not None:
            wilson_fit_central["label"] = fps_wilson_fit_label(False)
    elif args.spectrum_w and args.wflow_w:
        wilson_points = collect_wilson_points(args.spectrum_w, args.wflow_w)
        wilson_points, removed_first, removed_last = exclude_wilson_endpoints(
            wilson_points,
            n_first=args.exclude_first_wilson,
            m_last=args.exclude_last_wilson,
            x_key="x",
        )
        print_removed_wilson_points(removed_first, which="beginning")
        print_removed_wilson_points(removed_last, which="end")

        wilson_fit_linear = fit_wilson_complete_model_linear(wilson_points)
        wilson_fit = fit_wilson_complete_model_nonlinear(
            wilson_points,
            wilson_fit_linear,
            fix_q_to_zero=fix_wilson_q_zero,
        )
        wilson_fit_bootstrap = wilson_fit
        wilson_fit_starting_parameters = derive_wilson_start_parameters(wilson_fit_linear)
        wilson_fit_central = wilson_fit

    plot_fit_keys = select_bootstrap_plot_fit_keys(has_wilson=wilson_fit is not None)

    all_fits = {
        "dw": dw_fit,
        "dw2": physical_dw2_to_plot_fit(dw2_fit_bootstrap),
    }
    if wilson_fit is not None:
        all_fits["wilson_physical"] = wilson_fit

    plot_points_and_fits_bootstrap(
        dw_points=dw_points,
        wilson_points=wilson_points,
        all_fits=all_fits,
        plot_fit_keys=plot_fit_keys,
        output_plot=args.output_plot,
        dw2_fit_central=physical_dw2_to_plot_fit(dw2_fit_central),
        wilson_fit_central=wilson_fit_central,
    )

    save_fit_results_json(
        output_data=args.output_data,
        plot_fit_keys=plot_fit_keys,
        dw_points=dw_points,
        bootstrap_point_sets=bootstrap_point_sets,
        dw_fit_linear=dw_fit_linear,
        dw_fit_central=dw2_fit_central,
        dw2_fit_bootstrap=dw2_fit_bootstrap,
        wilson_points=wilson_points,
        wilson_fit_linear=wilson_fit_linear,
        wilson_fit_nonlinear=wilson_fit_central,
        wilson_fit_bootstrap=wilson_fit_bootstrap,
        wilson_fit_starting_parameters=wilson_fit_starting_parameters,
    )

    print(f"✓ Saved plot → {args.output_plot}")
    print(f"Fits shown on plot: {plot_fit_keys}")
    print_dw_fit_summary(dw_fit)
    print_dw2_fit_summary(physical_dw2_to_plot_fit(dw2_fit_central))
    print_dw2_fit_summary(physical_dw2_to_plot_fit(dw2_fit_bootstrap))
    print_starting_parameters(
        "DWF/MDWF starting parameters from linearized fit",
        start_params,
    )
    if wilson_fit_linear is not None and wilson_fit_bootstrap is not None:
        print_starting_parameters(
            "Wilson starting parameters from linearized fit",
            wilson_fit_starting_parameters,
        )
        if wilson_fit_central is not None:
            print_wilson_fit_summary(wilson_fit_central, "Wilson complete model [nonlinear]")


if __name__ == "__main__":
    main()
