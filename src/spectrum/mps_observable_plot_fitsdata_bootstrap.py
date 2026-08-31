#!/usr/bin/env python3
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
from mps_mv_plot_fitsdata_bootstrap import (
    plot_points_and_fits as plot_mv_points_and_fits,
)
from fps_mps_plot_fitsdata_bootstrap import (
    physical_dw2_to_plot_fit as physical_dw2_to_plot_fit_fps,
    plot_points_and_fits_bootstrap as plot_fps_points_and_fits_bootstrap,
    select_bootstrap_plot_fit_keys as select_fps_plot_fit_keys,
)
from fps_mps_plot import fit_dw_continuum

plt.style.use("tableau-colorblind10")
DEFAULT_WILSON_SHARED_NONLINEAR_P0 = [0.320, 2.9, -20.0, -0.183, 0.03, -1.0]

OBSERVABLE_CONFIG = {
    "mv": {
        "description": (
            "Bootstrap MDWF chiral-continuum fit for (m_V w0)^2 vs (m_PS w0)^2. "
            "The displayed MDWF points are bootstrap means with bootstrap standard "
            "deviations, and the MDWF continuum fit is obtained by fitting each "
            "bootstrap replica and summarising the fitted parameters."
        ),
        "dw_linear_label": (
            r"MDWF linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a^2$"
        ),
        "dw_physical_label": (
            r"MDWF: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + R_{m_M} a^2$"
        ),
        "dw_bootstrap_label": (
            r"MDWF bootstrap: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + R_{m_M} a^2$"
        ),
        "wilson_linear_label": (
            r"Wilson linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a m_{PS}^2 + E a + F a^2$"
        ),
        "wilson_physical_label": (
            r"Wilson: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "y_axis_label": r"$(m_{\rm V} w_0)^2$",
        "mdwf_formula": (
            r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            "\n"
            r"$\qquad\qquad + R_{m_M} a^2$"
        ),
        "wilson_formula": (
            r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            "\n"
            r"$\qquad\qquad + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "limit_key_sq": "mvw0_sq",
        "limit_key": "mvw0",
        "mdwf_extra_sample_paths": {
            "vv_samples": ["results", "bootstrap_fit", "VV", "samples"],
        },
        "point_stat_fields": (),
        "use_precomputed_wilson_fit": True,
        "set_ylim": (0.0, None),
    },
    "fps": {
        "description": (
            "Bootstrap MDWF chiral-continuum fit for (f_PS w0)^2 vs (m_PS w0)^2. "
            "The MDWF points are built from bootstrap summaries of PP, Z_A, "
            "simultaneous PP+A0P, and w0, then each bootstrap replica is refit "
            "to the MDWF dw2 ansatz."
        ),
        "dw_linear_label": (
            r"MDWF linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a^2$"
        ),
        "dw_physical_label": (
            r"MDWF: $f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + R_{m_M} a^2$"
        ),
        "dw_bootstrap_label": (
            r"MDWF bootstrap: $f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + R_{m_M} a^2$"
        ),
        "wilson_linear_label": (
            r"Wilson linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a m_{PS}^2 + E a + F a^2$"
        ),
        "wilson_physical_label": (
            r"Wilson: $f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "y_axis_label": r"$(f_{\rm PS} w_0)^2$",
        "mdwf_formula": (
            r"$f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            "\n"
            r"$\qquad\qquad + R_{m_M} a^2$"
        ),
        "wilson_formula": (
            r"$f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            "\n"
            r"$\qquad\qquad + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "limit_key_sq": "fpsw0_sq",
        "limit_key": "fpsw0",
        "mdwf_extra_sample_paths": {
            "sim_samples": ["results", "bootstrap_fit", "simultaneous_PP_A0P", "samples"],
            "za_samples": ["results", "bootstrap_fit", "Z_A", "samples"],
        },
        "point_stat_fields": ("Z_A", "fps"),
        "use_precomputed_wilson_fit": True,
        "set_ylim": (0.0, 0.0200),
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
        out_row = {
            "beta": float(row["beta"]),
            "m0": float(row["m0"]),
            "am_ps": am_ps,
            "am_ps_err": am_ps_err,
        }

        if "amv" in row:
            am_v, am_v_err = parse_pair(row, "amv", filename)
            out_row["am_v"] = am_v
            out_row["am_v_err"] = am_v_err

        if "afps" in row:
            af_ps, af_ps_err = parse_pair(row, "afps", filename)
            out_row["af_ps"] = af_ps
            out_row["af_ps_err"] = af_ps_err

        out[ens] = out_row

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


def collect_wilson_points(observable, spectrum_file, wflow_file):
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
        if observable == "mv":
            if "am_v" not in srow or "am_v_err" not in srow:
                raise ValueError(
                    f"Wilson spectrum JSON '{spectrum_file}' does not contain 'amv' for ensemble '{ens}'."
                )
            y, yerr = mw0_sq_and_error_from_w0a(
                srow["am_v"], srow["am_v_err"], wrow["w0a"], wrow["w0a_err"]
            )
        else:
            if "af_ps" not in srow or "af_ps_err" not in srow:
                raise ValueError(
                    f"Wilson spectrum JSON '{spectrum_file}' does not contain 'afps' for ensemble '{ens}'."
                )
            y, yerr = fw0_sq_and_error_from_w0a(
                srow["af_ps"], srow["af_ps_err"], wrow["w0a"], wrow["w0a_err"]
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


def read_spectrum_bootstrap_json(filename, observable):
    data = read_json_file(filename)
    cfg = get_config(observable)
    out = {
        "bootstrap": _require_key(data, ["bootstrap"], filename),
        "pp_samples": _require_key(data, ["results", "bootstrap_fit", "PP", "samples"], filename),
    }

    for name, path in cfg["mdwf_extra_sample_paths"].items():
        out[name] = _require_key(data, path, filename)

    return out


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


def ensure_bootstrap_sample_counts(spec, w0_samples, n_boot, spec_path, wflow_path, sample_keys):
    if any(len(spec[key]) != n_boot for key in sample_keys) or len(w0_samples) != n_boot:
        raise ValueError(
            f"Inconsistent bootstrap sample count for ensemble:\n"
            f"  spectrum: {spec_path}\n"
            f"  wflow:    {wflow_path}"
        )


def build_mv_replica(beta, mass, index, pp_b, extra_samples, w0_b):
    m_ps = pp_b.get("m_ps")
    m_v = extra_samples["vv_samples"].get("m_v")
    w0 = w0_b.get("w0")
    if m_ps is None or m_v is None or w0 in (None, 0.0):
        return None

    return {
        "index": int(index),
        "beta": beta,
        "m0": mass,
        "x": float((m_ps * w0) ** 2),
        "y": float((m_v * w0) ** 2),
        "a_over_w0": float(1.0 / w0),
        "w0": float(w0),
        "m_ps": float(m_ps),
        "m_v": float(m_v),
    }


def build_fps_replica(beta, mass, index, pp_b, extra_samples, w0_b):
    m_ps = pp_b.get("m_ps")
    f_ps = extra_samples["sim_samples"].get("f_ps")
    z_a = extra_samples["za_samples"].get("Z_A")
    w0 = w0_b.get("w0")
    if m_ps is None or f_ps is None or z_a is None or w0 in (None, 0.0):
        return None

    fps = float(z_a) * float(f_ps)
    return {
        "index": int(index),
        "beta": beta,
        "m0": mass,
        "x": float((float(m_ps) * float(w0)) ** 2),
        "y": float((fps * float(w0)) ** 2),
        "a_over_w0": float(1.0 / float(w0)),
        "w0": float(w0),
        "m_ps": float(m_ps),
        "f_ps": float(f_ps),
        "Z_A": float(z_a),
        "fps": fps,
    }


DW_REPLICA_BUILDERS = {
    "mv": build_mv_replica,
    "fps": build_fps_replica,
}


def build_dw_bootstrap_ensemble(spec_path, wflow_path, observable):
    cfg = get_config(observable)
    beta, mass = extract_beta_mass_from_path(spec_path)
    if beta is None or mass is None:
        raise ValueError(f"Could not extract beta/mass from path: {spec_path}")

    spec = read_spectrum_bootstrap_json(spec_path, observable)
    wflow = read_wflow_bootstrap_json(wflow_path)
    ensure_bootstrap_alignment(spec["bootstrap"], wflow["bootstrap"], spec_path, wflow_path)

    pp_samples = spec["pp_samples"]
    w0_samples = wflow["w0_samples"]
    n_boot = int(spec["bootstrap"]["n_boot"])
    sample_keys = ("pp_samples", *cfg["mdwf_extra_sample_paths"].keys())
    ensure_bootstrap_sample_counts(
        spec,
        w0_samples,
        n_boot,
        spec_path,
        wflow_path,
        sample_keys,
    )

    replica_points = []
    build_replica = DW_REPLICA_BUILDERS[observable]
    for b in range(n_boot):
        pp_b = pp_samples[b]
        w0_b = w0_samples[b]
        extra_samples = {
            key: spec[key][b]
            for key in cfg["mdwf_extra_sample_paths"]
        }
        if pp_b is None or w0_b is None or any(sample is None for sample in extra_samples.values()):
            replica_points.append(None)
            continue

        replica_points.append(
            build_replica(beta, mass, b, pp_b, extra_samples, w0_b)
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

    for key in cfg["point_stat_fields"]:
        stats = summary_stats([sample[key] for sample in valid])
        point[key] = stats["mean"]
        point[f"{key}_err"] = stats["sdev"]

    return {
        "point": point,
        "bootstrap_samples": replica_points,
        "bootstrap_meta": spec["bootstrap"],
        "paths": {"spectrum": spec_path, "wflow": wflow_path},
    }


def collect_dw_bootstrap_ensembles(spectrum_files, wflow_files, observable):
    if len(spectrum_files) != len(wflow_files):
        raise ValueError("Number of --spectrum files must equal number of --wflow files.")

    ensembles = [
        build_dw_bootstrap_ensemble(spec_path, wflow_path, observable)
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


def fit_dw2_bootstrap_replica(points, p0, fit_label):
    fit = fit_dw2_continuum_nonlinear(points, p0=p0, fit_label=fit_label)
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


def _robust_keep_mask(params):
    params = np.asarray(params, dtype=float)
    n_rows = params.shape[0]
    keep = np.all(np.isfinite(params), axis=1)

    if n_rows < 5:
        return keep, []

    med = np.median(params[keep], axis=0)
    mad = np.median(np.abs(params[keep] - med), axis=0)
    robust_sigma = 1.4826 * mad

    rejected = []
    for i in range(n_rows):
        if not keep[i]:
            rejected.append(
                {
                    "row_index": int(i),
                    "reason": "non_finite_parameters",
                }
            )
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


def fit_dw2_bootstrap_summary(
    bootstrap_point_sets,
    dw_points,
    central_fit,
    start_params,
    observable,
):
    cfg = get_config(observable)
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
                cfg["dw_physical_label"],
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

    raw_params = np.asarray([row[1] for row in success_rows], dtype=float)
    keep_mask, rejected = _robust_keep_mask(raw_params)

    param_rows = []
    chi2_values = []
    for row_idx, (boot_index, popt, chi2, sample) in enumerate(success_rows):
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
        raise RuntimeError("All bootstrap MDWF continuum fits were rejected by the outlier filter.")

    params = np.asarray(param_rows, dtype=float)
    mean_params = np.mean(params, axis=0)
    if params.shape[0] > 1:
        cov = np.cov(params, rowvar=False, ddof=1)
    else:
        cov = np.zeros((4, 4), dtype=float)

    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)

    mean_x = np.array([p["x"] for p in dw_points], dtype=float)
    mean_a = np.array([p["a_over_w0"] for p in dw_points], dtype=float)
    mean_y = np.array([p["y"] for p in dw_points], dtype=float)
    mean_ye = np.array([p["yerr"] for p in dw_points], dtype=float)
    central_residuals = mean_y - dw2_physical_model((mean_x, mean_a), *mean_params)
    final_chi2 = float(np.sum((central_residuals / mean_ye) ** 2))
    final_dof = int(len(mean_y) - len(mean_params))

    errs = np.sqrt(np.diag(cov))
    fit = {
        "model_key": "dw2_physical_bootstrap",
        "stage": "bootstrap_summary",
        "label": cfg["dw_bootstrap_label"],
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


def plot_points_and_fits(
    dw_points,
    dw_fit,
    output_file,
    observable,
    wilson_points=None,
    wilson_fit=None,
    dw_fit_central=None,
    wilson_fit_central=None,
):
    wilson_points = wilson_points or []
    cfg = get_config(observable)

    if observable == "mv":
        plot_mv_points_and_fits(
            dw_points=dw_points,
            dw_fit=dw_fit_central if dw_fit_central is not None else dw_fit,
            wilson_points=wilson_points,
            wilson_fit=(
                wilson_fit_central if wilson_fit_central is not None else wilson_fit
            ),
            dw_fit_central=dw_fit_central,
            wilson_fit_central=wilson_fit_central,
            output_file=output_file,
            set_ylim=cfg["set_ylim"],
        )
        return

    if observable == "fps":
        plot_fit_keys = select_fps_plot_fit_keys(has_wilson=wilson_fit is not None)
        all_fits = {
            "dw2": physical_dw2_to_plot_fit_fps(
                dw_fit_central if dw_fit_central is not None else dw_fit
            ),
        }
        # Keep the original fps single-script plot content, including the
        # extra linearized MDWF guide curve.
        if dw_fit_central is not None:
            all_fits["dw"] = fit_dw_continuum(dw_points)
        if wilson_fit is not None:
            all_fits["wilson_physical"] = (
                wilson_fit_central if wilson_fit_central is not None else wilson_fit
            )

        plot_fps_points_and_fits_bootstrap(
            dw_points=dw_points,
            wilson_points=wilson_points,
            all_fits=all_fits,
            plot_fit_keys=plot_fit_keys,
            output_plot=output_file,
            dw2_fit_central=(
                physical_dw2_to_plot_fit_fps(dw_fit_central)
                if dw_fit_central is not None
                else None
            ),
            wilson_fit_central=wilson_fit_central,
        )
        return

    raise ValueError(f"Unsupported observable '{observable}'")


def bootstrap_fit_to_json_dict(fit, observable):
    cfg = get_config(observable)
    out = physical_fit_to_json_dict(fit)
    out["bootstrap_meta"] = fit["bootstrap_meta"]
    out["bootstrap_samples"] = fit["bootstrap_samples"]
    out["bootstrap_failures"] = fit["bootstrap_failures"]

    limit_block = {
        cfg["limit_key_sq"]: {
            "mean": fit["m_M_chi_sq"],
            "sdev": fit["m_M_chi_sq_err"],
        },
        cfg["limit_key"]: {
            "mean": float(np.sqrt(fit["m_M_chi_sq"])) if fit["m_M_chi_sq"] >= 0.0 else None,
            "sdev": (
                float(0.5 * fit["m_M_chi_sq_err"] / np.sqrt(fit["m_M_chi_sq"]))
                if fit["m_M_chi_sq"] > 0.0
                else None
            ),
        },
    }
    out["continuum_limit"] = limit_block
    out["chiral_continuum_limit"] = limit_block
    return to_serializable(out)


def save_fit_results_json(
    output_data,
    observable,
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
    output_dir = os.path.dirname(output_data)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "observable": observable,
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
                "bootstrap_summary": bootstrap_fit_to_json_dict(dw_fit_bootstrap, observable),
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
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap MDWF chiral-continuum plot/fit driver for either "
            "(m_V w0)^2 or (f_PS w0)^2 as a function of (m_PS w0)^2."
        )
    )
    parser.add_argument(
        "--observable",
        required=True,
        help="Which observable to extrapolate: 'mv' or 'fps'.",
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
        default="intermediary_data/NF2/spectrum/wilson/wilson_extrapolation_fps.json",
        help="Optional precomputed Wilson bootstrap JSON for the selected observable.",
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
    args = parser.parse_args()
    cfg = get_config(args.observable)

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    dw_points, bootstrap_point_sets, bootstrap_input_failures = collect_dw_bootstrap_ensembles(
        args.spectrum,
        args.wflow,
        args.observable,
    )

    dw_fit_linear = fit_dw2_continuum_linear(
        dw_points,
        fit_label=cfg["dw_linear_label"],
    )
    start_params = derive_dw2_start_parameters(dw_fit_linear)
    dw_fit_central = fit_dw2_continuum_nonlinear(
        dw_points,
        dw_fit_linear,
        fit_label=cfg["dw_physical_label"],
    )
    dw_fit_bootstrap = fit_dw2_bootstrap_summary(
        bootstrap_point_sets,
        dw_points,
        dw_fit_central,
        start_params,
        args.observable,
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
    if cfg["use_precomputed_wilson_fit"] and wilsons_json_path and wilsons_json_path.exists():
        wilson_data = read_precomputed_wilson_bootstrap_json(str(wilsons_json_path))
        wilson_points = wilson_data["wilson_points"]
        wilson_fit_linear = wilson_data["linearized"]
        wilson_fit = wilson_data["bootstrap_summary"]
        wilson_fit_bootstrap = wilson_data["bootstrap_summary"]
        wilson_fit_starting_parameters = wilson_data["starting_parameters"]
        wilson_fit_central = wilson_data["central_nonlinear"]
    elif args.spectrum_w and args.wflow_w:
        wilson_points = collect_wilson_points(args.observable, args.spectrum_w, args.wflow_w)
        wilson_fit_linear = fit_wilson_complete_model_linear(
            wilson_points,
            fit_label=cfg["wilson_linear_label"],
        )
        if args.observable == "mv":
            wilson_fit = fit_wilson_complete_model_nonlinear(
                wilson_points,
                p0=DEFAULT_WILSON_SHARED_NONLINEAR_P0,
                fit_label=cfg["wilson_physical_label"],
            )
        else:
            wilson_fit = fit_wilson_complete_model_nonlinear(
                wilson_points,
                wilson_fit_linear,
                fit_label=cfg["wilson_physical_label"],
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
        observable=args.observable,
    )

    save_fit_results_json(
        output_data=args.output_data,
        observable=args.observable,
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
    print(f"Observable = {args.observable}")
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
