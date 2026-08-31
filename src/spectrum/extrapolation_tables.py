#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        value = float(x)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def format_phys_err(value, error, force_decimals=None):
    value = to_float(value)
    error = abs(to_float(error))

    if not np.isfinite(value):
        return "—"
    if not np.isfinite(error) or error == 0:
        if force_decimals is not None:
            return f"{value:.{force_decimals}f}"
        return f"{value:g}"

    exp = int(np.floor(np.log10(error)))
    norm = error / 10**exp
    sig = 2 if norm < 3 else 1
    decimals = max(0, -exp + sig - 1)

    if force_decimals is not None:
        decimals = force_decimals

    value_r = round(value, decimals)
    error_r = round(error, decimals)

    value_str = f"{value_r:.{decimals}f}"
    if error_r < 1:
        err_digits = int(round(error_r * 10**decimals))
        return f"{value_str}({err_digits})"

    error_str = f"{error_r:.{decimals}f}"
    return f"{value_str}({error_str})"


def format_chi2_over_dof(chi2, dof):
    chi2 = to_float(chi2)
    dof = to_float(dof)
    if not np.isfinite(chi2) or not np.isfinite(dof) or dof <= 0:
        return "—"
    return f"{chi2 / dof:.2f}"


def ratio_or_nan(num, den):
    num = to_float(num)
    den = to_float(den)
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return np.nan
    return num / den


def extract_plot_fit_keys(fit_data, fit_bootstrap_data):
    keys = fit_bootstrap_data.get("plot_fits") or fit_data.get("plot_fits")
    if isinstance(keys, list) and keys:
        return keys
    return sorted(set(fit_data.get("fits", {})) | set(fit_bootstrap_data.get("fits", {})))


def choose_mdwf_key(data, fallback_keys=None):
    fit_keys = set(data.get("fits", {}))
    ordered = list(fallback_keys or []) + ["dw2", "dw"]
    for key in ordered:
        if key in fit_keys and isinstance(key, str) and key.startswith("dw"):
            return key
    return None


def _match_legacy_wilson_key(data, include_q_term):
    fits = data.get("fits", {})
    for key in sorted(fits):
        if not isinstance(key, str) or not key.startswith("wilson_"):
            continue
        payload = fits.get(key, {})
        if not isinstance(payload, dict):
            continue
        terms = set(payload.get("basis_terms", []))
        if not {"1", "x", "x_a_over_w0", "a_over_w0", "a_over_w0_sq"}.issubset(terms):
            continue
        has_q_term = "x2" in terms
        if has_q_term == include_q_term:
            return key
    return None


def choose_wilson_key(data, preferred=None, preferred_fit=None):
    fit_keys = set(data.get("fits", {}))
    ordered = []
    if preferred is not None:
        ordered.append(preferred)
    ordered.append("wilson_physical")

    if preferred == "wilson_physical":
        include_q_term = True
        if isinstance(preferred_fit, dict):
            q_val = to_float(preferred_fit.get("Q_m_M"))
            q_err = abs(to_float(preferred_fit.get("Q_m_M_err")))
            include_q_term = not (
                np.isfinite(q_val)
                and np.isfinite(q_err)
                and q_val == 0.0
                and q_err == 0.0
            )
        legacy_match = _match_legacy_wilson_key(data, include_q_term=include_q_term)
        if legacy_match is not None:
            ordered.append(legacy_match)

    plot_fit_keys = data.get("plot_fits", [])
    if isinstance(plot_fit_keys, list):
        ordered.extend([key for key in plot_fit_keys if isinstance(key, str)])
    ordered.extend(sorted(key for key in fit_keys if isinstance(key, str) and key.startswith("wilson")))

    seen = set()
    for key in ordered:
        if key in seen:
            continue
        seen.add(key)
        if key in fit_keys and key.startswith("wilson"):
            return key
    return None


def extract_central_fit(data, fit_key):
    payload = data.get("fits", {}).get(fit_key)
    if not isinstance(payload, dict):
        return None
    if "model_key" in payload:
        return payload
    for key in ("central", "central_nonlinear", "nonlinear", "linearized"):
        fit = payload.get(key)
        if isinstance(fit, dict):
            return fit
    return None


def extract_bootstrap_fit(data, fit_key):
    payload = data.get("fits", {}).get(fit_key)
    if not isinstance(payload, dict):
        return None
    fit = payload.get("bootstrap_summary")
    if isinstance(fit, dict):
        return fit
    if "model_key" in payload:
        return payload
    for key in ("nonlinear", "central", "central_nonlinear", "linearized"):
        fit = payload.get(key)
        if isinstance(fit, dict):
            return fit
    return None


def extract_fit_row(fit_key, parameter_fit, chi2_fit):
    row = {
        "discretization": "MDWF" if fit_key.startswith("dw") else "Wilson",
        "m_M_chi_sq": "—",
        "L_m_M": "—",
        "Q_m_M": "—",
        "W_m_M": "—",
        "R_m_M": "—",
        "C_m_M": "—",
        "chi2_over_dof": format_chi2_over_dof(
            chi2_fit.get("chi2") if isinstance(chi2_fit, dict) else None,
            chi2_fit.get("dof") if isinstance(chi2_fit, dict) else None,
        ),
    }

    fit = parameter_fit if isinstance(parameter_fit, dict) else {}

    if fit_key == "dw2":
        if "m_M_chi_sq" in fit:
            row["m_M_chi_sq"] = format_phys_err(
                fit.get("m_M_chi_sq"), fit.get("m_M_chi_sq_err")
            )
            row["L_m_M"] = format_phys_err(
                fit.get("L_m_M"), fit.get("L_m_M_err"), force_decimals=2
            )
            row["Q_m_M"] = format_phys_err(fit.get("Q_m_M"), fit.get("Q_m_M_err"))
            row["R_m_M"] = format_phys_err(fit.get("R_m_M"), fit.get("R_m_M_err"))
            return row

        a = fit.get("A")
        a_err = fit.get("A_err")
        b = fit.get("B")
        c = fit.get("C")
        d = fit.get("D")
        d_err = fit.get("D_err")

        row["m_M_chi_sq"] = format_phys_err(a, a_err)
        row["L_m_M"] = format_phys_err(ratio_or_nan(b, a), None, force_decimals=2)
        row["Q_m_M"] = format_phys_err(ratio_or_nan(c, a), None)
        row["R_m_M"] = format_phys_err(d, d_err)
        return row

    if fit_key == "dw":
        a = fit.get("A")
        a_err = fit.get("A_err")
        b = fit.get("B")
        c = fit.get("C")
        c_err = fit.get("C_err")

        row["m_M_chi_sq"] = format_phys_err(a, a_err)
        row["L_m_M"] = format_phys_err(ratio_or_nan(b, a), None, force_decimals=2)
        row["R_m_M"] = format_phys_err(c, c_err)
        return row

    if "m_M_chi_sq" in fit:
        row["m_M_chi_sq"] = format_phys_err(
            fit.get("m_M_chi_sq"), fit.get("m_M_chi_sq_err")
        )
        row["L_m_M"] = format_phys_err(
            fit.get("L_m_M"), fit.get("L_m_M_err"), force_decimals=2
        )
        row["Q_m_M"] = format_phys_err(fit.get("Q_m_M"), fit.get("Q_m_M_err"))
        row["W_m_M"] = format_phys_err(fit.get("W_m_M"), fit.get("W_m_M_err"))
        row["R_m_M"] = format_phys_err(fit.get("R_m_M"), fit.get("R_m_M_err"))
        row["C_m_M"] = format_phys_err(fit.get("C_m_M"), fit.get("C_m_M_err"))
        return row

    basis_terms = list(fit.get("basis_terms", []))
    coeffs = list(fit.get("coeffs", []))
    coeff_errs = list(fit.get("coeff_errs", []))
    term_map = {
        term: (coeffs[i], coeff_errs[i] if i < len(coeff_errs) else None)
        for i, term in enumerate(basis_terms)
    }

    a, a_err = term_map.get("1", (None, None))
    bx, _bx_err = term_map.get("x", (None, None))
    cx2, _cx2_err = term_map.get("x2", (None, None))
    wa, wa_err = term_map.get("a_over_w0", (None, None))
    ra2, ra2_err = term_map.get("a_over_w0_sq", (None, None))
    camp, camp_err = term_map.get("x_a_over_w0", (None, None))

    row["m_M_chi_sq"] = format_phys_err(a, a_err)
    row["L_m_M"] = format_phys_err(ratio_or_nan(bx, a), None, force_decimals=2)
    if fit_key.startswith("wilson") and "x2" not in basis_terms:
        row["Q_m_M"] = "0"
    else:
        row["Q_m_M"] = format_phys_err(ratio_or_nan(cx2, a), None)
    row["W_m_M"] = format_phys_err(wa, wa_err)
    row["R_m_M"] = format_phys_err(ra2, ra2_err)
    row["C_m_M"] = format_phys_err(camp, camp_err)
    return row


def collect_comparison_rows(
    fit_data,
    fit_bootstrap_data,
    *,
    include_mdwf=True,
    include_wilson=True,
):
    plot_fit_keys = extract_plot_fit_keys(fit_data, fit_bootstrap_data)

    mdwf_key_boot = choose_mdwf_key(fit_bootstrap_data, fallback_keys=plot_fit_keys)
    mdwf_key_central = choose_mdwf_key(
        fit_data, fallback_keys=[mdwf_key_boot] + plot_fit_keys
    )

    wilson_key_boot = choose_wilson_key(fit_bootstrap_data)
    wilson_boot = (
        extract_bootstrap_fit(fit_bootstrap_data, wilson_key_boot)
        if wilson_key_boot is not None
        else None
    )
    wilson_key_central = choose_wilson_key(
        fit_data,
        preferred=wilson_key_boot,
        preferred_fit=wilson_boot,
    )

    rows = []
    fit_pairs = []
    if include_wilson:
        fit_pairs.append((wilson_key_boot, wilson_key_central))
    if include_mdwf:
        fit_pairs.append((mdwf_key_boot, mdwf_key_central))

    for boot_key, central_key in fit_pairs:
        fit_key = boot_key or central_key
        if fit_key is None:
            continue

        central_fit = None
        if central_key is not None:
            central_fit = extract_central_fit(fit_data, central_key)
        if not isinstance(central_fit, dict):
            central_fit = extract_central_fit(fit_bootstrap_data, fit_key)

        if isinstance(central_fit, dict):
            row = extract_fit_row(fit_key, central_fit, central_fit)
            row["label"] = "Central values fit"
            rows.append(row)

        bootstrap_fit = extract_bootstrap_fit(fit_bootstrap_data, fit_key)
        if isinstance(bootstrap_fit, dict):
            row = extract_fit_row(fit_key, bootstrap_fit, bootstrap_fit)
            row["label"] = "Bootstrap"
            rows.append(row)

    return rows


def collect_observable_rows(fit_data, fit_bootstrap_data):
    plot_fit_keys = extract_plot_fit_keys(fit_data, fit_bootstrap_data)

    mdwf_key_boot = choose_mdwf_key(fit_bootstrap_data, fallback_keys=plot_fit_keys)
    mdwf_key_central = choose_mdwf_key(
        fit_data, fallback_keys=[mdwf_key_boot] + plot_fit_keys
    )

    wilson_key_boot = choose_wilson_key(fit_bootstrap_data)
    wilson_boot = (
        extract_bootstrap_fit(fit_bootstrap_data, wilson_key_boot)
        if wilson_key_boot is not None
        else None
    )
    wilson_key_central = choose_wilson_key(
        fit_data,
        preferred=wilson_key_boot,
        preferred_fit=wilson_boot,
    )

    rows = []
    for boot_key, central_key in (
        (wilson_key_boot, wilson_key_central),
        (mdwf_key_boot, mdwf_key_central),
    ):
        fit_key = boot_key or central_key
        if fit_key is None:
            continue

        parameter_fit = None
        if central_key is not None:
            parameter_fit = extract_central_fit(fit_data, central_key)
        if not isinstance(parameter_fit, dict):
            parameter_fit = extract_central_fit(fit_bootstrap_data, fit_key)
        if not isinstance(parameter_fit, dict):
            parameter_fit = extract_bootstrap_fit(fit_bootstrap_data, fit_key)

        chi2_fit = None
        if central_key is not None:
            chi2_fit = extract_central_fit(fit_data, central_key)
        if not isinstance(chi2_fit, dict):
            chi2_fit = extract_central_fit(fit_bootstrap_data, fit_key)
        if not isinstance(chi2_fit, dict):
            chi2_fit = parameter_fit

        if isinstance(parameter_fit, dict):
            rows.append(extract_fit_row(fit_key, parameter_fit, chi2_fit))

    return rows


def build_combined_table(mv_rows, fps_rows, output_file):
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("%%%\\begin{table}[t]\n")
        f.write("%%%\\centering\n")
        f.write("\\begin{tabular}{|lccccccc|}\n")
        f.write("\\hline\\hline\n")
        f.write(
            "Discretisation & $(w_0 m^{\\chi}_{\\rm V})^2$ & "
            "$L_{m,\\rm V}$ & $Q_{m,\\rm V}$ & "
            "$W_{m,\\rm V}$ & $R_{m,\\rm V}$ & "
            "$C_{m,\\rm V}$ & "
            "$\\chi^2/\\mathrm{d.o.f.}$ \\\\\n"
        )
        f.write("\\hline\n")
        for row in mv_rows:
            if row["discretization"] == "MDWF":
                disc = "MDWF"
                w_coeff = row["R_m_M"]
                r_coeff = "—"
                c_coeff = "—"
            else:
                disc = "Wilson"
                w_coeff = row["W_m_M"]
                r_coeff = row["R_m_M"]
                c_coeff = row["C_m_M"]
            f.write(
                f"{disc} & "
                f"{row['m_M_chi_sq']} & "
                f"{row['L_m_M']} & "
                f"{row['Q_m_M']} & "
                f"{w_coeff} & "
                f"{r_coeff} & "
                f"{c_coeff} & "
                f"{row['chi2_over_dof']} \\\\\n"
            )
        f.write("\\hline\\hline\n")
        f.write(
            "Discretisation & $(w_0 f^{\\chi}_{\\rm PS})^2$ & "
            "$L_{f,\\rm PS}$ & $Q_{f,\\rm PS}$ & "
            "$W_{f,\\rm PS}$ & $R_{f,\\rm PS}$ & "
            "$C_{f,\\rm PS}$ & "
            "$\\chi^2/\\mathrm{d.o.f.}$ \\\\\n"
        )
        f.write("\\hline\n")
        for row in fps_rows:
            if row["discretization"] == "MDWF":
                disc = "MDWF"
                w_coeff = row["R_m_M"]
                r_coeff = "—"
                c_coeff = "—"
            else:
                disc = "Wilson"
                w_coeff = row["W_m_M"]
                r_coeff = row["R_m_M"]
                c_coeff = row["C_m_M"]
            f.write(
                f"{disc} & "
                f"{row['m_M_chi_sq']} & "
                f"{row['L_m_M']} & "
                f"{row['Q_m_M']} & "
                f"{w_coeff} & "
                f"{r_coeff} & "
                f"{c_coeff} & "
                f"{row['chi2_over_dof']} \\\\\n"
            )
        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(
            "%%%\\caption{Fit parameters for the extrapolations of "
            "$\\left(w_0 m_{\\mathrm{V}}\\right)^2$ and "
            "$\\left(w_0 f_{\\mathrm{PS}}\\right)^2$.}\n"
        )
        f.write("%%%\\label{tab:mv_fps_extrapolation}\n")
        f.write("%%%\\end{table}\n")


def build_bootstrap_comparison_table(mv_rows, fps_rows, output_file):
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _compact_w_coeff(row):
        if row.get("discretization") == "MDWF":
            return row.get("R_m_M", "—")
        return row.get("W_m_M", "—")

    mv_by_label = {row["label"]: row for row in mv_rows}
    fps_by_label = {row["label"]: row for row in fps_rows}
    ordered_labels = ["Central values fit", "Bootstrap"]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("%%%\\begin{table}[t]\n")
        f.write("%%%\\centering\n")
        f.write(
            "\\begin{tabular}{|l@{\\hspace{0.6em}}|@{\\hspace{0.6em}}cccc@{\\hspace{0.6em}}|@{\\hspace{0.6em}}cccc|}\n"
        )
        f.write("\\hline\\hline\n")
        f.write(
            "Fit & $(w_0 m^{\\chi}_{\\rm V})^2$ & "
            "$L_{m,\\rm V}$ & $Q_{m,\\rm V}$ & "
            "$W_{m,\\rm V}$ & "
            "$(w_0 f^{\\chi}_{\\rm PS})^2$ & "
            "$L_{f,\\rm PS}$ & $Q_{f,\\rm PS}$ & "
            "$W_{f,\\rm PS}$ \\\\\n"
        )
        f.write("\\hline\n")
        for label in ordered_labels:
            mv_row = mv_by_label.get(label, {})
            fps_row = fps_by_label.get(label, {})
            f.write(
                f"{label} & "
                f"{mv_row.get('m_M_chi_sq', '—')} & "
                f"{mv_row.get('L_m_M', '—')} & "
                f"{mv_row.get('Q_m_M', '—')} & "
                f"{_compact_w_coeff(mv_row) if mv_row else '—'} & "
                f"{fps_row.get('m_M_chi_sq', '—')} & "
                f"{fps_row.get('L_m_M', '—')} & "
                f"{fps_row.get('Q_m_M', '—')} & "
                f"{_compact_w_coeff(fps_row) if fps_row else '—'} \\\\\n"
            )
        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(
            "%%%\\caption{Central-fit and bootstrap-fit parameter comparison for the "
            "MDWF extrapolations of $\\left(w_0 m_{\\mathrm{V}}\\right)^2$ and "
            "$\\left(w_0 f_{\\mathrm{PS}}\\right)^2$.}\n"
        )
        f.write("%%%\\label{tab:mv_fps_extrapolation_bootstrap_compare_mdwf}\n")
        f.write("%%%\\end{table}\n")


def collect_wilson_point_rows(fit_bootstrap_data):
    rows = []
    for point in fit_bootstrap_data.get("points", {}).get("wilson", []):
        rows.append(
            {
                "ensemble": point.get("Ensemble", "—"),
                "beta": (
                    f"{to_float(point.get('beta')):.2f}"
                    if np.isfinite(to_float(point.get("beta")))
                    else "—"
                ),
                "m0": (
                    f"{to_float(point.get('m0')):.3f}"
                    if np.isfinite(to_float(point.get("m0")))
                    else "—"
                ),
                "x": format_phys_err(point.get("x"), point.get("xerr"), force_decimals=5),
                "y": format_phys_err(point.get("y"), point.get("yerr"), force_decimals=5),
                "a_over_w0": format_phys_err(
                    point.get("a_over_w0"), point.get("a_over_w0_err"), force_decimals=5
                ),
            }
        )
    return rows


def build_wilson_points_table(rows, output_file):
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("%%%\\begin{table}[t]\n")
        f.write("%%%\\centering\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\\hline\n")
        f.write(
            "Ensemble & $\\beta$ & $am_0$ & $(m_{\\rm PS} w_0)^2$ & "
            "$(m_V w_0)^2$ & $a/w_0$ \\\\\n"
        )
        f.write("\\hline\n")
        for row in rows:
            f.write(
                f"{row['ensemble']} & "
                f"{row['beta']} & "
                f"{row['m0']} & "
                f"{row['x']} & "
                f"{row['y']} & "
                f"{row['a_over_w0']} \\\\\n"
            )
        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("%%%\\caption{Wilson points entering the $m_V$ extrapolation.}\n")
        f.write("%%%\\label{tab:mv_wilson_points}\n")
        f.write("%%%\\end{table}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate combined LaTeX tables for the mV and fps extrapolation fits, "
            "including a central-fit table and a central-vs-bootstrap comparison table."
        )
    )
    parser.add_argument(
        "--mv_fit",
        required=True,
        help="JSON file with mV-mPS bootstrap fit results",
    )
    parser.add_argument(
        "--fps_fit",
        required=True,
        help="JSON file with fps-mPS bootstrap fit results",
    )
    parser.add_argument(
        "--mv_fit_bootstrap",
        default="",
        help="Optional legacy override for the mV bootstrap fit JSON",
    )
    parser.add_argument(
        "--fps_fit_bootstrap",
        default="",
        help="Optional legacy override for the fps bootstrap fit JSON",
    )
    parser.add_argument(
        "--fit_table",
        default="",
        help="Output LaTeX table file for the central-fit chiPT table",
    )
    parser.add_argument(
        "--boot_table",
        default="",
        help="Output LaTeX table file comparing central and bootstrap fit results",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="Legacy alias for --fit_table",
    )
    parser.add_argument(
        "--output_wilson_points",
        default="",
        help="Optional output LaTeX table file for Wilson points",
    )
    args = parser.parse_args()

    mv_fit_data = read_json(args.mv_fit)
    mv_fit_bootstrap_data = read_json(args.mv_fit_bootstrap or args.mv_fit)
    fps_fit_data = read_json(args.fps_fit)
    fps_fit_bootstrap_data = read_json(args.fps_fit_bootstrap or args.fps_fit)

    fit_table = args.fit_table or args.output_file
    if not fit_table:
        parser.error("one of --fit_table or --output_file is required")
    if not args.boot_table:
        parser.error("--boot_table is required")

    mv_rows = collect_observable_rows(mv_fit_data, mv_fit_bootstrap_data)
    fps_rows = collect_observable_rows(fps_fit_data, fps_fit_bootstrap_data)
    build_combined_table(mv_rows, fps_rows, fit_table)

    mv_comparison_rows = collect_comparison_rows(
        mv_fit_data,
        mv_fit_bootstrap_data,
        include_mdwf=True,
        include_wilson=False,
    )
    fps_comparison_rows = collect_comparison_rows(
        fps_fit_data,
        fps_fit_bootstrap_data,
        include_mdwf=True,
        include_wilson=False,
    )
    build_bootstrap_comparison_table(
        mv_comparison_rows,
        fps_comparison_rows,
        args.boot_table,
    )

    if args.output_wilson_points:
        wilson_point_rows = collect_wilson_point_rows(mv_fit_bootstrap_data)
        build_wilson_points_table(wilson_point_rows, args.output_wilson_points)


if __name__ == "__main__":
    main()
