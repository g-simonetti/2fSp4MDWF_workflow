#!/usr/bin/env python3

import argparse
import json
import math
import os

import numpy as np


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_central_covariance(path):
    data = read_json(path)
    central_fit = (
        data.get("fits", {})
        .get("dw2", {})
        .get("central_nonlinear", {})
    )
    cov = np.asarray(central_fit.get("cov"), dtype=float)
    parameter_order = list(
        central_fit.get(
            "parameter_order",
            ["m_M_chi_sq", "L_m_M", "Q_m_M", "R_m_M"],
        )
    )

    if cov.shape != (4, 4):
        raise ValueError(
            f"Expected a 4x4 central-fit covariance matrix in '{path}', got shape {cov.shape}."
        )

    return cov, parameter_order


def format_tex_number(value, sig_figs=4):
    if not np.isfinite(value):
        return r"\mathrm{nan}"
    if value == 0.0:
        return "0"

    sign = "-" if value < 0 else ""
    value = abs(float(value))
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10**exponent)
    mantissa_str = f"{mantissa:.{sig_figs - 1}f}".rstrip("0").rstrip(".")

    if exponent == 0:
        return f"{sign}{mantissa_str}"
    return rf"{sign}{mantissa_str}\times 10^{{{exponent}}}"


def matrix_to_pmatrix(matrix):
    rows = []
    for row in matrix:
        rows.append(" & ".join(format_tex_number(x) for x in row))
    body = " \\\\\n".join(rows)
    return "\\begin{pmatrix}\n" + body + "\n\\end{pmatrix}"


def write_covariance_tex(output_path, cov, matrix_symbol, basis_labels):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    basis_tex = ",\\; ".join(basis_labels)
    matrix_tex = matrix_to_pmatrix(cov)

    content = "\n".join(
        [
            "% Central-fit covariance matrix.",
            r"%\begin{equation*}",
            rf"{matrix_symbol} = {matrix_tex}",
            r"%\end{equation*}",
            r"%\begin{equation*}",
            rf"%\text{{basis}} = \left({basis_tex}\right)",
            r"%\end{equation*}",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)


def correlation_from_covariance(cov, i, j):
    den = float(np.sqrt(cov[i, i] * cov[j, j]))
    if not np.isfinite(den) or den <= 0.0:
        raise ValueError("Cannot compute correlation coefficient from covariance matrix.")
    return float(cov[i, j] / den)


def write_rho_lq_tex(output_path, rho_mv, rho_fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content = (
        r"\rho_{L,Q}^{m_{\rm V}} = "
        + f"{rho_mv:.3f}"
        + r"\qquad "
        + r"\rho_{L,Q}^{f_{\rm PS}} = "
        + f"{rho_fps:.3f}"
        + "\n"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read the MDWF central-fit covariance matrices stored in the "
            "spectrum extrapolation JSON files and write them as TeX equations."
        )
    )
    parser.add_argument("--mv_fit", required=True, help="Input mv_mps_fit_bootstrap.json")
    parser.add_argument("--fps_fit", required=True, help="Input fps_mps_fit_bootstrap.json")
    parser.add_argument("--cov_v", required=True, help="Output TeX file for the vector covariance matrix")
    parser.add_argument("--cov_ps", required=True, help="Output TeX file for the pseudoscalar-decay-constant covariance matrix")
    parser.add_argument("--rho_LQ", required=True, help="Output TeX file for rho(L,Q) in the mv and fps fits")
    args = parser.parse_args()

    cov_mv, order_mv = load_central_covariance(args.mv_fit)
    cov_fps, order_fps = load_central_covariance(args.fps_fit)

    if order_mv != ["m_M_chi_sq", "L_m_M", "Q_m_M", "R_m_M"]:
        raise ValueError(f"Unexpected mv parameter order: {order_mv}")
    if order_fps != ["m_M_chi_sq", "L_m_M", "Q_m_M", "R_m_M"]:
        raise ValueError(f"Unexpected fps parameter order: {order_fps}")

    rho_mv = correlation_from_covariance(cov_mv, 1, 2)
    rho_fps = correlation_from_covariance(cov_fps, 1, 2)

    write_covariance_tex(
        args.cov_v,
        cov_mv,
        matrix_symbol=r"\mathrm{Cov}_{m_{\rm V}}",
        basis_labels=[
            r"(w_0 m^{\chi}_{\rm V})^2",
            r"L_{m,\rm V}",
            r"Q_{m,\rm V}",
            r"W_{m,\rm V}",
        ],
    )
    write_covariance_tex(
        args.cov_ps,
        cov_fps,
        matrix_symbol=r"\mathrm{Cov}_{f_{\rm PS}}",
        basis_labels=[
            r"(w_0 f^{\chi}_{\rm PS})^2",
            r"L_{f,\rm PS}",
            r"Q_{f,\rm PS}",
            r"W_{f,\rm PS}",
        ],
    )
    write_rho_lq_tex(args.rho_LQ, rho_mv, rho_fps)


if __name__ == "__main__":
    main()
