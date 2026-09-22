#!/usr/bin/env python3

import argparse
import json
import math
import os


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_command(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def write_commands(path, commands):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(commands) + "\n")


def format_float(value, ndigits=3):
    return f"{float(value):.{ndigits}f}"


def get_mobius_fit(mres_fit_json, beta):
    data = read_json(mres_fit_json)
    beta_data = data["betas"][beta]
    return beta_data["mobius_min_fit"]


def get_floor_mps_inf_l_at_ns24(fv_json):
    data = read_json(fv_json)
    ns_values = data["Ns"]
    inf_l_values = data["m_ps_inf_L"]
    for ns, inf_l in zip(ns_values, inf_l_values):
        if float(ns) == 24.0:
            return math.floor(float(inf_l))
    raise ValueError(f"No Ns=24 entry found in '{fv_json}'.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read selected fit outputs and write single-number TeX definitions."
        )
    )
    parser.add_argument("--mres_fit_json", required=True)
    parser.add_argument("--fv_74006", required=True)
    parser.add_argument("--fv_76006", required=True)
    parser.add_argument("--fv_74002", required=True)
    parser.add_argument("--fv_76002", required=True)
    parser.add_argument("--output_mres_fits", required=True)
    parser.add_argument("--output_fve", required=True)
    args = parser.parse_args()

    fit_b74 = get_mobius_fit(args.mres_fit_json, "7.4")
    fit_b76 = get_mobius_fit(args.mres_fit_json, "7.6")

    chi_b74 = float(fit_b74["chi2_dof"])
    chi_b76 = float(fit_b76["chi2_dof"])
    nu_b74 = float(fit_b74["nu"])
    nu_b76 = float(fit_b76["nu"])

    fve_006 = min(
        get_floor_mps_inf_l_at_ns24(args.fv_74006),
        get_floor_mps_inf_l_at_ns24(args.fv_76006),
    )
    fve_002 = min(
        get_floor_mps_inf_l_at_ns24(args.fv_74002),
        get_floor_mps_inf_l_at_ns24(args.fv_76002),
    )

    write_commands(
        args.output_mres_fits,
        [
            format_command("ChiLargestBeta", format_float(chi_b76, ndigits=1)),
            format_command("ChiSmallestBeta", format_float(chi_b74, ndigits=1)),
            format_command("NuLargestBeta", format_float(nu_b76, ndigits=1)),
            format_command("NuSmallestBeta", format_float(nu_b74, ndigits=1)),
        ],
    )
    write_commands(
        args.output_fve,
        [
            format_command("FveLargestMass", str(fve_006)),
            format_command("FveSmallestMass", str(fve_002)),
        ],
    )


if __name__ == "__main__":
    main()
