#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from pathlib import Path


TRUE_VALUES = {"true", "1", "yes", "y"}
PATH_RE = re.compile(
    r"NF(?P<NF>\d+)/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
)
KEY_FIELDS = ["NF", "Nt", "Ns", "Ls", "beta", "mass", "mpv", "alpha", "a5", "M5"]


def is_true(value):
    return str(value).strip().lower() in TRUE_VALUES


def to_float(value, default=""):
    try:
        if value is None:
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def to_int(value, default=""):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_params(path):
    match = PATH_RE.search(path)
    if match is None:
        raise ValueError(f"Cannot parse metadata from path: {path}")
    return match.groupdict()


def metadata_key(parts):
    return tuple(str(parts.get(field, "")).strip() for field in KEY_FIELDS)


def path_key(path):
    return metadata_key(parse_params(path))


def load_metadata(metadata_csv):
    with open(metadata_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    lookup = {}
    for row in rows:
        lookup[metadata_key(row)] = row
    return rows, lookup


def unique_join(values):
    return " | ".join(sorted(dict.fromkeys(value for value in values if value)))


def value_and_error(block, key):
    return to_float(block.get(key)), to_float(block.get(f"{key}_err"))


def simplified_spectrum_formula(observable, fit_key):
    if observable == "mv" and fit_key == "dw2":
        return "m_M^2 = m_M,chi^2(1 + L_m_M m_PS^2 + Q_m_M m_PS^4) + R_m_M a^2"
    if observable == "mv" and fit_key == "wilson_physical":
        return (
            "m_M^2 = m_M,chi^2(1 + L_m_M m_PS^2 + Q_m_M m_PS^4) "
            "+ W_m_M a + R_m_M a^2 + C_m_M a m_PS^2"
        )
    if observable == "fps" and fit_key == "dw2":
        return "f_PS^2 = f_PS,chi^2(1 + L_m_M m_PS^2 + Q_m_M m_PS^4) + R_m_M a^2"
    if observable == "fps" and fit_key == "wilson_physical":
        return (
            "f_PS^2 = f_PS,chi^2(1 + L_m_M m_PS^2 + Q_m_M m_PS^4) "
            "+ W_m_M a + R_m_M a^2 + C_m_M a m_PS^2"
        )
    return ""


def empty_row():
    return {
        "analysis_group": "",
        "use": "",
        "observable": "",
        "fit_family": "",
        "fit_stage": "",
        "beta": "",
        "mass": "",
        "fit_model": "",
        "model_key": "",
        "n_points_used": "",
        "chi2": "",
        "dof": "",
        "chi2_dof": "",
        "c1": "",
        "c1_err": "",
        "lambda_c": "",
        "lambda_c_err": "",
        "c2": "",
        "c2_err": "",
        "nu": "",
        "nu_err": "",
        "m_M_chi_sq": "",
        "m_M_chi_sq_err": "",
        "L_m_M": "",
        "L_m_M_err": "",
        "Q_m_M": "",
        "Q_m_M_err": "",
        "W_m_M": "",
        "W_m_M_err": "",
        "R_m_M": "",
        "R_m_M_err": "",
        "C_m_M": "",
        "C_m_M_err": "",
        "n_bootstrap_copies": "",
        "Ns_used": "",
        "mps_values": "",
        "mps_err_values": "",
        "m_ps_inf": "",
        "m_ps_inf_err_proxy": "",
        "fv_A": "",
        "fv_A_err": "",
        "ensemble_names": "",
        "Ls_used": "",
        "alpha_used": "",
    }


def build_mres_rows(fit_json, metadata_lookup, use_name):
    data = read_json(fit_json)
    rows = []
    fit_models = data.get("fit_model", {})

    for beta_key in sorted(data.get("betas", {}), key=float):
        block = data["betas"][beta_key]
        for family_key, family_label in [("shamir", "Shamir"), ("mobius_min", "Mobius_min")]:
            entries = block.get(f"{family_key}_entries", [])
            fit = block.get(f"{family_key}_fit")
            row = empty_row()
            row["analysis_group"] = "mres_vs_Ls"
            row["use"] = use_name
            row["fit_family"] = family_label
            row["fit_stage"] = "final"
            row["beta"] = to_float(block.get("beta"))
            row["mass"] = block.get("mass", "")
            row["fit_model"] = fit_models.get(family_key, "")
            row["n_points_used"] = len(entries)
            row["Ls_used"] = ", ".join(
                str(to_int(entry.get("Ls"), "")) for entry in entries
            )
            row["alpha_used"] = ", ".join(
                str(entry.get("alpha", "")) for entry in entries
            )

            row["ensemble_names"] = unique_join(
                metadata_lookup.get(path_key(entry.get("filepath", "")), {}).get("name", "")
                for entry in entries
                if entry.get("filepath")
            )

            if fit:
                params = list(fit.get("params", []))
                errors = list(fit.get("errors", []))
                row["chi2"] = to_float(fit.get("chi2"))
                row["dof"] = to_int(fit.get("dof"))
                row["chi2_dof"] = to_float(fit.get("chi2_dof"))
                if len(params) > 0:
                    row["c1"] = to_float(params[0])
                    row["c1_err"] = to_float(errors[0] if len(errors) > 0 else "")
                if len(params) > 1:
                    row["lambda_c"] = to_float(params[1])
                    row["lambda_c_err"] = to_float(errors[1] if len(errors) > 1 else "")
                if len(params) > 2:
                    row["c2"] = to_float(params[2])
                    row["c2_err"] = to_float(errors[2] if len(errors) > 2 else "")
                row["nu"] = to_float(fit.get("nu"))
                row["nu_err"] = to_float(fit.get("nu_err"))

            rows.append(row)

    return rows


def safe_chi2_dof(stage):
    chi2 = to_float(stage.get("chi2"))
    dof = to_float(stage.get("dof"))
    if chi2 == "" or dof in ("", 0.0):
        return ""
    return chi2 / dof


def build_spectrum_rows(fit_path, metadata_rows):
    data = read_json(fit_path)
    rows = []
    observable = str(data.get("observable", ""))
    mdwf_names = unique_join(
        row.get("name", "") for row in metadata_rows if is_true(row.get("use_in_spectrum", ""))
    )

    family_configs = [
        ("dw2", "MDWF", "n_dw_points_used", mdwf_names),
        ("wilson_physical", "Wilson", "n_wilson_points_used", ""),
    ]

    for fit_key, fit_label, count_key, ensemble_names in family_configs:
        fit_block = data.get("fits", {}).get(fit_key, {})
        for fit_stage in ["central_nonlinear", "bootstrap_summary"]:
            stage = fit_block.get(fit_stage)
            if not isinstance(stage, dict):
                continue

            row = empty_row()
            row["analysis_group"] = "spectrum_chiral_continuum"
            row["observable"] = observable
            row["fit_family"] = fit_label
            row["fit_stage"] = fit_stage
            row["fit_model"] = simplified_spectrum_formula(observable, fit_key)
            row["model_key"] = stage.get("model_key", "")
            row["n_points_used"] = to_int(data.get(count_key))
            row["dof"] = to_int(stage.get("dof"))
            row["ensemble_names"] = ensemble_names

            if fit_stage != "bootstrap_summary":
                row["chi2"] = to_float(stage.get("chi2"))
                row["chi2_dof"] = safe_chi2_dof(stage)
            else:
                row["n_bootstrap_copies"] = to_int(
                    stage.get("bootstrap_meta", {}).get("n_requested")
                )

            for key in ["m_M_chi_sq", "L_m_M", "Q_m_M", "W_m_M", "R_m_M", "C_m_M"]:
                value, error = value_and_error(stage, key)
                row[key] = value
                row[f"{key}_err"] = error

            rows.append(row)

    return rows


def build_finite_volume_rows(fv_paths, metadata_lookup):
    rows = []

    for fv_path in fv_paths:
        data = read_json(fv_path)
        row = empty_row()
        row["analysis_group"] = "finite_volume"
        row["use"] = str(data.get("use", ""))
        row["fit_family"] = "finite_volume"
        row["fit_stage"] = "summary"
        row["beta"] = to_float(data.get("beta"))
        row["mass"] = to_float(data.get("mass"))
        row["fit_model"] = "m_PS(L) = m_PS,inf * (1 + A exp(-m_PS,inf L) / (m_PS,inf L)^1.5)"

        ns_values = [to_int(v) for v in data.get("Ns", [])]
        mps_values = [to_float(v) for v in data.get("m_ps", [])]
        mps_err_values = [to_float(v) for v in data.get("m_ps_err", [])]
        input_files = list(data.get("input_files", []))

        row["n_points_used"] = len(ns_values)
        row["Ns_used"] = ", ".join(str(v) for v in ns_values if v != "")
        row["mps_values"] = ", ".join(str(v) for v in mps_values if v != "")
        row["mps_err_values"] = ", ".join(str(v) for v in mps_err_values if v != "")
        row["m_ps_inf"] = to_float(data.get("m_ps_inf"))
        row["m_ps_inf_err_proxy"] = (
            to_float(mps_err_values[-1]) if mps_err_values else ""
        )

        fit = data.get("fit") or {}
        row["fv_A"] = to_float(fit.get("A_mean"))
        row["fv_A_err"] = to_float(fit.get("A_sdev"))

        row["ensemble_names"] = unique_join(
            metadata_lookup.get(path_key(path), {}).get("name", "")
            for path in input_files
        )

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export a CSV collecting mres(Ls) fit summaries and spectrum "
            "chiral-continuum extrapolation summaries."
        )
    )
    parser.add_argument("--mres_fit_json", required=True)
    parser.add_argument("--mv_fit_json", required=True)
    parser.add_argument("--fps_fit_json", required=True)
    parser.add_argument("--finite_volume_jsons", nargs="*", default=[])
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--use", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    metadata_rows, metadata_lookup = load_metadata(args.metadata_csv)
    output_rows = []
    output_rows.extend(build_mres_rows(args.mres_fit_json, metadata_lookup, args.use))
    output_rows.extend(build_spectrum_rows(args.mv_fit_json, metadata_rows))
    output_rows.extend(build_spectrum_rows(args.fps_fit_json, metadata_rows))
    output_rows.extend(build_finite_volume_rows(args.finite_volume_jsons, metadata_lookup))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(empty_row().keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[fit_summary_csv] wrote {output_path}")


if __name__ == "__main__":
    main()
