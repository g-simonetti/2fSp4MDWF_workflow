#!/usr/bin/env python3
import argparse
import json
import math
import os
import re

import pandas as pd


PATH_RE = re.compile(
    r"NF(?P<NF>\d+)/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
)


def parse_params(path: str) -> dict:
    match = PATH_RE.search(path)
    if match is None:
        raise ValueError(f"Cannot parse metadata from path: {path}")
    data = match.groupdict()
    out = {}
    for key, value in data.items():
        if key in {"NF", "Nt", "Ns", "Ls"}:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def key_from_params(params: dict):
    return (
        int(params["NF"]),
        int(params["Nt"]),
        int(params["Ns"]),
        int(params["Ls"]),
        float(params["beta"]),
        float(params["mass"]),
        float(params["mpv"]),
        float(params["alpha"]),
        float(params["a5"]),
        float(params["M5"]),
    )


def format_floatish(value, fmt=".3g"):
    try:
        if value is None:
            return "--"
        if pd.isna(value):
            return "--"
        return format(float(value), fmt)
    except Exception:
        return "--"


def format_value_error(value, error, max_decimals=8):
    try:
        if value is None or error is None:
            return "--"
        if pd.isna(value) or pd.isna(error):
            return "--"
        value = float(value)
        error = abs(float(error))
    except Exception:
        return "--"

    if error == 0.0:
        return format(value, ".6g")

    exponent = math.floor(math.log10(error))
    decimals = max(0, min(max_decimals, -exponent + 1))
    err_digits = int(round(error * (10**decimals)))
    value_str = f"{value:.{decimals}f}"
    return f"{value_str}({err_digits})"


def format_fit_parameter(fit, index):
    if fit is None:
        return "--"
    params = fit.get("params")
    errors = fit.get("errors")
    if params is None or errors is None:
        return "--"
    return format_value_error(params[index], errors[index])


def format_nu(fit):
    if fit is None:
        return "--"
    if fit.get("free_nu"):
        return format_value_error(fit.get("nu"), fit.get("nu_err"))
    return format_floatish(fit.get("nu"), ".3g")


def build_metadata_lookup(metadata_csv: str):
    df = pd.read_csv(metadata_csv, sep=r"\t|,", engine="python").convert_dtypes()
    lookup = {}
    for _, row in df.iterrows():
        try:
            params = {
                "NF": int(row["NF"]),
                "Nt": int(row["Nt"]),
                "Ns": int(row["Ns"]),
                "Ls": int(row["Ls"]),
                "beta": float(row["beta"]),
                "mass": float(row["mass"]),
                "mpv": float(row["mpv"]),
                "alpha": float(row["alpha"]),
                "a5": float(row["a5"]),
                "M5": float(row["M5"]),
            }
        except Exception:
            continue
        lookup[key_from_params(params)] = str(row.get("name", ""))
    return lookup


def format_entry_names(entries, metadata_lookup):
    names = []
    for entry in entries:
        filepath = entry.get("filepath", "")
        name = None
        if filepath:
            try:
                name = metadata_lookup.get(key_from_params(parse_params(filepath)))
            except Exception:
                name = None
        if not name:
            name = f"N5{int(entry['Ls'])},a{format_floatish(entry['alpha'], '.3g')}"
        names.append(name)
    return ", ".join(names)


def format_ls_list(entries):
    values = [str(int(entry["Ls"])) for entry in entries]
    return ", ".join(values)


def format_alpha_list(entries, family):
    if family == "Shamir":
        return "1"
    values = [format_floatish(entry["alpha"], ".3g") for entry in entries]
    return ", ".join(values)


def format_ls_alpha_pairs(entries, family):
    pairs = []
    for entry in entries:
        ls = int(entry["Ls"])
        alpha = "1" if family == "Shamir" else format_floatish(entry["alpha"], ".3g")
        pairs.append(f"({ls}, {alpha})")
    if len(pairs) <= 2:
        return ", ".join(pairs)

    split = math.ceil(len(pairs) / 2)
    first = ", ".join(pairs[:split])
    second = ", ".join(pairs[split:])
    return rf"\shortstack[c]{{{first} \\ {second}}}"


def format_ls_alpha_lines(entries, family, n_lines=2):
    pairs = []
    for entry in entries:
        ls = int(entry["Ls"])
        alpha = "1" if family == "Shamir" else format_floatish(entry["alpha"], ".3g")
        pairs.append(rf"$({ls}, {alpha})$")

    if not pairs:
        return ["--"] + [""] * (int(n_lines) - 1)

    n_lines = max(1, int(n_lines))
    per_line = math.ceil(len(pairs) / n_lines)
    lines = []
    for idx in range(n_lines):
        chunk = pairs[idx * per_line : (idx + 1) * per_line]
        lines.append(", ".join(chunk))

    while len(lines) < n_lines:
        lines.append("")

    return lines[:n_lines]


def build_rows(fit_json: str):
    with open(fit_json, "r") as handle:
        data = json.load(handle)
    rows = []

    for beta_key in sorted(data["betas"].keys(), key=float):
        block = data["betas"][beta_key]
        beta = float(block["beta"])
        mass = block.get("mass", "--")

        shamir_entries = block.get("shamir_entries", [])
        shamir_fit = block.get("shamir_fit")
        rows.append(
            {
                "family": "Shamir",
                "beta": beta,
                "mass": mass,
                "entries": shamir_entries,
                "fit": shamir_fit,
                "N5_used": format_ls_list(shamir_entries),
                "alpha_used": format_alpha_list(shamir_entries, "Shamir"),
                "ls_alpha_used": format_ls_alpha_pairs(shamir_entries, "Shamir"),
            }
        )

        mobius_entries = block.get("mobius_min_entries", [])
        mobius_fit = block.get("mobius_min_fit")
        rows.append(
            {
                "family": "Möbius",
                "beta": beta,
                "mass": mass,
                "entries": mobius_entries,
                "fit": mobius_fit,
                "N5_used": format_ls_list(mobius_entries),
                "alpha_used": format_alpha_list(mobius_entries, "Möbius"),
                "ls_alpha_used": format_ls_alpha_pairs(mobius_entries, "Möbius"),
            }
        )

    return rows


def write_fit_table(rows, output_table):
    out_dir = os.path.dirname(output_table)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header_line = (
        "Fit & $\\beta$ & $am_0$ & $(N_5,\\alpha)$ used & "
        "$c_1$ & $\\lambda_c$ & $c_2$ & $\\nu$ & $N_{\\rm d.o.f.}$ & $\\chi^2/N_{\\rm d.o.f.}$ \\\\\n"
    )
    tabular_spec = "|l|c|c|c|c|c|c|c|c|c|"

    with open(output_table, "w") as handle:
        handle.write("%%%\\begin{table}[t]\n")
        handle.write("%%%\\centering\n")
        handle.write(f"\\begin{{tabular}}{{{tabular_spec}}}\n")
        handle.write("\\hline\\hline\n")
        handle.write(header_line)
        handle.write("\\hline\n")
        for row in rows:
            fit = row["fit"]
            chi2_red = "--" if fit is None else format_floatish(fit.get("chi2_dof"), ".3f")
            ls_alpha_lines = format_ls_alpha_lines(row.get("entries", []), row["family"], n_lines=2)
            dof = "--" if fit is None else int(fit.get("dof", 0))

            first_line = (
                rf"\multirow{{2}}{{*}}{{{row['family']}}} & "
                rf"\multirow{{2}}{{*}}{{{format_floatish(row['beta'], '.1f')}}} & "
                rf"\multirow{{2}}{{*}}{{{row['mass']}}} & "
                f"{ls_alpha_lines[0]} & "
                rf"\multirow{{2}}{{*}}{{{format_fit_parameter(fit, 0)}}} & "
                rf"\multirow{{2}}{{*}}{{{format_fit_parameter(fit, 1)}}} & "
                rf"\multirow{{2}}{{*}}{{{format_fit_parameter(fit, 2)}}} & "
                rf"\multirow{{2}}{{*}}{{{format_nu(fit)}}} & "
                rf"\multirow{{2}}{{*}}{{{dof}}} & "
                rf"\multirow{{2}}{{*}}{{{chi2_red}}}"
            )
            second_line = f"& & & {ls_alpha_lines[1]} & & & & & & "

            handle.write(first_line + r" \\" + "\n")
            handle.write(second_line + r" \\" + "\n")

        handle.write("\\hline\\hline\n")
        handle.write("\\end{tabular}\n")
        handle.write(
            "%%%\\caption{Fit results for the residual-mass dependence on $N_5$.}\n"
        )
        handle.write("%%%\\label{tab:mres_ls_fit_results}\n")
        handle.write("%%%\\end{table}\n")


def write_selection_table(rows, metadata_lookup, output_table):
    out_dir = os.path.dirname(output_table)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header_line = (
        "Fit & $\\beta$ & $am_0$ & Ensembles used & $(N_5,\\alpha)$ used \\\\\n"
    )
    longtable_spec = (
        "|>{\\centering\\arraybackslash}m{1.7cm}|"
        ">{\\centering\\arraybackslash}m{0.9cm}|"
        ">{\\centering\\arraybackslash}m{1.0cm}|"
        ">{\\centering\\arraybackslash}m{6.0cm}|"
        ">{\\centering\\arraybackslash}m{4.2cm}|"
    )

    with open(output_table, "w") as handle:
        handle.write("%%%\\color{red}\n")
        handle.write(f"%%%\\begin{{longtable}}{{{longtable_spec}}}\n")
        handle.write("%%%\\caption\n")
        handle.write("%%%\\label \\\\\n\n")

        handle.write("% ================= FIRST PAGE HEADER =================\n")
        handle.write(header_line)
        handle.write("\\hline\n")
        handle.write("\\endfirsthead\n\n")

        handle.write("% ================ HEADER FOR PAGE 2+ =================\n")
        handle.write("\\hline\n")
        handle.write(header_line)
        handle.write("\\hline\n")
        handle.write("\\endhead\n\n")

        handle.write("% ================= FOOTER FOR INTERMEDIATE PAGES =================\n")
        handle.write("\\hline\n")
        handle.write("\\endfoot\n\n")

        handle.write("% ================= FINAL FOOTER =================\n")
        handle.write("\\hline\\hline\n")
        handle.write("\\endlastfoot\n\n")

        handle.write("% ===================== TABLE BODY =====================\n")
        for index, row in enumerate(rows):
            line = (
                f"{row['family']} & "
                f"{format_floatish(row['beta'], '.1f')} & "
                f"{row['mass']} & "
                f"{format_entry_names(row['entries'], metadata_lookup)} & "
                f"{row['ls_alpha_used']}"
            )
            if index < len(rows) - 1:
                line += r" \\"
            handle.write(line + "\n")

        handle.write("%%%\\end{longtable}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX longtable summarizing the N5-scan fit results."
    )
    parser.add_argument("--fit_json", required=True, help="Fit summary JSON from the N5 scan fit script")
    parser.add_argument("--metadata_csv", required=True, help="Metadata CSV for ensemble names")
    parser.add_argument("--output_table", required=True, help="Output LaTeX fit-summary table path")
    parser.add_argument("--output_selection_table", default=None, help="Optional output LaTeX table path for the ensembles used in the fits")
    args = parser.parse_args()

    rows = build_rows(args.fit_json)
    write_fit_table(rows, args.output_table)
    if args.output_selection_table:
        metadata_lookup = build_metadata_lookup(args.metadata_csv)
        write_selection_table(rows, metadata_lookup, args.output_selection_table)


if __name__ == "__main__":
    main()
