#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


MATCH_COLS = ["NF", "beta", "mass", "Nt", "Ns", "Ls", "alpha", "a5", "m5", "mpv"]


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def safe_get(dct, *keys, default=np.nan):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def to_float(value, default=np.nan):
    try:
        if value is None:
            return default
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def normalize_flag_series(series):
    values = series.astype("string")
    return values.str.strip().str.lower().isin(["true", "1", "yes", "y"])


def format_floatish(value, fmt):
    value = to_float(value)
    if np.isfinite(value):
        return format(value, fmt)
    return "—"


def format_intish(value):
    value = to_float(value)
    if np.isfinite(value):
        return str(int(round(value)))
    return "—"


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
    norm = error / (10**exp)
    sig = 2 if norm < 3 else 1
    decimals = max(0, -exp + sig - 1)

    if force_decimals is not None:
        decimals = force_decimals

    value_r = round(value, decimals)
    error_r = round(error, decimals)
    if value_r == 0:
        value_r = 0.0
    value_str = f"{value_r:.{decimals}f}"

    if error_r < 1:
        err_digits = int(round(error_r * (10**decimals)))
        return f"{value_str}({err_digits})"

    err_str = f"{error_r:.{decimals}f}"
    return f"{value_str}({err_str})"


def extract_from_path(path):
    values = {
        "NF": np.nan,
        "beta": np.nan,
        "mass": np.nan,
        "Nt": np.nan,
        "Ns": np.nan,
        "Ls": np.nan,
        "alpha": np.nan,
        "a5": np.nan,
        "m5": np.nan,
        "mpv": np.nan,
    }

    for part in Path(path).parts:
        if part.startswith("NF"):
            values["NF"] = to_float(part[2:], values["NF"])
        elif part.startswith("B"):
            values["beta"] = to_float(part[1:], values["beta"])
        elif part.startswith("Nt"):
            values["Nt"] = to_float(part[2:], values["Nt"])
        elif part.startswith("Ns"):
            values["Ns"] = to_float(part[2:], values["Ns"])
        elif part.startswith("Ls"):
            values["Ls"] = to_float(part[2:], values["Ls"])
        elif part.startswith("alpha"):
            values["alpha"] = to_float(part[5:], values["alpha"])
        elif part.startswith("a5"):
            values["a5"] = to_float(part[2:], values["a5"])
        elif part.startswith("mpv"):
            values["mpv"] = to_float(part[3:], values["mpv"])
        elif part.startswith("M5"):
            values["m5"] = to_float(part[2:], values["m5"])
        elif part.startswith("M") and not part.startswith("M5") and not np.isfinite(
            values["mass"]
        ):
            values["mass"] = to_float(part[1:], values["mass"])

    return values


def merge_prefer_finite(primary, fallback):
    merged = dict(primary)
    for key, value in fallback.items():
        if not np.isfinite(to_float(merged.get(key, np.nan))):
            merged[key] = value
    return merged


def read_common_parameters(data, path):
    params = safe_get(data, "parameters", default={})
    if not isinstance(params, dict) or not params:
        params = safe_get(data, "inputs", default={})

    from_json = {
        "NF": to_float(safe_get(params, "NF", default=np.nan)),
        "beta": to_float(safe_get(params, "beta", default=np.nan)),
        "mass": to_float(safe_get(params, "mass", default=np.nan)),
        "Nt": to_float(safe_get(params, "Nt", default=np.nan)),
        "Ns": to_float(safe_get(params, "Ns", default=np.nan)),
        "Ls": to_float(safe_get(params, "Ls", default=np.nan)),
        "alpha": to_float(safe_get(params, "alpha", default=np.nan)),
        "a5": to_float(safe_get(params, "a5", default=np.nan)),
        "m5": to_float(
            safe_get(params, "m5", default=safe_get(params, "M5", default=np.nan))
        ),
        "mpv": to_float(safe_get(params, "mpv", default=np.nan)),
    }

    return merge_prefer_finite(from_json, extract_from_path(path))


def read_wflow_json(path):
    data = read_json(path)
    summary = safe_get(data, "summary", default={})

    record = read_common_parameters(data, path)
    record["w0"] = to_float(safe_get(summary, "w0", default=np.nan))
    record["w0_err"] = to_float(safe_get(summary, "w0_err", default=np.nan))
    record["qw0_mean"] = to_float(safe_get(summary, "Qw0_mean", default=np.nan))
    record["qw0_err"] = to_float(safe_get(summary, "Qw0_err", default=np.nan))
    record["tau_q"] = to_float(
        safe_get(data, "tau_int", "Qw0", "tau_int", default=np.nan)
    )
    record["tau_q_err"] = to_float(
        safe_get(data, "tau_int", "Qw0", "tau_int_err", default=np.nan)
    )
    record["tau_w0"] = to_float(
        safe_get(data, "tau_int", "w0", "tau_int", default=np.nan)
    )
    record["tau_w0_err"] = to_float(
        safe_get(data, "tau_int", "w0", "tau_int_err", default=np.nan)
    )
    record["_source_file_wflow"] = str(path)
    return record


def read_mres_json(path):
    data = read_json(path)

    record = read_common_parameters(data, path)
    record["mres"] = to_float(safe_get(data, "mres_extract", "value", default=np.nan))
    record["mres_err"] = to_float(
        safe_get(data, "mres_extract", "error", default=np.nan)
    )
    record["_source_file_mres"] = str(path)
    return record


def read_metadata(metadata_csv, use_name):
    metadata = pd.read_csv(metadata_csv, sep=r"\t|,", engine="python")

    flagcol = f"use_in_{use_name}"
    if flagcol not in metadata.columns:
        raise ValueError(f"Column '{flagcol}' not found in {metadata_csv}")

    metadata = metadata[normalize_flag_series(metadata[flagcol])].copy()
    if metadata.empty:
        raise ValueError(f"No rows selected by column '{flagcol}'")

    metadata["name"] = metadata["name"].astype(str).str.strip()
    numeric_cols = ["NF", "beta", "mass", "Nt", "Ns", "Ls", "alpha", "a5", "mpv"]
    for col in numeric_cols:
        if col in metadata.columns:
            metadata[col] = pd.to_numeric(metadata[col], errors="coerce")

    if "M5" in metadata.columns:
        metadata["m5"] = pd.to_numeric(metadata["M5"], errors="coerce")
    elif "m5" in metadata.columns:
        metadata["m5"] = pd.to_numeric(metadata["m5"], errors="coerce")
    else:
        metadata["m5"] = np.nan

    metadata["_meta_order"] = np.arange(len(metadata))
    return metadata


def record_key(record):
    key = []
    for col in MATCH_COLS:
        value = to_float(record.get(col, np.nan))
        key.append(round(value, 12) if np.isfinite(value) else np.nan)
    return tuple(key)


def build_record_map(paths, reader, label):
    mapping = {}

    for path in paths:
        try:
            record = reader(path)
        except Exception:
            continue

        key = record_key(record)
        if all(not np.isfinite(v) for v in key):
            continue

        mapping[key] = record

    return mapping


def build_dataframe(wflow_files, mres_files, metadata_csv, use_name):
    metadata = read_metadata(metadata_csv, use_name)
    wflow_map = build_record_map(wflow_files, read_wflow_json, "wflow")
    mres_map = build_record_map(mres_files, read_mres_json, "m_res")

    rows = []
    for _, meta_row in metadata.sort_values("_meta_order").iterrows():
        row = meta_row.to_dict()
        key = record_key(row)

        if key in wflow_map:
            row.update(wflow_map[key])
        if key in mres_map:
            row.update(mres_map[key])

        row["w0_fmt"] = format_phys_err(row.get("w0"), row.get("w0_err"))
        row["qw0_fmt"] = format_phys_err(row.get("qw0_mean"), row.get("qw0_err"))
        row["tau_q_fmt"] = format_phys_err(
            row.get("tau_q"), row.get("tau_q_err"), force_decimals=2
        )
        row["tau_w0_fmt"] = format_phys_err(
            row.get("tau_w0"), row.get("tau_w0_err"), force_decimals=2
        )
        row["mres_fmt"] = format_phys_err(row.get("mres"), row.get("mres_err"))
        rows.append(row)

    if not rows:
        raise RuntimeError("No metadata-selected ensembles were available for the table.")

    return pd.DataFrame(rows)


def build_table(df, output_file):
    header_line = (
        "Ensemble & $\\beta$ & $L_s$ & $N_t$ & $N_s$ & $am_0$ & $w_0/a$ & "
        "$\\langle Q_L(w_0^2) \\rangle$ & $\\tau_{\\rm int}^{Q}$ & "
        "$\\tau_{\\rm int}^{w_0}$ & $am_{\\rm res}$ \\\\\n"
    )
    tabular_spec = "|l|c|c|c|c|c|c|c|c|c|c|"

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w") as handle:
        handle.write("%%%\\begin{table}[t]\n")
        handle.write("%%%\\centering\n")
        handle.write(f"\\begin{{tabular}}{{{tabular_spec}}}\n")
        handle.write("\\hline\\hline\n")
        handle.write(header_line)
        handle.write("\\hline\n")

        for idx, (_, row) in enumerate(df.iterrows()):
            line = (
                f"{row['name']} & "
                f"{format_floatish(row.get('beta'), '.1f')} & "
                f"{format_intish(row.get('Ls'))} & "
                f"{format_intish(row.get('Nt'))} & "
                f"{format_intish(row.get('Ns'))} & "
                f"{format_floatish(row.get('mass'), '.2f')} & "
                f"{row['w0_fmt']} & "
                f"{row['qw0_fmt']} & "
                f"{row['tau_q_fmt']} & "
                f"{row['tau_w0_fmt']} & "
                f"{row['mres_fmt']}"
            )
            line += r" \\"
            handle.write(line + "\n")

        handle.write("\\hline\\hline\n")
        handle.write("\\end{tabular}\n")
        handle.write(
            "%%%\\caption{Spectrum ensembles with Wilson-flow and residual-mass observables.}\n"
        )
        handle.write("%%%\\label{tab:spectrum_ensembles}\n")
        handle.write("%%%\\end{table}\n")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX longtable for the spectrum ensembles using "
            "wflow_extract.json, m_res.json, and metadata ordering."
        )
    )
    parser.add_argument("--wflow", nargs="+", required=True, help="List of wflow JSON files")
    parser.add_argument("--mres", nargs="+", required=True, help="List of m_res JSON files")
    parser.add_argument("--metadata_csv", required=True, help="Path to metadata/ensembles.csv")
    parser.add_argument("--output_file", required=True, help="Output LaTeX file")
    parser.add_argument(
        "--use",
        default="spectrum",
        help="Metadata selection name; rows are filtered using use_in_<use>.",
    )
    parser.add_argument(
        "--spectrum",
        nargs="*",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    df = build_dataframe(args.wflow, args.mres, args.metadata_csv, args.use)
    build_table(df, args.output_file)


if __name__ == "__main__":
    main()
