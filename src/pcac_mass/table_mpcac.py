#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def safe_get(dct, *keys, default=np.nan):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def normalize_flag_series(series):
    s = series.astype("string")
    return s.str.strip().str.lower().isin(["true", "1", "yes", "y"])


def format_phys_err(value, error, force_decimals=None):
    value = float(value)
    error = abs(float(error))

    if not np.isfinite(value) or not np.isfinite(error):
        return "—"
    if error == 0:
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
    err_str = f"{error_r:.{decimals}f}"
    return f"{value_str}({err_str})"


def format_intish(x):
    x = to_float(x)
    if np.isfinite(x):
        return str(int(round(x)))
    return "—"


def format_floatish(x, fmt=".3f"):
    x = to_float(x)
    if np.isfinite(x):
        return format(x, fmt)
    return "—"


def format_floatish_trimmed(x, decimals=3):
    x = to_float(x)
    if not np.isfinite(x):
        return "—"
    s = f"{x:.{int(decimals)}f}"
    return s.rstrip("0").rstrip(".")


def selection_mask(df, use_name):
    candidates = []
    if use_name:
        candidates.append(f"use_in_{use_name}")
        candidates.append(use_name)

    alias_map = {
        "tuned_Mobius": ["use_in_bulkphase_tuned"],
        "bulkphase_tuned": ["use_in_tuned_Mobius"],
    }
    candidates.extend(alias_map.get(use_name, []))

    for col in candidates:
        if col in df.columns:
            return normalize_flag_series(df[col])

    raise KeyError(
        f"Could not find a selection column for use='{use_name}'. "
        f"Tried: {', '.join(candidates)}"
    )


def build_json_path(row, leaf_dir, filename):
    return (
        f"intermediary_data/NF{int(row['NF'])}/"
        f"Nt{int(row['Nt'])}/Ns{int(row['Ns'])}/Ls{int(row['Ls'])}/"
        f"B{row['beta']}/M{row['mass']}/mpv{row['mpv']}/"
        f"alpha{row['alpha']}/a5{row['a5']}/M5{row['M5']}/"
        f"{leaf_dir}/{filename}"
    )


def canonical_key(beta, mass, Nt, Ns, Ls, alpha, a5, m5, mpv):
    return (
        round(to_float(beta), 12),
        round(to_float(mass), 12),
        int(round(to_float(Nt))),
        int(round(to_float(Ns))),
        int(round(to_float(Ls))),
        round(to_float(alpha), 12),
        round(to_float(a5), 12),
        round(to_float(m5), 12),
        round(to_float(mpv), 12),
    )


def key_from_metadata_row(row):
    return canonical_key(
        row["beta"],
        row["mass"],
        row["Nt"],
        row["Ns"],
        row["Ls"],
        row["alpha"],
        row["a5"],
        row["M5"],
        row["mpv"],
    )


def key_from_record(rec):
    return canonical_key(
        rec.get("beta", np.nan),
        rec.get("mass", np.nan),
        rec.get("Nt", np.nan),
        rec.get("Ns", np.nan),
        rec.get("Ls", np.nan),
        rec.get("alpha", np.nan),
        rec.get("a5", np.nan),
        rec.get("m5", np.nan),
        rec.get("mpv", np.nan),
    )


def read_mpcac_json(path):
    data = read_json(path)
    return {
        "beta": to_float(safe_get(data, "parameters", "beta", default=np.nan)),
        "mass": to_float(safe_get(data, "parameters", "mass", default=np.nan)),
        "Nt": to_float(safe_get(data, "parameters", "Nt", default=np.nan)),
        "Ns": to_float(safe_get(data, "parameters", "Ns", default=np.nan)),
        "Ls": to_float(safe_get(data, "parameters", "Ls", default=np.nan)),
        "alpha": to_float(safe_get(data, "parameters", "alpha", default=np.nan)),
        "a5": to_float(safe_get(data, "parameters", "a5", default=np.nan)),
        "m5": to_float(safe_get(data, "parameters", "m5", default=np.nan)),
        "mpv": to_float(safe_get(data, "parameters", "mpv", default=np.nan)),
        "mpcac": to_float(safe_get(data, "mpcac_extract", "value", default=np.nan)),
        "mpcac_err": to_float(safe_get(data, "mpcac_extract", "error", default=np.nan)),
        "pcac_plateau_start": to_float(
            safe_get(
                data,
                "mpcac_extract",
                "plateau_start",
                default=safe_get(data, "analysis_settings", "plateau_start", default=np.nan),
            )
        ),
        "pcac_plateau_end": to_float(
            safe_get(
                data,
                "mpcac_extract",
                "plateau_end",
                default=safe_get(data, "analysis_settings", "plateau_end", default=np.nan),
            )
        ),
        "pcac_chi2_red": to_float(
            safe_get(data, "mpcac_extract", "reduced_chi2", default=np.nan)
        ),
        "_source_file": str(path),
    }


def read_mres_json(path):
    data = read_json(path)
    return {
        "beta": to_float(safe_get(data, "parameters", "beta", default=np.nan)),
        "mass": to_float(safe_get(data, "parameters", "mass", default=np.nan)),
        "Nt": to_float(safe_get(data, "parameters", "Nt", default=np.nan)),
        "Ns": to_float(safe_get(data, "parameters", "Ns", default=np.nan)),
        "Ls": to_float(safe_get(data, "parameters", "Ls", default=np.nan)),
        "alpha": to_float(safe_get(data, "parameters", "alpha", default=np.nan)),
        "a5": to_float(safe_get(data, "parameters", "a5", default=np.nan)),
        "m5": to_float(
            safe_get(
                data,
                "parameters",
                "m5",
                default=safe_get(data, "parameters", "M5", default=np.nan),
            )
        ),
        "mpv": to_float(safe_get(data, "parameters", "mpv", default=np.nan)),
        "mres": to_float(safe_get(data, "mres_extract", "value", default=np.nan)),
        "mres_err": to_float(safe_get(data, "mres_extract", "error", default=np.nan)),
        "_mres_source_file": str(path),
    }


def build_record_map(file_list, reader, kind_name):
    out = {}
    for path in file_list:
        try:
            rec = reader(path)
        except Exception as e:
            print(f"Warning: could not read {kind_name} JSON {path}: {e}")
            continue
        key = key_from_record(rec)
        if key not in out:
            out[key] = rec
    return out


def build_dataframe(metadata_csv, use_name, mpcac_files=None, mres_files=None):
    df_meta = pd.read_csv(metadata_csv, sep="\t|,", engine="python")
    mask = selection_mask(df_meta, use_name)
    df_meta = df_meta.loc[mask].copy()
    df_meta["_meta_order"] = np.arange(len(df_meta))

    mpcac_map = (
        build_record_map(mpcac_files, read_mpcac_json, "PCAC")
        if mpcac_files else None
    )
    mres_map = (
        build_record_map(mres_files, read_mres_json, "mres")
        if mres_files else None
    )

    rows = []
    for _, row in df_meta.iterrows():
        meta_key = key_from_metadata_row(row)
        mpcac_path = build_json_path(row, "pcac_mass", "m_pcac.json")
        mres_path = build_json_path(row, "residual_mass", "m_res.json")

        try:
            rec = (
                dict(mpcac_map[meta_key])
                if mpcac_map is not None and meta_key in mpcac_map
                else read_mpcac_json(mpcac_path)
            )
        except Exception as e:
            print(f"Warning: could not read PCAC JSON for {row['name']}: {e}")
            rec = {
                "beta": to_float(row["beta"]),
                "mass": to_float(row["mass"]),
                "Nt": to_float(row["Nt"]),
                "Ns": to_float(row["Ns"]),
                "Ls": to_float(row["Ls"]),
                "alpha": to_float(row["alpha"]),
                "a5": to_float(row["a5"]),
                "m5": to_float(row["M5"]),
                "mpv": to_float(row["mpv"]),
                "mpcac": np.nan,
                "mpcac_err": np.nan,
                "pcac_plateau_start": np.nan,
                "pcac_plateau_end": np.nan,
                "pcac_chi2_red": np.nan,
                "_source_file": str(mpcac_path),
            }

        try:
            rec.update(
                dict(mres_map[meta_key])
                if mres_map is not None and meta_key in mres_map
                else read_mres_json(mres_path)
            )
        except Exception as e:
            print(f"Warning: could not read mres JSON for {row['name']}: {e}")
            rec["mres"] = np.nan
            rec["mres_err"] = np.nan
            rec["_mres_source_file"] = str(mres_path)

        rec["name"] = str(row["name"])
        rec["_meta_order"] = int(row["_meta_order"])
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["am_pcac_fmt"] = df.apply(
        lambda r: format_phys_err(r["mpcac"], r["mpcac_err"]), axis=1
    )
    df["am_res_fmt"] = df.apply(
        lambda r: format_phys_err(r["mres"], r["mres_err"]), axis=1
    )
    df["am0_plus_amres_fmt"] = df.apply(
        lambda r: format_phys_err(r["mass"] + r["mres"], r["mres_err"])
        if np.isfinite(r["mass"]) and np.isfinite(r["mres"]) and np.isfinite(r["mres_err"])
        else "—",
        axis=1,
    )

    df = (
        df.sort_values(["_meta_order", "name"])
        .drop_duplicates(subset=["name"], keep="first")
        .reset_index(drop=True)
    )
    return df


def build_table(df, output_table, use_name):
    is_tuned_mobius = use_name == "tuned_Mobius"

    header_line = (
        "Ensemble & $\\beta$ & $am_0$ & $N_t$ & $N_s$ & $N_5$ & "
        "$\\alpha$ & $a_5/a$ & $am_5$ & $am_{\\rm PV}$ & "
        "$am_{\\rm PCAC}$ & $am_0 + am_{\\rm res}$ & "
        "$t^{m_{\\rm PCAC}}_{\\rm start}/a$ & "
        "$t^{m_{\\rm PCAC}}_{\\rm end}/a$ & "
        "$\\chi^2/N_{\\rm d.o.f.}$ \\\\\n"
    )
    tabular_spec = "|l|c|c|c|c|c|c|c|c|c|c|c|c|c|c|"

    out_dir = os.path.dirname(output_table)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_table, "w") as f:
        if is_tuned_mobius:
            f.write("%%%\\begin{table}[t]\n")
            f.write("%%%\\centering\n")
            f.write(f"\\begin{{tabular}}{{{tabular_spec}}}\n")
            f.write("\\hline\\hline\n")
            f.write(header_line)
            f.write("\\hline\n")
        else:
            f.write("%%%\\begin{table}[t]\n")
            f.write("%%%\\centering\n")
            f.write(f"\\begin{{tabular}}{{{tabular_spec}}}\n")
            f.write("\\hline\\hline\n")
            f.write(header_line)
            f.write("\\hline\n")

        for _, r in df.iterrows():
            line = (
                f"{r['name']} & "
                f"{format_floatish(r['beta'], '.1f')} & "
                f"{format_floatish(r['mass'], '.3g')} & "
                f"{format_intish(r['Nt'])} & "
                f"{format_intish(r['Ns'])} & "
                f"{format_intish(r['Ls'])} & "
                f"{format_floatish_trimmed(r['alpha'], 3)} & "
                f"{format_floatish(r['a5'], '.3g')} & "
                f"{format_floatish(r['m5'], '.3g')} & "
                f"{format_floatish(r['mpv'], '.3g')} & "
                f"{r['am_pcac_fmt']} & "
                f"{r['am0_plus_amres_fmt']} & "
                f"{format_intish(r['pcac_plateau_start'])} & "
                f"{format_intish(r['pcac_plateau_end'])} & "
                f"{format_floatish(r['pcac_chi2_red'], '.3f')}"
            )
            f.write(line + r" \\" + "\n")

        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(
            "%%%\\caption{Ensembles used to tune the Mobius algorithm, showing "
            "the PCAC and residual-mass extractions.}\n"
        )
        f.write("%%%\\label{tab:pcac_tuned_mobius}\n")
        f.write("%%%\\end{table}\n")

    print(f"[table_pcac] wrote {output_table} with {len(df)} ensembles")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX table for PCAC and residual-mass results, using "
            "the tuned Mobius ensemble selection from metadata."
        )
    )
    parser.add_argument(
        "--metadata_csv",
        "--ensembles_csv",
        dest="metadata_csv",
        required=True,
        help="Path to ensembles.csv",
    )
    parser.add_argument(
        "--output_table",
        "--output_file",
        dest="output_table",
        required=True,
        help="Output LaTeX file",
    )
    parser.add_argument(
        "--use",
        default="tuned_Mobius",
        help="Selection name; metadata rows are filtered using use_in_<use>.",
    )
    parser.add_argument(
        "--mpcac",
        nargs="+",
        default=None,
        help="Optional list of m_pcac.json files supplied directly by the workflow.",
    )
    parser.add_argument(
        "--mres",
        nargs="+",
        default=None,
        help="Optional list of m_res.json files supplied directly by the workflow.",
    )
    args = parser.parse_args()

    df = build_dataframe(
        args.metadata_csv,
        args.use,
        mpcac_files=args.mpcac,
        mres_files=args.mres,
    )
    build_table(df, args.output_table, args.use)


if __name__ == "__main__":
    main()
