#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


TRUE_VALUES = {"true", "1", "yes", "y"}
CSV_METADATA_COLUMNS_TO_DROP = {
    "therm",
    "delta_traj",
    "delta_traj_conf",
    "delta_traj_w0",
    "delta_traj_q",
    "delta_traj_ps",
    "mres_p_start",
    "mres_p_end",
    "ps_p_start",
    "ps_p_end",
    "v_p_start",
    "v_p_end",
    "fps_p_start",
    "fps_p_end",
    "Z_p_start",
    "Z_p_end",
}


def safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


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


def first_present(*values, default=""):
    for value in values:
        out = to_float(value, default="")
        if out != "":
            return out
    return default


def extract_value_err(obj, key):
    if not isinstance(obj, dict):
        return "", ""

    if key in obj and isinstance(obj[key], dict):
        inner = obj[key]
        if "mean" in inner or "sdev" in inner:
            return to_float(inner.get("mean")), to_float(inner.get("sdev"))
        if "value" in inner or "err" in inner:
            return to_float(inner.get("value")), to_float(inner.get("err"))

    if key in obj:
        return to_float(obj.get(key)), to_float(obj.get(f"{key}_err"))

    return "", ""


def metadata_value(row, key):
    value = row.get(key, "")
    return str(value).strip()


KEY_FIELDS = ["NF", "Nt", "Ns", "Ls", "beta", "mass", "mpv", "alpha", "a5", "M5"]
YM_KEY_FIELDS = ["NF", "Nt", "Ns", "beta"]


def row_key(row):
    return tuple(metadata_value(row, field) for field in KEY_FIELDS)


def row_key_ym(row):
    return tuple(metadata_value(row, field) for field in YM_KEY_FIELDS)


def path_key(path):
    parts = Path(path).parts
    found = {
        "NF": "",
        "Nt": "",
        "Ns": "",
        "Ls": "",
        "beta": "",
        "mass": "",
        "mpv": "",
        "alpha": "",
        "a5": "",
        "M5": "",
    }

    for part in parts:
        if part.startswith("NF"):
            found["NF"] = part[2:]
        elif part.startswith("Nt"):
            found["Nt"] = part[2:]
        elif part.startswith("Ns"):
            found["Ns"] = part[2:]
        elif part.startswith("Ls"):
            found["Ls"] = part[2:]
        elif part.startswith("B"):
            found["beta"] = part[1:]
        elif part.startswith("mpv"):
            found["mpv"] = part[3:]
        elif part.startswith("alpha"):
            found["alpha"] = part[5:]
        elif part.startswith("a5"):
            found["a5"] = part[2:]
        elif part.startswith("M5"):
            found["M5"] = part[2:]
        elif part.startswith("M") and not part.startswith("M5") and found["mass"] == "":
            found["mass"] = part[1:]

    return tuple(found[field] for field in KEY_FIELDS)


def path_key_ym(path):
    parts = Path(path).parts
    found = {
        "NF": "",
        "Nt": "",
        "Ns": "",
        "beta": "",
    }

    for part in parts:
        if part.startswith("NF"):
            found["NF"] = part[2:]
        elif part.startswith("Nt"):
            found["Nt"] = part[2:]
        elif part.startswith("Ns"):
            found["Ns"] = part[2:]
        elif part.startswith("B"):
            found["beta"] = part[1:]

    return tuple(found[field] for field in YM_KEY_FIELDS)


def read_json(path):
    with path.open("r") as handle:
        return json.load(handle)


def read_mres(path):
    data = read_json(path)
    fit = safe_get(data, "mres_extract", default={})
    full = safe_get(data, "ensembles", "full", default={})
    meas = safe_get(data, "ensembles", "meas", default={})
    full_traj_numbers = full.get("traj_numbers", [])

    n_traj = to_int(full.get("n_cfg"))
    if n_traj == "" and isinstance(full_traj_numbers, list):
        n_traj = len(full_traj_numbers)

    return {
        "has_mres_json": True,
        "mres": to_float(fit.get("value")),
        "mres_err": to_float(fit.get("error")),
        "chi2_mres": to_float(fit.get("reduced_chi2")),
        "tau_mres": to_float(safe_get(fit, "mres_tau_int", "tau_int")),
        "tau_mres_err": to_float(safe_get(fit, "mres_tau_int", "tau_int_err")),
        "n_traj": n_traj,
        "n_cfg_mres": to_int(meas.get("n_cfg")),
    }


def read_spectrum(path):
    data = read_json(path)
    results = safe_get(data, "results", default={})
    summary = safe_get(results, "summary", default={})
    standard = safe_get(results, "standard_fit", default={})
    bootstrap = safe_get(results, "bootstrap_fit", default={})
    n_cfg_spectrum = to_int(
        safe_get(
            data,
            "selection",
            "n_used",
            default=safe_get(data, "data_shape", "Ncfg"),
        )
    )

    mps, mps_err = extract_value_err(summary, "am_ps")
    if mps == "":
        mps, mps_err = extract_value_err(safe_get(standard, "PP", default={}), "am_ps")
    if mps == "":
        mps, mps_err = extract_value_err(safe_get(bootstrap, "PP", default={}), "am_ps")

    mv, mv_err = extract_value_err(summary, "am_v")
    if mv == "":
        mv, mv_err = extract_value_err(safe_get(standard, "VV", default={}), "am_v")
    if mv == "":
        mv, mv_err = extract_value_err(safe_get(bootstrap, "VV", default={}), "am_v")

    fps, fps_err = extract_value_err(summary, "af_ps")
    if fps == "":
        fps, fps_err = extract_value_err(
            safe_get(standard, "simultaneous_PP_A0P", default={}),
            "af_ps",
        )
    if fps == "":
        fps, fps_err = extract_value_err(
            safe_get(bootstrap, "simultaneous_PP_A0P", default={}),
            "af_ps",
        )

    z_a, z_a_err = extract_value_err(summary, "Z_A")
    if z_a == "":
        z_a, z_a_err = extract_value_err(safe_get(standard, "Z_A", default={}), "Z_A")
    if z_a == "":
        z_a, z_a_err = extract_value_err(safe_get(bootstrap, "Z_A", default={}), "Z_A")

    return {
        "has_spectrum_json": True,
        "n_cfg_spectrum": n_cfg_spectrum,
        "mps": mps,
        "mps_err": mps_err,
        "chi2_ps": first_present(
            safe_get(standard, "PP", "fit_stats", "chi2_over_dof"),
            safe_get(bootstrap, "PP", "fit_stats", "chi2_over_dof"),
        ),
        "mv": mv,
        "mv_err": mv_err,
        "chi2_v": first_present(
            safe_get(standard, "VV", "fit_stats", "chi2_over_dof"),
            safe_get(bootstrap, "VV", "fit_stats", "chi2_over_dof"),
            safe_get(standard, "V", "fit_stats", "chi2_over_dof"),
            safe_get(bootstrap, "V", "fit_stats", "chi2_over_dof"),
        ),
        "fps": fps,
        "fps_err": fps_err,
        "chi2_fps": first_present(
            safe_get(standard, "simultaneous_PP_A0P", "fit_stats", "chi2_over_dof"),
            safe_get(bootstrap, "simultaneous_PP_A0P", "fit_stats", "chi2_over_dof"),
        ),
        "Z_A": z_a,
        "Z_A_err": z_a_err,
        "chi2_Z": first_present(
            safe_get(standard, "Z_A", "fit_stats", "chi2_over_dof"),
            safe_get(bootstrap, "Z_A", "fit_stats", "chi2_over_dof"),
        ),
    }


def read_wflow(path):
    data = read_json(path)
    summary = safe_get(data, "summary", default={})
    w0, w0_err = extract_value_err(summary, "w0")

    if w0 == "":
        w0 = to_float(summary.get("w0"))
        w0_err = to_float(summary.get("w0_err"))

    return {
        "has_wflow_json": True,
        "w0": w0,
        "w0_err": w0_err,
        "tau_w0": to_float(safe_get(data, "tau_int", "w0", "tau_int")),
        "tau_w0_err": to_float(safe_get(data, "tau_int", "w0", "tau_int_err")),
    }


def read_plaq(path):
    data = read_json(path)
    hmc = safe_get(data, "hmc_extract", default={})

    return {
        "has_plaq_json": True,
        "n_traj": to_int(hmc.get("n_traj_total")),
        "plaq": to_float(hmc.get("plaq")),
        "plaq_err": to_float(hmc.get("plaq_err")),
        "tau_plaq": to_float(hmc.get("tau_int_plaq")),
        "tau_plaq_err": to_float(hmc.get("tau_int_plaq_err")),
        "acceptance": to_float(hmc.get("accept_ratio")),
        "n_cfg_plaq": to_int(hmc.get("n_conf")),
    }


def build_file_map(paths, key_fn):
    out = {}

    for raw_path in paths:
        key = key_fn(raw_path)
        out[key] = Path(raw_path)

    return out


def build_output_row(row, flag_columns, mres_map, spectrum_map, wflow_map, plaq_dyn_map, plaq_ym_map):
    out = dict(row)
    out["selected_any_flag"] = any(bool(row[column]) for column in flag_columns)
    key = row_key(row)
    ym_key = row_key_ym(row)
    is_ym = metadata_value(row, "NF") == "0"

    out.update(
        {
            "has_mres_json": False,
            "has_spectrum_json": False,
            "has_wflow_json": False,
            "mres": "",
            "mres_err": "",
            "chi2_mres": "",
            "tau_mres": "",
            "tau_mres_err": "",
            "n_traj": "",
            "n_cfg_mres": "",
            "mps": "",
            "mps_err": "",
            "chi2_ps": "",
            "n_cfg_spectrum": "",
            "mv": "",
            "mv_err": "",
            "chi2_v": "",
            "fps": "",
            "fps_err": "",
            "chi2_fps": "",
            "Z_A": "",
            "Z_A_err": "",
            "chi2_Z": "",
            "w0": "",
            "w0_err": "",
            "tau_w0": "",
            "tau_w0_err": "",
            "has_plaq_json": False,
            "plaq": "",
            "plaq_err": "",
            "tau_plaq": "",
            "tau_plaq_err": "",
            "acceptance": "",
            "n_cfg_plaq": "",
        }
    )

    mres_path = mres_map.get(key)
    spectrum_path = spectrum_map.get(key)
    wflow_path = wflow_map.get(key)
    plaq_path = plaq_ym_map.get(ym_key) if is_ym else plaq_dyn_map.get(key)

    if mres_path is not None and mres_path.exists():
        try:
            out.update(read_mres(mres_path))
        except Exception as exc:
            print(f"[WARN] Failed to read {mres_path}: {exc}")

    if spectrum_path is not None and spectrum_path.exists():
        try:
            out.update(read_spectrum(spectrum_path))
        except Exception as exc:
            print(f"[WARN] Failed to read {spectrum_path}: {exc}")

    if wflow_path is not None and wflow_path.exists():
        try:
            out.update(read_wflow(wflow_path))
        except Exception as exc:
            print(f"[WARN] Failed to read {wflow_path}: {exc}")

    if plaq_path is not None and plaq_path.exists():
        try:
            out.update(read_plaq(plaq_path))
        except Exception as exc:
            print(f"[WARN] Failed to read {plaq_path}: {exc}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", "--input_file", dest="metadata_csv", required=True)
    parser.add_argument("--mres", nargs="*", default=[])
    parser.add_argument("--spectrum", nargs="*", default=[])
    parser.add_argument("--wflow", nargs="*", default=[])
    parser.add_argument("--plaq_YM", nargs="*", default=[])
    parser.add_argument("--plaq_NF", nargs="*", default=[])
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--include_unflagged",
        action="store_true",
        help="Keep ensembles with no TRUE value in any use_in_* column.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata_csv)
    output_path = Path(args.output_file)

    with metadata_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        metadata_columns = reader.fieldnames or []

    flag_columns = [column for column in metadata_columns if column.startswith("use_in_")]
    if not flag_columns:
        raise ValueError(f"No 'use_in_' flag columns found in {metadata_path}")

    for row in rows:
        for column in flag_columns:
            row[column] = str(row.get(column, "")).strip().lower() in TRUE_VALUES
        row["selected_any_flag"] = any(row[column] for column in flag_columns)

    if not args.include_unflagged:
        rows = [row for row in rows if row["selected_any_flag"]]

    if not rows:
        raise ValueError("No ensembles selected after applying metadata flags.")

    mres_map = build_file_map(args.mres, path_key)
    spectrum_map = build_file_map(args.spectrum, path_key)
    wflow_map = build_file_map(args.wflow, path_key)
    plaq_dyn_map = build_file_map(args.plaq_NF, path_key)
    plaq_ym_map = build_file_map(args.plaq_YM, path_key_ym)

    output_rows = [
        build_output_row(
            row,
            flag_columns,
            mres_map,
            spectrum_map,
            wflow_map,
            plaq_dyn_map,
            plaq_ym_map,
        )
        for row in rows
    ]

    n_mres = sum(bool(row["has_mres_json"]) for row in output_rows)
    n_spectrum = sum(bool(row["has_spectrum_json"]) for row in output_rows)
    n_wflow = sum(bool(row["has_wflow_json"]) for row in output_rows)
    n_plaq = sum(bool(row["has_plaq_json"]) for row in output_rows)

    drop_columns = set(flag_columns)
    drop_columns.add("selected_any_flag")
    drop_columns.update(
        column for column in output_rows[0] if column.startswith("has_")
    )
    drop_columns.update(CSV_METADATA_COLUMNS_TO_DROP)

    output_rows = [
        {key: value for key, value in row.items() if key not in drop_columns}
        for row in output_rows
    ]

    metric_order = [
        "n_traj",
        "plaq",
        "plaq_err",
        "tau_plaq",
        "tau_plaq_err",
        "acceptance",
        "n_cfg_plaq",
        "mres",
        "mres_err",
        "chi2_mres",
        "n_cfg_mres",
        "tau_mres",
        "tau_mres_err",
        "mps",
        "mps_err",
        "chi2_ps",
        "n_cfg_spectrum",
        "mv",
        "mv_err",
        "chi2_v",
        "fps",
        "fps_err",
        "chi2_fps",
        "Z_A",
        "Z_A_err",
        "chi2_Z",
        "w0",
        "w0_err",
        "tau_w0",
        "tau_w0_err",
    ]

    fieldnames = list(output_rows[0].keys())
    metric_columns = [name for name in metric_order if name in fieldnames]
    metadata_columns = [name for name in fieldnames if name not in metric_columns]
    fieldnames = metadata_columns + metric_columns
    output_rows = [{name: row.get(name, "") for name in fieldnames} for row in output_rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"Wrote {output_path} "
        f"with {len(output_rows)} ensembles "
        f"(mres={n_mres}, spectrum={n_spectrum}, wflow={n_wflow}, plaq={n_plaq})"
    )


if __name__ == "__main__":
    main()
