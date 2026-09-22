#!/usr/bin/env python3
"""
Extract HMC/MD statistics from compressed Grid logs and write ONE JSON file.
Most quantities come from the preferred machine payload (tursa if present,
otherwise sunbird), while fullbcs and t_traj retain per-machine values.

Also runs autocorr_time/tau_int.py (via compute_tau_from_file) on the plaquette history,
using therm and the plaquette-spacing metadata from ensembles.csv as the main estimate.
"""

import argparse
import glob
import io
import json
import os
import re
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import zstandard as zstd
import matplotlib.pyplot as plt

# Ensure src/ is importable so we can import autocorr_time.tau_int
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from autocorr_time.tau_int import compute_tau_from_file  # noqa: E402


DEVICE_IDENTIFIER_RE = re.compile(r"Device identifier:\s*(.+)")
HOSTNAME_RE = re.compile(r"This rank is running on host\s+(\S+)")
KNOWN_MACHINES = ("tursa", "sunbird")


class EmptyLogError(ValueError):
    """Raised when a compressed log file contains no readable text lines."""


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------
def apply_plot_styles(plot_styles_arg: str | None):
    if not plot_styles_arg:
        return
    parts = [p.strip() for p in str(plot_styles_arg).split(",") if p.strip()]
    if parts:
        plt.style.use(parts)
    if plt.rcParams.get("text.usetex", False) and shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def normpath_posix(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def read_zst_lines(path: str):
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                yield line.rstrip("\n")


def bootstrap_mean_err(x, n_boot: int = 1000, rng=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(means.std(ddof=1))


def slice_therm_delta(x, therm: int, delta: int, n_conf: int):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or n_conf <= 0:
        return x[:0]
    start = min(int(therm), x.size)
    idx = start + int(delta) * np.arange(int(n_conf), dtype=int)
    idx = idx[idx < x.size]
    return x[idx]


def select_series_by_coordinate(
    x: np.ndarray,
    y: np.ndarray,
    therm: int,
    delta: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if x.size == 0:
        return x[:0], y[:0]

    keep_therm = x >= float(therm)
    x = x[keep_therm]
    y = y[keep_therm]
    if x.size == 0:
        return x[:0], y[:0]

    delta = int(delta)
    if delta <= 1:
        return x, y

    x_int = np.rint(x).astype(int)
    start = int(x_int[0])
    keep_delta = ((x_int - start) % delta) == 0
    return x[keep_delta], y[keep_delta]


def detect_machine_from_log_header(path: str) -> tuple[str, str | None, str | None]:
    device_identifier = None
    host_name = None
    saw_any_line = False

    for i, line in enumerate(read_zst_lines(path), start=1):
        saw_any_line = True
        if device_identifier is None and (m := DEVICE_IDENTIFIER_RE.search(line)):
            device_identifier = m.group(1).strip()
        if host_name is None and (m := HOSTNAME_RE.search(line)):
            host_name = m.group(1).strip()
        if (device_identifier is not None and host_name is not None) or i >= 120:
            break

    if not saw_any_line:
        raise EmptyLogError(f"Compressed log file is empty: {path}")

    if device_identifier is not None:
        if "A100-PCIE" in device_identifier:
            return "sunbird", device_identifier, host_name
        if "A100-SXM4" in device_identifier:
            return "tursa", device_identifier, host_name

    if host_name is not None:
        host_l = host_name.lower()
        if "sunbird" in host_l:
            return "sunbird", device_identifier, host_name
        if host_l.startswith("tu-") or "tursa" in host_l:
            return "tursa", device_identifier, host_name

    raise ValueError(
        f"Could not classify machine for log file:\n  {path}\n"
        f"device_identifier={device_identifier!r}, host_name={host_name!r}\n"
        "Expected an A100-PCIE* header for sunbird or A100-SXM4* header for tursa."
    )


def init_machine_accumulator() -> dict[str, Any]:
    return {
        "accept": 0,
        "reject": 0,
        "fullbcs_raw": [],
        "fullbcs_incr": [],
        "traj_times": [],
        "plaq_pairs": [],
        "traj_numbers": [],
        "traj_length": None,
        "md_steps": None,
        "device_identifiers": set(),
        "hosts": set(),
        "log_files": [],
        "_last_bcs": None,
    }


def machine_has_data(data: dict[str, Any]) -> bool:
    return len(data.get("log_files", [])) > 0


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return make_json_safe(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def choose_primary_machine(data_by_machine: dict[str, dict[str, Any]]) -> str:
    for machine in KNOWN_MACHINES:
        if machine_has_data(data_by_machine[machine]):
            return machine
    raise RuntimeError("No machine data available to choose a primary payload")


def short_device_identifier(device_identifier: str | None) -> str:
    if not device_identifier:
        return "unknown-device"
    return device_identifier.removeprefix("NVIDIA ").strip()


def identifier_label(machine: str, device_identifier: str | None) -> str:
    return f"{machine}-{short_device_identifier(device_identifier)}"


# -----------------------------------------------------------------------------
# Parse log_dir into metadata keys
# -----------------------------------------------------------------------------
def parse_keys_from_log_dir(log_dir: str) -> dict:
    p = normpath_posix(log_dir)

    m = re.search(
        r"(?:^|/)raw_data/NF(?P<NF>\d+)/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/(?P<subdir>.+)/log$",
        p,
    )
    if not m:
        raise ValueError(
            f"Could not parse log_dir:\n  {log_dir}\n"
            "Expected: .../raw_data/NF*/Nt*/Ns*/<subdir>/log"
        )

    NF = int(m.group("NF"))
    Nt = int(m.group("Nt"))
    Ns = int(m.group("Ns"))
    subdir = m.group("subdir")

    if NF == 0:
        m0 = re.match(r"^B(?P<beta>[^/]+)$", subdir)
        if not m0:
            raise ValueError(
                f"NF=0 but subdir does not look like 'B{{beta}}'.\n"
                f"subdir={subdir}"
            )
        return {
            "NF": 0,
            "Nt": Nt,
            "Ns": Ns,
            "beta": m0.group("beta"),
        }

    m1 = re.match(
        r"^Ls(?P<Ls>\d+)/"
        r"B(?P<beta>[^/]+)/"
        r"M(?P<mass>[^/]+)/"
        r"mpv(?P<mpv>[^/]+)/"
        r"alpha(?P<alpha>[^/]+)/"
        r"a5(?P<a5>[^/]+)/"
        r"M5(?P<M5>[^/]+)"
        r"(?:/.*)?$",
        subdir,
    )
    if not m1:
        raise ValueError(
            f"NF>0 but subdir does not start with expected dynamical pattern.\n"
            f"subdir={subdir}\n"
            "Expected start like:\n"
            "  Ls8/B7.4/M0.1/mpv1.0/alpha1.75/a51.0/M51.8\n"
            "Optionally followed by /<run>/..."
        )

    d = m1.groupdict()
    return {
        "NF": NF,
        "Nt": Nt,
        "Ns": Ns,
        "Ls": int(d["Ls"]),
        "beta": d["beta"],
        "mass": d["mass"],
        "mpv": d["mpv"],
        "alpha": d["alpha"],
        "a5": d["a5"],
        "M5": d["M5"],
    }


def lookup_metadata_from_csv(ensembles_csv: str, keys: dict) -> tuple[str, int, int]:
    df = pd.read_csv(ensembles_csv, sep=r"\t|,", engine="python")

    core_required = ("name", "therm", "NF", "Nt", "Ns", "beta")
    for col in core_required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {ensembles_csv}")

    for col in ("NF", "Nt", "Ns", "beta"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def close(series, x):
        return np.isclose(series.to_numpy(dtype=float), float(x), rtol=0.0, atol=1e-12)

    NF = int(keys["NF"])

    if NF == 0:
        sel = (
            close(df["NF"], 0)
            & close(df["Nt"], keys["Nt"])
            & close(df["Ns"], keys["Ns"])
            & close(df["beta"], keys["beta"])
        )
        dfq = df[sel]
    else:
        dyn_required = ("Ls", "mass", "mpv", "alpha", "a5", "M5")
        for col in dyn_required:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {ensembles_csv} "
                    f"(required for dynamical NF>0 ensembles)"
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        sel = (
            close(df["NF"], keys["NF"])
            & close(df["Nt"], keys["Nt"])
            & close(df["Ns"], keys["Ns"])
            & close(df["Ls"], keys["Ls"])
            & close(df["beta"], keys["beta"])
            & close(df["mass"], keys["mass"])
            & close(df["mpv"], keys["mpv"])
            & close(df["alpha"], keys["alpha"])
            & close(df["a5"], keys["a5"])
            & close(df["M5"], keys["M5"])
        )
        dfq = df[sel]

    if len(dfq) != 1:
        cols_show = [
            c
            for c in (
                "NF",
                "Nt",
                "Ns",
                "Ls",
                "beta",
                "mass",
                "mpv",
                "alpha",
                "a5",
                "M5",
                "name",
                "therm",
                "delta_plaq",
                "delta_traj_conf",
                "delta_traj",
            )
            if c in df.columns
        ]
        preview = dfq[cols_show].head(20).to_string(index=False) if len(dfq) else "(no rows)"
        raise ValueError(
            f"Metadata lookup expected 1 row, got {len(dfq)}.\n"
            f"Parsed keys: {keys}\n"
            f"Matching rows preview:\n{preview}\n"
        )

    row = dfq.iloc[0]
    for delta_col in ("delta_traj_conf", "delta_plaq", "delta_traj"):
        if delta_col in row.index and not pd.isna(row[delta_col]):
            return str(row["name"]), int(row["therm"]), int(row[delta_col])

    raise ValueError(
        "Could not find a plaquette trajectory-spacing column in "
        f"{ensembles_csv}. Expected one of: delta_traj_conf, delta_plaq, delta_traj."
    )


# -----------------------------------------------------------------------------
# Log parsing
# -----------------------------------------------------------------------------
def extract_from_logs(log_dir: str):
    patterns = ("log-*.zst", "GridLog-*.out.zst")
    zst_files: list[str] = []
    for pattern in patterns:
        zst_files.extend(glob.glob(os.path.join(log_dir, pattern)))
    if not zst_files:
        raise FileNotFoundError(
            f"No compressed Grid logs found under {log_dir} "
            f"(looked for {', '.join(patterns)})"
        )

    def _key(p: str):
        base = os.path.basename(p)
        for log_name_re in (
            re.compile(r"^log-(\d+)(?:-(\d+))?\.out\.zst$"),
            re.compile(r"^GridLog-(\d+)\.out\.zst$"),
        ):
            match = log_name_re.match(base)
            if match:
                jobid = int(match.group(1))
                run = int(match.group(2)) if match.lastindex and match.lastindex >= 2 and match.group(2) is not None else 0
                return jobid, run
        raise ValueError(
            f"Unexpected log filename format: {base}\n"
            "Expected 'log-<jobid>.out.zst', 'log-<jobid>-<rank>.out.zst', "
            "or 'GridLog-<jobid>.out.zst'."
        )

    zst_paths = sorted(zst_files, key=_key)

    grouped: dict[str, dict[str, Any]] = {
        machine: init_machine_accumulator() for machine in KNOWN_MACHINES
    }
    grouped_by_identifier: dict[str, dict[str, Any]] = {}

    re_acc = re.compile(r"Metropolis_test\s*--\s*ACCEPTED")
    re_rej = re.compile(r"Metropolis_test\s*--\s*REJECTED")
    re_fullbcs = re.compile(r"Full BCs\s*:\s*(\d+)")
    re_traj_time = re.compile(r"Total time for trajectory \(s\)\s*:\s*([0-9.eE+-]+)")
    re_traj_len = re.compile(r"\[Integrator\]\s*Trajectory length\s*:\s*([0-9.eE+-]+)")
    re_md_steps = re.compile(r"\[Integrator\]\s*Number of MD steps\s*:\s*(\d+)")
    re_plaq_val = re.compile(r"Plaquette:\s*\[\s*(\d+)\s*\]\s*([0-9.eE+-]+)")
    re_traj_num = re.compile(r"#\s*Trajectory\s*=\s*(\d+)")

    for path in zst_paths:
        try:
            machine, device_identifier, host_name = detect_machine_from_log_header(path)
        except EmptyLogError:
            # Some directories contain zero-content compressed stubs. Skip them
            # and continue with the real logs for the ensemble.
            print(f"[WARN] Skipping empty compressed log: {path}")
            continue
        bucket = grouped[machine]
        id_key = identifier_label(machine, device_identifier)
        id_bucket = grouped_by_identifier.setdefault(id_key, init_machine_accumulator())
        bucket["log_files"].append(path)
        id_bucket["log_files"].append(path)
        if device_identifier:
            bucket["device_identifiers"].add(device_identifier)
            id_bucket["device_identifiers"].add(device_identifier)
        if host_name:
            bucket["hosts"].add(host_name)
            id_bucket["hosts"].add(host_name)

        want_unsmeared = False

        for line in read_zst_lines(path):
            if re_acc.search(line):
                bucket["accept"] += 1
                id_bucket["accept"] += 1
                continue
            if re_rej.search(line):
                bucket["reject"] += 1
                id_bucket["reject"] += 1
                continue

            if (m := re_traj_len.search(line)):
                bucket["traj_length"] = float(m.group(1))
                id_bucket["traj_length"] = float(m.group(1))
                continue
            if (m := re_md_steps.search(line)):
                bucket["md_steps"] = int(m.group(1))
                id_bucket["md_steps"] = int(m.group(1))
                continue

            if (m := re_fullbcs.search(line)):
                val = int(m.group(1))
                bucket["fullbcs_raw"].append(val)
                id_bucket["fullbcs_raw"].append(val)
                if bucket["_last_bcs"] is not None:
                    diff = val - bucket["_last_bcs"]
                    if diff >= 0:
                        bucket["fullbcs_incr"].append(diff)
                if id_bucket["_last_bcs"] is not None:
                    diff = val - id_bucket["_last_bcs"]
                    if diff >= 0:
                        id_bucket["fullbcs_incr"].append(diff)
                bucket["_last_bcs"] = val
                id_bucket["_last_bcs"] = val
                continue

            if (m := re_traj_time.search(line)):
                bucket["traj_times"].append(float(m.group(1)))
                id_bucket["traj_times"].append(float(m.group(1)))
                continue

            if (m := re_traj_num.search(line)):
                bucket["traj_numbers"].append(int(m.group(1)))
                id_bucket["traj_numbers"].append(int(m.group(1)))
                continue

            if "Unsmeared plaquette" in line:
                want_unsmeared = True
                continue
            if "Smeared plaquette" in line:
                want_unsmeared = False
                continue

            if want_unsmeared and (m := re_plaq_val.search(line)):
                ti = int(m.group(1))
                pv = float(m.group(2))
                bucket["plaq_pairs"].append((ti, pv))
                id_bucket["plaq_pairs"].append((ti, pv))
                continue

    out: dict[str, dict[str, Any]] = {}
    for machine, bucket in grouped.items():
        acc = int(bucket["accept"])
        rej = int(bucket["reject"])
        accept_ratio = acc / (acc + rej) if (acc + rej) > 0 else np.nan
        out[machine] = {
            "accept": acc,
            "reject": rej,
            "accept_ratio": accept_ratio,
            "fullbcs_raw": np.array(bucket["fullbcs_raw"], float),
            "fullbcs_incr": np.array(bucket["fullbcs_incr"], float),
            "traj_times": np.array(bucket["traj_times"], float),
            "plaq_pairs": np.array(bucket["plaq_pairs"], dtype=object),
            "traj_length": bucket["traj_length"],
            "md_steps": bucket["md_steps"],
            "traj_numbers": np.array(bucket["traj_numbers"], int),
            "device_identifiers": sorted(bucket["device_identifiers"]),
            "hosts": sorted(bucket["hosts"]),
            "log_files": list(bucket["log_files"]),
        }

    out_by_identifier: dict[str, dict[str, Any]] = {}
    for label, bucket in grouped_by_identifier.items():
        acc = int(bucket["accept"])
        rej = int(bucket["reject"])
        accept_ratio = acc / (acc + rej) if (acc + rej) > 0 else np.nan
        out_by_identifier[label] = {
            "accept": acc,
            "reject": rej,
            "accept_ratio": accept_ratio,
            "fullbcs_raw": np.array(bucket["fullbcs_raw"], float),
            "fullbcs_incr": np.array(bucket["fullbcs_incr"], float),
            "traj_times": np.array(bucket["traj_times"], float),
            "plaq_pairs": np.array(bucket["plaq_pairs"], dtype=object),
            "traj_length": bucket["traj_length"],
            "md_steps": bucket["md_steps"],
            "traj_numbers": np.array(bucket["traj_numbers"], int),
            "device_identifiers": sorted(bucket["device_identifiers"]),
            "hosts": sorted(bucket["hosts"]),
            "log_files": list(bucket["log_files"]),
        }

    return {
        "by_machine": out,
        "by_identifier": out_by_identifier,
    }


def build_full_series_for_plaquette(data: dict) -> tuple[np.ndarray, np.ndarray]:
    traj_numbers = data["traj_numbers"]
    if traj_numbers.size > 0:
        t_min = int(traj_numbers.min())
        t_max = int(traj_numbers.max())
        n_traj_total = t_max - t_min + 1
    else:
        lengths = [data["fullbcs_raw"].size, data["traj_times"].size]
        n_traj_total = int(max(lengths)) if lengths else 0
        t_min = 0

    plaq_pairs = data["plaq_pairs"]
    if plaq_pairs.size == 0 or n_traj_total <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    traj_idx = plaq_pairs[:, 0].astype(int)
    plaq_vals = plaq_pairs[:, 1].astype(float)

    full_series = np.full(n_traj_total, np.nan, dtype=float)
    for ti, pv in zip(traj_idx, plaq_vals):
        j = ti - t_min
        if 0 <= j < n_traj_total:
            full_series[j] = pv

    # Forward fill
    for i in range(1, n_traj_total):
        if np.isnan(full_series[i]):
            full_series[i] = full_series[i - 1]

    # Fill early segment
    if n_traj_total > 0 and np.isnan(full_series[0]):
        first_valid = np.where(~np.isnan(full_series))[0]
        if first_valid.size > 0:
            full_series[: first_valid[0]] = full_series[first_valid[0]]

    mc_times = np.arange(t_min, t_min + n_traj_total, dtype=int)
    return mc_times, full_series


def build_machine_payload(
    machine: str,
    keys: dict,
    name: str,
    therm: int,
    delta_traj: int,
    data: dict[str, Any],
    out_dir: str,
    plot_styles: str | None,
) -> dict[str, Any]:
    mc_times, plaq_full = build_full_series_for_plaquette(data)

    machine_out_dir = os.path.join(out_dir, machine)
    os.makedirs(machine_out_dir, exist_ok=True)

    plaq_out_dir = os.path.join(machine_out_dir, "plaq")
    os.makedirs(plaq_out_dir, exist_ok=True)

    n_traj_total = int(plaq_full.size)
    usable = max(0, n_traj_total - therm)
    n_conf = usable // int(delta_traj) if int(delta_traj) > 0 else 0

    fullbcs_incr_s = slice_therm_delta(data["fullbcs_incr"], therm, delta_traj, n_conf)
    traj_times_s = slice_therm_delta(data["traj_times"], therm, delta_traj, n_conf)
    plaq_s = slice_therm_delta(plaq_full, therm, delta_traj, n_conf)

    fullbcs_mean, fullbcs_err = bootstrap_mean_err(fullbcs_incr_s)
    bcs_mean, bcs_err = bootstrap_mean_err(fullbcs_incr_s)
    ttraj_mean, ttraj_err = bootstrap_mean_err(traj_times_s)
    plaq_mean, plaq_err = bootstrap_mean_err(plaq_s)

    length_traj = data["traj_length"] if data["traj_length"] is not None else np.nan
    n_steps = data["md_steps"] if data["md_steps"] is not None else np.nan
    accept_ratio = float(data["accept_ratio"]) if np.isfinite(data["accept_ratio"]) else np.nan

    mc_times_tau, plaq_tau = select_series_by_coordinate(
        mc_times,
        plaq_full,
        therm=therm,
        delta=delta_traj,
    )

    if mc_times_tau.size > 0 and np.isfinite(plaq_tau).any():
        tmp_plaq_path = os.path.join(plaq_out_dir, "plaq_history_tmp_for_tau_int.txt")
        with open(tmp_plaq_path, "w") as fpl:
            fpl.write("# traj_number\tplaquette\n")
            for t, pv in zip(mc_times_tau, plaq_tau):
                fpl.write(f"{int(t)}\t{float(pv):.16e}\n")

        tau_int_plaq, tau_int_plaq_err, Nb_est, Nbs_est, found = compute_tau_from_file(
            input_file=tmp_plaq_path,
            out_dir=plaq_out_dir,
            therm=therm,
            plot_styles=plot_styles,
            base_name="tau_int",
        )

        try:
            os.remove(tmp_plaq_path)
        except OSError:
            pass
    else:
        tau_int_plaq = np.nan
        tau_int_plaq_err = np.nan
        Nb_est = 0
        Nbs_est = 0
        found = False

    return {
        "keys_from_path": keys,
        "ensemble": {
            "name": name,
            "therm": int(therm),
            "delta_traj_conf": int(delta_traj),
            "delta_traj": int(delta_traj),
        },
        "hmc_extract": {
            "accept": int(data["accept"]),
            "reject": int(data["reject"]),
            "accept_ratio": float(accept_ratio),
            "n_traj_total": int(n_traj_total),
            "n_conf": int(n_conf),
            "fullbcs": float(fullbcs_mean),
            "fullbcs_err": float(fullbcs_err),
            "bcs": float(bcs_mean),
            "bcs_err": float(bcs_err),
            "t_traj": float(ttraj_mean),
            "t_traj_err": float(ttraj_err),
            "plaq": float(plaq_mean),
            "plaq_err": float(plaq_err),
            "tau_int_plaq": float(tau_int_plaq),
            "tau_int_plaq_err": float(tau_int_plaq_err),
            "Nb_est": None if Nb_est is None else int(Nb_est),
            "Nbs_est": None if Nbs_est is None else int(Nbs_est),
            "found_window": bool(found),
            "length_traj": float(length_traj),
            "n_steps": int(n_steps) if np.isfinite(n_steps) else None,
        },
        "plaq_history": {
            "t": [int(x) for x in mc_times.tolist()],
            "plaq": [float(x) for x in plaq_full.tolist()],
            "forward_filled": True,
            "includes_pre_therm": True,
        },
        "tau_int_outputs_dir": plaq_out_dir,
    }


def infer_n_traj_total(data: dict[str, Any]) -> int:
    traj_numbers = data["traj_numbers"]
    if traj_numbers.size > 0:
        t_min = int(traj_numbers.min())
        t_max = int(traj_numbers.max())
        return t_max - t_min + 1

    lengths = [data["fullbcs_raw"].size, data["traj_times"].size]
    return int(max(lengths)) if lengths else 0


def build_identifier_timing_payload(data: dict[str, Any], therm: int, delta_traj: int) -> dict[str, float]:
    n_traj_total = infer_n_traj_total(data)
    usable = max(0, n_traj_total - therm)
    n_conf = usable // int(delta_traj) if int(delta_traj) > 0 else 0

    fullbcs_incr_s = slice_therm_delta(data["fullbcs_incr"], therm, delta_traj, n_conf)
    traj_times_s = slice_therm_delta(data["traj_times"], therm, delta_traj, n_conf)

    fullbcs_mean, fullbcs_err = bootstrap_mean_err(fullbcs_incr_s)
    ttraj_mean, ttraj_err = bootstrap_mean_err(traj_times_s)

    return {
        "fullbcs": float(fullbcs_mean),
        "fullbcs_err": float(fullbcs_err),
        "t_traj": float(ttraj_mean),
        "t_traj_err": float(ttraj_err),
    }


def combine_machine_payloads(
    payload_by_machine: dict[str, dict[str, Any]],
    identifier_payloads: dict[str, dict[str, float]],
    primary_machine: str,
) -> dict[str, Any]:
    primary = payload_by_machine[primary_machine]
    combined = {
        "keys_from_path": primary["keys_from_path"],
        "ensemble": primary["ensemble"],
        "hmc_extract": dict(primary["hmc_extract"]),
        "plaq_history": primary["plaq_history"],
        "tau_int_outputs_dir": primary["tau_int_outputs_dir"],
    }

    for field in ("fullbcs", "fullbcs_err", "t_traj", "t_traj_err"):
        combined["hmc_extract"][field] = {
            identifier: payload[field]
            for identifier, payload in identifier_payloads.items()
        }

    return combined


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract HMC statistics from compressed Grid logs into one machine-grouped JSON output."
    )
    parser.add_argument("log_dir")
    parser.add_argument("--ensembles_csv", required=True, help="Path to metadata/ensembles.csv")
    parser.add_argument("--plot_styles", default=None, help="Matplotlib style(s): comma-separated ok")
    parser.add_argument(
        "--hmc_out",
        required=True,
        help="Output JSON file",
    )
    args = parser.parse_args()

    apply_plot_styles(args.plot_styles)

    keys = parse_keys_from_log_dir(args.log_dir)
    name, therm, delta_traj = lookup_metadata_from_csv(args.ensembles_csv, keys)
    print(f"[meta] name={name} therm={therm} delta_traj={delta_traj}")

    extracted = extract_from_logs(args.log_dir)
    data_by_machine = extracted["by_machine"]
    data_by_identifier = extracted["by_identifier"]

    out_dir = os.path.dirname(os.path.abspath(args.hmc_out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    payload_by_machine: dict[str, dict[str, Any]] = {}

    for machine in KNOWN_MACHINES:
        if not machine_has_data(data_by_machine[machine]):
            print(f"[log_ensembles_extract] {machine}: no data found, skipping JSON section")
            continue

        payload_by_machine[machine] = build_machine_payload(
            machine=machine,
            keys=keys,
            name=name,
            therm=therm,
            delta_traj=delta_traj,
            data=data_by_machine[machine],
            out_dir=out_dir,
            plot_styles=args.plot_styles,
        )

    if not payload_by_machine:
        raise RuntimeError(f"No machine-specific output was produced for log_dir={args.log_dir}")

    primary_machine = choose_primary_machine(data_by_machine)
    identifier_payloads = {
        label: build_identifier_timing_payload(data, therm, delta_traj)
        for label, data in data_by_identifier.items()
        if machine_has_data(data)
    }
    payload = combine_machine_payloads(payload_by_machine, identifier_payloads, primary_machine)

    with open(args.hmc_out, "w") as f:
        json.dump(make_json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)

    print(f"[log_ensembles_extract] wrote JSON → {args.hmc_out}")
    print(f"[log_ensembles_extract] primary machine for shared quantities → {primary_machine}")
    for machine in KNOWN_MACHINES:
        if machine not in payload_by_machine:
            continue
        section = payload_by_machine[machine]
        h = section["hmc_extract"]
        print(f"[log_ensembles_extract] {machine} tau_int outputs written in → {section['tau_int_outputs_dir']}")
        print(
            f"[log_ensembles_extract] {machine} plaq tau_int (Wolff Gamma-method): "
            f"{h['tau_int_plaq']:.6g} ± {h['tau_int_plaq_err']:.3g}  "
            f"(W={h['Nb_est']}, found={h['found_window']})"
        )

if __name__ == "__main__":
    main()
