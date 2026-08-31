#!/usr/bin/env python3
import argparse
import json
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FLOAT_TOKEN = r"[0-9]+(?:\.[0-9]+)?"
PAT_DYN = re.compile(
    r".*/NF(?P<NF>\d+)/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>{ft})/M(?P<mass>{ft})/mpv(?P<mpv>{ft})/"
    r"alpha(?P<alpha>{ft})/a5(?P<a5>{ft})/M5(?P<M5>{ft})/.*".format(ft=FLOAT_TOKEN)
)
PAT_YM = re.compile(
    r".*/NF(?P<NF>\d+)/Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/B(?P<beta>{ft})/.*".format(ft=FLOAT_TOKEN)
)
MARKERS = ["^", "v", "<", ">", "D", "P", "X"]
MRES_MARKERS = ["*", "o", "s", "D", "p"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build bulk-phase summary plots from JSON inputs.")
    parser.add_argument("--ensembles_csv", required=True)
    parser.add_argument("--plaq_avg", nargs="+", required=True, help="Input log_hmc_extract.json files.")
    parser.add_argument("--mres_data", nargs="*", default=[], help="Input m_res.json files.")
    parser.add_argument("--label", default="no")
    parser.add_argument("--plot_styles", default=None)
    parser.add_argument("--tuned_masses", required=True)
    parser.add_argument("--tuned_history", required=True)
    parser.add_argument("--shamir_summary", required=True)
    parser.add_argument("--history_masses", nargs="*", type=float, default=[0.01, 0.10])
    return parser.parse_args()


def is_true(value) -> bool:
    return str(value).strip().upper() in {"TRUE", "T", "1", "YES", "Y"}


def sfloat(value): return str(float(str(value).strip()))
def sint(value): return str(int(float(str(value).strip())))


def make_dyn_key(NF, Nt, Ns, Ls, beta, mass, mpv, alpha, a5, M5):
    return (sint(NF), sint(Nt), sint(Ns), sint(Ls), sfloat(beta), sfloat(mass), sfloat(mpv), sfloat(alpha), sfloat(a5), sfloat(M5))


def parse_info(path):
    match = PAT_DYN.match(str(path))
    if match:
        info = match.groupdict()
        return "dyn", make_dyn_key(**info), info

    match = PAT_YM.match(str(path))
    if match:
        info = match.groupdict()
        return "ym", (sint(info["Nt"]), sint(info["Ns"]), sfloat(info["beta"])), info

    return None, None, None


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def read_hmc_observables(path):
    data = load_json(path)
    extract = data.get("hmc_extract", {})
    history = data.get("plaq_history", {})

    plaq = float(extract["plaq"]) if "plaq" in extract else np.nan
    plaq_err = float(extract.get("plaq_err", 0.0)) if np.isfinite(plaq) else np.nan
    observables = {"plaq": plaq, "plaq_err": plaq_err, "history_t": None, "history_p": None}

    t = history.get("t", history.get("mc_time"))
    p = history.get("plaq")
    if t is None or p is None:
        return observables

    history_t = np.asarray(t, dtype=float)
    history_p = np.asarray(p, dtype=float)
    if history_t.size == 0 or history_p.size == 0:
        return observables

    size = min(history_t.size, history_p.size)
    observables["history_t"] = history_t[:size]
    observables["history_p"] = history_p[:size]
    return observables


def read_mres_observables(path):
    extract = load_json(path).get("mres_extract", {})
    value = float(extract["value"]) if "value" in extract else np.nan
    error = float(extract["error"]) if "error" in extract else np.nan
    return {"mres": value, "mres_err": error}


def build_meta_map(ensembles_csv):
    # Read the release metadata once so the plotting pass can stay path-driven.
    frame = pd.read_csv(ensembles_csv, sep=r"\t|,", engine="python")
    meta_map = {}

    for _, row in frame.iterrows():
        try:
            if float(row["NF"]) <= 0:
                continue
            key = make_dyn_key(row["NF"], row["Nt"], row["Ns"], row["Ls"], row["beta"], row["mass"], row["mpv"], row["alpha"], row["a5"], row["M5"])
        except Exception:
            continue
        meta_map[key] = row

    return meta_map


def collect_bulk_phase_entries(plaq_paths, mres_paths, meta_map):
    # Cache averages and histories during collection so each JSON file is parsed once.
    tuned_entries, shamir_entries, ym_entries, mres_entries = [], [], [], []

    for path in plaq_paths:
        kind, key, info = parse_info(path)
        if kind == "ym":
            ym_entries.append({"beta": float(info["beta"]), "path": path, **read_hmc_observables(path)})
            continue

        if kind != "dyn" or key not in meta_map:
            continue

        row = meta_map[key]
        base_entry = {"beta": float(info["beta"]), "mass": float(info["mass"]), **read_hmc_observables(path)}

        if is_true(row.get("use_in_bulkphase_tuned", False)):
            tuned_entries.append(base_entry.copy())
        if is_true(row.get("use_in_bulkphase_Shamir", False)):
            shamir_entries.append(base_entry.copy())

    for path in mres_paths:
        kind, key, info = parse_info(path)
        if kind != "dyn" or key not in meta_map:
            continue
        if not is_true(meta_map[key].get("use_in_bulkphase_mres", False)):
            continue
        mres_entries.append({"beta": float(info["beta"]), "mass": float(info["mass"]), **read_mres_observables(path)})

    return tuned_entries, shamir_entries, ym_entries, mres_entries


def build_color_maps(tuned_entries, shamir_entries, ym_entries, mres_entries):
    # Reuse fixed color assignments across panels so the release plots stay visually aligned.
    beta_values = sorted({entry["beta"] for entry in tuned_entries + shamir_entries + ym_entries})
    beta_cmap = dict(zip(beta_values, mpl.cm.viridis_r(np.linspace(0.1, 1.0, max(1, len(beta_values))))))
    mass_values = sorted({entry["mass"] for entry in tuned_entries + shamir_entries + mres_entries})
    mass_colors = mpl.cm.inferno(np.linspace(0.1, 0.85, max(1, len(mass_values))))
    return beta_cmap, {mass: mass_colors[index] for index, mass in enumerate(mass_values)}


def sorted_points(entries, x_key, y_key, err_key):
    return sorted(
        (entry[x_key], entry[y_key], entry[err_key])
        for entry in entries
        if np.isfinite(entry[y_key]) and np.isfinite(entry[err_key])
    )


def plot_grouped_errorbars(ax, entries, group_key, x_key, y_key, err_key, color_map, labels, markers, linestyle):
    for index, value in enumerate(sorted({entry[group_key] for entry in entries})):
        group = [entry for entry in entries if np.isclose(entry[group_key], value)]
        points = sorted_points(group, x_key, y_key, err_key)
        if not points:
            continue
        xs, ys, yerr = zip(*points)
        ax.errorbar(
            xs, ys, yerr=yerr, fmt=markers[index % len(markers)], ls=linestyle,
            color=color_map[value], label=labels(value)
        )


def set_history_ylim(ax, histories):
    if not histories:
        return

    values = np.concatenate(histories)
    lo, hi = np.percentile(values, [1, 99])
    if np.isclose(lo, hi):
        pad = 0.02 * max(1.0, abs(lo))
    else:
        pad = 0.5 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def plot_tuned_masses(tuned_entries, beta_cmap, output_path):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    plot_grouped_errorbars(
        ax, tuned_entries, "beta", "mass", "plaq", "plaq_err", beta_cmap,
        lambda beta: rf"$\beta={beta}$", MARKERS, ":"
    )
    ax.set_xlabel(r"$am_0$")
    ax.set_ylabel(r"$\langle \mathcal{P} \rangle$")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[::-1], labels[::-1], fontsize="x-small", bbox_to_anchor=(1, 0.9))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_tuned_histories(tuned_entries, mass_cmap, history_masses, show_legend, output_path):
    # Keep only the tuned masses highlighted in the release figure.
    beta_values = sorted({entry["beta"] for entry in tuned_entries})
    fig, axes = plt.subplots(max(1, len(beta_values)), 1, figsize=(3.5, 2.5), sharex=True, layout="constrained")
    if len(beta_values) == 1:
        axes = [axes]

    selected_masses = sorted(history_masses)
    for ax, beta in zip(axes[::-1], beta_values):
        group = sorted(
            [entry for entry in tuned_entries if np.isclose(entry["beta"], beta) and any(np.isclose(entry["mass"], mass) for mass in selected_masses) and entry["history_t"] is not None and entry["history_p"] is not None],
            key=lambda entry: entry["mass"],
        )

        histories = []
        for index, entry in enumerate(group):
            linestyle = "-" if index == 0 else (":" if index == len(group) - 1 else "-")
            ax.plot(entry["history_t"], entry["history_p"], color=mass_cmap[entry["mass"]], ls=linestyle, alpha=0.7, label=rf"$am_0={entry['mass']}$")
            histories.append(entry["history_p"])

        set_history_ylim(ax, histories)
        ax.set_ylabel(rf"$ \mathcal{{P}} [\beta={beta}]$")
        if show_legend and group:
            ax.legend(loc="upper right", fontsize="x-small", ncol=2)

    axes[-1].set_xlabel("Monte Carlo time")
    axes[-1].set_xlim(150, 6900)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_shamir_summary(shamir_entries, ym_entries, mres_entries, beta_cmap, show_legend, output_path):
    mass_values = sorted({entry["mass"] for entry in shamir_entries})
    mass_cmap = dict(zip(mass_values, mpl.cm.inferno(np.linspace(0.1, 0.85, max(1, len(mass_values))))))

    fig = plt.figure(figsize=(7, 3.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1])

    left_ax = fig.add_subplot(grid[:, 0])
    plot_grouped_errorbars(
        left_ax, shamir_entries, "beta", "mass", "plaq", "plaq_err", beta_cmap,
        lambda beta: rf"$\beta={beta}$", MARKERS, ":"
    )
    for entry in ym_entries:
        if np.isfinite(entry["plaq"]) and entry["beta"] in beta_cmap:
            left_ax.axhline(entry["plaq"], color=beta_cmap[entry["beta"]], ls="-", alpha=0.3, lw=1)

    left_ax.set_xlabel(r"$am_0$")
    left_ax.set_ylabel(r"$\langle \mathcal{P} \rangle$")
    left_ax.set_ylim(0.36, 0.68)
    if show_legend:
        left_ax.legend(ncol=3, loc="upper left", fontsize="x-small", columnspacing=0.5)

    top_right_ax = fig.add_subplot(grid[0, 1])
    bottom_right_ax = fig.add_subplot(grid[1, 1], sharex=top_right_ax)
    plot_grouped_errorbars(
        top_right_ax, shamir_entries, "mass", "beta", "plaq", "plaq_err", mass_cmap,
        lambda mass: rf"$am_0={mass}$", MARKERS, ":"
    )
    for index, mass in enumerate(mass_values):
        points = sorted_points([entry for entry in mres_entries if np.isclose(entry["mass"], mass)], "beta", "mres", "mres_err")
        if not points:
            continue
        xs, ys, yerr = zip(*points)
        bottom_right_ax.errorbar(
            xs, ys, yerr=yerr, marker=MRES_MARKERS[index % len(MRES_MARKERS)], ls="-",
            color=mass_cmap[mass], alpha=0.7, label=rf"$am_0={mass}$"
        )

    top_right_ax.set_ylabel(r"$\langle \mathcal{P} \rangle$")
    top_right_ax.set_ylim(0.36, 0.68)
    plt.setp(top_right_ax.get_xticklabels(), visible=False)

    bottom_right_ax.set_xlabel(r"$\beta$")
    bottom_right_ax.set_ylabel(r"$am_{\rm res}^{\rm fit}$")
    bottom_right_ax.set_ylim(0.0, 0.14)
    bottom_right_ax.set_yticks([0, 0.04, 0.08, 0.12])
    bottom_right_ax.grid(False)
    bottom_right_ax.axhline(0.02, color="gray", linestyle="--", alpha=0.7)
    bottom_right_ax.text(0.24, 0.025, r"$am_{\rm res}^{\rm fit} = 0.02$", fontsize="x-small", color="dimgrey", ha="right", va="bottom", transform=bottom_right_ax.get_yaxis_transform())

    if show_legend:
        top_right_ax.legend(fontsize="x-small", loc="upper left", ncol=2)
        bottom_right_ax.legend(fontsize="x-small", loc="upper left", ncol=1)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    plt.style.use(args.plot_styles if args.plot_styles else "tableau-colorblind10")
    meta_map = build_meta_map(args.ensembles_csv)
    tuned_entries, shamir_entries, ym_entries, mres_entries = collect_bulk_phase_entries(args.plaq_avg, args.mres_data, meta_map)
    beta_cmap, mass_cmap = build_color_maps(tuned_entries, shamir_entries, ym_entries, mres_entries)
    show_legend = str(args.label).strip().lower() == "yes"
    plot_tuned_masses(tuned_entries, beta_cmap, args.tuned_masses)
    plot_tuned_histories(tuned_entries, mass_cmap, args.history_masses, show_legend, args.tuned_history)
    plot_shamir_summary(shamir_entries, ym_entries, mres_entries, beta_cmap, show_legend, args.shamir_summary)


if __name__ == "__main__":
    main()
