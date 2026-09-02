#!/usr/bin/env python3
"""
Plot the coarse residual-mass scan selected by ``use_in_scan_beta``.

This script reads the full ``mres_series`` from each ``m_res.json`` and
produces a panel for each volume. Within a panel, curves are colored by mass
and annotated by beta so the output matches the release-style coarse scan.
"""

import argparse
import json
import re
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("tableau-colorblind10")


parser = argparse.ArgumentParser(description="Plot the coarse residual-mass beta scan.")
parser.add_argument("--mres", nargs="*", default=[], help="List of m_res.json files.")
parser.add_argument("--output_filename", required=True, help="Output plot filename.")
parser.add_argument("--label", type=str, default="no", help="Set to 'yes' to include legends.")
parser.add_argument("--plot_styles", default=None, help="Matplotlib style file to use.")
args = parser.parse_args()

show_legend = args.label.lower() == "yes"

if args.plot_styles:
    plt.style.use(args.plot_styles)


PATTERN_MRES = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
    r"residual_mass/m_res\.json"
)


def load_mres_series(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "r") as handle:
        data = json.load(handle)

    series = data.get("mres_series", {})
    t_values = np.asarray(series.get("t", []), dtype=float)
    values = np.asarray(series.get("mres", []), dtype=float)
    errors = np.asarray(series.get("mres_err", []), dtype=float)

    if len(t_values) == 0 or len(values) == 0 or len(errors) == 0:
        raise ValueError(f"{path} missing mres_series data")
    if len(t_values) != len(values) or len(values) != len(errors):
        raise ValueError(f"{path} has inconsistent mres_series lengths")

    if bool(series.get("folded")):
        t_values = np.append(t_values, t_values[-1] + 1.0)
        values = np.append(values, values[0])
        errors = np.append(errors, errors[0])

    return t_values, values, errors


def parse_entries(paths):
    entries = []
    for path in paths:
        match = PATTERN_MRES.search(path)
        if match is None:
            raise ValueError(f"Cannot parse parameters from path:\n{path}")

        entry = {
            key: int(value) if key in {"Nt", "Ns", "Ls"} else float(value)
            for key, value in match.groupdict().items()
        }
        entry["path"] = path
        entries.append(entry)

    return entries


def build_mass_color_map(sample_masses: np.ndarray):
    sample_masses = np.array(sample_masses, dtype=float)
    mass_cmap = mpl.cm.inferno(np.linspace(0.1, 0.85, len(sample_masses)))
    return sample_masses, mass_cmap


S_MASSES_DEFAULT = np.round(np.linspace(0.01, 0.10, 10), 12)
S_MASSES, M_CMAP = build_mass_color_map(S_MASSES_DEFAULT)


def mass_to_color(mass: float):
    index = int(np.argmin(np.abs(S_MASSES - float(mass))))
    return M_CMAP[index]


def mass_to_marker(mass: float) -> str:
    return "o" if float(mass) <= 0.02 + 1e-12 else "s"


def build_panel_groups(entries):
    panel_keys = ["Nt", "Ns", "Ls", "alpha", "a5", "M5", "mpv"]
    panel_groups = defaultdict(list)

    for entry in entries:
        key = tuple((name, entry[name]) for name in panel_keys)
        panel_groups[key].append(entry)

    groups = []
    for subset in panel_groups.values():
        beta_values = {entry["beta"] for entry in subset}
        if len(beta_values) >= 2:
            groups.append(
                sorted(
                    subset,
                    key=lambda entry: (
                        entry["beta"],
                        entry["mass"],
                    ),
                )
            )

    groups.sort(key=lambda subset: (subset[0]["Nt"], subset[0]["Ns"], subset[0]["Ls"]))
    return groups


def get_beta_label_position(beta_entries):
    x_max = min(entry["t"][-1] for entry in beta_entries)
    x_pos = 0.5 * x_max
    midpoint_values = [entry["y"][len(entry["y"]) // 2] for entry in beta_entries]
    y_pos = min(float(max(midpoint_values) + 0.006), 0.151)
    return x_pos, y_pos


def plot_beta_scan(entries, outname):
    panel_groups = build_panel_groups(entries)
    if not panel_groups:
        raise ValueError("No beta-scan groups with at least two beta values were found.")

    for entry in entries:
        t_values, values, errors = load_mres_series(entry["path"])
        entry["t"] = t_values
        entry["y"] = values
        entry["e"] = errors

    fig, axes = plt.subplots(
        1,
        len(panel_groups),
        figsize=(3.6 * len(panel_groups), 2.6),
        sharey=True,
        layout="constrained",
    )
    if len(panel_groups) == 1:
        axes = [axes]

    for index, (axis, subset) in enumerate(zip(axes, panel_groups)):
        seen_masses = set()
        beta_groups = defaultdict(list)

        for entry in subset:
            beta_groups[entry["beta"]].append(entry)
            color = mass_to_color(entry["mass"])
            marker = mass_to_marker(entry["mass"])
            label = None
            if show_legend and entry["mass"] not in seen_masses:
                label = rf"$am_0={entry['mass']:.2f}$"
                seen_masses.add(entry["mass"])

            axis.errorbar(
                entry["t"],
                entry["y"],
                yerr=entry["e"],
                fmt=marker,
                ms=3.0,
                mew=0.6,
                elinewidth=0.8,
                capsize=1.8,
                linestyle=":",
                linewidth=0.8,
                color=color,
                mec=color,
                mfc=color,
                label=label,
            )

        for beta_value in sorted(beta_groups):
            x_pos, y_pos = get_beta_label_position(beta_groups[beta_value])
            axis.text(
                x_pos,
                y_pos,
                rf"$\beta = {beta_value:.1f}$",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        axis.set_xlabel(r"$t/a$")
        axis.set_xlim(-0.4, max(entry["t"][-1] for entry in subset) + 0.4)
        axis.set_ylim(0.01, 0.157)

        if index == 0:
            axis.set_ylabel(r"$a m_{\rm res}$")

        if show_legend:
            axis.legend(loc="upper left", fontsize=7)

    plt.savefig(outname, dpi=300)
    plt.close(fig)


entries = parse_entries(args.mres)
if not entries:
    raise ValueError("No valid m_res.json files provided via --mres.")

plot_beta_scan(entries, args.output_filename)
