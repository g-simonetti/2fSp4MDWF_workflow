#!/usr/bin/env python3
"""
Plot residual-mass parameter scans from per-ensemble ``m_res.json`` outputs.

This script is used by the data-release workflow to make the 1x4 scan plot over
the Mobius/Shamir parameters ``alpha``, ``a5``, ``M5``, and ``mpv``.
"""

import re
import argparse
from collections import defaultdict
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("tableau-colorblind10")


# Keep the CLI narrow: this release script only consumes ``m_res.json`` inputs
# and produces the 1x4 residual-mass parameter-scan summary.
parser = argparse.ArgumentParser(
    description="Plot residual mass scans, grouping from filepath."
)
parser.add_argument(
    "--use",
    default="scan_param",
    help="Accepted aliases: scan_param or merged_m. Other modes are no longer supported.",
)
parser.add_argument("--mres", nargs="*", default=[], help="List of m_res.json files")
parser.add_argument("--output_filename", required=True, help="Output plot filename")
parser.add_argument("--label", type=str, default="no", help="Set to 'yes' to include legend")
parser.add_argument("--plot_styles", default=None, help="Matplotlib style file to use")
args = parser.parse_args()

show_legend = args.label.lower() == "yes"

if args.plot_styles:
    plt.style.use(args.plot_styles)


# Recover the ensemble coordinates directly from the workflow path layout so the
# plot stays reproducible without a separate metadata sidecar.
pattern_mres = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
    r"residual_mass/m_res\.json"
)


def load_mres_fit(path: str) -> tuple[float, float]:
    with open(path, "r") as f:
        data = json.load(f)
    try:
        y = data["mres_extract"]["value"]
        err = data["mres_extract"]["error"]
    except Exception as e:
        raise ValueError(f"{path} missing mres_extract.value/error") from e
    return float(y), float(err)


def _k(v):
    return round(v, 12) if isinstance(v, float) else v


def parse_entries(paths, pat):
    entries = []
    for path in paths:
        m = pat.search(path)
        if m is None:
            raise ValueError(f"Cannot parse parameters from path:\n{path}")

        d = m.groupdict()
        entry = {
            k: int(v) if k in {"Nt", "Ns", "Ls"} else float(v)
            for k, v in d.items()
        }
        entry["path"] = path
        entries.append(entry)
    return entries


# Use a fixed reference color map in mass so the same mass values appear with
# the same colors across regenerated release figures.
def build_mass_color_map(s_masses: np.ndarray):
    s_masses = np.array(s_masses, dtype=float)
    m_cmap = mpl.cm.inferno(np.linspace(0.1, 0.85, len(s_masses)))
    return s_masses, m_cmap


S_MASSES_DEFAULT = np.round(np.linspace(0.01, 0.10, 10), 12)
_S_MASSES, _M_CMAP = build_mass_color_map(S_MASSES_DEFAULT)


def mass_to_color(mass: float):
    mass = float(mass)
    idx = int(np.argmin(np.abs(_S_MASSES - mass)))
    return _M_CMAP[idx]


# Group ensembles by the one parameter that changes while the others stay fixed.
# This is the grouping shown panel-by-panel in the release scan figure.
SCAN_PARAMS = ["alpha", "a5", "M5", "mpv"]
ALL_KEYS = ["Nt", "Ns", "Ls", "beta", "mass", "mpv", "alpha", "a5", "M5"]


def build_merged_m_groups(entries_list):
    groups = defaultdict(list)
    for param in SCAN_PARAMS:
        fixed_keys = [k for k in ALL_KEYS if k != param]

        buckets = defaultdict(list)
        for e in entries_list:
            fixed = tuple((k, _k(e[k])) for k in fixed_keys)
            buckets[fixed].append(e)

        for subset in buckets.values():
            distinct_x = {_k(e[param]) for e in subset}
            if len(distinct_x) >= 2:
                groups[param].append(subset)

    return groups


xlabels = {
    "mpv": r"$am_{\rm PV}$",
    "M5": r"$am_5$",
    "alpha": r"$\alpha$",
    "a5": r"$a_5/a$",
}


def make_subplot_title(param, ref):
    alpha, a5, M5, mpv_val = ref["alpha"], ref["a5"], ref["M5"], ref["mpv"]

    if param == "a5":
        return rf"$\begin{{array}}{{c}} \alpha={alpha},\; am_5={M5}, \\ am_{{\mathrm{{PV}}}}={mpv_val} \\[6pt] \end{{array}}$"
    elif param == "alpha":
        return rf"$\begin{{array}}{{c}} a_5/a={a5},\; am_5={M5}, \\ am_{{\mathrm{{PV}}}}={mpv_val} \\[6pt] \end{{array}}$"
    elif param == "M5":
        return rf"$\begin{{array}}{{c}} \alpha={alpha},\; a_5/a={a5}, \\ am_{{\mathrm{{PV}}}}={mpv_val} \\[6pt] \end{{array}}$"
    elif param == "mpv":
        return rf"$\begin{{array}}{{c}} \alpha={alpha},\; a_5/a={a5}, \\  am_5={M5} \\[6pt] \end{{array}}$"
    return ""


# Plot the release summary as four panels, one for each scan direction.
def plot_merged_m(groups, outname):
    line_styles = ["--", ":"]
    fig, axes = plt.subplots(1, 4, figsize=(7, 2), sharey=True, layout="constrained")
    params = ["alpha", "a5", "M5", "mpv"]

    for i, param in enumerate(params):
        ax = axes[i]

        if param not in groups or len(groups[param]) == 0:
            ax.axis("off")
            continue

        mass_groups = defaultdict(list)
        for subset in groups[param]:
            mass_groups[_k(subset[0]["mass"])].append(subset)

        masses = sorted(mass_groups.keys())

        for j, mass_value in enumerate(masses):
            color = mass_to_color(mass_value)
            linestyle = line_styles[j % len(line_styles)]
            marker = "o" if j == 0 else "s"

            mass_label = rf"$am_0={mass_value}$"

            all_x = []
            for subset in mass_groups[mass_value]:
                for e in subset:
                    all_x.append(e[param])
            span = max(all_x) - min(all_x) if len(all_x) > 1 else 1.0
            dx = 0.01 * span * (-1 if j == 0 else 1)

            for idx_subset, subset in enumerate(mass_groups[mass_value]):
                x_vals, y_vals, y_err = [], [], []

                for e in subset:
                    y, err = load_mres_fit(e["path"])
                    x_vals.append(e[param] + dx)
                    y_vals.append(y)
                    y_err.append(err)

                xs, ys, es = zip(*sorted(zip(x_vals, y_vals, y_err)))

                if j == 0 and idx_subset == 0:
                    ax.set_title(make_subplot_title(param, subset[0]), fontsize=8)

                fmt = marker + linestyle
                ax.errorbar(
                    xs, ys, yerr=es,
                    fmt=fmt, mec=color, mfc=color, color=color,
                    label=mass_label if idx_subset == 0 else None
                )
                ax.plot(xs, ys, linestyle, color=color)

        ax.set_xlabel(xlabels[param])
        ax.set_yscale("log")
        ax.set_ylim(1.2e-3, 1.2e-1)

        if i == 0:
            ax.set_ylabel(r"$a m_{\rm res}$")

        if show_legend:
            ax.legend(loc="upper center", fontsize=7)

    plt.savefig(outname, dpi=300)
    plt.close()


use = args.use.strip().lower()
if use not in {"scan_param", "merged_m"}:
    raise ValueError(
        f"Unsupported mode '{args.use}'. This script now supports only parameter-scan plots."
    )

entries = parse_entries(args.mres, pattern_mres)
if not entries:
    raise ValueError("No valid m_res.json files provided via --mres.")

merged_groups = build_merged_m_groups(entries)
plot_merged_m(merged_groups, args.output_filename)
