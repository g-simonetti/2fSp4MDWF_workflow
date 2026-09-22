#!/usr/bin/env python3
"""
Plot the N5-scan cost summary from the selected m_res fit JSON and HMC outputs.

This release script rebuilds the two-panel comparison used for the Shamir and
minimum-m_res Mobius selections:
  - top row: trajectory time t_traj
  - bottom row: tau_int(plaq) * t_traj

Inputs:
  - fit summary JSON from fit_mres_scan_N5.py
  - HMC summary JSON files with plaquette and timing observables

Output:
  - combined PDF with one column per beta value
"""

import argparse
import json
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

plt.style.use("tableau-colorblind10")


# Keep the CLI focused on release products: a fit-summary JSON plus optional
# HMC summaries are enough to rebuild the cost-versus-m_res figure.
parser = argparse.ArgumentParser(
    description=(
        "Build the cost/bcs plot from a fit-summary JSON and optional HMC JSON files.\n"
        "Uses the selected Shamir and minimum-m_res Möbius ensembles stored in the fit JSON."
    )
)

parser.add_argument("--fit_json", required=True, help="Fit summary JSON from the N5 scan fit script.")
parser.add_argument("--hmc", nargs="*", default=[], help="Optional HMC JSON files.")
# Keep these CLI flags for workflow compatibility even though this release
# figure now has a single plaquette-based cost definition.
parser.add_argument("--label", default="no")
parser.add_argument("--plot_styles", default=None)
parser.add_argument(
    "--plaq",
    action="store_true",
    help="Use tau_int of the plaquette in the bottom cost panel. This is the only supported mode.",
)
parser.add_argument(
    "--machine",
    default="any",
    help="Machine section to read from grouped HMC JSON files; 'any' accepts files with exactly one machine section.",
)
parser.add_argument("--costs", required=True, help="Output PDF for combined cost/bcs vs m_res figure.")

args = parser.parse_args()

if args.plot_styles:
    plt.style.use(args.plot_styles)


pattern = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
)


def parse_params(path: str) -> dict:
    # Recover ensemble coordinates from the workflow path layout so HMC JSON
    # files can be matched back to the selected fit entries without extra tables.
    match = pattern.search(path)
    if match is None:
        raise ValueError(f"Cannot parse metadata from path: {path}")
    data = match.groupdict()
    return {k: (int(v) if k in ["Nt", "Ns", "Ls"] else float(v)) for k, v in data.items()}


def params_key(p: dict):
    return tuple(sorted((k, p[k]) for k in ["Nt", "Ns", "Ls", "beta", "mass", "mpv", "alpha", "a5", "M5"]))


def load_hmc_extract(path: str, machine: str | None = None):
    with open(path, "r") as handle:
        data = json.load(handle)

    # Support both flat per-machine HMC JSON files and grouped JSON files that
    # carry multiple machine sections in the same release artifact.
    if "hmc_extract" in data and isinstance(data["hmc_extract"], dict):
        h = data["hmc_extract"]
    else:
        machine_sections = [
            name
            for name in ("sunbird", "tursa")
            if name in data and isinstance(data[name], dict)
        ]

        if machine in (None, "any"):
            if len(machine_sections) != 1:
                raise ValueError(
                    f"{path}: --machine any requires exactly one machine section, found {machine_sections}"
                )
            machine = machine_sections[0]

        if machine is None:
            raise ValueError(f"{path}: grouped HMC JSON detected but no --machine was provided")
        if machine not in data or not isinstance(data[machine], dict):
            raise ValueError(f"{path}: missing machine section '{machine}' in grouped HMC JSON")
        if "hmc_extract" not in data[machine] or not isinstance(data[machine]["hmc_extract"], dict):
            raise ValueError(f"{path}: expected '{machine}.hmc_extract' object in grouped HMC JSON")
        h = data[machine]["hmc_extract"]

    def resolve_machine_value(name: str, value):
        if not isinstance(value, dict):
            return value

        family_matches = {"sunbird": [], "tursa": []}
        for key, entry in value.items():
            if entry is None:
                continue
            if key == "sunbird" or key.startswith("sunbird-"):
                family_matches["sunbird"].append(float(entry))
            elif key == "tursa" or key.startswith("tursa-"):
                family_matches["tursa"].append(float(entry))

        if machine in (None, "any"):
            present = [fam for fam, matches in family_matches.items() if matches]
            if len(present) != 1:
                raise ValueError(
                    f"{path}: hmc_extract.{name} requires values from exactly one machine family for --machine any, found {present}"
                )
            matches = family_matches[present[0]]
        else:
            matches = family_matches.get(machine, [])
            if not matches:
                raise ValueError(f"{path}: hmc_extract.{name} is missing machine value '{machine}'")

        if len(matches) > 1:
            warnings.warn(
                f"{path}: hmc_extract.{name} matched {len(matches)} identifier values; using their arithmetic mean",
                RuntimeWarning,
            )
        return float(np.mean(matches))

    def get(name: str, required: bool = True, default=np.nan) -> float:
        if name in h and h[name] is not None:
            return float(resolve_machine_value(name, h[name]))
        if required:
            raise ValueError(f"{path}: missing required hmc_extract.{name}")
        return float(default)

    return {
        "t_traj": get("t_traj", required=True),
        "t_traj_err": get("t_traj_err", required=True),
        "tau_int_plaq": get("tau_int_plaq", required=False),
        "tau_int_plaq_err": get("tau_int_plaq_err", required=False),
        "bcs": get("bcs", required=False),
        "bcs_err": get("bcs_err", required=False),
        "plaq": get("plaq", required=False),
        "plaq_err": get("plaq_err", required=False),
    }


def product_with_err(a, da, b, db):
    z = a * b
    dz = np.sqrt((b * da) ** 2 + (a * db) ** 2)
    return float(z), float(dz)


def finite_pair(x, y):
    return np.isfinite(x) and np.isfinite(y)


BLACK_GREY_CMAP = LinearSegmentedColormap.from_list("black_grey", ["black", "0.65"])
SHAMIR_MARKER_GREY = "0.25"


def build_alpha_style_for_panel(entries_panel):
    # Keep alpha colors and markers stable within each panel so the Shamir and
    # Möbius selections can be compared visually across regenerated figures.
    markers = ["s", "^", "v", "D", "P", "X"]
    panel_alphas = sorted({e["alpha"] for e in entries_panel if not np.isclose(e["alpha"], 1.0)})
    style = {}
    if not panel_alphas:
        return style

    cmap = plt.colormaps.get_cmap("viridis_r").resampled(len(panel_alphas))
    for i, alpha in enumerate(panel_alphas):
        style[alpha] = (markers[i % len(markers)], cmap(i))
    return style


def add_gradient_connector_straight(ax, xs, ys, cmap, linewidth=0.6, linestyle=":", reverse=False, zorder=1):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]

    if len(xs) < 2:
        return

    trans = ax.transData
    inv = trans.inverted()
    segs_all = []

    for i in range(len(xs) - 1):
        p0_disp = trans.transform((xs[i], ys[i]))
        p1_disp = trans.transform((xs[i + 1], ys[i + 1]))

        t = np.linspace(0.0, 1.0, 120, endpoint=False)
        pts_disp = p0_disp[None, :] + (p1_disp - p0_disp)[None, :] * t[:, None]
        if i == len(xs) - 2:
            pts_disp = np.vstack([pts_disp, p1_disp[None, :]])

        pts_data = inv.transform(pts_disp)
        segs_all.append(np.stack([pts_data[:-1], pts_data[1:]], axis=1))

    segs_all = np.concatenate(segs_all, axis=0)
    tcol = np.linspace(1.0, 0.0, len(segs_all)) if reverse else np.linspace(0.0, 1.0, len(segs_all))
    lc = LineCollection(
        segs_all,
        cmap=plt.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap,
        array=tcol,
        linewidths=linewidth,
        linestyles=linestyle,
        capstyle="round",
        zorder=zorder,
    )
    ax.add_collection(lc)


def make_legend_errorbar(ax, marker, color, include_xerr=False, include_yerr=True):
    kwargs = {
        "fmt": marker,
        "color": color,
        "ecolor": color,
        "mec": color,
        "linestyle": "None",
    }
    if include_xerr:
        kwargs["xerr"] = [1.0]
    if include_yerr:
        kwargs["yerr"] = [1.0]
    return ax.errorbar([np.nan], [np.nan], **kwargs)


def _mass_str(entries_panel) -> str:
    masses = sorted({f"{e['mass']:.8g}" for e in entries_panel})
    return masses[0] if len(masses) == 1 else ",".join(masses)


def compute_plaquette_cost(entry, hmc):
    # The lower panel uses the plaquette integrated autocorrelation time only,
    # combined with the HMC trajectory time to form the effective cost.
    tau = hmc.get("tau_int_plaq", np.nan)
    tau_err = hmc.get("tau_int_plaq_err", np.nan)

    if (
        np.isfinite(hmc.get("t_traj", np.nan))
        and np.isfinite(hmc.get("t_traj_err", np.nan))
        and np.isfinite(tau)
        and np.isfinite(tau_err)
    ):
        return product_with_err(
            hmc["t_traj"],
            hmc["t_traj_err"],
            tau,
            tau_err,
        )

    return np.nan, np.nan


TAU_COST_YLABEL = r"$\tau_{\mathrm{int}}^{\mathrm{plaq}} \times t_{\mathrm{traj}}\;[\mathrm{s}]$"


def load_selected_entries(fit_json_path):
    # The fit-summary JSON already records which Shamir and Möbius ensembles are
    # the selected representatives, so reuse that selection directly here.
    with open(fit_json_path, "r") as handle:
        fit_data = json.load(handle)

    entries = []
    beta_vals = []
    for _, beta_block in fit_data["betas"].items():
        beta = float(beta_block["beta"])
        beta_vals.append(beta)
        for entry in beta_block.get("shamir_entries", []):
            entries.append({**entry, "family": "shamir"})
        for entry in beta_block.get("mobius_min_entries", []):
            entries.append({**entry, "family": "mobius_min"})

    entries.sort(key=lambda e: (e["beta"], e["family"], e["Ls"], e["alpha"]))
    beta_vals = sorted(beta_vals)
    return entries, beta_vals


def build_hmc_lookup(paths, machine):
    # Build one lookup keyed by ensemble coordinates so the plotting code only
    # needs a compact dictionary lookup rather than repeated file parsing.
    hmc_lookup = {}
    for fp in paths:
        params = parse_params(fp)
        try:
            hmc_lookup[params_key(params)] = load_hmc_extract(fp, machine=machine)
        except Exception:
            continue
    return hmc_lookup


def attach_cost_observables(entries, hmc_lookup):
    # Enrich the selected fit entries with HMC timing data and the effective
    # cost metric used in the lower panel of the release figure.
    missing_hmc = {
        "t_traj": np.nan,
        "t_traj_err": np.nan,
        "bcs": np.nan,
        "bcs_err": np.nan,
        "plaq": np.nan,
        "plaq_err": np.nan,
        "x_eff": np.nan,
        "x_eff_err": np.nan,
    }

    for entry in entries:
        hmc = hmc_lookup.get(params_key(entry))
        if hmc is None:
            entry.update(missing_hmc)
            continue

        entry.update(hmc)
        x_eff, x_eff_err = compute_plaquette_cost(entry, hmc)
        entry["x_eff"] = x_eff
        entry["x_eff_err"] = x_eff_err


def plot_combined_metric_panel(ax, entries_panel, y_key, y_err_key=None, ylabel=None, title=None):
    # Both panels share the same x-axis (am_res) and differ only in the y-axis
    # observable, so use one plotting helper for timing and effective cost.
    ax.set_xscale("log")
    alpha_style = build_alpha_style_for_panel(entries_panel)

    shamir_entries = sorted(
        [e for e in entries_panel if e["family"] == "shamir"],
        key=lambda e: e["Ls"],
    )
    mobius_entries = sorted(
        [e for e in entries_panel if e["family"] == "mobius_min"],
        key=lambda e: e["Ls"],
    )

    sh_x, sh_y = [], []
    mo_x, mo_y = [], []

    for entry in shamir_entries:
        x, dx = entry["mres"], entry["mres_err"]
        y = entry.get(y_key, np.nan)
        dy = entry.get(y_err_key, np.nan) if y_err_key is not None else None

        if not finite_pair(x, y):
            continue

        sh_x.append(x)
        sh_y.append(y)
        ax.errorbar(
            x,
            y,
            xerr=dx if np.isfinite(dx) else None,
            yerr=dy if (dy is not None and np.isfinite(dy)) else None,
            fmt="o",
            color=SHAMIR_MARKER_GREY,
            ecolor=SHAMIR_MARKER_GREY,
            mec=SHAMIR_MARKER_GREY,
            zorder=3,
        )

    for entry in mobius_entries:
        x, dx = entry["mres"], entry["mres_err"]
        y = entry.get(y_key, np.nan)
        dy = entry.get(y_err_key, np.nan) if y_err_key is not None else None
        alpha = entry["alpha"]

        if not finite_pair(x, y):
            continue

        mo_x.append(x)
        mo_y.append(y)
        marker, color = alpha_style.get(alpha, ("s", "C1"))
        ax.errorbar(
            x,
            y,
            xerr=dx if np.isfinite(dx) else None,
            yerr=dy if (dy is not None and np.isfinite(dy)) else None,
            fmt=marker,
            color=color,
            ecolor=color,
            mec=color,
            zorder=3,
        )

    # Connect the selected points in Ls order to show the fitted scan path
    # without hiding the point-by-point error bars.
    add_gradient_connector_straight(ax, sh_x, sh_y, cmap=BLACK_GREY_CMAP, linewidth=0.6, linestyle=":", reverse=True, zorder=1)
    add_gradient_connector_straight(ax, mo_x, mo_y, cmap="viridis_r", linewidth=0.6, linestyle=":", zorder=1)

    if title is not None:
        ax.set_title(title, pad=plt.rcParams["axes.titlepad"])
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if not sh_x and not mo_x:
        ax.text(0.5, 0.5, "No cost data", ha="center", va="center", transform=ax.transAxes)


def add_panel_alpha_legend(ax, entries_panel, beta):
    mobius_entries = sorted(
        [e for e in entries_panel if e["family"] == "mobius_min"],
        key=lambda e: e["Ls"],
    )
    alpha_style = build_alpha_style_for_panel(entries_panel)
    has_shamir = any(e["family"] == "shamir" for e in entries_panel)
    if not mobius_entries and not has_shamir:
        return

    handles = []
    labels = []
    if has_shamir:
        handles.append(make_legend_errorbar(ax, "o", SHAMIR_MARKER_GREY, include_xerr=False, include_yerr=True))
        labels.append(r"$\alpha=1$ (Shamir)")

    panel_alphas = sorted({e["alpha"] for e in mobius_entries})
    for alpha in panel_alphas:
        marker, color = alpha_style.get(alpha, ("s", "C1"))
        handles.append(make_legend_errorbar(ax, marker, color, include_xerr=False, include_yerr=True))
        labels.append(rf"$\alpha={alpha}$")

    legend_loc = "lower left" if np.isclose(beta, 7.4) else "upper right"
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc=legend_loc,
        fontsize=5.8,
        ncol=2,
        frameon=True,
        borderpad=0.25,
        handletextpad=0.3,
        labelspacing=0.25,
        columnspacing=0.8,
    )
    legend.get_frame().set_alpha(0.95)


def main():
    entries, beta_vals = load_selected_entries(args.fit_json)
    entries_by_beta = {beta: [e for e in entries if np.isclose(e["beta"], beta)] for beta in beta_vals}
    hmc_lookup = build_hmc_lookup(args.hmc, args.machine)
    attach_cost_observables(entries, hmc_lookup)

    # The release figure is a fixed 2x2 layout: top row for trajectory cost,
    # bottom row for the effective cost built from the selected tau_int choice.
    fig, axs = plt.subplots(2, 2, figsize=(7, 3.3), sharex="col", sharey="row", layout="constrained")
    fig.set_constrained_layout_pads(hspace=0.02, h_pad=0.02)

    xs_all = np.array([e["mres"] for e in entries], dtype=float)
    xs_all = xs_all[np.isfinite(xs_all) & (xs_all > 0)]
    if xs_all.size > 0:
        xmin = max(xs_all.min() * 0.6, 1e-12)
        xmax = xs_all.max() * 1.8
        for ax in axs.ravel():
            ax.set_xlim(xmin, xmax)

    for j, beta in enumerate(beta_vals):
        panel_entries = entries_by_beta[beta]
        title = rf"$\beta={beta}\; am_0={_mass_str(panel_entries)}$"
        plot_combined_metric_panel(
            axs[0, j],
            panel_entries,
            y_key="t_traj",
            y_err_key="t_traj_err",
            ylabel=r"$t_{\mathrm{traj}}\;[\mathrm{s}]$" if j == 0 else None,
            title=title,
        )
        add_panel_alpha_legend(axs[0, j], panel_entries, beta)
        plot_combined_metric_panel(
            axs[1, j],
            panel_entries,
            y_key="x_eff",
            y_err_key="x_eff_err",
            ylabel=TAU_COST_YLABEL if j == 0 else None,
        )

    ttraj_vals = np.array([e.get("t_traj", np.nan) for e in entries], dtype=float)
    ttraj_errs = np.array([e.get("t_traj_err", np.nan) for e in entries], dtype=float)
    ttraj_mask = np.isfinite(ttraj_vals) & np.isfinite(ttraj_errs)
    if np.any(ttraj_mask):
        ttraj_top = np.max(ttraj_vals[ttraj_mask] + ttraj_errs[ttraj_mask])
        for ax in axs[0, :]:
            ax.set_ylim(0.0, ttraj_top * 1.05)

    bottom_entries = [e for e in entries if np.isfinite(e.get("x_eff", np.nan))]
    if bottom_entries:
        bottom_top = max(
            e["x_eff"] + (e["x_eff_err"] if np.isfinite(e.get("x_eff_err", np.nan)) else 0.0)
            for e in bottom_entries
        )
        for ax in axs[1, :]:
            ax.set_ylim(0.0, bottom_top * 1.05)

    axs[1, 0].set_xlabel(r"$a m_{\rm res}$")
    axs[1, 1].set_xlabel(r"$a m_{\rm res}$")

    plt.savefig(args.costs, dpi=300)
    plt.close()


main()
