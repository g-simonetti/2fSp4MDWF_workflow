#!/usr/bin/env python3
"""
Fit and plot the Ls scan from per-ensemble residual-mass summaries.

This release script reads the selected ``m_res.json`` files, reconstructs the
Shamir and representative Mobius branches for each beta value, fits their Ls
dependence, and stores both the figure and a compact JSON summary.
"""

import argparse
import json
import re
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import OptimizeWarning, curve_fit

plt.style.use("tableau-colorblind10")


parser = argparse.ArgumentParser(
    description=(
        "Fit and plot the Ls scan using only m_res inputs.\n"
        "Shamir is fitted with nu=1.\n"
        "The minimum-m_res Möbius branch is fitted with free nu.\n"
        "Also writes a JSON summary with fit results and the selected ensembles."
    )
)

parser.add_argument("--mres", nargs="+", required=True, help="m_res.json files (one per ensemble).")
# Keep the label flag for workflow compatibility. In this release script it is
# only used to decide whether the fit legend should be drawn.
parser.add_argument("--label", default="no")
parser.add_argument("--plot_styles", default=None)
parser.add_argument("--ls_scan", required=True, help="Output PDF for m_res vs Ls.")
parser.add_argument("--fit_json", required=True, help="Output JSON with fit summaries and selected ensembles.")

args = parser.parse_args()
show_legend = args.label.lower() == "yes"

if args.plot_styles:
    plt.style.use(args.plot_styles)


pattern = re.compile(
    r"Nt(?P<Nt>\d+)/Ns(?P<Ns>\d+)/Ls(?P<Ls>\d+)/"
    r"B(?P<beta>[0-9\.]+)/M(?P<mass>[0-9\.]+)/mpv(?P<mpv>[0-9\.]+)/"
    r"alpha(?P<alpha>[0-9\.]+)/a5(?P<a5>[0-9\.]+)/M5(?P<M5>[0-9\.]+)/"
)


def parse_params(path: str) -> dict:
    # The release workflow encodes ensemble metadata in directory names, so we
    # recover the plotting and fitting coordinates directly from the input path.
    match = pattern.search(path)
    if match is None:
        raise ValueError(f"Cannot parse metadata from path: {path}")
    data = match.groupdict()
    return {k: (int(v) if k in ["Nt", "Ns", "Ls"] else float(v)) for k, v in data.items()}


def load_mres_extract(path: str):
    # Read the per-ensemble residual-mass summary and keep the optional tau_int
    # values alongside the fit data so downstream release products can reuse
    # the same compact summary JSON.
    with open(path, "r") as handle:
        data = json.load(handle)

    try:
        mres = data["mres_extract"]["value"]
        mres_err = data["mres_extract"]["error"]
    except Exception as exc:
        raise ValueError(f"{path}: expected JSON with mres_extract.value/error") from exc

    try:
        tau = data["mres_extract"]["mres_tau_int"]["tau_int"]
        tau_err = data["mres_extract"]["mres_tau_int"]["tau_int_err"]
    except Exception:
        tau = np.nan
        tau_err = np.nan

    return float(mres), float(mres_err), float(tau), float(tau_err)


def mres_fit_ansatz(Ls, c1, lambda_c, c2, nu):
    Ls = np.asarray(Ls, dtype=float)
    return c1 * np.exp(-lambda_c * Ls) + c2 / np.power(Ls, nu)


def guess_mres_fit_parameters(Ls, y, nu):
    # Build stable starting values from the large-Ls tail and the remaining
    # exponential component so the non-linear fit is reproducible in release runs.
    Ls = np.asarray(Ls, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(Ls)
    Ls = Ls[order]
    y = y[order]

    tail_count = min(2, len(Ls))
    c2_guess = float(np.median(y[-tail_count:] * np.power(Ls[-tail_count:], nu)))
    c2_guess = max(c2_guess, 0.0)

    residual = np.maximum(y - c2_guess / np.power(Ls, nu), 1e-12)
    c1_guess = float(max(np.max(residual), y[0], 1e-12))

    if len(Ls) > 1 and not np.isclose(Ls[-1], Ls[0]):
        lambda_guess = np.log(residual[0] / residual[-1]) / (Ls[-1] - Ls[0])
    else:
        lambda_guess = 0.1
    lambda_guess = float(np.clip(lambda_guess, 1e-6, 5.0))

    return [c1_guess, lambda_guess, c2_guess]


def prepare_fit_inputs(Ls, y, yerr, min_points):
    # Apply one shared input sanitization step for both fit branches so the
    # later fitting code only sees finite, Ls-ordered data with safe errors.
    Ls = np.asarray(Ls, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    finite = np.isfinite(Ls) & np.isfinite(y) & np.isfinite(yerr)
    Ls = Ls[finite]
    y = y[finite]
    yerr = yerr[finite]

    order = np.argsort(Ls)
    Ls = Ls[order]
    y = y[order]
    yerr = yerr[order]

    if len(Ls) < min_points:
        return None

    sigma_floor = np.maximum(np.abs(y) * 1e-3, 1e-12)
    sigma = np.where(yerr > 0, yerr, sigma_floor)
    return Ls, y, sigma


def fit_mres_vs_Ls(Ls, y, yerr, *, family, nu, free_nu=False):
    # Use one shared fitter for both branches: Shamir keeps nu fixed at 1,
    # while the representative Mobius branch lets nu float.
    prepared = prepare_fit_inputs(Ls, y, yerr, min_points=4 if free_nu else 3)
    if prepared is None:
        return None
    Ls, y, sigma = prepared

    c1_guess, lambda_guess, c2_guess = guess_mres_fit_parameters(Ls, y, nu)
    if free_nu:
        model = mres_fit_ansatz
        p0 = [c1_guess, lambda_guess, c2_guess, nu]
        bounds = ([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 6.0])
        maxfev = 40000
    else:
        model = lambda x, c1, lambda_c, c2: mres_fit_ansatz(x, c1, lambda_c, c2, nu)
        p0 = [c1_guess, lambda_guess, c2_guess]
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])
        maxfev = 20000

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                model,
                Ls,
                y,
                p0=p0,
                sigma=sigma,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=maxfev,
            )
    except Exception:
        return None

    n_params = len(popt)
    diag = np.diag(pcov) if pcov is not None else np.full(n_params, np.nan)
    perr = np.sqrt(diag) if np.all(np.isfinite(diag)) else np.full(n_params, np.nan)
    y_fit = model(Ls, *popt)
    chi2 = float(np.sum(((y - y_fit) / sigma) ** 2))
    dof = int(len(Ls) - len(popt))

    result = {
        "family": family,
        "free_nu": bool(free_nu),
        "nu": float(popt[3]) if free_nu else float(nu),
        "params": popt[:3] if free_nu else popt,
        "errors": perr[:3] if free_nu else perr,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": (chi2 / dof) if dof > 0 else np.nan,
        "Ls_min": float(np.min(Ls)),
        "Ls_max": float(np.max(Ls)),
    }
    if free_nu:
        result["nu_err"] = float(perr[3])
    return result


def serialize_fit(fit):
    if fit is None:
        return None

    out = {}
    for key, value in fit.items():
        if isinstance(value, np.ndarray):
            out[key] = [float(v) for v in value.tolist()]
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = float(value) if isinstance(value, np.floating) else int(value)
        else:
            out[key] = value
    return out


BLACK_GREY_CMAP = LinearSegmentedColormap.from_list("black_grey", ["black", "0.65"])
SHAMIR_MARKER_GREY = "0.25"


def make_legend_errorbar(ax, marker, color):
    # Build legend entries with the same marker/error-bar style used in the
    # panel without introducing extra data points.
    return ax.errorbar(
        [np.nan],
        [np.nan],
        yerr=[1.0],
        fmt=marker,
        color=color,
        ecolor=color,
        mec=color,
        linestyle="None",
    )


def add_gradient_fit_curve(ax, xs, ys, cmap, linewidth=0.8, linestyle="solid", reverse=False, zorder=1):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]

    if len(xs) < 2:
        return

    points = np.column_stack([xs, ys])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    tcol = np.linspace(1.0, 0.0, len(segments)) if reverse else np.linspace(0.0, 1.0, len(segments))
    lc = LineCollection(
        segments,
        cmap=plt.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap,
        array=tcol,
        linewidths=linewidth,
        linestyles=linestyle,
        capstyle="round",
        zorder=zorder,
    )
    ax.add_collection(lc)


def fit_curve_xmax(entries, fit, extension_fraction=0.18):
    xs_data = np.asarray(sorted(float(e["Ls"]) for e in entries), dtype=float)
    x_max = float(fit["Ls_max"])
    if len(xs_data) >= 2:
        x_prev = float(xs_data[-2])
        pad = max(extension_fraction * (x_max - x_prev), 0.03 * x_max)
    else:
        pad = 0.05 * x_max
    return x_max + pad


def build_alpha_style_for_panel(entries_panel):
    markers = ["s", "^", "v", "D", "P", "X"]
    panel_alphas = sorted({e["alpha"] for e in entries_panel if not np.isclose(e["alpha"], 1.0)})
    style = {}
    if not panel_alphas:
        return style

    cmap = plt.colormaps.get_cmap("viridis_r").resampled(len(panel_alphas))
    for i, alpha in enumerate(panel_alphas):
        style[alpha] = (markers[i % len(markers)], cmap(i))
    return style


def _mass_str(entries_panel) -> str:
    masses = sorted({f"{e['mass']:.8g}" for e in entries_panel})
    return masses[0] if len(masses) == 1 else ",".join(masses)


def load_entries(paths):
    # Preload the per-ensemble summaries once so the plotting code can work from
    # a compact in-memory table instead of reopening JSON files inside loops.
    entries = []
    for fp in paths:
        params = parse_params(fp)
        mres, mres_err, tau_int_mres, tau_int_mres_err = load_mres_extract(fp)
        entries.append(
            {
                **params,
                "filepath": fp,
                "mres": mres,
                "mres_err": mres_err,
                "tau_int_mres": tau_int_mres,
                "tau_int_mres_err": tau_int_mres_err,
            }
        )

    entries.sort(
        key=lambda e: (
            e["beta"],
            e["mass"],
            e["Ls"],
            e["alpha"],
            e["mpv"],
            e["a5"],
            e["M5"],
            e["filepath"],
        )
    )
    return entries


def choose_beta_panels(entries):
    # Preserve the usual two-panel release layout when beta=7.4 and 7.6 are
    # present, and otherwise fall back to the first two beta values available.
    betas_present = sorted({e["beta"] for e in entries})
    preferred = [7.4, 7.6]
    if all(any(np.isclose(b, bp) for bp in betas_present) for b in preferred):
        return preferred
    if len(betas_present) < 2:
        raise ValueError(f"Need at least 2 beta values, found {betas_present}")
    return betas_present[:2]


def plot_Ls_panel(ax, entries_panel, beta):
    # Each panel shows all measured points at fixed beta, plus one Mobius
    # representative per Ls chosen by the minimum measured residual mass.
    alpha_style = build_alpha_style_for_panel(entries_panel)

    for alpha in sorted(alpha_style):
        marker, color = alpha_style[alpha]
        handle = make_legend_errorbar(ax, marker, color)
        handle.set_label(rf"$\alpha={alpha}$")

    Ls_groups = defaultdict(list)
    for entry in entries_panel:
        Ls_groups[entry["Ls"]].append(entry)

    shamir_entries = []
    mobius_min_entries = []
    for Ls in sorted(Ls_groups.keys()):
        group = sorted(
            Ls_groups[Ls],
            key=lambda g: (g["alpha"], g["mres"], g["mres_err"], g["filepath"]),
        )
        alphas = np.array([g["alpha"] for g in group], dtype=float)
        y_vals = np.array([g["mres"] for g in group], dtype=float)
        y_errs = np.array([g["mres_err"] for g in group], dtype=float)

        shamir_mask = np.isclose(alphas, 1.0)
        if shamir_mask.any():
            idx = int(np.where(shamir_mask)[0][0])
            shamir_entries.append(group[idx])
            ax.errorbar(
                Ls,
                float(y_vals[idx]),
                yerr=float(y_errs[idx]),
                fmt="o",
                color=SHAMIR_MARKER_GREY,
                ecolor=SHAMIR_MARKER_GREY,
                mec=SHAMIR_MARKER_GREY,
                label=r"$\alpha=1$ (Shamir)" if len(shamir_entries) == 1 else None,
                zorder=3,
            )

        non_indices = [i for i in range(len(alphas)) if not shamir_mask[i]]
        for local_i, idx in enumerate(non_indices):
            alpha = float(alphas[idx])
            marker, color = alpha_style.get(alpha, ("s", "C0"))
            offset = -0.12 + local_i * 0.12
            ax.errorbar(
                Ls + offset,
                float(y_vals[idx]),
                yerr=float(y_errs[idx]),
                fmt=marker,
                color=color,
                ecolor=color,
                mec=color,
                zorder=3,
            )

        if non_indices:
            # For the Mobius comparison we keep one representative point per Ls:
            # the ensemble with the smallest measured residual mass at that Ls.
            idx_min = min(
                non_indices,
                key=lambda i: (y_vals[i], alphas[i], y_errs[i], group[i]["filepath"]),
            )
            mobius_min_entries.append(group[idx_min])

    shamir_fit = fit_mres_vs_Ls(
        [e["Ls"] for e in shamir_entries],
        [e["mres"] for e in shamir_entries],
        [e["mres_err"] for e in shamir_entries],
        family="shamir",
        nu=1.0,
    )
    mobius_fit = fit_mres_vs_Ls(
        [e["Ls"] for e in mobius_min_entries],
        [e["mres"] for e in mobius_min_entries],
        [e["mres_err"] for e in mobius_min_entries],
        family="mobius_min",
        nu=2.0,
        free_nu=True,
    )

    ax.set_title(rf"$\beta={beta},\; am_0={_mass_str(entries_panel)}$")
    ax.set_xlabel(r"$L_s$")
    ax.set_yscale("log")

    panel_Ls = np.array(sorted({e["Ls"] for e in entries_panel}), dtype=float)
    if panel_Ls.size > 0:
        ax.set_xlim(float(panel_Ls.min()) * 0.95, float(panel_Ls.max()) * 1.05)

    fit_legend_handles = []
    fit_legend_labels = []

    if shamir_fit is not None:
        xs_stop = fit_curve_xmax(shamir_entries, shamir_fit)
        xs = np.linspace(shamir_fit["Ls_min"], xs_stop, 400)
        ys = mres_fit_ansatz(xs, *shamir_fit["params"], nu=1.0)
        add_gradient_fit_curve(ax, xs, ys, cmap=BLACK_GREY_CMAP, linewidth=0.8, linestyle="--", reverse=True, zorder=1)
        if show_legend:
            fit_legend_handles.append(plt.Line2D([], [], linestyle="-", color="0.35", linewidth=0.8))
            fit_legend_labels.append("Shamir fit")

    if mobius_fit is not None:
        xs_stop = fit_curve_xmax(mobius_min_entries, mobius_fit)
        xs = np.linspace(mobius_fit["Ls_min"], xs_stop, 400)
        ys = mres_fit_ansatz(xs, *mobius_fit["params"], nu=mobius_fit["nu"])
        add_gradient_fit_curve(ax, xs, ys, cmap="viridis_r", linewidth=0.8, zorder=1)
        if show_legend:
            fit_legend_handles.append(
                plt.Line2D([], [], linestyle="-", color=plt.colormaps.get_cmap("viridis_r")(0.5), linewidth=0.8)
            )
            fit_legend_labels.append("Möbius fit")

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        legend_loc = "lower left" if np.isclose(beta, 7.4) else "upper right"
        ax.legend(
            handles + fit_legend_handles,
            labels + fit_legend_labels,
            loc=legend_loc,
            fontsize=5.8,
            ncol=3,
            frameon=True,
            borderpad=0.25,
            handletextpad=0.3,
            labelspacing=0.25,
            columnspacing=0.8,
        )

    return {
        "beta": float(beta),
        "mass": _mass_str(entries_panel),
        "shamir_entries": shamir_entries,
        "mobius_min_entries": mobius_min_entries,
        "shamir_fit": serialize_fit(shamir_fit),
        "mobius_min_fit": serialize_fit(mobius_fit),
    }


def main():
    entries = load_entries(args.mres)
    if not entries:
        print("WARNING: no entries found.")
        raise SystemExit(0)

    beta_vals = choose_beta_panels(entries)
    entries_by_beta = {b: [e for e in entries if np.isclose(e["beta"], b)] for b in beta_vals}
    # Store the selected points and fit results in a compact JSON artifact so
    # later release plots can reuse the same branch definitions directly.
    fit_payload = {
        "format": "mres_scan_ls_fit_summary_v1",
        # Keep the fit formulas explicit in the release JSON so downstream
        # tables or archive scripts do not need to infer the model choice.
        "fit_model": {
            "shamir": "c1 * exp(-lambda_c * Ls) + c2 / Ls",
            "mobius_min": "c1 * exp(-lambda_c * Ls) + c2 / Ls^nu",
        },
        "betas": {},
    }

    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True, layout="constrained")

    for ax, beta in zip(axs, beta_vals):
        fit_payload["betas"][str(beta)] = plot_Ls_panel(ax, entries_by_beta[beta], beta)

    axs[0].set_ylabel(r"$a m_{\rm res}$")

    ys_all = np.array([e["mres"] for e in entries], dtype=float)
    ys_all = ys_all[np.isfinite(ys_all) & (ys_all > 0)]
    if ys_all.size > 0:
        ymin = max(ys_all.min() * 0.4, 1e-12)
        ymax = ys_all.max() * 1.6
        axs[0].set_ylim(ymin, ymax)

    plt.savefig(args.ls_scan, dpi=300)
    plt.close()

    with open(args.fit_json, "w") as handle:
        json.dump(fit_payload, handle, indent=2)


main()
