#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


plt.style.use("tableau-colorblind10")


def _apply_plot_styles(plot_styles):
    if not plot_styles:
        return

    if isinstance(plot_styles, str):
        styles = [part.strip() for part in plot_styles.split(",") if part.strip()]
    else:
        styles = [str(part).strip() for part in plot_styles if str(part).strip()]

    if styles:
        plt.style.use(styles)


def plot_residual_mass_fit(
    output_path,
    t_vals,
    ratio_mean,
    ratio_err,
    *,
    plot_styles=None,
    data_label=None,
    title=None,
    plateau_start=None,
    plateau_end=None,
    fit_value=None,
    fit_error=None,
):
    _apply_plot_styles(plot_styles)

    t_plot_max = int(max(t_vals))

    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")

    if title:
        ax.set_title(title, fontsize=10)

    ax.errorbar(
        t_vals,
        ratio_mean,
        yerr=ratio_err,
        fmt="o",
        color="C4",
        label=data_label,
    )

    have_plateau = (
        plateau_start is not None
        and plateau_end is not None
        and fit_value is not None
        and fit_error is not None
    )
    if have_plateau:
        fit_label = rf"$am_{{\rm res}}^{{\rm fit}} = {fit_value:.5f}\,\pm\,{fit_error:.5f}$"
        ax.axvspan(plateau_start, plateau_end, color="C2", alpha=0.2, label="Plateau range")
        ax.fill_between(
            [plateau_start, plateau_end],
            [fit_value - fit_error, fit_value - fit_error],
            [fit_value + fit_error, fit_value + fit_error],
            color="C1",
            alpha=0.25,
            linewidth=0,
        )
        ax.hlines(
            fit_value,
            plateau_start,
            plateau_end,
            color="C1",
            linestyle="--",
            label=fit_label,
        )

    ax.set_xlim(-0.5, t_plot_max + 0.5)
    ax.set_xlabel("$t/a$")
    ax.set_ylabel("$am_{\\rm res}$")

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if have_plateau or data_label:
        ax.legend()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
