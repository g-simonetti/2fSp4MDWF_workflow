#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared_continuum_models import (
    derive_dw2_start_parameters,
    derive_wilson_start_parameters,
    fit_dw2_continuum_linear,
    fit_dw2_continuum_nonlinear,
)
from shared_fit_serialization import (
    linear_fit_to_json_dict,
    physical_fit_to_json_dict,
    to_serializable,
)
from mv_mps_plot_fitsdata_bootstrap import (
    collect_dw_bootstrap_ensembles as collect_dw_bootstrap_ensembles_mv,
    fit_dw2_bootstrap_summary as fit_dw2_bootstrap_summary_mv,
    plot_points_and_fits as plot_mv_points_and_fits,
    read_precomputed_wilson_bootstrap_json,
)
from fps_mps_plot_fitsdata_bootstrap import (
    collect_dw_bootstrap_ensembles as collect_dw_bootstrap_ensembles_fps,
    fit_dw_continuum as fit_dw_continuum_fps,
    fit_dw2_bootstrap_summary as fit_dw2_bootstrap_summary_fps,
    physical_dw2_to_plot_fit as physical_dw2_to_plot_fit_fps,
    plot_points_and_fits_bootstrap as plot_fps_points_and_fits_bootstrap,
    select_bootstrap_plot_fit_keys as select_fps_plot_fit_keys,
)

plt.style.use("tableau-colorblind10")
DEFAULT_WILSON_SHARED_NONLINEAR_P0 = [0.320, 2.9, -20.0, -0.183, 0.03, -1.0]

OBSERVABLE_CONFIG = {
    "mv": {
        "description": (
            "Bootstrap MDWF chiral-continuum fit for (m_V w0)^2 vs (m_PS w0)^2. "
            "The displayed MDWF points are bootstrap means with bootstrap standard "
            "deviations, and the MDWF continuum fit is obtained by fitting each "
            "bootstrap replica and summarising the fitted parameters."
        ),
        "dw_linear_label": (
            r"MDWF linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a^2$"
        ),
        "dw_physical_label": (
            r"MDWF: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + R_{m_M} a^2$"
        ),
        "dw_bootstrap_label": (
            r"MDWF bootstrap: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + R_{m_M} a^2$"
        ),
        "wilson_linear_label": (
            r"Wilson linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a m_{PS}^2 + E a + F a^2$"
        ),
        "wilson_physical_label": (
            r"Wilson: $m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "y_axis_label": r"$(m_{\rm V} w_0)^2$",
        "mdwf_formula": (
            r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            "\n"
            r"$\qquad\qquad + R_{m_M} a^2$"
        ),
        "wilson_formula": (
            r"$m_M^2 = m_{M,\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            "\n"
            r"$\qquad\qquad + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "limit_key_sq": "mvw0_sq",
        "limit_key": "mvw0",
        "mdwf_extra_sample_paths": {
            "vv_samples": ["results", "bootstrap_fit", "VV", "samples"],
        },
        "point_stat_fields": (),
        "use_precomputed_wilson_fit": True,
        "set_ylim": (0.0, None),
    },
    "fps": {
        "description": (
            "Bootstrap MDWF chiral-continuum fit for (f_PS w0)^2 vs (m_PS w0)^2. "
            "The MDWF points are built from bootstrap summaries of PP, Z_A, "
            "simultaneous PP+A0P, and w0, then each bootstrap replica is refit "
            "to the MDWF dw2 ansatz."
        ),
        "dw_linear_label": (
            r"MDWF linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a^2$"
        ),
        "dw_physical_label": (
            r"MDWF: $f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + R_{m_M} a^2$"
        ),
        "dw_bootstrap_label": (
            r"MDWF bootstrap: $f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + R_{m_M} a^2$"
        ),
        "wilson_linear_label": (
            r"Wilson linearized: $A + B m_{PS}^2 + C m_{PS}^4 + D a m_{PS}^2 + E a + F a^2$"
        ),
        "wilson_physical_label": (
            r"Wilson: $f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            r" + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "y_axis_label": r"$(f_{\rm PS} w_0)^2$",
        "mdwf_formula": (
            r"$f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            "\n"
            r"$\qquad\qquad + R_{m_M} a^2$"
        ),
        "wilson_formula": (
            r"$f_{\rm PS}^2 = f_{{\rm PS},\chi}^2(1 + L_{m_M} m_{PS}^2 + Q_{m_M} m_{PS}^4)$"
            "\n"
            r"$\qquad\qquad + W_{m_M} a + R_{m_M} a^2 + C_{m_M} a m_{PS}^2$"
        ),
        "limit_key_sq": "fpsw0_sq",
        "limit_key": "fpsw0",
        "mdwf_extra_sample_paths": {
            "sim_samples": ["results", "bootstrap_fit", "simultaneous_PP_A0P", "samples"],
            "za_samples": ["results", "bootstrap_fit", "Z_A", "samples"],
        },
        "point_stat_fields": ("Z_A", "fps"),
        "use_precomputed_wilson_fit": True,
        "set_ylim": (0.0, 0.0200),
    },
}


def get_config(observable):
    try:
        return OBSERVABLE_CONFIG[observable]
    except KeyError as exc:
        valid = ", ".join(sorted(OBSERVABLE_CONFIG))
        raise ValueError(
            f"Unsupported observable '{observable}'. Expected one of: {valid}."
        ) from exc


def collect_dw_bootstrap_ensembles(observable, spectrum_files, wflow_files):
    if observable == "mv":
        return collect_dw_bootstrap_ensembles_mv(spectrum_files, wflow_files)
    if observable == "fps":
        return collect_dw_bootstrap_ensembles_fps(spectrum_files, wflow_files)
    raise ValueError(f"Unsupported observable '{observable}'")


def fit_dw2_bootstrap_summary(
    observable,
    bootstrap_point_sets,
    dw_points,
    central_fit,
    start_params,
):
    if observable == "mv":
        return fit_dw2_bootstrap_summary_mv(
            bootstrap_point_sets,
            dw_points,
            central_fit,
            start_params,
        )
    if observable == "fps":
        return fit_dw2_bootstrap_summary_fps(
            bootstrap_point_sets,
            dw_points,
            central_fit,
            start_params,
        )
    raise ValueError(f"Unsupported observable '{observable}'")


def print_wilson_fit_summary(fit, title):
    print(f"{title}:")
    if "m_M_chi_sq" in fit:
        print(f"  m_M_chi_sq = {fit['m_M_chi_sq']:.8g} ± {fit['m_M_chi_sq_err']:.3g}")
        print(f"  L_m_M = {fit['L_m_M']:.8g} ± {fit['L_m_M_err']:.3g}")
        print(f"  Q_m_M = {fit['Q_m_M']:.8g} ± {fit['Q_m_M_err']:.3g}")
        print(f"  W_m_M = {fit['W_m_M']:.8g} ± {fit['W_m_M_err']:.3g}")
        print(f"  R_m_M = {fit['R_m_M']:.8g} ± {fit['R_m_M_err']:.3g}")
        print(f"  C_m_M = {fit['C_m_M']:.8g} ± {fit['C_m_M_err']:.3g}")
    else:
        for i, (term, coeff, err) in enumerate(
            zip(fit["basis_terms"], fit["coeffs"], fit["coeff_errs"])
        ):
            name = chr(ord("A") + i)
            print(f"  {name} = {coeff:.8g} ± {err:.3g}   [{term}]")

    if fit["dof"] > 0:
        print(f"  chi2/dof = {fit['chi2']:.3f}/{fit['dof']}")


def print_starting_parameters(title, params):
    print(f"{title}:")
    for key, value in params.items():
        print(f"  {key} = {value:.8g}")




def plot_points_and_fits(
    dw_points,
    dw_fit,
    output_file,
    observable,
    wilson_points=None,
    wilson_fit=None,
    dw_fit_central=None,
    wilson_fit_central=None,
):
    wilson_points = wilson_points or []
    cfg = get_config(observable)

    if observable == "mv":
        plot_mv_points_and_fits(
            dw_points=dw_points,
            dw_fit=dw_fit_central if dw_fit_central is not None else dw_fit,
            wilson_points=wilson_points,
            wilson_fit=(
                wilson_fit_central if wilson_fit_central is not None else wilson_fit
            ),
            dw_fit_central=dw_fit_central,
            wilson_fit_central=wilson_fit_central,
            output_file=output_file,
            set_ylim=cfg["set_ylim"],
        )
        return

    if observable == "fps":
        plot_fit_keys = select_fps_plot_fit_keys(has_wilson=wilson_fit is not None)
        all_fits = {
            "dw2": physical_dw2_to_plot_fit_fps(
                dw_fit_central if dw_fit_central is not None else dw_fit
            ),
        }
        # Keep the original fps single-script plot content, including the
        # extra linearized MDWF guide curve.
        if dw_fit_central is not None:
            all_fits["dw"] = fit_dw_continuum_fps(dw_points)
        if wilson_fit is not None:
            all_fits["wilson_physical"] = (
                wilson_fit_central if wilson_fit_central is not None else wilson_fit
            )

        plot_fps_points_and_fits_bootstrap(
            dw_points=dw_points,
            wilson_points=wilson_points,
            all_fits=all_fits,
            plot_fit_keys=plot_fit_keys,
            output_plot=output_file,
            dw2_fit_central=(
                physical_dw2_to_plot_fit_fps(dw_fit_central)
                if dw_fit_central is not None
                else None
            ),
            wilson_fit_central=wilson_fit_central,
        )
        return

    raise ValueError(f"Unsupported observable '{observable}'")


def bootstrap_fit_to_json_dict(fit, observable):
    cfg = get_config(observable)
    out = physical_fit_to_json_dict(fit)
    out["bootstrap_meta"] = fit["bootstrap_meta"]
    out["bootstrap_samples"] = fit["bootstrap_samples"]
    out["bootstrap_failures"] = fit["bootstrap_failures"]

    limit_block = {
        cfg["limit_key_sq"]: {
            "mean": fit["m_M_chi_sq"],
            "sdev": fit["m_M_chi_sq_err"],
        },
        cfg["limit_key"]: {
            "mean": float(np.sqrt(fit["m_M_chi_sq"])) if fit["m_M_chi_sq"] >= 0.0 else None,
            "sdev": (
                float(0.5 * fit["m_M_chi_sq_err"] / np.sqrt(fit["m_M_chi_sq"]))
                if fit["m_M_chi_sq"] > 0.0
                else None
            ),
        },
    }
    out["continuum_limit"] = limit_block
    out["chiral_continuum_limit"] = limit_block
    return to_serializable(out)


def save_fit_results_json(
    output_data,
    observable,
    dw_points,
    bootstrap_point_sets,
    dw_fit_linear,
    dw_fit_central,
    dw_fit_bootstrap,
    wilson_points=None,
    wilson_fit_linear=None,
    wilson_fit_nonlinear=None,
    wilson_fit_bootstrap=None,
    wilson_fit_starting_parameters=None,
):
    output_dir = os.path.dirname(output_data)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "observable": observable,
        "n_dw_points_used": len(dw_points),
        "n_dw_bootstrap_requested": len(bootstrap_point_sets),
        "n_dw_bootstrap_success": dw_fit_bootstrap["bootstrap_meta"]["n_success"],
        "n_dw_bootstrap_failed": dw_fit_bootstrap["bootstrap_meta"]["n_failed"],
        "points": {
            "dw": to_serializable(dw_points),
            "wilson": to_serializable(wilson_points or []),
        },
        "fits": {
            "dw2": {
                "linearized": linear_fit_to_json_dict(dw_fit_linear),
                "starting_parameters": to_serializable(
                    derive_dw2_start_parameters(dw_fit_linear)
                ),
                "central_nonlinear": physical_fit_to_json_dict(dw_fit_central),
                "bootstrap_summary": bootstrap_fit_to_json_dict(dw_fit_bootstrap, observable),
            },
        },
    }

    if wilson_points is not None and wilson_fit_linear is not None:
        payload["n_wilson_points_used"] = len(wilson_points)
        wilson_payload = {
            "linearized": linear_fit_to_json_dict(wilson_fit_linear),
            "starting_parameters": to_serializable(
                wilson_fit_starting_parameters
                if wilson_fit_starting_parameters is not None
                else derive_wilson_start_parameters(wilson_fit_linear)
            ),
        }
        if wilson_fit_nonlinear is not None:
            wilson_payload["central_nonlinear"] = physical_fit_to_json_dict(
                wilson_fit_nonlinear
            )
        if wilson_fit_bootstrap is not None:
            wilson_payload["bootstrap_summary"] = to_serializable(wilson_fit_bootstrap)
        payload["fits"]["wilson_physical"] = wilson_payload

    with open(output_data, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"✓ Saved fit data → {output_data}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap MDWF chiral-continuum plot/fit driver for either "
            "(m_V w0)^2 or (f_PS w0)^2 as a function of (m_PS w0)^2."
        )
    )
    parser.add_argument(
        "--observable",
        required=True,
        help="Which observable to extrapolate: 'mv' or 'fps'.",
    )
    parser.add_argument("--plot_styles", default="")
    parser.add_argument(
        "--spectrum",
        nargs="+",
        required=True,
        help="List of DWF/MDWF spectrum.json files",
    )
    parser.add_argument(
        "--wflow",
        nargs="+",
        required=True,
        help="List of DWF/MDWF wflow_extract.json files",
    )
    parser.add_argument(
        "--wilsons_data",
        required=True,
        help="Precomputed Wilson bootstrap JSON for the selected observable.",
    )
    parser.add_argument(
        "--output_plot",
        "--output_file",
        dest="output_plot",
        required=True,
        help="Output plot file",
    )
    parser.add_argument(
        "--output_data",
        required=True,
        help="Output JSON file storing fit results",
    )
    args = parser.parse_args()
    cfg = get_config(args.observable)

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    dw_points, bootstrap_point_sets, bootstrap_input_failures = collect_dw_bootstrap_ensembles(
        args.observable,
        args.spectrum,
        args.wflow,
    )

    dw_fit_linear = fit_dw2_continuum_linear(
        dw_points,
        fit_label=cfg["dw_linear_label"],
    )
    start_params = derive_dw2_start_parameters(dw_fit_linear)
    dw_fit_central = fit_dw2_continuum_nonlinear(
        dw_points,
        dw_fit_linear,
        fit_label=cfg["dw_physical_label"],
    )
    dw_fit_bootstrap = fit_dw2_bootstrap_summary(
        args.observable,
        bootstrap_point_sets,
        dw_points,
        dw_fit_central,
        start_params,
    )
    dw_fit_bootstrap["bootstrap_failures"] = (
        bootstrap_input_failures + dw_fit_bootstrap["bootstrap_failures"]
    )
    dw_fit_bootstrap["bootstrap_meta"]["n_failed"] = len(dw_fit_bootstrap["bootstrap_failures"])

    wilson_points = []
    wilson_fit_linear = None
    wilson_fit = None
    wilson_fit_bootstrap = None
    wilson_fit_starting_parameters = None
    wilson_fit_central = None

    wilsons_json_path = Path(args.wilsons_data)
    if cfg["use_precomputed_wilson_fit"] and wilsons_json_path.exists():
        wilson_data = read_precomputed_wilson_bootstrap_json(str(wilsons_json_path))
        wilson_points = wilson_data["wilson_points"]
        wilson_fit_linear = wilson_data["linearized"]
        wilson_fit = wilson_data["bootstrap_summary"]
        wilson_fit_bootstrap = wilson_data["bootstrap_summary"]
        wilson_fit_starting_parameters = wilson_data["starting_parameters"]
        wilson_fit_central = wilson_data["central_nonlinear"]
    elif cfg["use_precomputed_wilson_fit"]:
        raise FileNotFoundError(
            f"Required Wilson input JSON not found: {wilsons_json_path}"
        )

    plot_points_and_fits(
        dw_points=dw_points,
        dw_fit=dw_fit_bootstrap,
        wilson_points=wilson_points,
        wilson_fit=wilson_fit,
        dw_fit_central=dw_fit_central,
        wilson_fit_central=wilson_fit_central,
        output_file=args.output_plot,
        observable=args.observable,
    )

    save_fit_results_json(
        output_data=args.output_data,
        observable=args.observable,
        dw_points=dw_points,
        bootstrap_point_sets=bootstrap_point_sets,
        dw_fit_linear=dw_fit_linear,
        dw_fit_central=dw_fit_central,
        dw_fit_bootstrap=dw_fit_bootstrap,
        wilson_points=wilson_points,
        wilson_fit_linear=wilson_fit_linear,
        wilson_fit_nonlinear=wilson_fit_central,
        wilson_fit_bootstrap=wilson_fit_bootstrap,
        wilson_fit_starting_parameters=wilson_fit_starting_parameters,
    )

    print(f"✓ Saved plot → {args.output_plot}")
    print(f"Observable = {args.observable}")
    print_starting_parameters(
        "DWF/MDWF starting parameters from linearized fit",
        start_params,
    )
    if wilson_fit_linear is not None and wilson_fit_starting_parameters is not None:
        print_starting_parameters(
            "Wilson initial fit parameters",
            wilson_fit_starting_parameters,
        )
        if wilson_fit_central is not None:
            print_wilson_fit_summary(
                wilson_fit_central,
                "Wilson complete model [central-value fit]",
            )
        if wilson_fit_bootstrap is not None:
            print_wilson_fit_summary(
                wilson_fit_bootstrap,
                "Wilson complete model [bootstrap mean ± std]",
            )
        elif wilson_fit is not None:
            print_wilson_fit_summary(wilson_fit, "Wilson complete model")


if __name__ == "__main__":
    main()
