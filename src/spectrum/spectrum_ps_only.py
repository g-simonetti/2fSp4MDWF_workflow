#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt

from ps_fit import fit_with_bootstrap_PP
from spectrum import (
    _bootstrap_failures_from_samples,
    _fit_stats,
    _find_file_maps,
    _gvar_to_obj,
    _select_pairs_by_number,
    _summary_from_sample_list,
    bootstrap_effmass,
    bootstrap_from_path,
    bootstrap_to_jsonable,
    fold_even,
    gv,
    plot_effmass,
    read_ps_corr,
)

plt.style.use("tableau-colorblind10")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--label", default="")
    parser.add_argument("--spectrum_out", required=True)
    parser.add_argument("--plot_ps", required=True, nargs="+")
    parser.add_argument("--plot_styles", default="")
    parser.add_argument("--plateau_start_ps", type=float, required=True)
    parser.add_argument("--plateau_end_ps", type=float, required=True)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--mass", type=float, default=0)
    parser.add_argument("--Ns", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=200)
    parser.add_argument("--svdcut", type=float, default=1e-8)
    parser.add_argument("--therm", type=int, default=0)
    parser.add_argument("--delta_traj", type=int, default=1)
    args = parser.parse_args()

    if args.plot_styles:
        plt.style.use(args.plot_styles)

    ps0, ps1 = int(args.plateau_start_ps), int(args.plateau_end_ps)
    mesons_dir = Path(args.input_dir)

    result: Dict[str, Any] = {
        "ok": False,
        "input": {
            "input_dir": str(mesons_dir),
            "label": args.label,
            "beta": float(args.beta),
            "mass": float(args.mass),
            "Ns": int(args.Ns),
            "therm": int(args.therm),
            "delta_traj": int(args.delta_traj),
            "n_boot": int(args.n_boot),
            "svdcut": float(args.svdcut),
            "plot_styles": args.plot_styles,
        },
        "windows": {
            "ps": {"t0": int(ps0), "t1": int(ps1)},
        },
        "outputs": {
            "spectrum_out": str(args.spectrum_out),
            "plot_ps": [str(x) for x in args.plot_ps],
        },
    }

    try:
        pt_map, mr_map, common_nums = _find_file_maps(mesons_dir)
        pt_files, mres_files, nums_used = _select_pairs_by_number(
            pt_map, mr_map, common_nums, therm=args.therm, delta_traj=args.delta_traj
        )

        result["selection"] = {
            "n_common": int(len(common_nums)),
            "nums_common_min": int(common_nums[0]),
            "nums_common_max": int(common_nums[-1]),
            "n_used": int(len(nums_used)),
            "nums_used": [int(n) for n in nums_used],
            "pt_files": [str(p) for p in pt_files],
            "mres_files": [str(p) for p in mres_files],
        }

        ps = np.array([read_ps_corr(str(fpt)) for fpt in pt_files], dtype=float)
        _, T_full = ps.shape
        ps = fold_even(ps)

        _, T = ps.shape
        result["data_shape"] = {
            "Ncfg": int(ps.shape[0]),
            "T_full": int(T_full),
            "T_folded": int(T),
            "ps_folded": True,
        }

        if not (0 <= ps0 < ps1 < T):
            raise RuntimeError(
                f"Invalid PS plateau window after folding: [{ps0}, {ps1}] with folded T={T}"
            )

        bootstrap = bootstrap_from_path(mesons_dir, nums_used, args.n_boot)
        boot_idx = np.asarray(bootstrap["boot_idx"], dtype=int)
        result["bootstrap"] = bootstrap_to_jsonable(bootstrap)

        tps, meps, eeps = bootstrap_effmass(ps, args.n_boot, boot_idx)

        pp_res = fit_with_bootstrap_PP(
            ps,
            ps0,
            ps1,
            Nt_full=T_full,
            n_boot=args.n_boot,
            boot_idx=boot_idx,
            svdcut=args.svdcut,
        )

        fit_pp = pp_res["fit"]
        pp_samples = pp_res["bootstrap_samples"]
        pp_failures = pp_res.get(
            "bootstrap_failures",
            _bootstrap_failures_from_samples(pp_samples),
        )
        mps_pp_gv = fit_pp.p["m_ps"][0]

        plot_effmass(
            tps,
            meps,
            eeps,
            ps0,
            ps1,
            float(gv.mean(mps_pp_gv)),
            float(gv.sdev(mps_pp_gv)),
            args.plot_ps,
            args.label,
            args.beta,
            args.mass,
            "PS",
        )

        result["results"] = {
            "standard_fit": {
                "PP": {
                    "am_ps": _gvar_to_obj(mps_pp_gv),
                    "Afit": _gvar_to_obj(fit_pp.p["Afit"][0]),
                    "fit_stats": _fit_stats(fit_pp),
                },
            },
            "bootstrap_fit": {
                "PP": {
                    "am_ps": _summary_from_sample_list(pp_samples, "m_ps"),
                    "Afit": _summary_from_sample_list(pp_samples, "Afit"),
                    "fit_stats": pp_res["bootstrap_fit_stats"],
                    "meta": pp_res["bootstrap_meta"],
                    "samples": pp_samples,
                    "failures": pp_failures,
                },
            },
            "summary": {
                "am_ps": _gvar_to_obj(mps_pp_gv),
            },
        }

        result["ok"] = True

    except Exception as e:
        result["ok"] = False
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }

    out_path = Path(args.spectrum_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    if not result.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
