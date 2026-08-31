#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


FIRST_TABLE_CAPTION = (
    r"Representative result of the test of Shamir's kernel, "
    r"$\tilde{D}_{\rm S}$, as in Eq.~(\ref{eq:fourier_dwf_formula}). "
    r"Squared norm of the difference between Dirac kernels computed from "
    r"unitary gauge, random gauge transformation, and analytical "
    r"momentum-space results. The discrepancies are small, comparable with "
    r"the numerical machine precision."
)

SECOND_TABLE_CAPTION = (
    r"Representative result of the tests of the free propagator. "
    r"Squared norm of the difference between free field propagators "
    r"obtained from a unitary gauge configuration, a random gauge "
    r"transformation, and the CG-based result."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the FFT cross-check LaTeX tables from FFT test logs."
    )
    parser.add_argument(
        "--log",
        nargs="+",
        required=True,
        help="One or more FFT log files or directories containing files matching FFT_test.*",
    )
    parser.add_argument("--output_table", required=True)
    parser.add_argument("--output_table_prop", required=True)
    return parser.parse_args()


def parse_log(path):
    text = Path(path).read_text()

    kernel_diffs = re.findall(r"^diff\s+([0-9.eE+-]+)\s*$", text, flags=re.MULTILINE)
    prop_diffs = re.findall(
        r"^result - ref\s+([0-9.eE+-]+)\s*$", text, flags=re.MULTILINE
    )

    gauge_match = re.search(
        r"Check the output gauge transformation matrices applied to the original "
        r"field produce the xformed field\s+([0-9.eE+-]+)\s+\(expect 0\)",
        text,
    )
    prop_xform_match = re.search(
        r"^unit - xform\s+([0-9.eE+-]+)\s*$", text, flags=re.MULTILINE
    )

    if len(kernel_diffs) < 2:
        raise ValueError(f"{path}: expected at least two 'diff' lines, found {len(kernel_diffs)}")
    if len(prop_diffs) < 2:
        raise ValueError(
            f"{path}: expected at least two 'result - ref' lines, found {len(prop_diffs)}"
        )
    if gauge_match is None:
        raise ValueError(f"{path}: could not find the gauge-transform cross-check line")
    if prop_xform_match is None:
        raise ValueError(f"{path}: could not find the 'unit - xform' propagator line")

    return {
        "kernel_unit_textbook": kernel_diffs[0],
        "kernel_xform_textbook": kernel_diffs[1],
        "kernel_unit_xform": gauge_match.group(1),
        "prop_unit_cg": prop_diffs[0],
        "prop_xform_cg": prop_diffs[1],
        "prop_unit_xform": prop_xform_match.group(1),
    }


def resolve_log_paths(log_args):
    log_paths = []
    for log_arg in log_args:
        path = Path(log_arg)
        if path.is_dir():
            log_paths.extend(sorted(path.glob("FFT_test.*")))
        elif path.exists():
            log_paths.append(path)

    if not log_paths:
        raise ValueError("No FFT log files found from the supplied --log paths")

    return log_paths


def choose_record(log_paths):
    errors = []
    for log_path in reversed(log_paths):
        try:
            return parse_log(log_path)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError("\n".join(errors))


def format_scientific(value):
    mantissa, exponent = f"{float(value):.5e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def first_table_tex(values):
    return "\n".join(
        [
            "%%%\\begin{table}[t]",
            "%%%\\centering",
            "\\begin{tabular}{ |c|c|c| }",
            "\\hline \\hline",
            "$~~~~$sqnorm(\\texttt{unit},  \\texttt{textbook}) $~~~~$&$~~~~$ "
            "sqnorm(\\texttt{xform}, \\texttt{textbook}) $~~~~$& $~~~~$sqnorm(\\texttt{unit},  "
            "\\texttt{xform}) $~~~~$   \\\\",
            "\\hline",
            f"${format_scientific(values['kernel_unit_textbook'])}$ & "
            f"${format_scientific(values['kernel_xform_textbook'])}$ & "
            f"${format_scientific(values['kernel_unit_xform'])}$ \\\\",
            "\\hline \\hline",
            "\\end{tabular}",
            f"%%%\\caption{{{FIRST_TABLE_CAPTION}}}",
            "%%%\\label{table:cross_check_fftw}",
            "%%%\\end{table}",
            "",
        ]
    )


def second_table_tex(values):
    return "\n".join(
        [
            "%%%\\begin{table}[t]",
            "%%%\\centering",
            "\\begin{tabular}{ |c|c|c| }",
            "\\hline \\hline",
            "sqnorm(\\texttt{unit}, CG) & "
            "sqnorm(\\texttt{xform}, CG) & "
            "sqnorm(\\texttt{unit}, \\texttt{xform})\\\\",
            "\\hline",
            f"$~~~~$${format_scientific(values['prop_unit_cg'])}$ $~~~~$&$~~~~$ "
            f"${format_scientific(values['prop_xform_cg'])}$ $~~~~$&$~~~~$ "
            f"${format_scientific(values['prop_unit_xform'])}$$~~~~$  \\\\",
            "\\hline \\hline",
            "\\end{tabular}",
            f"%%%\\caption{{{SECOND_TABLE_CAPTION}}}",
            "%%%\\label{table:cross_check_fftw2}",
            "%%%\\end{table}",
            "",
        ]
    )


def write_text(path_str, text):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    args = parse_args()
    values = choose_record(resolve_log_paths(args.log))
    write_text(args.output_table, first_table_tex(values))
    write_text(args.output_table_prop, second_table_tex(values))


if __name__ == "__main__":
    main()
