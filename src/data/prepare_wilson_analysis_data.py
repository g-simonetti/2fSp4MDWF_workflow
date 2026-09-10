#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


CHUNK_SIZE = 1024 * 1024
REQUIRED_ENSEMBLE_FILES = (
    "w0_samples.json",
    "meson_extraction_f_ps_samples.json",
    "meson_extraction_f_v_samples.json",
    "meson_gevp_f_ps_samples.json",
    "meson_gevp_f_v_samples.json",
    "decay_constant_f_ps_samples.json",
)
MIN_READY_ENSEMBLES = 20
OPTIONAL_ENSEMBLE_FILES = (
    "meson_extraction_f_ps_mean.csv",
    "meson_extraction_f_v_mean.csv",
    "meson_gevp_E0_f_ps_mean.csv",
    "meson_gevp_E0_f_v_mean.csv",
    "decay_constant_f_ps_mean.csv",
)


def log(message):
    print(f"[prepare_wilson_analysis_data] {message}", flush=True)


def format_bytes(size):
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def print_download_progress(filename, downloaded, total):
    if total:
        fraction = min(downloaded / total, 1.0)
        filled = int(30 * fraction)
        bar = "#" * filled + "-" * (30 - filled)
        message = (
            f"\r[prepare_wilson_analysis_data] Downloading {filename.name} "
            f"[{bar}] {100 * fraction:5.1f}% "
            f"({format_bytes(downloaded)} / {format_bytes(total)})"
        )
    else:
        message = (
            f"\r[prepare_wilson_analysis_data] Downloading {filename.name} "
            f"{format_bytes(downloaded)}"
        )
    print(message, end="", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare analysis-ready Wilson comparison data. Existing processed "
            "JSON files are reused; otherwise precomputed data are staged or "
            "the upstream Wilson workflow is run from local/downloaded archives."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "precomputed", "recompute"),
        default="auto",
        help=(
            "auto reuses existing data, then tries precomputed data, then "
            "recomputes; precomputed requires a precomputed archive; recompute "
            "runs the upstream workflow."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--precomputed-data-archive", default="")
    parser.add_argument("--local-precomputed-data-archive", default="")
    parser.add_argument("--precomputed-data-url", default="")
    parser.add_argument("--precomputed-data-md5", default="")
    parser.add_argument("--precomputed-data-source-subdir", default="")
    parser.add_argument("--workflow-dir", required=True)
    parser.add_argument("--workflow-archive", required=True)
    parser.add_argument("--workflow-url", default="")
    parser.add_argument("--workflow-md5", default="")
    parser.add_argument("--raw-data-archive", required=True)
    parser.add_argument("--local-raw-data-archive", default="")
    parser.add_argument("--raw-data-url", default="")
    parser.add_argument("--raw-data-md5", default="")
    parser.add_argument("--raw-data-dir", default="")
    parser.add_argument("--metadata-archive", required=True)
    parser.add_argument("--metadata-url", default="")
    parser.add_argument("--metadata-md5", default="")
    parser.add_argument("--upstream-target", default="required_wilson_jsons")
    parser.add_argument("--cores", default="1")
    return parser.parse_args()


def md5sum(filename):
    digest = hashlib.md5()
    with filename.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(filename, expected_md5):
    if not expected_md5:
        return

    observed_md5 = md5sum(filename)
    if observed_md5.lower() != expected_md5.lower():
        raise RuntimeError(
            f"MD5 mismatch for {filename}: expected {expected_md5}, got {observed_md5}"
        )


def configured_path(value):
    return Path(value).expanduser()


def download_if_missing(url, filename, expected_md5):
    if filename.is_file():
        log(f"Using existing archive: {filename}")
        verify_md5(filename, expected_md5)
        return

    if not url:
        raise RuntimeError(f"{filename} does not exist and no download URL was provided.")

    log(f"Downloading {url} -> {filename}")
    filename.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=filename.name + ".",
        suffix=".part",
        dir=str(filename.parent),
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                tmp_file.write(chunk)
                downloaded += len(chunk)
                print_download_progress(filename, downloaded, total)
            print()

    tmp_path.replace(filename)
    verify_md5(filename, expected_md5)
    log(f"Finished download: {filename}")


def existing_archive(candidates, expected_md5):
    for candidate in candidates:
        if not candidate:
            continue
        archive = configured_path(candidate)
        if archive.is_file():
            verify_md5(archive, expected_md5)
            return archive
    return None


def write_marker(marker, message):
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(message + "\n", encoding="utf-8")


def ensemble_dirs(output_dir):
    return sorted(path for path in output_dir.glob("Sp4b*nF2mF*T*L*") if path.is_dir())


def analysis_data_ready(output_dir):
    dirs = ensemble_dirs(output_dir)
    if len(dirs) < MIN_READY_ENSEMBLES:
        return False

    return all((path / filename).is_file() for path in dirs for filename in REQUIRED_ENSEMBLE_FILES)


def extract_workflow_archive(archive, workflow_dir):
    if (workflow_dir / "workflow" / "Snakefile").is_file():
        log(f"Using existing Wilson workflow: {workflow_dir}")
        return

    log(f"Extracting Wilson workflow archive: {archive}")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wilson-workflow-") as tmp:
        tmp_path = Path(tmp)
        shutil.unpack_archive(str(archive), str(tmp_path))
        candidates = [
            path.parent.parent
            for path in tmp_path.rglob("workflow/Snakefile")
            if path.is_file()
        ]
        if not candidates:
            raise RuntimeError(f"No workflow/Snakefile found in {archive}")

        source_root = candidates[0]
        for item in source_root.iterdir():
            destination = workflow_dir / item.name
            if destination.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)


def extract_tar_archive(archive, destination_root):
    log(f"Extracting tar archive: {archive} -> {destination_root}")
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise RuntimeError(f"Refusing unsafe tar path: {member.name}")
            tar.extract(member, path=destination_root)


def extract_zip_archive(archive, destination_root):
    log(f"Extracting zip archive: {archive} -> {destination_root}")
    with zipfile.ZipFile(archive) as zip_archive:
        for member in zip_archive.infolist():
            member_path = Path(member.filename)
            if member.filename.startswith("/") or ".." in member_path.parts:
                raise RuntimeError(f"Refusing unsafe zip path: {member.filename}")
        zip_archive.extractall(destination_root)


def extract_archive(archive, destination_root):
    if tarfile.is_tarfile(archive):
        extract_tar_archive(archive, destination_root)
    elif zipfile.is_zipfile(archive):
        extract_zip_archive(archive, destination_root)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive}")


def candidate_analysis_roots(extract_dir, source_subdir):
    if source_subdir:
        explicit = extract_dir / source_subdir
        if explicit.exists():
            yield explicit
        for path in extract_dir.rglob(Path(source_subdir).name):
            if path.is_dir() and path != explicit:
                yield path

    yield extract_dir
    for path in extract_dir.rglob("*"):
        if path.is_dir() and analysis_data_ready(path):
            yield path


def copy_analysis_data_tree(source_dir, output_dir):
    log(f"Copying precomputed Wilson data: {source_dir} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for ensemble_dir in ensemble_dirs(source_dir):
        destination_dir = output_dir / ensemble_dir.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for filename in REQUIRED_ENSEMBLE_FILES + OPTIONAL_ENSEMBLE_FILES:
            copy_if_present(ensemble_dir / filename, destination_dir / filename)

    for filename in ("ensembles.json", "spectrum_data.json"):
        copy_if_present(source_dir / filename, output_dir / filename)


def precomputed_archive(args, required):
    archive = existing_archive(
        (args.local_precomputed_data_archive, args.precomputed_data_archive),
        args.precomputed_data_md5,
    )
    if archive is not None:
        return archive

    if args.precomputed_data_url and args.precomputed_data_archive:
        archive = configured_path(args.precomputed_data_archive)
        download_if_missing(args.precomputed_data_url, archive, args.precomputed_data_md5)
        return archive

    if required:
        raise RuntimeError(
            "Wilson precomputed mode was requested, but no precomputed archive "
            "exists and no download URL was configured."
        )
    return None


def stage_precomputed_analysis_data(args, output_dir, required=False):
    archive = precomputed_archive(args, required)
    if archive is None:
        log("No precomputed Wilson archive configured/found; will recompute if needed.")
        return False

    log(f"Trying precomputed Wilson archive: {archive}")
    with tempfile.TemporaryDirectory(prefix="wilson-precomputed-") as tmp:
        extract_dir = Path(tmp)
        extract_archive(archive, extract_dir)
        for source_dir in candidate_analysis_roots(
            extract_dir,
            args.precomputed_data_source_subdir,
        ):
            if analysis_data_ready(source_dir):
                copy_analysis_data_tree(source_dir, output_dir)
                return True

    raise RuntimeError(
        f"No complete Wilson analysis-ready data tree found in {archive}."
    )


def ensure_upstream_inputs(args):
    log("Preparing upstream Wilson workflow inputs.")
    workflow_dir = configured_path(args.workflow_dir)
    workflow_archive = configured_path(args.workflow_archive)
    raw_archive = configured_path(args.raw_data_archive)
    local_raw_archive = (
        configured_path(args.local_raw_data_archive)
        if args.local_raw_data_archive
        else None
    )
    metadata_archive = configured_path(args.metadata_archive)

    download_if_missing(args.workflow_url, workflow_archive, args.workflow_md5)
    extract_workflow_archive(workflow_archive, workflow_dir)

    metadata_csv = workflow_dir / "metadata" / "spectrum" / "ensemble_metadata.csv"
    if not metadata_csv.is_file():
        log("Wilson metadata not found in workflow directory; preparing metadata archive.")
        download_if_missing(args.metadata_url, metadata_archive, args.metadata_md5)
        extract_tar_archive(metadata_archive, workflow_dir)
    else:
        log(f"Using existing Wilson metadata: {metadata_csv}")

    raw_data_dir = workflow_dir / "raw_data"
    if not raw_data_dir.exists():
        if args.raw_data_dir:
            source_raw_data_dir = configured_path(args.raw_data_dir)
            if not source_raw_data_dir.exists():
                raise RuntimeError(
                    f"Configured raw-data directory does not exist: {source_raw_data_dir}"
                )
            log(f"Linking Wilson raw data directory: {source_raw_data_dir} -> {raw_data_dir}")
            raw_data_dir.symlink_to(source_raw_data_dir.resolve(), target_is_directory=True)
        elif local_raw_archive is not None and local_raw_archive.is_file():
            log(f"Using local Wilson raw-data archive: {local_raw_archive}")
            verify_md5(local_raw_archive, args.raw_data_md5)
            extract_tar_archive(local_raw_archive, workflow_dir)
        else:
            log("Wilson raw_data directory is missing; downloading raw-data archive.")
            download_if_missing(args.raw_data_url, raw_archive, args.raw_data_md5)
            extract_tar_archive(raw_archive, workflow_dir)
    else:
        log(f"Using existing Wilson raw_data directory: {raw_data_dir}")


def upstream_required_targets(workflow_dir):
    metadata_csv = workflow_dir / "metadata" / "spectrum" / "ensemble_metadata.csv"
    return [
        f"intermediary_data/{ensemble_name}/{filename}"
        for ensemble_name in selected_ensemble_names(metadata_csv)
        for filename in REQUIRED_ENSEMBLE_FILES
    ]


def upstream_targets(workflow_dir, target):
    if target == "required_wilson_jsons":
        return upstream_required_targets(workflow_dir)
    return [target]


def run_upstream_workflow(workflow_dir, target, cores):
    targets = upstream_targets(workflow_dir, target)
    if targets and all((workflow_dir / target_path).exists() for target_path in targets):
        log("Required upstream Wilson JSON files already exist; skipping upstream workflow.")
        return

    log(
        "Running upstream Wilson workflow for "
        f"{len(targets)} target(s) with {cores} core(s)."
    )
    try:
        subprocess.run(
            ["snakemake", "--cores", str(cores), "--use-conda", *targets],
            cwd=workflow_dir,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not run the upstream Wilson workflow because 'snakemake' "
            "is not available on PATH."
        ) from exc


def row_selected(row):
    value = row.get("use_in_extrapolation", "TRUE")
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def ensemble_dir_name(row):
    return (
        f"Sp{row['Nc']}b{row['beta']}nF{row['nF']}mF{row['mF']}"
        f"T{row['Nt']}L{row['Ns']}"
    )


def selected_ensemble_names(metadata_csv):
    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row_selected(row)]
    return [ensemble_dir_name(row) for row in rows]


def copy_if_present(source, destination):
    if not source.is_file():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def write_top_level_jsons(workflow_dir, output_dir, ensemble_names):
    metadata_csv = workflow_dir / "metadata" / "spectrum" / "ensemble_metadata.csv"
    ensemble_data_csv = workflow_dir / "data_assets" / "spectrum" / "ensemble_data.csv"

    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        metadata_rows = [
            row
            for row in csv.DictReader(handle)
            if ensemble_dir_name(row) in set(ensemble_names)
        ]
    (output_dir / "ensembles.json").write_text(
        json.dumps(metadata_rows, indent=2) + "\n",
        encoding="utf-8",
    )

    if ensemble_data_csv.is_file():
        with ensemble_data_csv.open(newline="", encoding="utf-8-sig") as handle:
            spectrum_rows = list(csv.DictReader(handle))
    else:
        spectrum_rows = []

    (output_dir / "spectrum_data.json").write_text(
        json.dumps(spectrum_rows, indent=2) + "\n",
        encoding="utf-8",
    )


def stage_analysis_data(workflow_dir, output_dir):
    metadata_csv = workflow_dir / "metadata" / "spectrum" / "ensemble_metadata.csv"
    ensemble_names = selected_ensemble_names(metadata_csv)
    if not ensemble_names:
        raise RuntimeError(f"No selected Wilson ensembles found in {metadata_csv}")

    log(
        f"Staging {len(ensemble_names)} selected Wilson ensemble(s) "
        f"from {workflow_dir / 'intermediary_data'} to {output_dir}."
    )
    missing = []
    for name in ensemble_names:
        source_dir = workflow_dir / "intermediary_data" / name
        destination_dir = output_dir / name

        for filename in REQUIRED_ENSEMBLE_FILES:
            if not copy_if_present(source_dir / filename, destination_dir / filename):
                missing.append(str(source_dir / filename))

        for filename in OPTIONAL_ENSEMBLE_FILES:
            copy_if_present(source_dir / filename, destination_dir / filename)

    if missing:
        raise RuntimeError(
            "The upstream workflow did not produce required Wilson files:\n"
            + "\n".join(missing[:20])
        )

    write_top_level_jsons(workflow_dir, output_dir, ensemble_names)


def main():
    args = parse_args()
    output_dir = configured_path(args.output_dir)
    marker = configured_path(args.marker)

    log(f"Mode: {args.mode}")
    log(f"Output directory: {output_dir}")
    if analysis_data_ready(output_dir):
        log("Wilson analysis-ready data already present; skipping preparation.")
        write_marker(marker, "Wilson analysis-ready data already present.")
        return 0

    if args.mode in {"auto", "precomputed"}:
        staged = stage_precomputed_analysis_data(
            args,
            output_dir,
            required=args.mode == "precomputed",
        )
        if staged and analysis_data_ready(output_dir):
            log("Wilson analysis-ready data prepared from precomputed archive.")
            write_marker(marker, "Wilson analysis-ready data prepared from precomputed archive.")
            return 0

    if args.mode == "precomputed":
        print(
            f"Wilson precomputed data are incomplete in {output_dir}.",
            file=sys.stderr,
        )
        return 1

    ensure_upstream_inputs(args)
    workflow_dir = configured_path(args.workflow_dir)
    run_upstream_workflow(workflow_dir, args.upstream_target, args.cores)
    stage_analysis_data(workflow_dir, output_dir)

    if not analysis_data_ready(output_dir):
        print(
            f"Wilson analysis-ready data are still incomplete in {output_dir}.",
            file=sys.stderr,
        )
        return 1

    log("Wilson analysis-ready data prepared successfully.")
    write_marker(marker, "Wilson analysis-ready data prepared from upstream workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
