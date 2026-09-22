# Symplectic lattice gauge theories in the Grid framework: domain wall fermions and continuum extrapolations––analysis workflow


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22307749.svg)](https://doi.org/10.5281/zenodo.22307749)

The workflow in this repository performs
the analyses presented in the paper
[Symplectic lattice gauge theories in the Grid framework: domain wall fermions and continuum extrapolations][paper].

## Requirements

- Conda, for example, installed from [Miniforge][miniforge]
- [Snakemake][snakemake], which may be installed using Conda
- LaTeX, for example, from [TeX Live][texlive]

## Setup

1. Install the dependencies above.
2. Clone the workflow repository
   (or download and extract the [code release][coderelease] from Zenodo)
   and `cd` into it:

   ```shellsession
   git clone https://github.com/g-simonetti/2fSp4MDWF_workflow.git
   cd 2fSp4MDWF_workflow
   ```
3. Download the required `.tar.gz` data archives from [the data release][datarelease].
   To reproduce the workflow outputs from the released data,
   the repository root must contain `metadata/` and `raw_data/`.
   The optional `external_data/` archive may also be added to reuse the
   analysis-ready Wilson-fermion comparison inputs.

   Extract the following archives in the repository root:

   ```shellsession
   unzip metadata.zip
   unzip external_data.zip
   tar xzf raw_data.tar.gz
   ```

   After extraction,
   the expected layout is:

   ```text
   2fSp4MDWF_workflow/
   ├── config/
   ├── metadata/
   │   └── ensembles.csv
   ├── raw_data/
   ├── src/
   ├── styles/
   ├── workflow/
   └── external_data/
       └── wilson_fermions_data/
   ```

   The `external_data/` directory is optional.
   When `external_data/wilson_fermions_data/` is present and complete,
   the workflow uses the included analysis-ready Wilson comparison data.
   When this directory is absent or incomplete,
   Snakemake can recreate those Wilson inputs through the
   `prepare_wilson_analysis_data` rule.
   That rule stages the upstream [Wilson workflow release][Wilson-workflow],
   downloads the [Wilson raw-data release][Wilson-data],
   and requests only the Wilson JSON targets needed by this analysis.

## Running the workflow

The workflow is run using Snakemake:

``` shellsession
snakemake --cores 1 --use-conda
```

where the number `1`
may be replaced by
the number of CPU cores you wish to allocate to the computation.

Snakemake will automatically download and install
all required Python packages.
This requires an Internet connection;
if you are running in an HPC environment where you would need
to run the workflow without Internet access,
details on how to preinstall the environment
can be found in the [Snakemake documentation][snakemake-conda].

There are two possible ways to run the workflow.

1. If `external_data/wilson_fermions_data/` has been extracted from the data
   release,
   the workflow reuses those analysis-ready Wilson comparison inputs:

   ``` shellsession
   snakemake --cores all --use-conda
   ```

   With this directory already present,
   running the workflow with `--cores all` on a MacBook Pro with an
   Apple M3 Pro processor
   (12 CPU cores: 6 performance and 6 efficiency)
   took around 30 minutes.

2. If `external_data/wilson_fermions_data/` is not included,
   the workflow attempts to recreate the Wilson comparison inputs from the
   upstream Wilson releases. This requires Internet access and additional disk
   space. The number of cores passed to the upstream Wilson workflow follows
   the `--cores` value used for the main Snakemake command.

   To force regeneration of the Wilson comparison inputs from the upstream
   workflow, run:

   ``` shellsession
   snakemake --cores 6 --use-conda --forcerun prepare_wilson_analysis_data process_wilson_combined
   ```

   On Tursa HPC facility, using 6 cores and including the download of the Wilson raw data,
   the same regeneration step took approximately 3 hours.
   The full regeneration of the Wilson data is not supported on macOS.

## Output

Output plots, tables, equations, and definitions
are placed in the `assets/plots`, `assets/tables`, `assets/equations` and `assets/definitions` directories.

Output data assets are placed into the `data_assets` directory.

Intermediary data are placed in the `intermediary_data` directory.

## Reusability

This workflow is relatively tailored to the data
which it was originally written to analyse.
Additional ensembles may be added to the analysis
by adding relevant files to the `raw_data` directory,
and adding corresponding entries to the files in the `metadata` directory.
However,
extending the analysis in this way
has not been as fully tested as the rest of the workflow,
and is not guaranteed to be trivial for someone not already familiar with the code.

[datarelease]: https://doi.org/10.5281/zenodo.22308547
[coderelease]: https://doi.org/10.5281/zenodo.22307749
[github]: https://github.com/g-simonetti/2fSp4MDWF_workflow
[miniforge]: https://github.com/conda-forge/miniforge
[paper]: https://arxiv.org/abs/2609.19930
[snakemake]: https://snakemake.github.io
[snakemake-conda]: https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html
[texlive]: https://tug.org/texlive/
[Wilson-data]: https://doi.org/10.5281/zenodo.20111459
[Wilson-workflow]: https://doi.org/10.5281/zenodo.20638262
