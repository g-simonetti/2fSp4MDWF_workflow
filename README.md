# Möbius domain wall fermion parameter scan and meson spectroscopy in the Sp(4) gauge theory with two fundamental fermions&mdash;Analysis workflow


[![DOI](https://zenodo.org/badge/DOI/TODO DOI.svg)](https://doi.org/TODO DOI)

The workflow in this repository performs
the analyses presented in the paper
[TODO: paper title][paper].

## Requirements

- Conda, for example, installed from [Miniforge][miniforge]
- [Snakemake][snakemake], which may be installed using Conda
- LaTeX, for example, from [TeX Live][texlive]

## Setup

1. Install the dependencies above.
2. Clone this repository including submodules
   (or download its Zenodo release and `unzip` it)
   and `cd` into it:

   ```shellsession
   git clone --recurse-submodules https://github.com/telos-collaboration/TODO REPO NAME
   cd TODO REPO NAME
   ```
3. Download the `raw_data.zip` file from [the data release][datarelease],
   and extract it into the root of the repository,
4. Download the `ensemble_metadata.csv` file from [the data release].      
   [datarelease], and place it into the `metadata` directory.

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

Using --cores all on a MacBook Pro with an Apple M3 Pro processor
(12 CPU cores: 6 performance and 6 efficiency),
the analysis takes around 30 minutes starting from raw data.

## Output

Output plots, tables, and definitions
are placed in the `assets/plots`, `assets/tables`, and `assets/definitions` directories.

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

[datarelease]: https://doi.org/10.5281/zenodo.TODO_ZENODO_ID
[miniforge]: https://github.com/conda-forge/miniforge
[paper]: https://doi.org/10.48550/arXiv.TODO_ARXIV_ID
[snakemake]: https://snakemake.github.io
[snakemake-conda]: https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html
[texlive]: https://tug.org/texlive/

