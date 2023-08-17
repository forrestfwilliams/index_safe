# index_safe
A set of Python utilities for downloading Sentinel-1 Burst data and metadata via S3 with minimal `GET` requests and data
download:

The main innovation of these utilities is that they allow downloading of burst data **without downloading the
surrounding data from the swath they're contained in**. These has been difficult because `DEFLATE` compressed data is
not set up for random read access. This innovation is made possible by the
[`zran.c`](https://github.com/madler/zlib/blob/master/examples/zran.c) utility created by Mark Adler and its
implementation in Paul McCarthy's [`indexed_gzip`](https://github.com/pauldmccarthy/indexed_gzip) package.

# Installation
Install the requirements using Mamba or Conda using the command:

`mamba create -f envirnoment.yml`
or
`conda env create -f envirnoment.yml`

Activate the environment:
`mamba (or conda) activate bursts`

Then, from within the top-level package directory, run:
`python -m pip install -e .`

# Usage
Run `create_index.py` Ex:
```bash
create_index S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85
```
This will create a `{GRANULE_NAME}.json` and burst index json files (`{GRANULE_NAME}_{SWATH_NAME_{POLARIZATION}.json`) that contain the information needed to download the metadata/data directly.

Then, run `extract_burst.py` Ex:
```bash
extract_burst S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_IW2_VV_0.json
```
to get the burst image data

Then, run `extract_metadata.py` Ex:
```bash
extract_metadata S1A_IW_SLC__1SDV_20200604T022251_20200604T022318.json
```
to get the burst metadata
