# indexed_safe
A set of Python utilities for downloading Sentinel-1 Burst data and metadata via S3 with minimal `GET` requests and data
download:

The main innovation of these utilities is that they allow downloading of burst data **without downloading data from the
swath they're contained in**. These has been difficult because `DEFLATE` compressed data is not set up for random read
access. This innovation is made possible by the [`zran.c`](https://github.com/madler/zlib/blob/master/examples/zran.c)
utility created by Mark Adler and its implementation in Paul McCarthy's [`indexed_gzip`](https://github.com/pauldmccarthy/indexed_gzip) package.

# Installation
Install using Mamba or Conda using the command:

`mamba create -f envirnoment.yml`
or
`conda env create -f envirnoment.yml`

# Usage
With a Sentinel-1 SAFE stored locally, run `index_safe.py` (will add CLI soon), then with a copy of this SAFE in an
publicly-readable bucket, run `extract_burst.py` (will add CLI soon).
