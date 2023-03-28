#!/bin/bash --login
set -e
conda activate index_safe
exec python -um awslambdaric
# exec create_index --help
