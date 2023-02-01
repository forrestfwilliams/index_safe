import struct
import zlib

import indexed_gzip as igzip
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

LOOKBACK_LENGTH = 32_000


def get_data(gz_path: str, index1: int, index2: int) -> bytes:
    with igzip.IndexedGzipFile(gz_path) as f:
        f.build_full_index()
        seek_points = list(f.seek_points())
        f.export_index('out.gzidx')
    array = np.array(seek_points)
    pd.DataFrame(array, columns=['uncompressed', 'compressed']).to_csv('out.csv', index=False)

    start = array[index1, 1]
    stop = array[index2, 1]
    # stand-in for a ranged GET request~~~~ #
    with open(gz_path, 'rb') as f:
        f.seek(start)
        compressed = f.read(stop - start)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    decompressed = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(compressed)
    return decompressed


def get_data_check(gz_path):
    with igzip.IndexedGzipFile(gz_path, spacing=int(4194304 / 8)) as f:
        f.build_full_index()
        seek_points = list(f.seek_points())
    array = np.array(seek_points)

    trials = []
    for i in range(array.shape[0] - 1):
        start = array[i, 1]
        stop = array[i + 1, 1]
        # stand-in for a ranged GET request~~~~ #
        with open(gz_path, 'rb') as f:
            f.seek(start)
            compressed = f.read(stop - start)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        success = True
        try:
            decompressed = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(compressed)
        except:
            success = False

        trials.append((i, str(success)))

    df = pd.DataFrame(trials, columns=['offset', 'result'])

    return df


def get_data_v2(gz_path: str, index1, index2):
    with igzip.IndexedGzipFile(gz_path) as f:
        f.build_full_index()
        seek_points = list(f.seek_points())
    array = np.array(seek_points)

    start = array[index1, 1]
    stop = array[index2, 1]

    decompress_start = array[index1, 0]
    decompress_stop = array[index2, 0]
    decompress_length = decompress_stop - decompress_start

    header_length = array[0, 1] - array[0, 0]
    if start != header_length:
        lookback_max = start - LOOKBACK_LENGTH
        new_start = max(lookback_max, header_length)
        seek_offset = start - new_start
        start = new_start
    else:
        seek_offset = 0

    with open(gz_path, 'rb') as f:
        f.seek(start)
        compressed = f.read(stop - start)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    decompressed = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(compressed)
    return decompressed, (decompress_start, decompress_stop)


def parse_gzidx(gz_path):
    tmp = tempfile.NamedTemporaryFile()
    tmp = 'out.gzidx'
    
    if not Path(tmp).exists():
        with igzip.IndexedGzipFile(gz_path, spacing=int(4194304 / 8)) as f:
            f.build_full_index()
            f.export_index(tmp)

    with open(tmp, 'rb') as f:
        raw = f.read()

    header = raw[:35]
    n_points = struct.unpack('<I', header[31:])[0]
    raw_points = raw[35 : 35 + (n_points * 18)]
    parsed = np.array([struct.unpack('<QQBB', raw_points[18 * i : 18 * (i + 1)]) for i in range(n_points)])
    pd.DataFrame(parsed, columns=['compressed', 'uncompressed', 'bit_offset', 'has_window']).to_csv(
        'parsed.csv', index=False
    )
    return None


compressed_file = '../../tests/s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gz'
uncompressed_file = '../../tests/s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff'
index = 'out.gzidx'

# out = get_data(compressed_file, 0, 1)
parse_gzidx(compressed_file)

# df = get_data_check(compressed_file)
# df.to_csv('trials.csv', index=False)
print('done')
