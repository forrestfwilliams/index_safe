import struct
from pathlib import Path

import indexed_gzip as igzip
import numpy as np


def create_gzidx(gz_path):
    out_path = Path(gz_path).with_suffix('.gzidx')

    with igzip.IndexedGzipFile(gz_path, spacing=int(4194304 / 8)) as f:
        f.build_full_index()
        f.export_index(out_path)


def parse_gzidx(fobj):
    header = fobj.read(35)
    n_points = struct.unpack('<I', header[31:])[0]
    raw_points = fobj.read(n_points * 18)
    parsed = np.array([struct.unpack('<QQBB', raw_points[18 * i : 18 * (i + 1)]) for i in range(n_points)])
    fobj.seek(0)
    return parsed


def build_gzidx(input_gzidx_path, output_gzidx_path, points):
    header_length = 35
    point_size = 18

    with open(input_gzidx_path, 'rb') as fobj:
        array = parse_gzidx(fobj)
        data = fobj.read()

    window_size = struct.unpack('<I', data[27:31])[0]
    n_points = struct.unpack('<I', data[31:35])[0]

    points_length = point_size * n_points
    point_offsets = np.arange(0, (n_points) * point_size, point_size) + header_length

    # FIXME Assume only first window entry is zero
    window_offsets = np.arange(0, array[1:].shape[0] * window_size, window_size) + header_length + points_length
    window_offsets = np.append([0], window_offsets, axis=0)

    indexes = np.where(np.isin(array[:, 0], points))[0].tolist()
    point_bytes = []
    window_bytes = []
    for i in indexes:
        point_entry = data[point_offsets[i] : point_offsets[i] + point_size]

        if window_offsets[i] == 0:
            window_entry = b''
        else:
            window_entry = data[window_offsets[i] : window_offsets[i] + window_size]

        point_bytes.append(point_entry)
        window_bytes.append(window_entry)

    out_gzidx = data[:31] + struct.pack('<I', len(points)) + b''.join(point_bytes) + b''.join(window_bytes)

    with open(output_gzidx_path, 'wb') as fobj:
        fobj.write(out_gzidx)

    return output_gzidx_path


if __name__ == '__main__':
    compressed_file = '../../tests/s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gz'
    uncompressed_file = '../../tests/s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff'
    index = 'out.gzidx'

    parse_gzidx(compressed_file)
