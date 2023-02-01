import io
import struct
import tempfile
import zipfile
from pathlib import Path

import indexed_gzip as igzip
import numpy as np

from index_safe import index_safe, utils

GZIDX_HEADER_LENGTH = 35
GZIDX_POINT_LENGTH = 18


def create_gzidx(gz_path, spacing=2**20):
    out_path = Path(gz_path).with_suffix('.gzidx')

    with igzip.IndexedGzipFile(gz_path, spacing=spacing) as f:
        f.build_full_index()
        f.export_index(out_path)

    return out_path


def parse_gzidx(fobj):
    header = fobj.read(35)
    n_points = struct.unpack('<I', header[31:])[0]
    window_size = struct.unpack('<I', header[27:31])[0]
    raw_points = fobj.read(n_points * 18)
    parsed = np.array([struct.unpack('<QQBB', raw_points[18 * i : 18 * (i + 1)]) for i in range(n_points)])
    fobj.seek(0)
    return parsed, n_points, window_size


def build_gzidx(input_gzidx_path, output_gzidx_path, points=None, filesize=None, compressed_offset=0):
    with open(input_gzidx_path, 'rb') as fobj:
        array, n_points, window_size = parse_gzidx(fobj)
        data = fobj.read()

    # FIXME Assume only first window entry is zero
    length_to_window = GZIDX_HEADER_LENGTH + (GZIDX_POINT_LENGTH * n_points)
    window_offsets = np.arange(0, array[1:].shape[0] * window_size, window_size) + length_to_window
    window_offsets = np.append([0], window_offsets, axis=0)

    array = np.append(array, np.expand_dims(window_offsets, axis=1), axis=1)
    if points:
        array = array[np.isin(array[:, 0], points), :].copy()
    array[:, 0] += compressed_offset

    point_bytes = []
    window_bytes = []
    for row in array:
        point_entry = struct.pack('<QQBB', *row[:4].tolist())

        if row[4] == 0:
            window_entry = b''
        else:
            window_entry = data[row[4] : row[4] + window_size]

        point_bytes.append(point_entry)
        window_bytes.append(window_entry)

    if filesize:
        filesize_bytes = struct.pack('<Q', filesize)
    else:
        filesize_bytes = data[7:15]

    header = data[:7] + filesize_bytes + data[15:31] + struct.pack('<I', array.shape[0])
    out_gzidx = header + b''.join(point_bytes) + b''.join(window_bytes)
    with open(output_gzidx_path, 'wb') as fobj:
        fobj.write(out_gzidx)

    return output_gzidx_path


def create_gzidx_for_zip_member(zip_path, member_name, out_path):
    gz_header_length = 10
    with zipfile.ZipFile(zip_path) as f:
        zinfo = [x for x in f.infolist() if member_name in Path(x.filename).name][0]

    offset_zinfo = utils.OffsetZipInfo(zip_path, zinfo)
    offset = offset_zinfo.get_compressed_offset()
    archive_size = Path(zip_path).stat().st_size

    with open(zip_path, 'rb') as f:
        f.seek(offset.start)
        body = f.read(offset.stop - offset.start)

    gz_body = index_safe.wrap_as_gz(body, offset_zinfo)
    tmp = tempfile.NamedTemporaryFile()
    with igzip.IndexedGzipFile(io.BytesIO(gz_body), spacing=2**22) as f:
        f.build_full_index()
        f.export_index(tmp.name)

    adjusted_offset = offset.start - gz_header_length
    build_gzidx(tmp.name, out_path, filesize=archive_size, compressed_offset=adjusted_offset)
    return out_path


if __name__ == '__main__':
    gz_path = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gz'
    gzidx_path = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gzidx'
    create_gzidx(gz_path, 2**16)
