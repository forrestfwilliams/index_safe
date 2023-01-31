import tempfile
from pathlib import Path

import indexed_gzip as igzip
import numpy as np
import pytest

from index_safe import index

GZ_PATH = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gz'
GZIDX_PATH = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gzidx'


@pytest.fixture(scope='module')
def seek_point_array():
    if not Path(GZIDX_PATH).exists():
        with igzip.IndexedGzipFile(GZ_PATH, spacing=int(4194304 / 8)) as f:
            f.build_full_index()
            f.export_index(GZIDX_PATH)

    with igzip.IndexedGzipFile(GZ_PATH, spacing=int(4194304 / 8)) as f:
        f.import_index(GZIDX_PATH)
        seek_points = list(f.seek_points())
    array = np.array(seek_points)

    return array


def test_parse_gzidx(seek_point_array):
    with open(GZIDX_PATH, 'rb') as f:
        test_array = index.parse_gzidx(f)
    assert np.all(test_array[:, [1, 0]] == seek_point_array)


def test_build_gzidx(seek_point_array):
    points = [100,105]
    points_compressed = seek_point_array[points, 1].tolist()
    points_uncompressed = seek_point_array[points, 0].tolist()

    tmp = tempfile.NamedTemporaryFile()
    index.build_gzidx(GZIDX_PATH, tmp.name, points_compressed)

    with igzip.IndexedGzipFile(GZ_PATH) as fobj:
        fobj.import_index(GZIDX_PATH)
        fobj.seek(points_uncompressed[0] + 1)
        golden = fobj.read(points_uncompressed[1] - 1)

    with igzip.IndexedGzipFile(GZ_PATH) as fobj:
        fobj.import_index(tmp.name)
        fobj.seek(points_uncompressed[0] + 1)
        test = fobj.read(points_uncompressed[1] - 1)

    assert golden == test
