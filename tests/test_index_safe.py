import gzip
import io
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path

import indexed_gzip as igzip
import numpy as np
import pytest
import zran
from osgeo import gdal

from index_safe import create_index, extract_burst, utils

BURST_LENGTH = 153814955 - 109035
BURST_SHAPE = (1510, 25448)

BURST0_START = 109035
BURST0_STOP = BURST0_START + BURST_LENGTH
BURST7_START = 1076050475
BURST7_STOP = BURST7_START + BURST_LENGTH


ZIP_PATH = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
TIFF_PATH = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff'

BURST_NUMBER = 7
BURST_START = BURST7_START
BURST_STOP = BURST7_STOP
BURST_RAW_PATH = f'raw_0{BURST_NUMBER + 1}.slc.vrt'
BURST_VALID_PATH = f'valid_0{BURST_NUMBER + 1}.slc.vrt'
TEST_BURST_NAME = f'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_IW2_VV_{BURST_NUMBER}.tiff'

GZIDX_PATH = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gzidx'
GZ_PATH = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gz'


def load_geotiff(infile, band=1):
    ds = gdal.Open(infile, gdal.GA_ReadOnly)

    data = ds.GetRasterBand(band).ReadAsArray()
    nodata = ds.GetRasterBand(band).GetNoDataValue()
    projection = ds.GetProjection()
    transform = ds.GetGeoTransform()
    ds = None
    return data, transform, projection, nodata


@pytest.fixture(scope='module')
def golden_bytes():
    with open(TIFF_PATH, 'rb') as f:
        golden_bytes = f.read()
    return golden_bytes


@pytest.fixture(scope='module')
def golden_tiff():
    if not Path(TIFF_PATH).exists():
        with zipfile.ZipFile(ZIP_PATH, mode='r') as archive:
            info_list = archive.infolist()
            zinfo = [x for x in info_list if TIFF_PATH in x.filename][0]
            with archive.open(zinfo.filename, 'r') as f:
                body = f.read()

        with open(Path(zinfo.filename).name, 'wb') as f:
            f.write(body)

    return TIFF_PATH


@pytest.fixture(scope='module')
def golden_gz(golden_tiff):
    if not Path(GZ_PATH).exists():
        with open(TIFF_PATH, 'rb') as f_in:
            with gzip.open(GZ_PATH, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return GZ_PATH


@pytest.fixture(scope='module')
def zinfo():
    with zipfile.ZipFile(ZIP_PATH, mode="r") as archive:
        info_list = archive.infolist()
        zinfo = [x for x in info_list if TIFF_PATH in x.filename][0]
    return zinfo


@pytest.fixture(scope='module')
def seek_point_array(golden_gz):
    if not Path(GZIDX_PATH).exists():
        print('Creating full gzidx')
        with igzip.IndexedGzipFile(GZ_PATH, spacing=2**22) as f:
            f.build_full_index()
            f.export_index(GZIDX_PATH)

    with igzip.IndexedGzipFile(GZ_PATH, spacing=2**22) as f:
        f.import_index(GZIDX_PATH)
        seek_points = list(f.seek_points())
    array = np.array(seek_points)

    return array


def test_burst_specific_index():
    # # IW2 VV 3 golden
    # compressed = utils.Offset(start=2589238610, stop=2690292830)
    # uncompressed = utils.Offset(start=461226795, stop=614932715)

    # IW2 VV 7
    compressed = utils.Offset(start=2986776876, stop=3087580207)
    uncompressed = utils.Offset(start=1076050475, stop=1229756395)

    indexer = utils.ZipIndexer(ZIP_PATH, TIFF_PATH)
    indexer.create_full_dflidx()
    compressed_offset, uncompressed_offset, index = indexer.subset_dflidx(
        starts=[uncompressed.start], stops=[uncompressed.stop]
    )

    assert compressed_offset.start == compressed.start
    assert compressed_offset.stop == compressed.stop

    with open(ZIP_PATH, 'rb') as f:
        f.seek(compressed_offset.start)
        body = f.read(compressed_offset.stop - compressed_offset.start)

    length = uncompressed.stop - uncompressed.start
    start = uncompressed.start - uncompressed_offset.start
    test = zran.decompress(body, index, start, length)

    with zipfile.ZipFile(ZIP_PATH) as f:
        zinfo = [x for x in f.infolist() if TIFF_PATH in Path(x.filename).name][0]
        with f.open(zinfo.filename, 'r') as member:
            member.seek(uncompressed.start)
            golden = member.read(uncompressed.stop - uncompressed.start)

    assert golden == test

# Swath level
def test_get_zip_compressed_offset(zinfo, golden_bytes):
    offset = utils.get_zip_compressed_offset(ZIP_PATH, zinfo)
    with open(ZIP_PATH, 'rb') as f:
        f.seek(offset.start)
        bytes_compressed = f.read(offset.stop - offset.start)
    test_bytes = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(bytes_compressed)

    assert len(test_bytes) == len(golden_bytes)
    assert test_bytes == golden_bytes


# Burst level
def test_get_burst_annotation_data(zinfo):
    burst_shape, burst_offsets, burst_windows = create_index.get_burst_annotation_data(ZIP_PATH, zinfo.filename)

    assert burst_shape == BURST_SHAPE
    assert burst_offsets[7].start == BURST_START
    assert burst_offsets[7].stop == BURST_STOP


def test_burst_bytes_to_numpy(golden_bytes):
    test_array = extract_burst.burst_bytes_to_numpy(golden_bytes[BURST_START:BURST_STOP], BURST_SHAPE)
    golden_array = load_geotiff(BURST_RAW_PATH)[0]
    equal = np.isclose(golden_array, test_array)
    assert np.all(equal)


def test_invalid_to_nodata(golden_bytes):
    window = utils.Window(xstart=188, ystart=26, xend=24648, yend=1486)

    valid_data = load_geotiff('valid_01.slc.vrt')[0]

    golden_burst_bytes = golden_bytes[BURST0_START:BURST0_STOP]
    burst_array = extract_burst.burst_bytes_to_numpy(golden_burst_bytes, BURST_SHAPE)
    burst_array = extract_burst.invalid_to_nodata(burst_array, window)

    equal = np.isclose(valid_data, burst_array)
    assert np.all(equal)

# Golden
# @pytest.mark.skip()
def test_golden_by_burst(golden_tiff):
    safe_name = str(Path(ZIP_PATH).with_suffix(''))
    create_index.index_safe(safe_name)
    extract_burst.extract_burst(TEST_BURST_NAME)

    valid_data = load_geotiff(BURST_VALID_PATH)[0]
    test_data = load_geotiff(TEST_BURST_NAME)[0]

    equal = np.isclose(valid_data, test_data)
    assert np.all(equal)
