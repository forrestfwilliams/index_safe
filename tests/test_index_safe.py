import gzip
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from index_safe import extract_burst, index_safe, utils

BURST_LENGTH = 153814955 - 109035 - 1
BURST_SHAPE = (1510, 25448)

BURST0_START = 109035
BURST0_STOP = BURST0_START + BURST_LENGTH
BURST7_START = 1076050475
BURST7_STOP = BURST7_START + BURST_LENGTH

BURST_START = BURST0_START
BURST_STOP = BURST0_STOP

ZIP_PATH = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
TIFF_PATH = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff'

BURST_RAW_PATH = 'raw_01.slc.vrt'
BURST_VALID_PATH = 'valid_01.slc.vrt'
TEST_BURST_NAME = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_IW2_VV_0.tiff'


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


@pytest.fixture
def zinfo():
    with zipfile.ZipFile(ZIP_PATH, mode="r") as archive:
        info_list = archive.infolist()
        zinfo = [x for x in info_list if TIFF_PATH in x.filename][0]
    return zinfo


# Swath level
def test_compressed_offset(zinfo, golden_bytes):
    zinfo = utils.OffsetZipInfo(ZIP_PATH, zinfo)

    with open(ZIP_PATH, 'rb') as f:
        f.seek(zinfo.compressed_offset.start)
        test_bytes_compressed = f.read(zinfo.compressed_offset.stop)
    test_bytes = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(test_bytes_compressed)

    assert len(test_bytes) == len(golden_bytes)
    assert test_bytes == golden_bytes


def test_wrap_as_gz(zinfo, golden_bytes):
    zinfo = utils.OffsetZipInfo(ZIP_PATH, zinfo)

    with open(ZIP_PATH, 'rb') as f:
        f.seek(zinfo.compressed_offset.start)
        bytes_compressed = f.read(zinfo.compressed_offset.stop - zinfo.compressed_offset.start)

    gz_bytes = index_safe.wrap_as_gz(bytes_compressed, zinfo)
    test_bytes = gzip.decompress(gz_bytes)

    assert len(gz_bytes) == len(bytes_compressed) + 10 + 8  # wrap_as_gz should add 18 bytes of data
    assert len(test_bytes) == len(golden_bytes)
    assert test_bytes == golden_bytes


# def test_get_index(golden_bytes, zinfo):
#     zinfo = utils.OffsetZipInfo(ZIP_PATH, zinfo)
#     header_length = 10
#
#     with open(ZIP_PATH, 'rb') as f:
#         f.seek(zinfo.compressed_offset.start)
#         bytes_compressed = f.read(zinfo.compressed_offset.stop - zinfo.compressed_offset.start)
#
#     import pandas as pd
#     index = pd.read_csv('seek_points.csv').to_numpy()
#
#     index1 = 0
#     index2 = 10
#     start =  index[index1,1] - header_length
#     stop =  index[index2,1] - header_length + 1
#     body = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(bytes_compressed[start:stop])
#     assert True


# Burst level
def test_get_burst_annotation_data(zinfo):
    burst_shape, burst_offsets, burst_windows = index_safe.get_burst_annotation_data(ZIP_PATH, zinfo.filename)

    assert burst_shape == BURST_SHAPE
    assert burst_offsets[0].start == BURST_START
    assert burst_offsets[0].stop == BURST_STOP


def test_burst_bytes_to_numpy(golden_bytes):
    test_array = extract_burst.burst_bytes_to_numpy(golden_bytes[BURST_START : BURST_STOP + 1], (1510, 25448))

    golden_array = load_geotiff(BURST_RAW_PATH)[0]
    equal = np.isclose(golden_array, test_array)
    assert np.all(equal)


def test_extract_bytes_http(golden_bytes):
    url = (
        'https://sentinel1.asf.alaska.edu/SLC/SA/'
        'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    )
    metadata = utils.BurstMetadata(
        name='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_IW2_VV_0.tiff',
        slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
        shape=(1510, 25448),
        compressed_offset=utils.Offset(start=2289811182, stop=2391345202),
        decompressed_offset=utils.Offset(start=109035, stop=153814954),
        valid_window=utils.Window(xstart=188, ystart=26, xend=24648, yend=1486),
    )
    test_bytes = extract_burst.extract_bytes_http(url, metadata)
    golden_burst_bytes = golden_bytes[BURST_START : BURST_STOP + 1]
    assert test_bytes == golden_burst_bytes


def test_invalid_to_nodata(golden_bytes):
    window = utils.Window(xstart=188, ystart=26, xend=24648, yend=1486)

    valid_data = load_geotiff(BURST_VALID_PATH)[0]

    golden_burst_bytes = golden_bytes[BURST_START : BURST_STOP + 1]
    burst_array = extract_burst.burst_bytes_to_numpy(golden_burst_bytes, BURST_SHAPE)
    burst_array = extract_burst.invalid_to_nodata(burst_array, window)

    equal = np.isclose(valid_data, burst_array)
    assert np.all(equal)


# Golden
def test_golden():
    safe_name = str(Path(ZIP_PATH).with_suffix(''))
    index_safe.index_safe(safe_name)
    extract_burst.extract_burst_http(TEST_BURST_NAME, 'bursts.csv')

    valid_data = load_geotiff(BURST_VALID_PATH)[0]
    test_data = load_geotiff(TEST_BURST_NAME)[0]

    equal = np.isclose(valid_data, test_data)
    assert np.all(equal)
