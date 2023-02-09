import gzip
import io
import tempfile
import zipfile
import zlib
from pathlib import Path

import indexed_gzip as igzip
import numpy as np
import pytest
from osgeo import gdal

from index_safe import create_index, extract_burst, utils

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

BURST_NUMBER = 7
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
def zinfo():
    with zipfile.ZipFile(ZIP_PATH, mode="r") as archive:
        info_list = archive.infolist()
        zinfo = [x for x in info_list if TIFF_PATH in x.filename][0]
    return zinfo


@pytest.fixture(scope='module')
def seek_point_array():
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
    # IW2 VV 3 golden
    # compressed = utils.Offset(start=2589238610, stop=2690292831)
    # uncompressed = utils.Offset(start=461226795, stop=614932715)

    # # IW2 VV 7
    compressed = utils.Offset(start=2986776876, stop=3087580208)
    uncompressed = utils.Offset(start=1076050475, stop=1229756395)

    tmp_file = tempfile.NamedTemporaryFile()
    indexer = utils.ZipIndexer(ZIP_PATH, TIFF_PATH)
    indexer.build_gzidx(tmp_file.name, [uncompressed.start], [uncompressed.stop], relative=True)

    assert indexer.index_offset.start == compressed.start
    assert indexer.index_offset.stop == compressed.stop
    points, compressed_size, *_ = utils.parse_gzidx(Path(tmp_file.name).read_bytes())

    with open(ZIP_PATH, 'rb') as f:
        f.seek(indexer.index_offset.start)
        body = bytes(10) + f.read(indexer.index_offset.stop - indexer.index_offset.start)

    with igzip.IndexedGzipFile(io.BytesIO(body)) as igzip_fobj:
        igzip_fobj.import_index(tmp_file.name)
        igzip_fobj.seek(uncompressed.start)
        test = igzip_fobj.read(uncompressed.stop - uncompressed.start)

    with zipfile.ZipFile(ZIP_PATH) as f:
        zinfo = [x for x in f.infolist() if TIFF_PATH in Path(x.filename).name][0]
        with f.open(zinfo.filename, 'r') as member:
            member.seek(uncompressed.start)
            golden = member.read(uncompressed.stop - uncompressed.start)

    assert golden == test


def test_whole_file_index():
    uncompressed = utils.Offset(start=461226795, stop=614932715)

    tmp_file = tempfile.NamedTemporaryFile()
    indexer = utils.ZipIndexer(ZIP_PATH, TIFF_PATH)
    indexer.build_gzidx(tmp_file.name, relative=False)

    with open(ZIP_PATH, 'rb') as f:
        body = f.read()

    with igzip.IndexedGzipFile(io.BytesIO(body)) as igzip_fobj:
        igzip_fobj.import_index(tmp_file.name)
        igzip_fobj.seek(uncompressed.start)
        test = igzip_fobj.read(uncompressed.stop - uncompressed.start)

    with zipfile.ZipFile(ZIP_PATH) as f:
        zinfo = [x for x in f.infolist() if TIFF_PATH in Path(x.filename).name][0]
        with f.open(zinfo.filename, 'r') as member:
            member.seek(uncompressed.start)
            golden = member.read(uncompressed.stop - uncompressed.start)

    assert golden == test


def test_zip_indexer():
    start = 10
    length = 15

    with zipfile.ZipFile(ZIP_PATH) as f:
        zinfo = [x for x in f.infolist() if TIFF_PATH in Path(x.filename).name][0]
        with f.open(zinfo.filename, 'r') as member:
            member.seek(start)
            golden = member.read(length)

    with tempfile.NamedTemporaryFile() as tmp:
        zip_indexer = utils.ZipIndexer(ZIP_PATH)
        zip_indexer.build_gzidx(TIFF_PATH, tmp.name)
        with igzip.IndexedGzipFile(ZIP_PATH, skip_crc_check=True) as f:
            f.import_index(tmp.name)
            f.seek(start)
            test = f.read(length)

    assert golden == test


def test_parse_gzidx(seek_point_array):
    zip_indexer = utils.ZipIndexer(ZIP_PATH)
    with open(GZIDX_PATH, 'rb') as f:
        test_array, n_points, _ = zip_indexer.parse_gzidx(f.read())
    assert np.all(test_array[:, [1, 0]] == seek_point_array)
    assert n_points == seek_point_array.shape[0]


# Swath level
def test_get_zip_compressed_offset(zinfo, golden_bytes):
    offset = utils.get_zip_compressed_offset(ZIP_PATH, zinfo)
    with open(ZIP_PATH, 'rb') as f:
        f.seek(offset.start)
        bytes_compressed = f.read(offset.stop - offset.start)
    test_bytes = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(bytes_compressed)

    assert len(test_bytes) == len(golden_bytes)
    assert test_bytes == golden_bytes


def test_wrap_deflate_as_gz(zinfo, golden_bytes):
    offset = utils.get_zip_compressed_offset(ZIP_PATH, zinfo)
    with open(ZIP_PATH, 'rb') as f:
        f.seek(offset.start)
        bytes_compressed = f.read(offset.stop - offset.start)

    gz_bytes = utils.wrap_deflate_as_gz(bytes_compressed, zinfo)
    test_bytes = gzip.decompress(gz_bytes)

    assert len(gz_bytes) == len(bytes_compressed) + 10 + 8  # wrap_as_gz should add 18 bytes of data
    assert len(test_bytes) == len(golden_bytes)
    assert test_bytes == golden_bytes


# Burst level
def test_get_burst_annotation_data(zinfo):
    burst_shape, burst_offsets, burst_windows = create_index.get_burst_annotation_data(ZIP_PATH, zinfo.filename)

    assert burst_shape == BURST_SHAPE
    assert burst_offsets[0].start == BURST_START
    assert burst_offsets[0].stop == BURST_STOP


def test_burst_bytes_to_numpy(golden_bytes):
    test_array = extract_burst.burst_bytes_to_numpy(golden_bytes[BURST_START : BURST_STOP + 1], (1510, 25448))

    golden_array = load_geotiff(BURST_RAW_PATH)[0]
    equal = np.isclose(golden_array, test_array)
    assert np.all(equal)


@pytest.mark.skip()
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


def test_get_closest_index():
    array = np.arange(0, 105, 5)
    value = 14
    less_than = utils.get_closest_index(array, value)
    assert less_than == 2
    greater_than = utils.get_closest_index(array, value, less_than=False)
    assert greater_than == 3


# Golden
@pytest.mark.skip()
def test_golden_by_burst(golden_tiff):
    safe_name = str(Path(ZIP_PATH).with_suffix(''))
    create_index.index_safe(safe_name, by_burst = True)
    extract_burst.extract_burst_by_burst(TEST_BURST_NAME, 'bursts.csv')

    valid_data = load_geotiff(BURST_VALID_PATH)[0]
    test_data = load_geotiff(TEST_BURST_NAME)[0]

    equal = np.isclose(valid_data, test_data)
    assert np.all(equal)


@pytest.mark.skip()
def test_golden_by_swath(golden_tiff):
    safe_name = str(Path(ZIP_PATH).with_suffix(''))
    create_index.index_safe(safe_name, by_burst=False)
    extract_burst.extract_burst_by_swath(TEST_BURST_NAME, 'bursts.csv')

    valid_data = load_geotiff(BURST_VALID_PATH)[0]
    test_data = load_geotiff(TEST_BURST_NAME)[0]

    equal = np.isclose(valid_data, test_data)
    assert np.all(equal)
