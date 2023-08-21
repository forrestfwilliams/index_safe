import gzip
import shutil
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import requests
import zran
from osgeo import gdal

from index_safe import create_index, extract_burst, utils


gdal.UseExceptions()

BURST_LENGTH = 153814955 - 109035
BURST_SHAPE = (1510, 25448)

BURST0_START = 109035
BURST0_STOP = BURST0_START + BURST_LENGTH
BURST7_START = 1076050475
BURST7_STOP = BURST7_START + BURST_LENGTH

SCRIPT_DIR = Path(__file__).parent.absolute()
ZIP_PATH = str(SCRIPT_DIR / 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip')
ZIP_NAME = Path(ZIP_PATH).name
TIFF_PATH = str(SCRIPT_DIR / 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff')
TIFF_NAME = Path(TIFF_PATH).name

BURST_NUMBER = 7
BURST_START = BURST7_START
BURST_STOP = BURST7_STOP
BURST_RAW_PATH = str(SCRIPT_DIR / f'raw_0{BURST_NUMBER + 1}.slc.vrt')
BURST_VALID_PATH = str(SCRIPT_DIR / f'valid_0{BURST_NUMBER + 1}.slc.vrt')
TEST_BURST_NAME = str(
    SCRIPT_DIR / f'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_IW2_VV_{BURST_NUMBER}.tiff'
)

GZIDX_PATH = str(SCRIPT_DIR / 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gzidx')
GZ_PATH = str(SCRIPT_DIR / 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gz')


def load_geotiff(infile, band=1):
    ds = gdal.Open(infile, gdal.GA_ReadOnly)

    data = ds.GetRasterBand(band).ReadAsArray()
    nodata = ds.GetRasterBand(band).GetNoDataValue()
    projection = ds.GetProjection()
    transform = ds.GetGeoTransform()
    ds = None
    return data, transform, projection, nodata


@pytest.fixture(scope='module')
def golden_zip():
    if Path(ZIP_PATH).exists():
        return ZIP_PATH

    print('Downloading test SAFE archive')
    url = f'https://sentinel1.asf.alaska.edu/SLC/SA/{ZIP_NAME}'
    session = requests.Session()
    with session.get(url, stream=True) as s:
        s.raise_for_status()
        with open(ZIP_PATH, 'wb') as f:
            for chunk in s.iter_content(chunk_size=None):
                if chunk:
                    f.write(chunk)
        session.close()

    return ZIP_PATH


@pytest.fixture(scope='module')
def golden_tiff(golden_zip):
    if Path(TIFF_PATH).exists():
        return TIFF_PATH

    with zipfile.ZipFile(golden_zip, mode='r') as archive:
        info_list = archive.infolist()
        zinfo = [x for x in info_list if TIFF_NAME in x.filename][0]
        with archive.open(zinfo.filename, 'r') as f:
            body = f.read()

    with open(TIFF_PATH, 'wb') as f:
        f.write(body)

    return TIFF_PATH


@pytest.fixture(scope='module')
def golden_bytes(golden_tiff):
    with open(golden_tiff, 'rb') as f:
        golden_bytes = f.read()
    return golden_bytes


@pytest.fixture(scope='module')
def golden_gz(golden_tiff):
    if Path(GZ_PATH).exists():
        return GZ_PATH

    with open(golden_tiff, 'rb') as f_in:
        with gzip.open(GZ_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return GZ_PATH


@pytest.fixture(scope='module')
def zinfo():
    with zipfile.ZipFile(ZIP_PATH, mode="r") as archive:
        info_list = archive.infolist()
        zinfo = [x for x in info_list if TIFF_NAME in x.filename][0]
    return zinfo


def test_burst_specific_index(golden_zip):
    # # IW2 VV 3 golden
    # compressed = utils.Offset(start=2589238610, stop=2690292830)
    # uncompressed = utils.Offset(start=461226795, stop=614932715)

    # IW2 VV 7
    compressed = utils.Offset(start=2987161209, stop=3087580208)
    uncompressed = utils.Offset(start=1076050475, stop=1229756395)

    indexer = utils.ZipIndexer(golden_zip, TIFF_NAME)
    indexer.create_full_dflidx()
    compressed_offset, uncompressed_offset, index = indexer.subset_dflidx(
        locations=[uncompressed.start], end_location=uncompressed.stop
    )

    assert compressed_offset.start == compressed.start
    assert compressed_offset.stop == compressed.stop

    with open(golden_zip, 'rb') as f:
        f.seek(compressed_offset.start)
        body = f.read(compressed_offset.stop - compressed_offset.start)

    length = uncompressed.stop - uncompressed.start
    start = uncompressed.start - uncompressed_offset.start
    test = zran.decompress(body, index, start, length)

    with zipfile.ZipFile(golden_zip) as f:
        zinfo = [x for x in f.infolist() if TIFF_NAME in Path(x.filename).name][0]
        with f.open(zinfo.filename, 'r') as member:
            member.seek(uncompressed.start)
            golden = member.read(uncompressed.stop - uncompressed.start)

    assert golden == test


# Swath level
def test_get_zip_compressed_offset(golden_zip, zinfo, golden_bytes):
    offset = utils.get_zip_compressed_offset(golden_zip, zinfo)
    with open(golden_zip, 'rb') as f:
        f.seek(offset.start)
        bytes_compressed = f.read(offset.stop - offset.start)
    test_bytes = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(bytes_compressed)

    assert len(test_bytes) == len(golden_bytes)
    assert test_bytes == golden_bytes


# Burst level
def test_get_burst_annotation_data(golden_zip, zinfo):
    burst_shape, burst_offsets = create_index.get_burst_annotation_data(golden_zip, zinfo.filename)

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

    # FIXME: Hardcoded to work with first burst
    valid_data = load_geotiff(str(SCRIPT_DIR / 'valid_01.slc.vrt'))[0]

    golden_burst_bytes = golden_bytes[BURST0_START:BURST0_STOP]
    burst_array = extract_burst.burst_bytes_to_numpy(golden_burst_bytes, BURST_SHAPE)
    burst_array = extract_burst.invalid_to_nodata(burst_array, window)

    equal = np.isclose(valid_data, burst_array)
    assert np.all(equal)


# Golden, must be run from tests directory
@pytest.mark.skip(reason='Integration testing')
def test_golden_by_burst(golden_tiff):
    safe_name = str(Path(ZIP_NAME).with_suffix(''))
    create_index.index_safe(safe_name)
    index_path = str(Path(TEST_BURST_NAME).with_suffix('.json'))
    extract_burst.extract_burst(index_path)

    valid_data = load_geotiff(BURST_VALID_PATH)[0]
    test_data = load_geotiff(TEST_BURST_NAME)[0]

    equal = np.isclose(valid_data, test_data)
    assert np.all(equal)
