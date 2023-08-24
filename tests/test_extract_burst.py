import base64
import json

import numpy as np
from osgeo import gdal
from pathlib import Path
from unittest.mock import patch

from index_safe import extract_burst, utils


def test_compute_valid_window(annotation_xml):
    golden_window = utils.Window(xstart=10, xend=100, ystart=2, yend=7)
    burst_xmls = annotation_xml.findall('.//{*}burst')
    assert golden_window == extract_burst.compute_valid_window(0, burst_xmls[0])
    assert golden_window == extract_burst.compute_valid_window(1, burst_xmls[1])


def test_get_gcps_from_xml(annotation_xml):
    test_gcps = extract_burst.get_gcps_from_xml(annotation_xml)
    assert len(test_gcps) == 2

    test_gcp = test_gcps[0]
    assert isinstance(test_gcp.pixel, int)
    assert isinstance(test_gcp.lat, float)
    assert test_gcp == utils.GeoControlPoint(0, 0, 50, 20, 1000)


def test_format_gcps_for_burst(annotation_xml):
    burst_index = 1
    burst_length = 10
    test_gcps = extract_burst.get_gcps_from_xml(annotation_xml)
    reformatted_gcps = extract_burst.format_gcps_for_burst(burst_index, burst_length, test_gcps)

    assert reformatted_gcps[0].line == test_gcps[0].line - (burst_index * burst_length)
    assert reformatted_gcps[1].line == test_gcps[1].line - (burst_index * burst_length)


def test_burst_bytes_to_numpy():
    input_bytes = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int16).tobytes()
    test_array = extract_burst.burst_bytes_to_numpy(input_bytes, shape=(2, 2))

    golden_array = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.csingle)
    assert np.allclose(test_array, golden_array)


def test_invalid_to_nodata():
    input_array = np.ones((4, 4))
    valid_window = utils.Window(xstart=1, xend=3, ystart=1, yend=3)
    nodata_value = 0
    test_array = extract_burst.invalid_to_nodata(input_array, valid_window, nodata_value)
    golden_array = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    assert np.allclose(test_array, golden_array)


def test_json_to_burst_metadata(tmpdir):
    json_path = Path(tmpdir) / 'test.json'
    sample_json = {
        'name': 'test_name',
        'slc': 'test_slc',
        'swath': 'test_swath',
        'burst_index': 0,
        'n_rows': 1,
        'n_columns': 2,
        'index_offset': {'start': 3, 'stop': 4},
        'uncompressed_offset': {'start': 5, 'stop': 6},
        'annotation_offset': {'start': 7, 'stop': 8},
        'manifest_offset': {'start': 9, 'stop': 10},
        'dflidx_b64': base64.b64encode(b'0').decode('utf-8'),
    }
    json_path.write_text(json.dumps(sample_json))

    with patch('zran.Index.parse_index_file', return_value=None):
        index, burst_metadata = extract_burst.json_to_burst_metadata(str(json_path))

    golden_metadata = utils.BurstMetadata(
        'test_name',
        'test_slc',
        'test_swath',
        0,
        (1, 2),
        utils.Offset(3, 4),
        utils.Offset(5, 6),
        utils.Offset(7, 8),
        utils.Offset(9, 10),
    )
    assert burst_metadata == golden_metadata


def test_array_to_raster(tmpdir):
    raster_path = Path(tmpdir) / 'test.tiff'
    array = np.ones((3, 3))
    gcps = [utils.GeoControlPoint(0, 1, 2, 3, 4), utils.GeoControlPoint(5, 6, 7, 8, 9)]
    extract_burst.array_to_raster(raster_path, array, gcps)

    ds = gdal.Open(str(raster_path))
    assert ds.RasterXSize == 3
    assert ds.RasterYSize == 3

    epsg_4326 = gdal.osr.SpatialReference()
    epsg_4326.ImportFromEPSG(4326)
    assert ds.GetGCPProjection() == epsg_4326.ExportToWkt()

    gcps = ds.GetGCPs()
    assert len(gcps) == 2
    assert gcps[0].GCPPixel == 0
    assert gcps[1].GCPPixel == 5
