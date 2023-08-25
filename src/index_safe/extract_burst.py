import base64
import json
import os
import tempfile
import lxml.etree as ET
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

import boto3
import numpy as np
import requests
import zran
from osgeo import gdal

from index_safe.extract_metadata import extract_metadata_xml

try:
    from index_safe import utils
except ModuleNotFoundError:
    import utils


gdal.UseExceptions()

KB = 1024
MB = KB * KB


def extract_burst_data(
    url: str,
    metadata: utils.BurstMetadata,
    index: zran.Index,
    client: Union[boto3.client, requests.sessions.Session],
    range_get_func: Callable,
) -> bytes:
    """Extract bytes pertaining to a burst from a Sentinel-1 SLC archive using a zran index that represents
    a single burst.

    Args:
        url: url location of SLC archive
        metadata: metadata object for burst to extract
        index: zran index object for SLC archive
        client: boto3 S3 client or requests session
        range_get_func: function to use to get a range of bytes from SLC archive

    Returns:
        bytes of burst data
    """
    total_size = (metadata.index_offset.stop - 1) - metadata.index_offset.start
    range_headers = utils.calculate_range_parameters(total_size, metadata.index_offset.start, 20 * MB)
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(range_get_func, repeat(client), repeat(url), range_headers)
        body = b''.join(results)
    length = metadata.uncompressed_offset.stop - metadata.uncompressed_offset.start
    burst_bytes = zran.decompress(body, index, metadata.uncompressed_offset.start, length)
    return burst_bytes


def compute_valid_window(index: int, burst: ET.Element) -> utils.Window:
    """Written by Jason Ninneman for the ASF I&A team's burst extractor.
    Using the information contained within a SAFE annotation XML burst
    element, identify the window of a burst that contains valid data.

    Args:
        index: zero-indexed burst number to compute the valid window for
        burst: <burst> element of annotation XML for the given index

    Returns:
        Row and column ranges for the valid data within a burst
    """
    # all offsets, even invalid offsets
    offsets_range = utils.Offset(
        np.array([int(val) for val in burst.find('firstValidSample').text.split()]),
        np.array([int(val) for val in burst.find('lastValidSample').text.split()]),
    )

    # returns the indices of lines containing valid data
    lines_with_valid_data = np.flatnonzero(offsets_range.stop - offsets_range.start)

    # get first and last sample with valid data per line
    # x-axis, range
    valid_offsets_range = utils.Offset(
        offsets_range.start[lines_with_valid_data].min(),
        offsets_range.stop[lines_with_valid_data].max(),
    )

    # get the first and last line with valid data
    # y-axis, azimuth
    valid_offsets_azimuth = utils.Offset(
        lines_with_valid_data.min(),
        lines_with_valid_data.max(),
    )

    # x-length
    length_range = valid_offsets_range.stop - valid_offsets_range.start
    # y-length
    length_azimuth = len(lines_with_valid_data)

    valid_window = utils.Window(
        valid_offsets_range.start,
        valid_offsets_range.start + length_range,
        valid_offsets_azimuth.start,
        valid_offsets_azimuth.start + length_azimuth,
    )

    return valid_window


def get_gcps_from_xml(annotation_xml: ET.Element) -> Iterable[utils.GeoControlPoint]:
    """Get geolocation control points from annotation XML.

    Args:
        annotation_xml: root element of annotation XML

    Returns:
        List of geolocation control points
    """
    xml_gcps = annotation_xml.findall('.//{*}geolocationGridPoint')
    gcps = []
    for xml_gcp in xml_gcps:
        pixel = int(xml_gcp.findtext('.//{*}pixel'))
        line = int(xml_gcp.findtext('.//{*}line'))
        longitude = float(xml_gcp.findtext('.//{*}longitude'))
        latitude = float(xml_gcp.findtext('.//{*}latitude'))
        height = float(xml_gcp.findtext('.//{*}height'))
        gcps.append(utils.GeoControlPoint(pixel, line, longitude, latitude, height))
    return gcps


def format_gcps_for_burst(
    burst_number: int, burst_n_lines: int, swath_gcps: Iterable[utils.GeoControlPoint]
) -> Iterable[utils.GeoControlPoint]:
    """Format geolocation control points for a burst by making line numbers
    relative to the burst, and removing any GCPs that are outside of the burst.

    Args:
        burst_number: zero-indexed burst number
        burst_n_lines: number of lines in the burst
        swath_gcps: list of geolocation control points for the entire swath

    Returns:
        List of geolocation control points for a particular burst
    """
    gcps = []
    burst_starting_line = burst_number * burst_n_lines
    for gcp in swath_gcps:
        gcps.append(utils.GeoControlPoint(gcp.pixel, gcp.line - burst_starting_line, gcp.lon, gcp.lat, gcp.hgt))
    return gcps


def burst_bytes_to_numpy(burst_bytes: bytes, shape: Iterable[int]) -> np.ndarray:
    """Convert bytes representing a burst to numpy array.

    Args:
        burst_bytes: bytes of a burst
        shape: tuple representing shape of the burst array (n_rows, n_cols)

    Returns:
        burst array with a CFloat data type
    """
    array = np.frombuffer(burst_bytes, dtype=np.int16).astype(float)
    array.dtype = 'complex'
    array = array.reshape(shape).astype(np.csingle)
    return array


def invalid_to_nodata(array: np.ndarray, valid_window: utils.Window, nodata_value: int = 0) -> np.ndarray:
    """Use valid window information to set array values outside of valid window to nodata.

    Args:
        array: input burst array to modify
        valid_window: window that will not be set to nodata
        nodata: value used to represent nodata

    Returns
        modified burst array
    """
    is_not_valid = np.ones(array.shape).astype(bool)
    is_not_valid[valid_window.ystart : valid_window.yend, valid_window.xstart : valid_window.xend] = False
    array[is_not_valid] = nodata_value
    return array


def json_to_burst_metadata(burst_json_path: str) -> Tuple[zran.Index, utils.BurstMetadata]:
    """Convert burst metadata json file to a BurstMetadata object and zran index file.

    Args:
        burst_json_path: path to burst metadata json file

    Returns:
        zran index file and BurstMetadata object
    """
    with open(burst_json_path, 'r') as json_file:
        metadata_dict = json.load(json_file)

    shape = (metadata_dict['n_rows'], metadata_dict['n_columns'])
    index_offset = utils.Offset(metadata_dict['index_offset']['start'], metadata_dict['index_offset']['stop'])
    decompressed_offset = utils.Offset(
        metadata_dict['uncompressed_offset']['start'], metadata_dict['uncompressed_offset']['stop']
    )
    annotation_offset = utils.Offset(
        metadata_dict['annotation_offset']['start'], metadata_dict['annotation_offset']['stop']
    )
    manifest_offset = utils.Offset(metadata_dict['manifest_offset']['start'], metadata_dict['manifest_offset']['stop'])
    burst_metadata = utils.BurstMetadata(
        metadata_dict['name'],
        metadata_dict['slc'],
        metadata_dict['swath'],
        metadata_dict['burst_index'],
        shape,
        index_offset,
        decompressed_offset,
        annotation_offset,
        manifest_offset,
    )
    decoded_bytes = base64.b64decode(metadata_dict['dflidx_b64'])
    index = zran.Index.parse_index_file(decoded_bytes)
    return index, burst_metadata


def array_to_raster(
    out_path: Path, array: np.ndarray, gcps: Iterable[utils.GeoControlPoint], fmt: str = 'GTiff'
) -> str:
    """Save a burst array as gdal raster.

    Args:
        out_path: path to save file to
        array: array to save as raster
        gcps: list of gcps to use for georeferencing
        fmt: file format to use

    Returns:
        path to saved raster
    """
    n_rows, n_cols = array.shape
    epsg_4326 = gdal.osr.SpatialReference()
    epsg_4326.ImportFromEPSG(4326)

    driver = gdal.GetDriverByName(fmt)
    out_dataset = driver.Create(str(out_path), n_cols, n_rows, 1, gdal.GDT_CInt16)

    out_dataset.GetRasterBand(1).WriteArray(array)

    out_dataset.SetProjection(epsg_4326.ExportToWkt())

    band = out_dataset.GetRasterBand(1)
    band.SetNoDataValue(0)

    gdal_gcps = [gdal.GCP(point.lon, point.lat, point.hgt, point.pixel, point.line) for point in gcps]
    out_dataset.SetGCPs(gdal_gcps, epsg_4326.ExportToWkt())

    out_dataset = None
    return out_path


def extract_burst(
    burst_index_path: str, edl_token: str = None, strategy: str = 's3', working_dir: Path = Path('.')
) -> str:
    """Extract burst from SLC in ASF archive using a burst-level index
    file. Index must be available locally.

    Args:
        burst_index_path: path to burst index file on disk
        edl_token: EDL token to use for downloading SLC
        strategy: download strategy to use ('s3' | 'https')
        working_dir: directory to use for extract burst to

    Returns:
        path to saved burst raster
    """
    index, burst_metadata = json_to_burst_metadata(burst_index_path)
    url = utils.get_download_url(burst_metadata.slc)

    client, range_get_func = utils.setup_download_client(strategy=strategy)
    burst_bytes = extract_burst_data(url, burst_metadata, index, client, range_get_func)
    annotation_xml = extract_metadata_xml(url, burst_metadata.annotation_offset, client, range_get_func)

    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_metadata.shape))
    valid_window = compute_valid_window(
        burst_metadata.burst_index, annotation_xml.findall('.//{*}burst')[burst_metadata.burst_index]
    )
    burst_array = invalid_to_nodata(burst_array, valid_window)
    swath_gcps = get_gcps_from_xml(annotation_xml)
    gcps = format_gcps_for_burst(burst_metadata.burst_index, burst_metadata.shape[0], swath_gcps)
    out_path = array_to_raster(working_dir / burst_metadata.name, burst_array, gcps)
    return out_path


def lambda_handler(event, context):
    # TODO need to test with new interface
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)
    print('## PROCESS BEGIN...')
    s3 = boto3.client('s3')
    index_bucket_name = os.environ.get('IndexBucketName')
    extract_bucket_name = os.environ.get('ExtractBucketName')
    burst_json_name = Path(event['burst']).with_suffix('.json')
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        utils.lambda_get_credentials(event['edl_token'], tmpdir, s3, extract_bucket_name, 'credentials.json')
        s3.download_file(index_bucket_name, str(burst_json_name), str(tmpdir / burst_json_name))
        tmp_path = extract_burst(burst_json_name, event['edl_token'], working_dir=tmpdir)
        s3.upload_file(str(tmp_path), extract_bucket_name, tmp_path.name)
    print('## PROCESS COMPLETE!')


def main():
    """Example Command:

    extract_burst.py S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_IW2_VV_0.json
    """
    parser = ArgumentParser()
    parser.add_argument('index_path')
    args = parser.parse_args()

    extract_burst(args.index_path)


if __name__ == '__main__':
    main()
