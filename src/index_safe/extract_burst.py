import os
import struct
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import boto3
import numpy as np
import pandas as pd
import requests
# FIXME
# from . import utils
import utils
import zran
from osgeo import gdal

KB = 1024
MB = 1024 * KB

BUCKET = 'asf-ngap2w-p-s1-slc-7b420b89'


def extract_bytes_by_burst(
    url: str,
    metadata: utils.BurstMetadata,
    index: zran.Index,
    edl_token: str = None,
    working_dir: Path = Path('.'),
    strategy: str = 's3',
) -> bytes:
    """Extract bytes pertaining to a burst from a Sentinel-1 SLC archive using a GZIDX that represents
    a single burst. Index file must be in working directory.

    Args:
        url: url location of SLC archive
        metadata: metadata object for burst to extract
        strategy: strategy to use for download (s3 | http) s3 only works if runnning from us-west-2 region

    Returns:
        bytes representing a single burst
    """
    range_header = f'bytes={metadata.index_offset.start}-{metadata.index_offset.stop - 1}'
    if strategy == 's3':
        creds = utils.get_credentials(edl_token, working_dir)
        client = boto3.client(
            "s3",
            aws_access_key_id=creds["accessKeyId"],
            aws_secret_access_key=creds["secretAccessKey"],
            aws_session_token=creds["sessionToken"],
        )
        resp = client.get_object(Bucket=BUCKET, Key=Path(url).name, Range=range_header)
        body = resp['Body'].read()
    elif strategy == 'http':
        client = requests.session()
        resp = client.get(url, headers={'Range': range_header})
        body = resp.content

    length = metadata.uncompressed_offset.stop - metadata.uncompressed_offset.start
    burst_bytes = zran.decompress(body, index, metadata.uncompressed_offset.start, length)
    return burst_bytes


def burst_bytes_to_numpy(burst_bytes: bytes, shape: Iterable[int]) -> np.ndarray:
    """Convert bytes representing a burst to numpy array.

    Args:
        burst_bytes: bytes of a burst
        shape: tuple representing shape of the burst array (n_rows, n_cols)

    Returns:
        burst array with a CFloat data type
    """
    tmp_array = np.frombuffer(burst_bytes, dtype=np.int16).astype(float)
    array = tmp_array.copy()
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


def row_to_burst_entry(row: pd.Series) -> utils.BurstMetadata:
    """Convert row of burst metadata dataframe to a burst metadata
    object.

    Args:
        row: row of dataframe to convert

    Returns:
        burst metadata object
    """
    shape = (row['n_rows'], row['n_columns'])
    index_offset = utils.Offset(row['index_start'], row['index_stop'])
    decompressed_offset = utils.Offset(row['offset_start'], row['offset_stop'])
    window = utils.Window(row['valid_x_start'], row['valid_x_stop'], row['valid_y_start'], row['valid_y_stop'])

    burst_entry = utils.BurstMetadata(row['name'], row['slc'], shape, index_offset, decompressed_offset, window)
    return burst_entry


def bytes_to_burst_entry(burst_name: str):
    """Convert header bytes of burst-specifc
    index file to a burst metadata object.
    Index file must be in working directory.

    Args:
        burst_name: name of burst to get info for

    Returns:
        burst metadata object
    """
    with open(str(burst_name.with_suffix('.bstidx')), 'rb') as f:
        byte_data = f.read(85)
        index_data = f.read()

    index = zran.Index.parse_index_file(index_data)

    assert byte_data[0:5] == b'BURST'

    slc_name = '_'.join(burst_name.name.split('_')[:-3])
    data = struct.unpack('<QQQQQQQQQQ', byte_data[5:85])

    shape_y = data[0]
    shape_x = data[1]
    index_offset_start = data[2]
    index_offset_stop = data[3]
    data_offset_start = data[4]
    data_offset_stop = data[5]
    valid_xstart = data[6]
    valid_xend = data[7]
    valid_ystart = data[8]
    valid_yend = data[9]

    shape = (shape_y, shape_x)
    index_offset = utils.Offset(index_offset_start, index_offset_stop)
    decompressed_offset = utils.Offset(data_offset_start, data_offset_stop)
    window = utils.Window(valid_xstart, valid_xend, valid_ystart, valid_yend)

    burst_entry = utils.BurstMetadata(burst_name, slc_name, shape, index_offset, decompressed_offset, window)
    return index, burst_entry


def array_to_raster(out_path: Path, array: np.ndarray, fmt: str = 'GTiff') -> str:
    """Save a burst array as gdal raster.

    Args:
        out_path: path to save file to
        array: array to save as raster
        fmt: file format to use

    Returns:
        path to saved raster
    """
    driver = gdal.GetDriverByName(fmt)
    n_rows, n_cols = array.shape
    out_dataset = driver.Create(str(out_path), n_cols, n_rows, 1, gdal.GDT_CFloat32)
    out_dataset.GetRasterBand(1).WriteArray(array)
    out_dataset = None
    return out_path


def extract_burst(burst_name: str, edl_token: str = None, working_dir: Path = Path('.')) -> str:
    """Extract burst from SLC in ASF archive using a burst-level index
    file. Index must be in working directory.

    Args:
        burst_name: name of burst to extract

    Returns:
        path to saved burst raster
    """
    burst_path = working_dir / burst_name
    index, burst_metadata = bytes_to_burst_entry(burst_path)
    url = utils.get_download_url(burst_metadata.slc)
    burst_bytes = extract_bytes_by_burst(url, burst_metadata, index, edl_token, working_dir)
    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_metadata.shape))
    burst_array = invalid_to_nodata(burst_array, burst_metadata.valid_window)
    out_path = array_to_raster(burst_path, burst_array)
    return out_path


def lambda_handler(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)
    print('## PROCESS BEGIN...')
    s3 = boto3.client('s3')
    index_bucket_name = os.environ.get('IndexBucketName')
    extract_bucket_name = os.environ.get('ExtractBucketName')
    bstidx_name = Path(event['burst']).with_suffix('.bstidx')
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        s3.download_file(index_bucket_name, str(bstidx_name), str(tmpdir / bstidx_name))
        tmp_path = extract_burst(event['burst'], event['edl_token'], working_dir=tmpdir)
        s3.upload_file(str(tmp_path), extract_bucket_name, tmp_path.name)
    print('## PROCESS COMPLETE!')


def main():
    """Example Command:

    extract_burst.py S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_IW2_VV_0.tiff
    """
    parser = ArgumentParser()
    parser.add_argument('burst')
    args = parser.parse_args()

    extract_burst(args.burst)


if __name__ == '__main__':
    main()
