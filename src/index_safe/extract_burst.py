import io
import os
import struct
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import boto3
import fsspec
import indexed_gzip as igzip
import numpy as np
import pandas as pd
import requests
from osgeo import gdal

from . import utils

KB = 1024
MB = 1024 * KB

BUCKET = 'asf-ngap2w-p-s1-slc-7b420b89'


def extract_bytes_by_swath(url: str, metadata: utils.BurstMetadata, strategy: str = 's3') -> bytes:
    """Extract bytes pertaining to a burst from a Sentinel-1 SLC archive using a GZIDX that represents
    the entire swath. Index file must be in working directory.

    Args:
        url: url location of SLC archive
        metadata: metadata object for burst to extract
        strategy: strategy to use for download (s3 | http) s3 only works if runnning from us-west-2 region

    Returns:
        bytes representing a single burst
    """
    gzidx_name = '_'.join(metadata.name.split('_')[:-1]) + '.gzidx'

    if strategy == 's3':
        creds = utils.get_credentials()
        options = {'key': creds['accessKeyId'], 'secret': creds['secretAccessKey'], 'token': creds['sessionToken']}
        base_fs = fsspec.filesystem('s3', **options)
        url = f'{BUCKET}/{Path(url).name}'
    elif strategy == 'http':
        token = os.environ['EDL_TOKEN']
        options = {'block_size': 100 * MB, 'client_kwargs': {'headers': {'Authorization': f'Bearer {token}'}}}
        base_fs = fsspec.filesystem('https', **options)

    length = metadata.uncompressed_offset.stop - metadata.uncompressed_offset.start
    burst_bytes = bytearray(length)
    with base_fs.open(url, 'rb') as zip_fobj:
        with igzip.IndexedGzipFile(zip_fobj) as igzip_fobj:
            igzip_fobj.import_index(gzidx_name)
            igzip_fobj.seek(metadata.uncompressed_offset.start)
            igzip_fobj.readinto(burst_bytes)
    return burst_bytes


def extract_bytes_by_burst(url: str, metadata: utils.BurstMetadata, strategy: str = 's3') -> bytes:
    """Extract bytes pertaining to a burst from a Sentinel-1 SLC archive using a GZIDX that represents
    a single burst. Index file must be in working directory.

    Args:
        url: url location of SLC archive
        metadata: metadata object for burst to extract
        strategy: strategy to use for download (s3 | http) s3 only works if runnning from us-west-2 region

    Returns:
        bytes representing a single burst
    """
    gzidx_name = Path(metadata.name).with_suffix('.gzidx').name

    range_header = f'bytes={metadata.index_offset.start}-{metadata.index_offset.stop - 1}'
    if strategy == 's3':
        creds = utils.get_credentials()
        client = boto3.client(
            "s3",
            aws_access_key_id=creds["accessKeyId"],
            aws_secret_access_key=creds["secretAccessKey"],
            aws_session_token=creds["sessionToken"],
        )
        resp = client.get_object(Bucket=BUCKET, Key=Path(url).name, Range=range_header)
        body = bytes(10) + resp['Body'].read()
    elif strategy == 'http':
        client = requests.session()
        resp = client.get(url, headers={'Range': range_header})
        body = bytes(10) + resp.content

    with open(gzidx_name, 'rb') as f:
        f.seek(85)
        gzidx = f.read()
    assert gzidx[0:5] == b'GZIDX'

    length = metadata.uncompressed_offset.stop - metadata.uncompressed_offset.start
    burst_bytes = bytearray(length)
    with igzip.IndexedGzipFile(io.BytesIO(body)) as igzip_fobj:
        igzip_fobj.import_index(fileobj=io.BytesIO(gzidx))
        igzip_fobj.seek(metadata.uncompressed_offset.start)
        igzip_fobj.readinto(burst_bytes)
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
    with open(Path(burst_name).with_suffix('.gzidx').name, 'rb') as f:
        byte_data = f.read(85)
    assert byte_data[0:5] == b'BURST'

    slc_name = '_'.join(burst_name.split('_')[:-3])
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
    return burst_entry


def array_to_raster(out_path: str, array: np.ndarray, fmt: str = 'GTiff') -> str:
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
    out_dataset = driver.Create(out_path, n_cols, n_rows, 1, gdal.GDT_CFloat32)
    out_dataset.GetRasterBand(1).WriteArray(array)
    out_dataset = None
    return out_path


def extract_burst_by_swath(burst_name: str, df_file_name: str) -> str:
    """Extract burst from SLC in ASF archive using a swath-level index
    file and a burst metadata csv. Index must be in working directory.

    Args:
        burst_name: name of burst to extract
        df_file_name: path to csv file containing burst metadata

    Returns:
        path to saved burst raster
    """
    df = pd.read_csv(df_file_name)
    single_burst = df.loc[df.name == burst_name].squeeze()
    burst_metadata = row_to_burst_entry(single_burst)

    url = utils.get_download_url(single_burst['slc'])
    burst_bytes = extract_bytes_by_swath(url, burst_metadata)
    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_metadata.shape))
    burst_array = invalid_to_nodata(burst_array, burst_metadata.valid_window)
    out_name = array_to_raster(burst_name, burst_array)
    return out_name


def extract_burst_by_burst(burst_name: str) -> str:
    """Extract burst from SLC in ASF archive using a burst-level index
    file. Index must be in working directory.

    Args:
        burst_name: name of burst to extract

    Returns:
        path to saved burst raster
    """
    burst_metadata = bytes_to_burst_entry(burst_name)

    url = utils.get_download_url(burst_metadata.slc)
    burst_bytes = extract_bytes_by_burst(url, burst_metadata)
    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_metadata.shape))
    burst_array = invalid_to_nodata(burst_array, burst_metadata.valid_window)
    out_name = array_to_raster(burst_name, burst_array)
    return out_name


def main():
    """Example Command:

    extract_burst.py S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_IW2_VV_0.tiff
    """
    parser = ArgumentParser()
    parser.add_argument('burst')
    args = parser.parse_args()

    extract_burst_by_burst(args.burst)


if __name__ == '__main__':
    main()
