from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import boto3
import botocore
import fsspec
import indexed_gzip as igzip
import numpy as np
import pandas as pd
import requests
from osgeo import gdal

from . import utils

try:
    from isal import isal_zlib as zlib
except ImportError:
    import zlib


KB = 1024
MB = 1024 * KB
GB = 1024 * MB
MAX_WBITS = 15


def s3_download(client: botocore.client, bucket: str, key: str, range_header: str) -> bytes:
    resp = client.get_object(Bucket=bucket, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def extract_bytes_s3(client: botocore.client, bucket: str, key: str, metadata: utils.BurstMetadata) -> bytes:
    """ """
    range_header = f'bytes={metadata.compressed_offset.start}-{metadata.compressed_offset.stop}'
    body = s3_download(client, bucket, key, range_header)

    body = zlib.decompressobj(-1 * MAX_WBITS).decompress(body)
    # utils.Offset properties are end inclusive, unlike python indexes
    burst_bytes = body[metadata.decompressed_offset.start : metadata.decompressed_offset.stop + 1]

    return burst_bytes


def http_download(url: str, range_header: str) -> bytes:
    with requests.Session() as s:
        resp = s.get(url, headers={'Range': range_header})
        resp.raise_for_status()
        body = resp.content
    return body


def extract_bytes_http(url: str, metadata: utils.BurstMetadata) -> bytes:
    range_header = f'bytes={metadata.compressed_offset.start}-{metadata.compressed_offset.stop}'
    body = http_download(url, range_header)

    body = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(body)
    # utils.Offset properties are end inclusive, unlike python indexes so need to + 1
    burst_bytes = body[metadata.decompressed_offset.start : metadata.decompressed_offset.stop + 1]

    return burst_bytes


def extract_bytes_fsspec(url: str, metadata: utils.BurstMetadata) -> bytes:
    # name = f'ffwilliams2-shenanigans/bursts/{Path(url).name}'
    # base_fs = fsspec.filesystem('s3')

    name = f'https://ffwilliams2-shenanigans.s3.us-west-2.amazonaws.com/bursts/{Path(url).name}'
    base_fs = fsspec.filesystem('https', block_size=20 * MB)

    gzidx_name = Path(metadata.base_tiff).with_suffix('.gzidx').name

    length = metadata.uncompressed_offset.stop - metadata.uncompressed_offset.start
    burst_bytes = bytearray(length)
    with base_fs.open(name, 'rb') as zip_fobj:
        with igzip.IndexedGzipFile(zip_fobj) as igzip_fobj:
            igzip_fobj.import_index(gzidx_name)
            igzip_fobj.seek(metadata.uncompressed_offset.start)
            igzip_fobj.readinto(burst_bytes)
    return burst_bytes


def burst_bytes_to_numpy(burst_bytes: bytes, shape: Iterable[int]) -> np.ndarray:
    tmp_array = np.frombuffer(burst_bytes, dtype=np.int16).astype(float)
    array = tmp_array.copy()
    array.dtype = 'complex'
    array = array.reshape(shape).astype(np.csingle)
    return array


def invalid_to_nodata(array: np.ndarray, valid_window: utils.Window, nodata_value: int = 0) -> np.ndarray:
    is_not_valid = np.ones(array.shape).astype(bool)
    is_not_valid[valid_window.ystart : valid_window.yend, valid_window.xstart : valid_window.xend] = False
    array[is_not_valid] = nodata_value
    return array


def row_to_burst_entry(row: pd.Series) -> utils.BurstMetadata:
    shape = (row['n_rows'], row['n_columns'])
    decompressed_offset = utils.Offset(row['offset_start'], row['offset_stop'])

    window = utils.Window(row['valid_x_start'], row['valid_y_start'], row['valid_x_stop'], row['valid_y_stop'])

    burst_entry = utils.BurstMetadata(row['name'], row['base_tiff'], row['slc'], shape, decompressed_offset, window)
    return burst_entry


def array_to_raster(out_path: str, array: np.ndarray, fmt: str = 'GTiff') -> str:
    driver = gdal.GetDriverByName(fmt)
    n_rows, n_cols = array.shape
    out_dataset = driver.Create(out_path, n_cols, n_rows, 1, gdal.GDT_CFloat32)
    out_dataset.GetRasterBand(1).WriteArray(array)
    out_dataset = None
    return out_path


def extract_burst_s3(burst_name: str, df_file_name: str) -> str:
    bucket = 'ffwilliams2-shenanigans'
    key = 'bursts/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    df = pd.read_csv(df_file_name)
    single_burst = df.loc[df.name == burst_name].squeeze()
    burst_metadata = row_to_burst_entry(single_burst)

    client = boto3.client('s3')
    burst_bytes = extract_bytes_s3(client, bucket, key, burst_metadata)
    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_metadata.shape))
    burst_array = invalid_to_nodata(burst_array, burst_metadata.valid_window)
    out_name = array_to_raster(burst_name, burst_array)
    return out_name


def extract_burst_http(burst_name: str, df_file_name: str) -> str:
    df = pd.read_csv(df_file_name)
    single_burst = df.loc[df.name == burst_name].squeeze()
    burst_metadata = row_to_burst_entry(single_burst)

    url = utils.get_download_url(single_burst['slc'])
    burst_bytes = extract_bytes_http(url, burst_metadata)
    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_metadata.shape))
    burst_array = invalid_to_nodata(burst_array, burst_metadata.valid_window)
    out_name = array_to_raster(burst_name, burst_array)
    return out_name


def extract_burst_fsspec(burst_name: str, df_file_name: str) -> str:
    df = pd.read_csv(df_file_name)
    single_burst = df.loc[df.name == burst_name].squeeze()
    burst_metadata = row_to_burst_entry(single_burst)

    url = utils.get_download_url(single_burst['slc'])
    burst_bytes = extract_bytes_fsspec(url, burst_metadata)
    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_metadata.shape))
    burst_array = invalid_to_nodata(burst_array, burst_metadata.valid_window)
    out_name = array_to_raster(burst_name, burst_array)
    return out_name


def main():
    """Example Command:

    extract_burst.py \
        S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_IW2_VV_0.tiff \
        bursts.csv
    """
    parser = ArgumentParser()
    parser.add_argument('burst')
    parser.add_argument('df')
    args = parser.parse_args()

    extract_burst_fsspec(args.burst, args.df)


if __name__ == '__main__':
    main()
