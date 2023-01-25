import math
import zlib
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import boto3
import numpy as np
import pandas as pd
from isal import isal_zlib
from osgeo import gdal

import data

KB = 1024
MB = KB * KB


def calculate_range_parameters(download_start, download_stop, chunk_size=10 * MB):
    total_size = download_stop - download_start
    num_parts = int(math.ceil(total_size / float(chunk_size)))
    range_params = []
    for part_index in range(num_parts):
        start_range = (part_index * chunk_size) + download_start
        if part_index == num_parts - 1:
            end_range = str(total_size + download_start)
        else:
            end_range = start_range + chunk_size - 1

        range_params.append(f'bytes={start_range}-{end_range}')
    return range_params


def s3_download_multithread(client, bucket, key, metadata):
    range_params = calculate_range_parameters(metadata.compressed_offset.start, metadata.compressed_offset.stop)

    # Dispatch work tasks with our client
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(s3_download, repeat(client), repeat(bucket), repeat(key), range_params)

    content = b''.join(results)
    return content


def s3_download(client, bucket, key, range_header):
    resp = client.get_object(Bucket=bucket, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def extract_bytes(client, bucket, key, metadata, multithread=False):
    if multithread:
        body = s3_download_multithread(client, bucket, key, metadata)
    else:
        range_header = f'bytes={metadata.compressed_offset.start}-{metadata.compressed_offset.stop}'
        body = s3_download(client, bucket, key, range_header)

    body = isal_zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(body)
    burst_bytes = body[metadata.decompressed_offset.start : metadata.decompressed_offset.stop]

    return burst_bytes


def burst_bytes_to_numpy(burst_bytes, shape):
    tmp_array = np.frombuffer(burst_bytes, dtype=np.int16).astype(float)
    array = tmp_array.copy()
    array.dtype = 'complex'
    array = array.reshape(shape).astype(np.csingle)
    return array


def invalid_to_nodata(array: np.ndarray, valid_window: data.Window, nodata_value: int = 0):
    is_not_valid = np.ones(array.shape).astype(bool)
    is_not_valid[valid_window.ystart : valid_window.yend, valid_window.xstart : valid_window.xend] = False
    array[is_not_valid] = nodata_value
    return array


def row_to_burst_entry(row):
    shape = (row['n_rows'], row['n_columns'])
    compressed_offset = data.Offset(row['download_start'], row['download_stop'])
    decompressed_offset = data.Offset(row['offset_start'], row['offset_stop'])

    window = data.Window(row['valid_x_start'], row['valid_y_start'], row['valid_x_stop'], row['valid_y_stop'])

    burst_entry = data.BurstMetadata(
        row['name'], row['slc'], shape, compressed_offset, decompressed_offset, window
    )
    return burst_entry


def array_to_raster(out_path, array, fmt='GTiff'):
    driver = gdal.GetDriverByName(fmt)
    n_rows, n_cols = array.shape
    out_dataset = driver.Create(out_path, n_cols, n_rows, 1, gdal.GDT_CFloat32)
    out_dataset.GetRasterBand(1).WriteArray(array)
    out_dataset = None
    return out_path


def extract_burst(bucket, key, burst_name, df_file_name):
    df = pd.read_csv(df_file_name)
    single_burst = df.loc[df.name == burst_name].squeeze()
    burst_metadata = row_to_burst_entry(single_burst)

    client = boto3.client('s3')
    burst_bytes = extract_bytes(client, bucket, key, burst_metadata)
    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_metadata.shape))
    burst_array = invalid_to_nodata(burst_array, burst_metadata.valid_window)
    out_name = array_to_raster('extracted_01.tif', burst_array)
    return out_name


if __name__ == '__main__':
    bucket = 'ffwilliams2-shenanigans'
    key = 'bursts/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    burst = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_IW2_VV_0.tiff'
    df_filename = 'bursts.csv'
    extract_burst(bucket, key, burst, df_filename)
