import zlib

import boto3
import numpy as np
import pandas as pd
from isal import isal_zlib
from osgeo import gdal

import data


def s3_download(client, bucket, key, download_start, download_stop):
    range_header = f'bytes={download_start}-{download_stop}'
    resp = client.get_object(Bucket=bucket, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def extract_burst(client, bucket, key, metadata):
    body = s3_download(
        client,
        bucket,
        key,
        metadata.extraction_data.compressed_offset.start,
        metadata.extraction_data.compressed_offset.stop,
    )
    body = isal_zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(body)
    burst_bytes = body[
        metadata.extraction_data.decompressed_offset.start : metadata.extraction_data.decompressed_offset.stop
    ]

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
    download_offset = data.Offset(row['download_start'], row['download_stop'])
    data_offset = data.Offset(row['offset_start'], row['offset_stop'])
    extractor = data.Extraction(download_offset, data_offset)

    window = data.Window(row['valid_x_start'], row['valid_y_start'], row['valid_x_stop'], row['valid_y_stop'])

    burst_entry = data.BurstEntry(row['name'], row['slc'], row['n_rows'], row['n_columns'], extractor, window)
    return burst_entry


def array_to_raster(out_path, array, fmt='GTiff'):
    driver = gdal.GetDriverByName(fmt)
    n_rows, n_cols = array.shape
    out_dataset = driver.Create(out_path, n_cols, n_rows, 1, gdal.GDT_CFloat32)
    out_dataset.GetRasterBand(1).WriteArray(array)
    out_dataset = None
    return out_path


def read_burst_df(bucket, key, burst_name, df_file_name):
    df = pd.read_csv(df_file_name)
    single_burst = df.loc[df.name == burst_name].squeeze()
    burst_entry = row_to_burst_entry(single_burst)

    client = boto3.client('s3')
    burst_bytes = extract_burst(client, bucket, key, burst_entry)
    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_entry.n_rows, burst_entry.n_columns))
    burst_array = invalid_to_nodata(burst_array, burst_entry.valid_window)
    out_name = array_to_raster('extracted_01.tif', burst_array)
    return out_name


if __name__ == '__main__':
    bucket = 'ffwilliams2-shenanigans'
    key = 'bursts/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    burst = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_IW2_VV_0.tiff'
    df_filename = 'bursts.csv'
    read_burst_df(bucket, key, burst, df_filename)
