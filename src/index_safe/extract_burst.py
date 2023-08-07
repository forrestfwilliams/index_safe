import base64
import json
import os
import struct
import tempfile
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Iterable, Tuple

import boto3
import numpy as np
import requests
import zran
from osgeo import gdal


try:
    from index_safe import utils
except ModuleNotFoundError:
    import utils


gdal.UseExceptions()

KB = 1024
MB = KB * KB

BUCKET = 'asf-ngap2w-p-s1-slc-7b420b89'


def s3_range_get(client: boto3.client, key: str, range_header: str, bucket: str = BUCKET) -> bytes:
    """Get a range of bytes from an S3 object.
    Used in threading to download a large file in chunks.

    Args:
        client: boto3 S3 client
        key: S3 object key
        range_header: range header string
        bucket: S3 bucket name (default is ASF's S1 SLC bucket)
    Returns:
        bytes of object
    """
    resp = client.get_object(Bucket=BUCKET, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


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
    if strategy == 's3':
        creds = utils.get_credentials(edl_token, working_dir)
        client = boto3.client(
            "s3",
            aws_access_key_id=creds["accessKeyId"],
            aws_secret_access_key=creds["secretAccessKey"],
            aws_session_token=creds["sessionToken"],
        )
        total_size = (metadata.index_offset.stop - 1) - metadata.index_offset.start
        range_headers = utils.calculate_range_parameters(total_size, metadata.index_offset.start, 20 * MB)
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = executor.map(s3_range_get, repeat(client), repeat(Path(url).name), range_headers)
            body = b''.join(results)

    elif strategy == 'http':
        range_header = f'bytes={metadata.index_offset.start}-{metadata.index_offset.stop - 1}'
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
    window = utils.Window(
        metadata_dict['valid_window']['xstart'],
        metadata_dict['valid_window']['xend'],
        metadata_dict['valid_window']['ystart'],
        metadata_dict['valid_window']['yend'],
    )
    burst_metadata = utils.BurstMetadata(
        metadata_dict['name'], metadata_dict['slc'], shape, index_offset, decompressed_offset, window
    )
    decoded_bytes = base64.b64decode(metadata_dict['dflidx_64encoded'])
    index = zran.Index.parse_index_file(decoded_bytes)
    return index, burst_metadata


def bytes_to_burst_entry(burst_name: str) -> Tuple[zran.Index, utils.BurstMetadata]:
    """Convert header bytes of burst-specifc
    index file to a burst metadata object.
    Index file must be in working directory.

    Args:
        burst_name: name of burst to get info for

    Returns:
        burst metadata object
    """
    burst_name = Path(burst_name)
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


def extract_burst(burst_index_path: str, edl_token: str = None, working_dir: Path = Path('.')) -> str:
    """Extract burst from SLC in ASF archive using a burst-level index
    file. Index must be available locally.

    Args:
        burst_index_path: path to burst index file on disk

    Returns:
        path to saved burst raster
    """
    index, burst_metadata = json_to_burst_metadata(burst_index_path)
    url = utils.get_download_url(burst_metadata.slc)
    burst_bytes = extract_bytes_by_burst(url, burst_metadata, index, edl_token, working_dir)
    burst_array = burst_bytes_to_numpy(burst_bytes, (burst_metadata.shape))
    burst_array = invalid_to_nodata(burst_array, burst_metadata.valid_window)
    out_path = array_to_raster(burst_metadata.name, burst_array)
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
