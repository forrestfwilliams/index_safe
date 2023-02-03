from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import boto3
import botocore
import pandas as pd
import requests
from tqdm import tqdm

from . import utils

try:
    from isal import isal_zlib as zlib
except ImportError:
    import zlib


KB = 1024
MB = 1024 * KB
MAX_WBITS = 15


def s3_download(bucket: str, key: str, range_header: str) -> bytes:
    client = boto3.client('s3')
    resp = client.get_object(Bucket=bucket, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def extract_bytes_s3(bucket: str, key: str, metadata: utils.BurstMetadata) -> bytes:
    range_header = f'bytes={metadata.compressed_offset.start}-{metadata.compressed_offset.stop}'
    body = s3_download(bucket, key, range_header)
    body = zlib.decompressobj(-1 * MAX_WBITS).decompress(body)
    return body


def http_download(url: str, range_header: str) -> bytes:
    with requests.Session() as s:
        resp = s.get(url, headers={'Range': range_header})
        resp.raise_for_status()
        body = resp.content
    return body


def extract_bytes_http(url: str, offset: utils.Offset) -> bytes:
    range_header = f'bytes={offset.start}-{offset.stop}'
    body = http_download(url, range_header)
    body = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(body)
    return body


def row_to_metadata_entry(row: pd.Series) -> utils.XmlMetadata:
    compressed_offset = utils.Offset(row['offset_start'], row['offset_stop'])
    metadata_entry = utils.XmlMetadata(row['name'], row['slc'], compressed_offset)
    return metadata_entry


def extract_metadata(slc_name, df_file_name):
    url = utils.get_download_url(slc_name)
    df = pd.read_csv(df_file_name)
    slc_df = df.loc[df.slc == slc_name]
    offsets = [row_to_metadata_entry(row).offset for i, row in slc_df.iterrows()]

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(extract_bytes_http, repeat(url), offsets), total=len(offsets)))

    content = b''.join(results)
    with open(f'{slc_name}.xml', 'wb') as f:
        f.write(content)

    return None


def main():
    """Example Command:

    extract_metadata.py \
        S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85 \
        metadata.csv
    """
    parser = ArgumentParser()
    parser.add_argument('slc_name')
    parser.add_argument('df')
    args = parser.parse_args()

    extract_metadata(args.slc_name, args.df)


if __name__ == '__main__':
    main()
