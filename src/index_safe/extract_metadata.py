from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

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


def extract_bytes(
    url: str, offset: utils.Offset, client: botocore.client.BaseClient | requests.sessions.Session
) -> bytes:
    range_header = f'bytes={offset.start}-{offset.stop - 1}'

    if isinstance(client, botocore.client.BaseClient):
        resp = client.get_object(Bucket=utils.BUCKET, Key=Path(url).name, Range=range_header)
        body = resp['Body'].read()
    elif isinstance(client, requests.sessions.Session):
        resp = client.get(url, headers={'Range': range_header})
        body = resp.content

    body = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(body)
    return body


def row_to_metadata_entry(row: pd.Series) -> utils.XmlMetadata:
    compressed_offset = utils.Offset(row['offset_start'], row['offset_stop'])
    metadata_entry = utils.XmlMetadata(row['name'], row['slc'], compressed_offset)
    return metadata_entry


def extract_metadata(slc_name, df_file_name, strategy='s3'):
    url = utils.get_download_url(slc_name)
    df = pd.read_csv(df_file_name)
    slc_df = df.loc[df.slc == slc_name]
    offsets = [row_to_metadata_entry(row).offset for i, row in slc_df.iterrows()]

    if strategy == 's3':
        creds = utils.get_credentials()
        client = boto3.client(
            "s3",
            aws_access_key_id=creds["accessKeyId"],
            aws_secret_access_key=creds["secretAccessKey"],
            aws_session_token=creds["sessionToken"],
        )
    elif strategy == 'http':
        client = requests.session()

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(extract_bytes, repeat(url), offsets, repeat(client)), total=len(offsets)))

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
