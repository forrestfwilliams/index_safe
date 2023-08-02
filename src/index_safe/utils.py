import json
import math
import os
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import boto3
import botocore
import requests
import zran
from tqdm import tqdm


KB = 1024
MB = 1024 * KB
SENTINEL_DISTRIBUTION_URL = 'https://sentinel1.asf.alaska.edu'
BUCKET = 'asf-ngap2w-p-s1-slc-7b420b89'


# TODO can also be np.array
@dataclass(frozen=True)
class Offset:
    start: int
    stop: int


@dataclass(frozen=True)
class Window:
    xstart: int
    xend: int
    ystart: int
    yend: int


@dataclass(frozen=True)
class BurstMetadata:
    name: str
    slc: str
    shape: Iterable[int]  # n_row, n_column
    index_offset: Offset
    uncompressed_offset: Offset
    valid_window: Window

    def to_tuple(self):
        tuppled = (
            self.name,
            self.slc,
            self.shape[0],
            self.shape[1],
            self.index_offset.start,
            self.index_offset.stop,
            self.uncompressed_offset.start,
            self.uncompressed_offset.stop,
            self.valid_window.xstart,
            self.valid_window.xend,
            self.valid_window.ystart,
            self.valid_window.yend,
        )
        return tuppled

    def to_bytes(self):
        data = (
            self.shape[0],
            self.shape[1],
            self.index_offset.start,
            self.index_offset.stop,
            self.uncompressed_offset.start,
            self.uncompressed_offset.stop,
            self.valid_window.xstart,
            self.valid_window.xend,
            self.valid_window.ystart,
            self.valid_window.yend,
        )
        byte_metadata = b'BURST' + struct.pack('<QQQQQQQQQQ', *data)
        return byte_metadata

    def to_dict(self):
        dictionary = {
            'name': self.name,
            'slc': self.slc,
            'n_rows': int(self.shape[0]),
            'n_columns': int(self.shape[1]),
            'index_offset': {'start': int(self.index_offset.start), 'stop': int(self.index_offset.stop)},
            'uncompressed_offset': {
                'start': int(self.uncompressed_offset.start),
                'stop': int(self.uncompressed_offset.stop),
            },
            'valid_window': {
                'xstart': int(self.valid_window.xstart),
                'xend': int(self.valid_window.xend),
                'ystart': int(self.valid_window.ystart),
                'yend': int(self.valid_window.yend),
            },
        }
        return dictionary


@dataclass(frozen=True)
class XmlMetadata:
    name: str
    slc: str
    offset: Offset

    def to_tuple(self):
        return (self.name, self.slc, self.offset.start, self.offset.stop)

    def to_dict(self):
        return {self.slc: {self.name: {'offset_start': self.offset.start, 'offset_stop': self.offset.stop}}}

<<<<<<< HEAD

def calculate_range_parameters(total_size: int, offset: int, chunk_size: int) -> list[str]:
    """Calculate range parameters for HTTP range requests.
    Useful when downloading large files in chunks.

    Args:
        total_size: total size of request
        offset: offset to start request
        chunk_size: size of each chunk

    Returns:
        list of range parameters
    """
    num_parts = int(math.ceil(total_size / float(chunk_size)))
    range_params = []
    for part_index in range(num_parts):
        start_range = (part_index * chunk_size) + offset
        if part_index == num_parts - 1:
            end_range = str(total_size + offset - 1)
        else:
            end_range = start_range + chunk_size - 1

        range_params.append(f'bytes={start_range}-{end_range}')
    return range_params


def get_tmp_access_keys(save_path: Path = Path('./credentials.json'), edl_token: str = None) -> dict:
    """Get temporary AWS access keys for direct
    access to ASF data in S3.

    Args:
        save_path: path to save credentials to

    Returns:
        dictionary of credentials
    """
    if not edl_token:
        edl_token = os.environ['EDL_TOKEN']
    resp = requests.get(
        'https://sentinel1.asf.alaska.edu/s3credentials',
        headers={'Authorization': f'Bearer {edl_token}'},
    )
    resp.raise_for_status()
    save_path.write_bytes(resp.content)
    return resp.json()


def get_credentials(edl_token: str = None, working_dir=Path('.')) -> dict:
    """Gets temporary ASF AWS credentials from
    file or request new credentials if credentials
    are not present or expired.

    Returns:
        dictionary of credentials
    """
    credential_file = working_dir / 'credentials.json'
    if not credential_file.exists():
        credentials = get_tmp_access_keys(credential_file, edl_token)
        return credentials

    credentials = json.loads(credential_file.read_text())
    expiration_time = datetime.fromisoformat(credentials['expiration'])
    current_time = datetime.now(timezone.utc)

    if current_time >= expiration_time:
        credentials = get_tmp_access_keys(credential_file, edl_token)

    return credentials


def lambda_get_credentials(edl_token, working_dir, client, bucket, key):
    """Gets temporary ASF AWS credentials for a lambda function.
    Checks if credentials exist in S3 and are not expired. If neither are true,
    a new version of the credentials is uploaded to S3.

    Args:
        edl_token: EDL token
        working_dir: working directory
        client: boto3 client
        bucket: S3 bucket
        key: S3 key
    """
    credential_file = working_dir / 'credentials.json'
    credentials_need_update = False

    try:
        client.download_file(bucket, key, credential_file)
    except botocore.exceptions.ClientError:
        print('The credentials do not exist, will create.')
        credentials_need_update = True
        get_tmp_access_keys(credential_file, edl_token)

    credentials = json.loads(credential_file.read_text())
    expiration_time = datetime.fromisoformat(credentials['expiration'])
    current_time = datetime.now(timezone.utc)

    if current_time >= expiration_time:
        credentials_need_update = True
        get_tmp_access_keys(credential_file, edl_token)

    if credentials_need_update:
        client.upload_file(credential_file, bucket, key)


def get_download_url(scene: str) -> str:
    """Get download url for Sentinel-1 scene.

    Args:
        scene: scene name

    Returns:
        Scene url
    """
    mission = scene[0] + scene[2]
    product_type = scene[7:10]
    if product_type == 'GRD':
        product_type += '_' + scene[10] + scene[14]
    url = f'{SENTINEL_DISTRIBUTION_URL}/{product_type}/{mission}/{scene}.zip'
    return url


def download_slc(scene: str, edl_token: str = None, working_dir=Path('.'), strategy='s3') -> str:
    """Download an SLC zip file from ASF.

    Args:
        scene: SLC name (no exstension)
        strategy: strategy to use for download (s3 | http)
            s3 only works if runnning from us-west-2 region
    """
    zip_name = f'{scene}.zip'
    url = get_download_url(scene)

    if strategy == 's3':
        creds = get_credentials(edl_token, working_dir)
        client = boto3.client(
            "s3",
            aws_access_key_id=creds["accessKeyId"],
            aws_secret_access_key=creds["secretAccessKey"],
            aws_session_token=creds["sessionToken"],
        )

        metadata = client.head_object(Bucket=BUCKET, Key=zip_name)
        total_length = int(metadata.get('ContentLength', 0))
        with tqdm(total=total_length, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(working_dir / zip_name, 'wb') as f:
                client.download_fileobj(BUCKET, zip_name, f, Callback=pbar.update)

    elif strategy == 'http':
        session = requests.Session()
        with session.get(url, stream=True) as s:
            s.raise_for_status()
            total = int(s.headers.get('content-length', 0))
            with open(zip_name, "wb") as f:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=total) as pbar:
                    for chunk in s.iter_content(chunk_size=10 * MB):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        session.close()


def get_zip_compressed_offset(zip_path: str, zinfo: zipfile.ZipInfo) -> Offset:
    """Get the byte offset (beginning byte and end byte inclusive) for a member
    of a zip archive.

    Args:
        zip_path: path to zip file on disk
        zinfo: the ZipInfo object associated with the zip member to get offset for

    Returns:
        byte offset (start and end) for zip member
    """
    with open(zip_path, 'rb') as f:
        f.seek(zinfo.header_offset)
        data = f.read(30)

    n = int.from_bytes(data[26:28], "little")
    m = int.from_bytes(data[28:30], "little")

    data_start = zinfo.header_offset + n + m + 30
    data_stop = data_start + zinfo.compress_size
    return Offset(data_start, data_stop)


class ZipIndexer:
    """Class for creating dflidx indexes for zip archive members
    that are compatible with the zran library.
    """

    def __init__(self, zip_path: str, member_name: str, spacing: int = 2**20):
        self.zip_path = zip_path
        self.spacing = spacing
        self.member_name = member_name
        self.index = None

    def create_full_dflidx(self):
        with zipfile.ZipFile(self.zip_path) as f:
            zinfo = [x for x in f.infolist() if self.member_name in x.filename][0]

        self.file_offset = get_zip_compressed_offset(self.zip_path, zinfo)
        with open(self.zip_path, 'rb') as f:
            f.seek(self.file_offset.start)
            self.body = f.read(self.file_offset.stop - self.file_offset.start)

        self.index = zran.Index.create_index(self.body, span=self.spacing)

    def subset_dflidx(self, starts: Iterable[int] = [], stops: Iterable[int] = []) -> Iterable:
        """Build base DFLIDX index for a Zip member file that has
        been compresed using zlib (DEFLATE).

        Args:
            member_name: name of zip member
            starts: uncompressed locations to provide indexes before
            stops: uncompressed locations to provide indexes after

        Returns:
            bytes of dflidx, and compressed range of member in zip archive
        """
        compressed_offset, uncompressed_offset, modified_index = self.index.create_modified_index(starts, stops)
        compressed_offset = Offset(
            compressed_offset[0] + self.file_offset.start, compressed_offset[1] + self.file_offset.start
        )
        uncompressed_offset = Offset(uncompressed_offset[0], uncompressed_offset[1])
        return compressed_offset, uncompressed_offset, modified_index
