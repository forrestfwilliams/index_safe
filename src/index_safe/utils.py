import json
import math
import os
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


def check_if_in_aws_and_region(region='us-west-2'):
    using_ec2 = False
    try:
        with open('/var/lib/cloud/instance/datasource') as f:
            line = f.readlines()
            if 'DataSourceEc2' in line[0]:
                using_ec2 = True
    except FileNotFoundError:
        pass

    using_lambda = False
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
        using_lambda = True

    in_aws = using_ec2 or using_lambda

    if not in_aws:
        return False

    if not boto3.Session().region_name == region:
        return False

    return True


@dataclass(frozen=True)
class GeoControlPoint:
    pixel: int
    line: int
    lon: float
    lat: float
    hgt: float


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
    swath: str
    burst_index: int
    shape: Iterable[int]  # n_row, n_column
    index_offset: Offset
    uncompressed_offset: Offset
    annotation_offset: Offset
    manifest_offset: Offset

    def to_dict(self):
        dictionary = {
            'name': self.name,
            'slc': self.slc,
            'swath': self.swath,
            'burst_index': self.burst_index,
            'n_rows': int(self.shape[0]),
            'n_columns': int(self.shape[1]),
            'index_offset': {'start': int(self.index_offset.start), 'stop': int(self.index_offset.stop)},
            'uncompressed_offset': {
                'start': int(self.uncompressed_offset.start),
                'stop': int(self.uncompressed_offset.stop),
            },
            'annotation_offset': {
                'start': int(self.annotation_offset.start),
                'stop': int(self.annotation_offset.stop),
            },
            'manifest_offset': {
                'start': int(self.manifest_offset.start),
                'stop': int(self.manifest_offset.stop),
            },
        }
        return dictionary


@dataclass(frozen=True)
class XmlMetadata:
    name: str
    slc: str
    offset: Offset

    def to_dict(self):
        return {self.slc: {self.name: {'offset_start': self.offset.start, 'offset_stop': self.offset.stop}}}


def calculate_range_parameters(total_size: int, offset: int, chunk_size: int) -> Iterable[str]:
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


def s3_range_get(client: boto3.client, url: str, range_header: str) -> bytes:
    """Get a range of bytes from an S1 SLC file in ASF's archive.
    Used in threading to download a large file in chunks.

    Args:
        client: boto3 S3 client
        url: url location of SLC file
        range_header: range header string

    Returns:
        bytes of object
    """
    key = Path(url).name
    resp = client.get_object(Bucket=BUCKET, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def http_range_get(client: requests.sessions.Session, url: str, range_header: str) -> bytes:
    """Get a range of bytes from an S1 SLC file in ASF's archive.
    Used in threading to download a large file in chunks.

    Args:
        client: requests session
        url: url location of SLC file
        range_header: range header string

    Returns:
        bytes of object
    """
    resp = client.get(url, headers={'Range': range_header})
    body = resp.content
    return body


def setup_download_client(strategy: str = 's3', edl_token: str = None, working_dir: Path = Path('.')) -> bytes:
    """Create client and range_get_func for downloading from SLC archive based on strategy (s3 | http).

    Args:
        strategy: strategy to use for download (s3 | http) s3 only works if runnning from us-west-2 region
        edl_token: EDL token for downloading from ASF's archive, if None will assume token is specified
                   in environment variable EDL_TOKEN
        working_dir: working directory where temparary credentials will be stored

    Returns:
        S3 client or http client, and matching *_range_get function
    """
    if strategy == 's3':
        creds = get_credentials(edl_token, working_dir)
        client = boto3.client(
            's3',
            aws_access_key_id=creds['accessKeyId'],
            aws_secret_access_key=creds['secretAccessKey'],
            aws_session_token=creds['sessionToken'],
        )
        range_get = s3_range_get

    elif strategy == 'http':
        client = requests.session()
        range_get = http_range_get

    return client, range_get


def download_slc(scene: str, edl_token: str = None, working_dir=Path('.'), strategy='s3') -> str:
    """Download an SLC zip file from ASF.

    Args:
        scene: SLC name (no exstension)
        strategy: strategy to use for download (s3 | http)
            s3 only works if runnning from us-west-2 region
    """
    zip_name = f'{scene}.zip'
    url = get_download_url(scene)

    client, _ = setup_download_client(strategy, edl_token, working_dir)
    if strategy == 's3':
        metadata = client.head_object(Bucket=BUCKET, Key=zip_name)
        total_length = int(metadata.get('ContentLength', 0))
        with tqdm(total=total_length, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(working_dir / zip_name, 'wb') as f:
                client.download_fileobj(BUCKET, zip_name, f, Callback=pbar.update)

    elif strategy == 'http':
        with client.get(url, stream=True) as s:
            s.raise_for_status()
            total = int(s.headers.get('content-length', 0))
            with open(working_dir / zip_name, 'wb') as f:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=total) as pbar:
                    for chunk in s.iter_content(chunk_size=10 * MB):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        client.close()


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

    n = int.from_bytes(data[26:28], 'little')
    m = int.from_bytes(data[28:30], 'little')

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

    def subset_dflidx(self, locations: Iterable[int], end_location: int) -> Iterable:
        """Build base DFLIDX index for a Zip member file that has
        been compresed using zlib (DEFLATE).

        Args:
            locations: A list of uncompressed locations to be included in the new index.
                       The closes point before each location will be selected.
            end_location: The uncompressed endpoint of the index. Used to determine file size.

        Returns:
            bytes of dflidx, and compressed range of member in zip archive
        """
        compressed_offset, uncompressed_offset, modified_index = self.index.create_modified_index(
            locations, end_location
        )
        compressed_offset = Offset(
            compressed_offset[0] + self.file_offset.start, compressed_offset[1] + self.file_offset.start
        )
        uncompressed_offset = Offset(uncompressed_offset[0], uncompressed_offset[1])
        return compressed_offset, uncompressed_offset, modified_index
