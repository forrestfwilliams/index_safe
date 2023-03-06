import io
import json
import os
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from gzip import _create_simple_gzip_header
from pathlib import Path
from typing import Iterable

import boto3
import indexed_gzip as igzip
import numpy as np
import requests
import zran
from tqdm import tqdm

KB = 1024
MB = 1024 * KB
SENTINEL_DISTRIBUTION_URL = 'https://sentinel1.asf.alaska.edu'
BUCKET = 'asf-ngap2w-p-s1-slc-7b420b89'

# FIXME json is not actual name of output
def get_tmp_access_keys(save_path: str = 'credentials.json') -> dict:
    """ Get temporary AWS access keys for direct
    access to ASF data in S3.

    Args:
        save_path: path to save credentials to

    Returns:
        dictionary of credentials
    """
    token = os.environ['EDL_TOKEN']
    resp = requests.get(
        'https://sentinel1.asf.alaska.edu/s3credentials',
        headers={'Authorization': f'Bearer {token}'},
    )
    resp.raise_for_status()
    save_path.write_bytes(resp.content)
    return resp.json()


def get_credentials() -> dict:
    """ Gets temporary ASF AWS credentials from
    file or request new credentials if credentials
    are not present or expired.

    Returns:
        dictionary of credentials
    """
    credential_file = Path('credentials.json')
    if not credential_file.exists():
        credentials = get_tmp_access_keys(credential_file)
        return credentials

    credentials = json.loads(credential_file.read_text())
    expiration_time = datetime.fromisoformat(credentials['expiration'])
    current_time = datetime.now(timezone.utc)

    if current_time >= expiration_time:
        credentials = get_tmp_access_keys(credential_file)

    return credentials


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


@dataclass(frozen=True)
class XmlMetadata:
    name: str
    slc: str
    offset: Offset

    def to_tuple(self):
        return (self.name, self.slc, self.offset.start, self.offset.stop)


def get_closest_index(array: np.ndarray, value: int, less_than: bool=True) -> int:
    """Identifies index of closest value in a numpy array to input value.

    Args:
        array: 1d np array
        value: value that you want to find closes index for in array
        less_than: whether to return closest index that <= or >= value

    Returns:
        index of closest value
    """
    if less_than:
        valid_options = array[array <= value]
    else:
        valid_options = array[array >= value]
    closest_number = valid_options[np.abs(valid_options - value).argmin()]
    closest_index = int(np.argwhere(array == closest_number))
    return closest_index


def wrap_deflate_as_gz(payload: bytes, zinfo: zipfile.ZipInfo) -> bytes:
    """Add a GZIP-style header and footer to a raw DEFLATE byte
    object based on information from a ZipInfo object.

    Args:
        payload: raw DEFLATE bytes (no header or footer)
        zinfo: the ZipInfo object associated with the payload

    Returns:
        {10-byte header}DEFLATE_PAYLOAD{CRC}{filesize % 2**32}
    """
    header = _create_simple_gzip_header(1)
    trailer = struct.pack("<LL", zinfo.CRC, (zinfo.file_size & 0xFFFFFFFF))
    gz_wrapped = header + payload + trailer
    return gz_wrapped


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


def parse_gzidx(gzidx: bytes) -> Iterable:
    """Parse bytes of GZIDX file to return relevant information.

    Args:
        gzidx: bytes of a gzidx file

    Returns
        np.array of index points, compressed file size,
        uncompressed file size, window size, and number of index points
    """
    compressed_size, uncompressed_size, point_spacing, window_size, n_points = struct.unpack('<QQIII', gzidx[7:35])
    raw_points = gzidx[35 : 35 + n_points * 18]
    points = np.array([struct.unpack('<QQBB', raw_points[18 * i : 18 * (i + 1)]) for i in range(n_points)])
    return points, compressed_size, uncompressed_size, window_size, n_points


class ZipIndexer:
    """Class for creating dflidx indexes for zip archive members
    that are compatible with the zran library.
    """

    def __init__(self, zip_path: str, spacing: int = 2**20):
        self.zip_path = zip_path
        self.spacing = spacing

    def create_dflidx(self, member_name: str, starts: Iterable[int] = [], stops: Iterable[int] = []) -> Iterable:
        """Build base DFLIDX index for a Zip member file that has
        been compresed using zlib (DEFLATE).

        Args:
            member_name: name of zip member
            starts: uncompressed locations to provide indexes before
            stops: uncompressed locations to provide indexes after

        Returns:
            bytes of dflidx, and compressed range of member in zip archive
        """
        with zipfile.ZipFile(self.zip_path) as f:
            zinfo = [x for x in f.infolist() if member_name in x.filename][0]

        offset = get_zip_compressed_offset(self.zip_path, zinfo)
        with open(self.zip_path, 'rb') as f:
            f.seek(offset.start)
            body = f.read(offset.stop - offset.start)
        
        index = zran.build_deflate_index(io.BytesIO(body), span = self.spacing)
        desired_points = zran.modify_points(index.points, starts, stops, offset = offset.start)
        new_length = desired_points[0].outloc - desired_points[-1].outloc
        dflidx = zran.create_index_file(index.mode, new_length, len(desired_points), desired_points)

        return dflidx


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


def download_slc(scene: str, strategy='s3') -> str:
    """Download an SLC zip file from ASF.

    Args:
        scene: SLC name (no exstension)
        strategy: strategy to use for download (s3 | http)
            s3 only works if runnning from us-west-2 region
    """
    zip_name = f'{scene}.zip'
    url = get_download_url(scene)

    if strategy == 's3':
        creds = get_credentials()
        client = boto3.client(
            "s3",
            aws_access_key_id=creds["accessKeyId"],
            aws_secret_access_key=creds["secretAccessKey"],
            aws_session_token=creds["sessionToken"],
        )

        metadata = client.head_object(Bucket=BUCKET, Key=zip_name)
        total_length = int(metadata.get('ContentLength', 0))
        with tqdm(total=total_length, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(zip_name, 'wb') as f:
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
