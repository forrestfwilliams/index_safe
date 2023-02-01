import io
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from gzip import _create_simple_gzip_header
from pathlib import Path
from typing import Iterable

import indexed_gzip as igzip
import numpy as np
import requests
from tqdm import tqdm

KB = 1024
MB = 1024 * KB
GB = 1024 * MB
SENTINEL_DISTRIBUTION_URL = 'https://sentinel1.asf.alaska.edu'


# TODO can also be np.array
@dataclass(frozen=True)
class Offset:
    start: int
    stop: int


@dataclass(frozen=True)
class Window:
    xstart: int
    ystart: int
    xend: int
    yend: int


@dataclass(frozen=True)
class BurstMetadata:
    name: str
    slc: str
    shape: Iterable[int]  # n_row, n_column
    compressed_offset: Offset
    decompressed_offset: Offset
    valid_window: Window

    def to_tuple(self):
        tuppled = (
            self.name,
            self.slc,
            self.shape[0],
            self.shape[1],
            self.compressed_offset.start,
            self.compressed_offset.stop,
            self.decompressed_offset.start,
            self.decompressed_offset.stop,
            self.valid_window.xstart,
            self.valid_window.xend,
            self.valid_window.ystart,
            self.valid_window.yend,
        )
        return tuppled


@dataclass(frozen=True)
class XmlMetadata:
    name: str
    slc: str
    offset: Offset

    def to_tuple(self):
        return (self.name, self.slc, self.offset.start, self.offset.stop)


class OffsetZipInfo:
    def __init__(self, zip_path: str, zinfo: zipfile.ZipInfo):
        self.zip_path: str = zip_path
        self.header_offset: int = zinfo.header_offset
        self.compress_size: int = zinfo.compress_size
        self.file_size: int = zinfo.file_size
        self.filename: str = zinfo.filename
        self.CRC: bytes = zinfo.CRC
        self.compressed_offset: Offset = self.get_compressed_offset()
        self.zinfo = zinfo

    def get_compressed_offset(self) -> Offset:
        with open(self.zip_path, 'rb') as f:
            f.seek(self.header_offset)
            data = f.read(30)

        n = int.from_bytes(data[26:28], "little")
        m = int.from_bytes(data[28:30], "little")

        data_start = self.header_offset + n + m + 30
        data_stop = data_start + self.compress_size
        return Offset(data_start, data_stop)


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
    """Get the byte offset (beginning byte and end byte inclusive)
    for a member of a zip archive.

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
    """Class for creating gzidx indexes for zip archive members
    that are compatible with the indexed_gzip library
    """

    def __init__(self, zip_path: str, spacing: int = 2**20):
        self.zip_path = zip_path
        self.archive_size = Path(self.zip_path).stat().st_size
        self.gz_header_length = 10
        self.spacing = spacing
        self.gzidx_header_length = 35
        self.gzidx_point_length = 18

    def parse_gzidx(self, gzidx: bytes) -> Iterable:
        header = gzidx[:35]
        n_points = struct.unpack('<I', header[31:])[0]
        window_size = struct.unpack('<I', header[27:31])[0]
        raw_points = gzidx[35 : 35 + n_points * 18]
        points = np.array([struct.unpack('<QQBB', raw_points[18 * i : 18 * (i + 1)]) for i in range(n_points)])
        return points, n_points, window_size

    def create_base_gzidx(self, member_name: str) -> Iterable[bytes, Offset]:
        with zipfile.ZipFile(self.zip_path) as f:
            zinfo = [x for x in f.infolist() if member_name in x.filename][0]

        offset = get_zip_compressed_offset(self.zip_path, zinfo)
        with open(self.zip_path, 'rb') as f:
            f.seek(offset.start)
            body = f.read(offset.stop - offset.start)
        gz_body = wrap_deflate_as_gz(body, zinfo)

        with tempfile.NamedTemporaryFile() as tmp_gzidx:
            with igzip.IndexedGzipFile(io.BytesIO(gz_body), spacing=self.spacing) as f:
                f.build_full_index()
                f.export_index(tmp_gzidx.name)

            with open(tmp_gzidx.name, 'rb') as f:
                base_gzidx = f.read()

        return base_gzidx, offset

    def build_gzidx(self, member_name: str, gzidx_path: str, points: Iterable[int] = None) -> str:
        base_gzidx, offset = self.create_base_gzidx(member_name)
        point_array, n_points, window_size = self.parse_gzidx(base_gzidx)

        # FIXME Assume only first window entry is zero
        length_to_window = self.gzidx_header_length + (self.gzidx_point_length * n_points)
        window_offsets = np.arange(0, point_array[1:].shape[0] * window_size, window_size) + length_to_window
        window_offsets = np.append([0], window_offsets, axis=0)

        point_array = np.append(point_array, np.expand_dims(window_offsets, axis=1), axis=1)
        if points:
            point_array = point_array[np.isin(point_array[:, 0], points), :].copy()
        point_array[:, 0] += offset.start - self.gz_header_length

        point_bytes = []
        window_bytes = []
        for row in point_array:
            point_entry = struct.pack('<QQBB', *row[:4].tolist())

            if row[4] == 0:
                window_entry = b''
            else:
                window_entry = base_gzidx[row[4] : row[4] + window_size]

            point_bytes.append(point_entry)
            window_bytes.append(window_entry)

        archive_size_bytes = struct.pack('<Q', self.archive_size)

        header = base_gzidx[:7] + archive_size_bytes + base_gzidx[15:31] + struct.pack('<I', point_array.shape[0])
        altered_gzidx = header + b''.join(point_bytes) + b''.join(window_bytes)

        with open(gzidx_path, 'wb') as fobj:
            fobj.write(altered_gzidx)
        return gzidx_path


def get_download_url(scene: str) -> str:
    """Get download url for Sentinel-1 Scene

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


def download_slc(scene: str, chunk_size=10 * MB) -> str:
    """Download a file

    Args:
        url: URL of the file to download
        directory: Directory location to place files into
        chunk_size: Size to chunk the download into
    Returns:
        download_path: The path to the downloaded file
    """
    url = get_download_url(scene)
    download_path = Path(url).name
    session = requests.Session()
    with session.get(url, stream=True) as s:
        s.raise_for_status()
        total = int(s.headers.get('content-length', 0))
        with open(download_path, "wb") as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=total) as pbar:
                for chunk in s.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    session.close()
    return url
