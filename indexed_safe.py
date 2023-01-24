import io
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import dataclass
from gzip import _create_simple_gzip_header
from pathlib import Path
from typing import Iterable

import boto3
import indexed_gzip as igzip
import numpy as np
import pandas as pd
from isal import isal_zlib  # noqa

MAGIC_NUMBER = 28


@dataclass(frozen=True)
class CompressedFile:
    path: str
    offset: int
    length: int
    compress_type: int
    crc: int
    uncompressed_size: int


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
class Extraction:
    compressed_offset: Offset
    decompressed_offset: Offset


@dataclass(frozen=True)
class BurstEntry:
    name: str
    slc: str
    n_rows: int
    n_columns: int
    extraction_data: Extraction
    valid_window: Window


@dataclass(frozen=True)
class MetadataEntry:
    name: str
    slc: str
    offset: Offset


def get_compressed_file_info(zip_path):
    files = []
    with zipfile.ZipFile(zip_path) as f:
        for zinfo in f.infolist():
            file_offset = len(zinfo.FileHeader()) + zinfo.header_offset + MAGIC_NUMBER - len(zinfo.extra)
            compressed_file = CompressedFile(
                zinfo.filename, file_offset, zinfo.compress_size, zinfo.compress_type, zinfo.CRC, zinfo.file_size
            )
            files.append(compressed_file)

    relevant = [x for x in files if 'xml' in Path(x.path).name or 'tiff' in Path(x.path).name]
    return relevant


def s3_download(client, bucket, key, file):
    range_header = f'bytes={file.offset}-{file.offset + file.length - 1}'
    resp = client.get_object(Bucket=bucket, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def wrap_as_gz(payload, file: CompressedFile):
    header = _create_simple_gzip_header(1)
    trailer = struct.pack("<LL", file.crc, (file.uncompressed_size & 0xFFFFFFFF))
    gzip_wrapped = header + payload + trailer
    return gzip_wrapped


def s3_extract(client, bucket, key, file, convert_gzip=False):
    out_name = Path(file.path).name
    body = s3_download(client, bucket, key, file)

    if convert_gzip:
        body = wrap_as_gz(body, file)
        out_name = out_name + '.gz'
    elif file.compress_type == zipfile.ZIP_STORED:
        pass
    elif file.compress_type == zipfile.ZIP_DEFLATED:
        body = isal_zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(body)
    else:
        raise ValueError('Only DEFLATE and uncompressed formats accepted')

    with open(out_name, 'wb') as f:
        f.write(body)

    return out_name


def build_index_old(file_name: str, save=False) -> str:
    with igzip.IndexedGzipFile(file_name) as f:
        f.build_full_index()
        seek_points = f.seek_points()
        df = pd.DataFrame(seek_points, columns=['uncompressed', 'compressed'])

    if save:
        out_name = str(Path(file_name).with_suffix('.csv'))
        df.to_csv(out_name)
    return df


def build_index(file_name: str, save=False) -> str:
    with igzip.IndexedGzipFile(file_name) as f:
        f.build_full_index()
        seek_points = list(f.seek_points())
        array = np.array(seek_points)  # first column is uncompressed, second is compressed
    return array


def compute_valid_window(index: int, burst: ET.Element) -> Window:
    """Written by Jason Ninneman for the I&A team's burst extractor"""

    # all offsets, even invalid offsets
    offsets_range = Offset(
        np.array([int(val) for val in burst.find('firstValidSample').text.split()]),
        np.array([int(val) for val in burst.find('lastValidSample').text.split()]),
    )

    # returns the indices of lines containing valid data
    lines_with_valid_data = np.flatnonzero(offsets_range.stop - offsets_range.start)

    # get first and last sample with valid data per line
    # x-axis, range
    valid_offsets_range = Offset(
        offsets_range.start[lines_with_valid_data].min(),
        offsets_range.stop[lines_with_valid_data].max(),
    )

    # get the first and last line with valid data
    # y-axis, azimuth
    valid_offsets_azimuth = Offset(
        lines_with_valid_data.min(),
        lines_with_valid_data.max(),
    )

    # x-length
    length_range = valid_offsets_range.stop - valid_offsets_range.start
    # y-length
    length_azimuth = len(lines_with_valid_data)

    valid_window = Window(
        valid_offsets_range.start,
        valid_offsets_azimuth.start,
        valid_offsets_range.start + length_range,
        valid_offsets_azimuth.start + length_azimuth,
    )

    return valid_window


def get_burst_annotation_data(archive_name, file_name):
    with zipfile.ZipFile(archive_name) as z:
        content = z.read(file_name)

    xml = ET.parse(io.BytesIO(content)).getroot()
    burst_xmls = xml.findall('.//{*}burst')
    n_lines = int(xml.findtext('.//{*}linesPerBurst'))
    n_samples = int(xml.findtext('.//{*}samplesPerBurst'))
    burst_shape = (n_lines, n_samples)  #  y, x for numpy
    burst_starts = [int(x.findtext('.//{*}byteOffset')) for x in burst_xmls]
    burst_lengths = burst_starts[1] - burst_starts[0]
    burst_offsets = [Offset(x, x + burst_lengths - 1) for x in burst_starts]
    burst_windows = [compute_valid_window(i, burst_xml) for i, burst_xml in enumerate(burst_xmls)]
    return burst_shape, burst_offsets, burst_windows


def get_extraction_offsets(index: np.ndarray, uncompressed_offsets: Iterable[Offset]):
    first_entry = index[index[:, 0].argmin()]
    header_size = first_entry[1] - first_entry[0]

    extractions = []
    for uncompressed_offset in uncompressed_offsets:
        less_than_start = index[index[:, 0] < uncompressed_offset.start]
        start_pair = less_than_start[less_than_start[:, 0].argmax()]

        greater_than_stop = index[index[:, 0] > uncompressed_offset.stop]
        stop_pair = greater_than_stop[greater_than_stop[:, 0].argmin()]

        decompress_offset = Offset(start_pair[1] - header_size, stop_pair[1] - header_size)
        data_offset = Offset(
            uncompressed_offset.start - start_pair[0],
            uncompressed_offset.stop - start_pair[0],
        )
        extractor = Extraction(decompress_offset, data_offset)
        extractions.append(extractor)

    return extractions


def index_safe(zipped_safe_path):
    breakpoint()
    with zipfile.ZipFile(zipped_safe_path) as f:
        tiffs = [x for x in f.infolist() if 'tiff' in Path(x.filename).name]
        xmls = [x for x in f.infolist() if 'xml' in Path(x.filename).name]

    return None


def burst_population(safe_path):
    return None

def index_xml():
    return None


# function to get interior and exterior offsets


if __name__ == '__main__':
    zip_filename = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    out = index_safe(zip_filename)
    # bucket = 'ffwilliams2-shenanigans'
    # key = 'bursts/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    # file_name = Path(key).name
    # client = boto3.client('s3')

    # files = get_compressed_file_info(file_name)
    # file = files[22]  # 3 is IW2 VV SLC
    # file_name = s3_extract(client, bucket, key, file, convert_gzip=True)
    # index_name = build_index(file_name)

    # zip_filename = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    # interior_path = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.SAFE/annotation/s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.xml'
    # name = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gz'
    # index = build_index(name)
    # burst_shape, uncompressed_offsets, valid_windows = get_burst_annotation_data(zip_filename, interior_path)
    # offsets = get_extraction_offsets(index, uncompressed_offsets)

    # zip_filename = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    # interior_path = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.SAFE/annotation/s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.xml'
    # shape, offsets = get_burst_annotation_data(zip_filename, interior_path)
    breakpoint()
    print('done')
