import io
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import dataclass
from gzip import _create_simple_gzip_header
from pathlib import Path

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


@dataclass(frozen=True)
class Offset:
    start: int
    stop: int


@dataclass(frozen=True)
class ExtractionData:
    slc: str
    compressed_offset: Offset
    decompressed_offset: Offset
    rows: int
    columns: int


def get_compressed_file_info(zip_path):
    files = []
    with zipfile.ZipFile(zip_path) as f:
        for zinfo in f.infolist():
            file_offset = len(zinfo.FileHeader()) + zinfo.header_offset + MAGIC_NUMBER - len(zinfo.extra)
            compressed_file = CompressedFile(
                zinfo.filename, file_offset, zinfo.compress_size, zinfo.compress_type, zinfo.CRC, zinfo.file_size
            )
            files.append(compressed_file)

    relevant = [x for x in files if 'xml' in Path(x.path).name or 'tif' in Path(x.path).name]
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
        # first column is uncompressed, second is compressed
        array = np.array(seek_points)
    return array


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

    return burst_shape, burst_offsets

# function to get interior and exterior offsets


if __name__ == '__main__':
    # bucket = 'ffwilliams2-shenanigans'
    # key = 'bursts/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    # file_name = Path(key).name
    # client = boto3.client('s3')

    # files = get_compressed_file_info(file_name)
    # file = files[22]  # 3 is IW2 VV SLC
    # file_name = s3_extract(client, bucket, key, file, convert_gzip=True)
    # index_name = build_index(file_name)

    # name = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gz'
    # index = build_index(name)

    zip_filename = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    interior_path = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.SAFE/annotation/s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.xml'
    shape, offsets = get_burst_annotation_data(zip_filename, interior_path)
    breakpoint()
    print('done')
