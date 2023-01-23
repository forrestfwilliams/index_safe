import struct
import zipfile
import zlib
from dataclasses import dataclass
from gzip import _create_simple_gzip_header
from pathlib import Path
import pandas as pd

import boto3
import indexed_gzip as igzip
from isal import isal_zlib  # noqa

MAGIC_NUMBER = 28


@dataclass
class CompressedFile:
    path: str
    offset: int
    length: int
    compress_type: int
    crc: int
    uncompressed_size: int


def get_compressed_file_info(zip_path):
    files = []
    with zipfile.ZipFile(file_name) as f:
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


def build_index(file_name: str, save=False) -> str:
    with igzip.IndexedGzipFile(file_name) as f:
        f.build_full_index()
        seek_points = f.seek_points()
        df = pd.DataFrame(seek_points, columns = ['uncompressed', 'compressed'])
    
    if save:
        out_name = str(Path(file_name).with_suffix('.csv'))
        df.to_csv(out_name)
    return df


if __name__ == '__main__':
    # bucket = 'ffwilliams2-shenanigans'
    # key = 'bursts/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    # file_name = Path(key).name
    # client = boto3.client('s3')
    #
    # files = get_compressed_file_info(file_name)
    # file = files[22]  # 3 is IW2 VV SLC
    # file_name = s3_extract(client, bucket, key, file, convert_gzip=True)
    # index_name = build_index(file_name)
    name = 's1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.tiff.gz'
    index = build_index(name)
