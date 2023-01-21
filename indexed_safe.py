import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import boto3
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
    range_header = f'bytes={file.offset}-{file.offset + file.length}'
    resp = client.get_object(Bucket=bucket, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def wrap_as_gz(payload, file: CompressedFile):
    GZIP_HEADER = b'\x1f\x8b\x08\x00!k\xccc\x02\xff'
    crc_bytes = file.crc.to_bytes(4, byteorder='big')
    modulo_size = file.uncompressed_size % (2^32)
    file_size_bytes = modulo_size.to_bytes(4, byteorder='big')
    return GZIP_HEADER + payload + crc_bytes + file_size_bytes


def s3_extract(client, bucket, key, file, convert_gzip=False):
    out_name = Path(file.path).name
    body = s3_download(client, bucket, key, file)

    if convert_gzip:
        body = wrap_as_gz(body, file)
    elif file.compress_type == zipfile.ZIP_STORED:
        pass
    elif file.compress_type == zipfile.ZIP_DEFLATED:
        body = isal_zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(body)
    else:
        raise ValueError('Only DEFLATE and uncompressed formats accepted')

    with open(out_name, 'wb') as f:
        f.write(body)

    return out_name


if __name__ == '__main__':
    bucket = 'ffwilliams2-shenanigans'
    key = 'bursts/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    file_name = Path(key).name
    client = boto3.client('s3')

    files = get_compressed_file_info(file_name)
    file = files[6]
    # with open('compressed.deflate', 'wb') as f:
    #     body = s3_download(client, bucket, key, file)
    #     f.write(body)
    out = s3_extract(client, bucket, key, file, convert_gzip=True)
