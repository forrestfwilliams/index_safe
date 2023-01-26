import gzip
import io
import struct
import zipfile
import zlib
from gzip import _create_simple_gzip_header
from pathlib import Path
import indexed_gzip as igzip

import boto3

MAGIC_NUMBER = 28

zip_path = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
bucket = 'ffwilliams2-shenanigans'
key = 'bursts/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
client = boto3.client('s3')


def s3_download(client, bucket, key, range_header):
    resp = client.get_object(Bucket=bucket, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def wrap_as_gz(payload: bytes, zinfo: zipfile.ZipInfo) -> bytes:
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


def test_equality(tiff_path, use_local=True, use_zip=True):
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        info_list = archive.infolist()
        zinfo = [x for x in info_list if x.filename == tiff_path][0]

    # test read
    # file_offset = len(zinfo.FileHeader()) + zinfo.header_offset
    file_offset = len(zinfo.FileHeader()) + zinfo.header_offset + MAGIC_NUMBER - len(zinfo.extra)
    file_length = zinfo.compress_size

    if use_local:
        with open(zip_path, 'rb') as f:
            f.seek(file_offset)
            compressed = f.read(file_length)
    else:
        range_header = f'bytes={file_offset}-{file_offset+file_length}'
        compressed = s3_download(client, bucket, key, range_header)

    if use_zip:
        decompressed = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(compressed)
    else:
        gzip_compressed = wrap_as_gz(compressed, zinfo)
        decompressed = gzip.decompress(gzip_compressed)
        gzf = igzip.IndexedGzipFile(fileobj=io.BytesIO(gzip_compressed))

    # golden read
    with open(tiff_path, 'rb') as f:
        golden = f.read()

    print(Path(tiff_path).name)
    print(golden == decompressed)


tiff_paths = [
    test_equality(str(x), use_zip=False) for x in Path(zip_path).with_suffix('.SAFE').glob('measurement/*tiff')
]

print('done')
