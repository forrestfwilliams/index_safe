from pathlib import Path
import zipfile
import zlib
import isal_zlib


def s3_download(client, bucket, key, file):
    range_header = f'bytes={file.offset}-{file.offset + file.length - 1}'
    resp = client.get_object(Bucket=bucket, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def s3_extract(client, bucket, key, file, convert_gzip=False):
    out_name = Path(file.path).name
    body = s3_download(client, bucket, key, file)

    if file.compress_type == zipfile.ZIP_STORED:
        pass
    elif file.compress_type == zipfile.ZIP_DEFLATED:
        body = isal_zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(body)
    else:
        raise ValueError('Only DEFLATE and uncompressed formats accepted')

    with open(out_name, 'wb') as f:
        f.write(body)

    return out_name
