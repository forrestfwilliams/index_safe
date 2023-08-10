import subprocess
import sys
import time
from pathlib import Path

import boto3
import numpy as np


KB = 1024
MB = KB * KB


def benchmark(fun, args, n=5):
    runtimes = []
    for i in range(5):
        start = time.time()
        result = fun(**args)
        end = time.time()
        runtimes.append(end - start)
    best_time = np.min(runtimes)
    print(f'using {fun.__name__} downloaded {sys.getsizeof(result) / MB:.2f}mb in {best_time:.2f}s')
    return result


def download_boto3(url, offset, length):
    _, _, bucket, *parts = url.split('/')
    bucket = bucket.split('.')[0]
    key = '/'.join(parts)
    range_header = f'bytes={offset}-{offset+length}'
    client = boto3.client('s3')
    resp = client.get_object(Bucket=bucket, Key=key, Range=range_header)
    body = resp['Body'].read()
    return body


def download_mnt(url, offset, length):
    _, _, bucket, *parts = url.split('/')
    bucket = bucket.split('.')[0]
    prefix = None
    if len(parts) > 1:
        prefix = '/'.join(parts[:-1]) + '/'
    name = parts[-1]

    mountpoint = Path('~/mnt').expanduser()
    cmd = f'mount-s3 {bucket} {str(mountpoint)}'
    if prefix:
        cmd += f' --prefix {prefix}'
    subprocess.run(cmd.split(' '))

    with open(mountpoint / name, 'rb') as file:
        file.seek(offset)
        body = file.read(length)

    cmd = f'umount {str(mountpoint)}'
    subprocess.run(cmd.split(' '))
    return body


if __name__ == '__main__':
    url_base = 'https://ffwilliams2-shenanigans.s3.us-west-2.amazonaws.com/bursts'
    name = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    url = f'{url_base}/{name}'
    offset = 1 * MB
    length = 96 * MB

    args_dict = {'url': url, 'offset': offset, 'length': length}
    benchmark(download_mnt, args_dict, n=5)
    benchmark(download_boto3, args_dict, n=5)
