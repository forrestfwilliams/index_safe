import sys
import time

import fsspec
import requests


def benchmark(fun, args):
    start = time.time()
    result = fun(**args)
    end = time.time()
    print(f'using {fun.__name__} downloaded {sys.getsizeof(result) * 1e-6:.2f}mb in {end - start:.2f}s')
    return result


def download_requests(url, offset, length):
    range_header = f'bytes={offset}-{offset+length}'

    with requests.session() as client:
        resp = client.get(url, headers={'Range': range_header})
        body = resp.content

    return body


def download_fsspec(url, offset, length):
    base_fs = fsspec.filesystem('https')
    with base_fs.open(url, 'rb') as f:
        f.seek(offset)
        body = f.read(length)
    return body


if __name__ == '__main__':
    url_base = 'https://ffwilliams2-shenanigans.s3.us-west-2.amazonaws.com/bursts'
    name = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    url = f'{url_base}/{name}'

    offset = 1_000
    length = 1_000_000_000
    
    args_dict = {'url':url, 'offset':offset, 'length':length}

    benchmark(download_requests, args_dict)
    benchmark(download_fsspec, args_dict)
