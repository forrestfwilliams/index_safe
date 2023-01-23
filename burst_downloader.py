import io
import math
import struct
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import boto3
import botocore
import requests
# from isal import isal_zlib  # noqa
import zlib  # noqa

# from uuid import uuid4

KB = 1024
MB = KB * KB


def bytes_to_xml(in_bytes):
    xml = ET.parse(io.BytesIO(in_bytes)).getroot()
    return xml


def calculate_range_parameters(total_size, offset, chunk_size):
    num_parts = int(math.ceil(total_size / float(chunk_size)))
    range_params = []
    for part_index in range(num_parts):
        start_range = (part_index * chunk_size) + offset
        if part_index == num_parts - 1:
            end_range = str(total_size + offset - 1)
        else:
            end_range = start_range + chunk_size - 1

        range_params.append(f'bytes={start_range}-{end_range}')
    return range_params


class S3Zip:
    def __init__(
        self,
        client: Union[botocore.client.BaseClient, requests.sessions.Session],
        bucket: str,
        key: str,
        multipart_threshold: int = 25 * MB,
        multipart_chunksize: int = 25 * MB,
    ):
        self.EOCD_RECORD_SIZE = 22
        self.ZIP64_EOCD_RECORD_SIZE = 56
        self.ZIP64_EOCD_LOCATOR_SIZE = 20
        self.MAX_STANDARD_ZIP_SIZE = 4_294_967_295
        self.ZLIB_MAX_WBITS = 15

        self.client = client
        self.bucket = bucket
        self.key = key
        self.multipart_threshold = multipart_threshold
        self.multipart_chunksize = multipart_chunksize
        self.url = f'https://{self.bucket}.s3.us-west-2.amazonaws.com/{self.key}'

        if isinstance(self.client, botocore.client.BaseClient):
            self.ranged_get = self._s3_ranged_get
            self.get_file_size = self._s3_get_file_size
        elif isinstance(self.client, requests.sessions.Session):
            self.ranged_get = self._http_ranged_get
            self.get_file_size = self._http_get_file_size
        else:
            raise TypeError(
                'Client must be either a botocore.client.S3 or requests.sessions.Session instance,'
                f'not a {type(client)} instance.'
            )

        self.zip_dir, self.cd_start = self.get_zip_dir()

    def parse_short(self, in_bytes):
        return ord(in_bytes[0:1]) + (ord(in_bytes[1:2]) << 8)

    def parse_little_endian_to_int(self, little_endian_bytes):
        format_character = "i" if len(little_endian_bytes) == 4 else "q"
        return struct.unpack("<" + format_character, little_endian_bytes)[0]

    def get_central_directory_metadata_from_eocd(self, eocd):
        cd_size = self.parse_little_endian_to_int(eocd[12:16])
        cd_start = self.parse_little_endian_to_int(eocd[16:20])
        return cd_start, cd_size

    def get_central_directory_metadata_from_eocd64(self, eocd64):
        cd_size = self.parse_little_endian_to_int(eocd64[40:48])
        cd_start = self.parse_little_endian_to_int(eocd64[48:56])
        return cd_start, cd_size

    def get_zip_content(self):
        files = [zi.filename for zi in self.zip_dir.filelist]
        return files

    def _s3_get_file_size(self):
        file_size = self.client.head_object(Bucket=self.bucket, Key=self.key)['ContentLength']
        return file_size

    def _http_get_file_size(self):
        file_size = int(self.client.head(self.url).headers['content-length'])
        return file_size

    def _s3_ranged_get(self, range_header):
        resp = self.client.get_object(Bucket=self.bucket, Key=self.key, Range=range_header)
        body = resp['Body'].read()
        return body

    def _http_ranged_get(self, range_header):
        resp = self.client.get(self.url, headers={'Range': range_header})
        body = resp.content
        return body

    def threaded_get(self, offset, file_size):
        range_params = calculate_range_parameters(file_size, offset, self.multipart_chunksize)

        # Dispatch work tasks with our client
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = executor.map(self.ranged_get, range_params)

        content = b''.join(results)
        return content

    def get(self, start, length):
        if length <= self.multipart_threshold:
            end = start + length - 1
            content = self.ranged_get(f'bytes={start}-{end}')
        else:
            content = self.threaded_get(start, length)
        return content

    def get_zip_dir(self):
        file_size = self.get_file_size()
        eocd_record = self.get(file_size - self.EOCD_RECORD_SIZE, self.EOCD_RECORD_SIZE)
        if file_size <= self.MAX_STANDARD_ZIP_SIZE:
            print('accessing zip')
            cd_start, cd_size = self.get_central_directory_metadata_from_eocd(eocd_record)
            central_directory = self.get(cd_start, cd_size)
            return zipfile.ZipFile(io.BytesIO(central_directory + eocd_record)), cd_start
        else:
            print('accessing zip64')
            zip64_eocd_record = self.get(
                file_size - (self.EOCD_RECORD_SIZE + self.ZIP64_EOCD_LOCATOR_SIZE + self.ZIP64_EOCD_RECORD_SIZE),
                self.ZIP64_EOCD_RECORD_SIZE,
            )
            zip64_eocd_locator = self.get(
                file_size - (self.EOCD_RECORD_SIZE + self.ZIP64_EOCD_LOCATOR_SIZE),
                self.ZIP64_EOCD_LOCATOR_SIZE,
            )
            cd_start, cd_size = self.get_central_directory_metadata_from_eocd64(zip64_eocd_record)
            central_directory = self.get(cd_start, cd_size)
            return (
                zipfile.ZipFile(io.BytesIO(central_directory + zip64_eocd_record + zip64_eocd_locator + eocd_record)),
                cd_start,
            )

    def extract_file(self, filename, outname=None):
        zi = [zi for zi in self.zip_dir.filelist if zi.filename == filename][0]
        file_head = self.get(self.cd_start + zi.header_offset + 26, 4)
        name_len = self.parse_short(file_head[0:2])
        extra_len = self.parse_short(file_head[2:4])
        content_offset = self.cd_start + zi.header_offset + 30 + name_len + extra_len
        breakpoint()
        content = self.get(content_offset, zi.compress_size)
        if zi.compress_type == zipfile.ZIP_DEFLATED:
            # content = isal_zlib.decompressobj(-1 * self.ZLIB_MAX_WBITS).decompress(content)
            content = zlib.decompressobj(-1 * self.ZLIB_MAX_WBITS).decompress(content)

        if outname:
            with open(outname, 'wb') as f:
                f.write(content)
            return outname

        return content


if __name__ == '__main__':
    bucket = 'ffwilliams2-shenanigans'
    key = 'bursts/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    annotation_path = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.SAFE/annotation/s1a-iw2-slc-vh-20200604t022253-20200604t022318-032861-03ce65-002.xml'
    client = boto3.client('s3')
    safe_zip = S3Zip(client, bucket, key)

    annotation_out = 'annotation.xml'
    annotation_bytes = safe_zip.extract_file(annotation_path, outname=annotation_out)
