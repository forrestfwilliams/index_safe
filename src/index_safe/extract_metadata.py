import json
import zlib
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Iterable

import boto3
import botocore
import lxml
import requests
from tqdm import tqdm


try:
    from index_safe import utils
except ModuleNotFoundError:
    import utils

KB = 1024
MB = 1024 * KB
MAX_WBITS = 15


def extract_bytes(
    url: str, offset: utils.Offset, client: botocore.client.BaseClient | requests.sessions.Session
) -> bytes:
    """Extract bytes pertaining to a metadata xml file from a Sentinel-1 SLC archive using offset
    information from a XmlMetadata object.

    Args:
        url: url location of SLC archive
        offset: offset for compressed data range in zip archive
        client: client to use for downloading the data (s3 | http) client

    Returns:
        bytes representing metadata xml
    """
    range_header = f'bytes={offset.start}-{offset.stop - 1}'

    if isinstance(client, botocore.client.BaseClient):
        resp = client.get_object(Bucket=utils.BUCKET, Key=Path(url).name, Range=range_header)
        body = resp['Body'].read()
    elif isinstance(client, requests.sessions.Session):
        resp = client.get(url, headers={'Range': range_header})
        body = resp.content

    body = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(body)
    return body


def json_to_metadata_entries(json_path: str) -> Iterable[utils.XmlMetadata]:
    """Convert json of XML metadata information to list of XmlMetadata objects.

    Args:
        json_path: path to json file containing xml metadata information

    Returns:
        list of XmlMetadata objects
    """
    with open(json_path, 'r') as json_file:
        metadata_dict = json.load(json_file)

    slc_name = list(metadata_dict.keys())[0]
    metadata_dict = metadata_dict[slc_name]
    xml_metadatas = []
    for key in metadata_dict:
        offset = utils.Offset(metadata_dict[key]['offset_start'], metadata_dict[key]['offset_stop'])
        xml_metadatas.append(utils.XmlMetadata(key, slc_name, offset))
    return xml_metadatas


def make_input_xml(manifest_name, metadata_names) -> lxml.etree._Element:
    root = lxml.etree.Element('files')

    for metadata_name in metadata_names:
        child = lxml.etree.SubElement(root, 'file')
        child.set('name', metadata_name)
        child.set('label', metadata_name)

    root.attrib['manifest'] = manifest_name

    return root


def extract_metadata(json_file_path: str, strategy='s3'):
    """Extract all xml metadata files from SLC in ASF archive
    using offset information.

    Args:
        json_file_path: path to json file containing extraction metadata
        strategy: strategy to use for download (s3 | http) s3 only
            works if runnning from us-west-2 region
    """
    metadatas = json_to_metadata_entries(json_file_path)
    slc_name = metadatas[0].slc
    url = utils.get_download_url(slc_name)
    offsets = [metadata.offset for metadata in metadatas]

    if strategy == 's3':
        creds = utils.get_credentials()
        client = boto3.client(
            "s3",
            aws_access_key_id=creds["accessKeyId"],
            aws_secret_access_key=creds["secretAccessKey"],
            aws_session_token=creds["sessionToken"],
        )
    elif strategy == 'http':
        client = requests.session()

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(extract_bytes, repeat(url), offsets, repeat(client)), total=len(offsets)))

    names = [metadata.name for metadata in metadatas]
    renderer = XsltRenderer(Path(__file__).parent / 'burst.xsl')
    burst_metadata_path = 'transformed.xml'
    renderer.render_to(burst_metadata_path, metadata_paths, manifest_path)
    # content = b''.join(results)
    # with open(f'{slc_name}.xml', 'wb') as f:
    #     f.write(content)


def main():
    """Example Command:

    extract_metadata.py S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_metadata.json
    """
    parser = ArgumentParser()
    parser.add_argument('metadata_path')
    args = parser.parse_args()

    extract_metadata(args.metadata_path)


if __name__ == '__main__':
    main()
