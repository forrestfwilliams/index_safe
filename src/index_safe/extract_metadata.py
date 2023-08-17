import json
import zlib
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Iterable, Optional

import boto3
import botocore
import lxml.etree
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


def make_input_xml(metadata_paths: Iterable[Path], manifest_path: Path) -> lxml.etree._Element:
    """Create input xml for xslt transformation.
    Origionally written by Gabe Clark and Rohan Weeden of ASF's Ingest and archive team.

    Args:
        metadata_paths: list of paths to metadata xml files (annotations, calibration, and noise)
        manifest_path: path to manifest.safe file

    Returns:
        lxml.etree._Element representing input xml
    """
    root = lxml.etree.Element('files')

    for metadata_path in metadata_paths:
        child = lxml.etree.SubElement(root, 'file')
        child.set('name', str(metadata_path))
        child.set('label', metadata_path.name)

    root.attrib['manifest'] = str(manifest_path)

    return root


class XsltRenderer:
    """Class for rendering xml using xslt transformation.
    Origionally written by Jason Ninneman and Rohan Weeden of ASF's Ingest and archive team.
    """

    def __init__(self, template_path: Path | str):
        self.template_path = Path(template_path)

    def render_to(self, out_path: Path | str, metadata_paths: list[Path | str], manifest_path: Path | str):
        """Render xml using xslt transformation and save to out_path.

        Args:
            out_path: path to write rendered xml to
            metadata_paths: list of paths to metadata xml files (annotations, calibration, and noise)
            manifest_path: path to manifest.safe file
        """
        out_path = Path(out_path)
        metadata_paths = [Path(p) for p in metadata_paths]
        manifest_path = Path(manifest_path)

        input_xml = make_input_xml(metadata_paths, manifest_path)
        parser = lxml.etree.XMLParser(remove_blank_text=True)
        xslt_root = lxml.etree.parse(self.template_path, parser)

        transform = lxml.etree.XSLT(xslt_root)
        transformed = transform(input_xml)
        transformed.write(out_path, pretty_print=True)


def combine_xml_metadata_files(results_dict: dict, output_name='transformed.xml', directory: Optional[Path] = None):
    """Combine xml metadata files into single xml file using xslt transformation.

    Args:
        results_dict: dictionary where keys are metadata filenames (including manifest.safe)
                      and values are the downloaded xml data as bytes
        output_name: name of combined xml file
        directory: directory to write combined xml file to
    """
    if not directory:
        directory = Path.cwd()
    metadata_paths = []
    for name in results_dict:
        path = directory / name
        if 'manifest' in name:
            manifest_path = path
        else:
            metadata_paths.append(path)
        path.write_bytes(results_dict[name])

    renderer = XsltRenderer(Path(__file__).parent / 'burst.xsl')
    renderer.render_to(output_name, metadata_paths, manifest_path)
    [path.unlink() for path in metadata_paths + [manifest_path]]


def select_and_reorder_metadatas(
    metadatas: Iterable[utils.XmlMetadata], polarization: str
) -> Iterable[utils.XmlMetadata]:
    """Select correct polarization, then reorder XmlMetadata objects swath, then by product, noise, calibration.

    Args:
        metadatas: list of XmlMetadata objects
        polarization: polarization to select (vv | vh | hv | hh)

    Returns:
        list of sorted XmlMetadata objects
    """
    polarization = polarization.lower()
    manifest = [metadata for metadata in metadatas if 'manifest' in metadata.name][0]
    products = []
    noises = []
    calibrations = []
    for metadata in metadatas:
        if metadata.name.startswith('s1') and metadata.name.split('-')[3] == polarization:
            products.append(metadata)
        elif metadata.name.startswith('noise') and metadata.name.split('-')[4] == polarization:
            noises.append(metadata)
        elif metadata.name.startswith('calibration') and metadata.name.split('-')[4] == polarization:
            calibrations.append(metadata)

    products = sorted(products, key=lambda x: x.name.split('-')[1])
    noises = sorted(noises, key=lambda x: x.name.split('-')[2])
    calibrations = sorted(calibrations, key=lambda x: x.name.split('-')[2])

    metadatas = [item for sublist in zip(products, noises, calibrations) for item in sublist]
    metadatas.append(manifest)
    return metadatas


def extract_metadata(json_file_path: str, polarization: str, strategy='s3'):
    """Extract all xml metadata files from SLC in ASF archive
    using offset information.

    Args:
        json_file_path: path to json file containing extraction metadata
        polarization: polarization to extract (vv | vh | hv | hh)
        strategy: strategy to use for download (s3 | http) s3 only
            works if runnning from us-west-2 region
    """
    metadatas = json_to_metadata_entries(json_file_path)
    metadatas = select_and_reorder_metadatas(metadatas, polarization)
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
    results = {name: result for name, result in zip(names, results)}
    combine_xml_metadata_files(results)


def main():
    """Example Command:

    extract_metadata.py S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_metadata.json
    """
    parser = ArgumentParser()
    parser.add_argument('metadata_path')
    parser.add_argument('--polarization', default='vv')
    args = parser.parse_args()

    extract_metadata(args.metadata_path, args.polarization)


if __name__ == '__main__':
    main()
