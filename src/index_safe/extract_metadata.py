import io
import json
import zlib
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import boto3
import lxml.etree as ET
import requests


try:
    from index_safe import utils
except ModuleNotFoundError:
    import utils

KB = 1024
MB = 1024 * KB
MAX_WBITS = 15


def extract_metadata_xml(
    url: str,
    offset: utils.Offset,
    client: Union[boto3.client, requests.sessions.Session],
    range_get_func: Callable,
) -> ET._Element:
    """Extract and decompress bytes pertaining to a metadata XML file from a Sentinel-1 SLC archive.

    Args:
        url: url location of SLC archive
        offset: offset for compressed data range in zip archive
        client: boto3 S3 client or requests session
        range_get_func: function to use to get a range of bytes from SLC archive

    Returns:
        bytes representing metadata xml
    """
    if offset.stop <= offset.start:
        raise ValueError('offset stop must be greater than offset start')
    elif offset.start < 0 or offset.stop < 0:
        raise ValueError('offset stop and offset start must be greater than 0')

    xml_range = f'bytes={offset.start}-{offset.stop-1}'
    compressed_bytes = range_get_func(client, url, xml_range)
    uncompressed_bytes = zlib.decompressobj(-1 * zlib.MAX_WBITS).decompress(compressed_bytes)
    metadata_xml = ET.parse(io.BytesIO(uncompressed_bytes))
    return metadata_xml


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


def make_input_xml(metadata_paths: Iterable[Path], manifest_path: Path) -> ET._Element:
    """Create input xml for xslt transformation.
    Origionally written by Gabe Clark and Rohan Weeden of ASF's Ingest and archive team.

    Args:
        metadata_paths: list of paths to metadata xml files (annotations, calibration, and noise)
        manifest_path: path to manifest.safe file

    Returns:
        lxml.etree._Element representing input xml
    """
    root = ET.Element('files')

    for metadata_path in metadata_paths:
        child = ET.SubElement(root, 'file')
        child.set('name', str(metadata_path))
        child.set('label', metadata_path.name)

    root.attrib['manifest'] = str(manifest_path)

    return root


class XsltRenderer:
    """Class for rendering xml using xslt transformation.
    Origionally written by Jason Ninneman and Rohan Weeden of ASF's Ingest and archive team.
    """

    def __init__(self, template_path: Union[Path, str]):
        self.template_path = Path(template_path)

    def render_to(
        self, out_path: Union[Path, str], metadata_paths: Iterable[Union[Path, str]], manifest_path: Union[Path, str]
    ):
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
        parser = ET.XMLParser(remove_blank_text=True)
        xslt_root = ET.parse(self.template_path, parser)

        transform = ET.XSLT(xslt_root)
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
        xml_string = ET.tostring(results_dict[name], pretty_print=True, encoding='utf-8', xml_declaration=True)
        path.write_bytes(xml_string)

    renderer = XsltRenderer(Path(__file__).parent / 'burst.xsl')
    renderer.render_to(directory / output_name, metadata_paths, manifest_path)
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


def extract_metadata(json_file_path: str, polarization: str, strategy='s3', working_dir: Optional[Path] = None):
    """Extract all xml metadata files from SLC in ASF archive
    using offset information.

    Args:
        json_file_path: path to json file containing extraction metadata
        polarization: polarization to extract (vv | vh | hv | hh)
        strategy: strategy to use for download (s3 | http) s3 only
            works if runnning from us-west-2 region
    """
    if not working_dir:
        working_dir = Path.cwd()

    metadatas = json_to_metadata_entries(json_file_path)
    metadatas = select_and_reorder_metadatas(metadatas, polarization)
    slc_name = metadatas[0].slc
    url = utils.get_download_url(slc_name)
    offsets = [metadata.offset for metadata in metadatas]

    client, range_get_func = utils.setup_download_client(strategy=strategy)
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(extract_metadata_xml, repeat(url), offsets, repeat(client), repeat(range_get_func))

    names = [metadata.name for metadata in metadatas]
    results = {name: result for name, result in zip(names, results)}
    combine_xml_metadata_files(results, directory=working_dir)


def main():
    """Example Command:

    extract_metadata.py S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_metadata.json
    """
    parser = ArgumentParser()
    parser.add_argument('metadata_path')
    parser.add_argument('polarization')
    args = parser.parse_args()

    extract_metadata(args.metadata_path, args.polarization)


if __name__ == '__main__':
    main()
