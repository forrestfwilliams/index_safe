import base64
import io
import json
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from argparse import ArgumentParser
from collections import ChainMap
from pathlib import Path
from typing import Iterable, Union

import boto3
from tqdm import tqdm


try:
    from index_safe import utils
except ModuleNotFoundError:
    import utils


def get_burst_annotation_data(zipped_safe_path: str, swath_path: str) -> Iterable:
    """Obtain the information needed to extract a burst that is contained within
    a SAFE annotation XML.

    Args:
        zipped_safe_path: path to a SAFE zip containing the annotation XML
        swath_path: The within the zip path to the swath tiff you need the
            annotation XML for

    Returns:
        burst_shape: numpy-style tuple of burst array size (n rows, n columns)
        burst_offsets: uncompressed byte offsets for the bursts contained within
            a swath
    """
    swath_path = Path(swath_path)
    annotation_path = swath_path.parent.parent / 'annotation' / swath_path.with_suffix('.xml').name
    with zipfile.ZipFile(zipped_safe_path) as f:
        annotation_bytes = f.read(str(annotation_path))

    xml = ET.parse(io.BytesIO(annotation_bytes)).getroot()
    burst_xmls = xml.findall('.//{*}burst')
    n_lines = int(xml.findtext('.//{*}linesPerBurst'))
    n_samples = int(xml.findtext('.//{*}samplesPerBurst'))
    burst_shape = (n_lines, n_samples)  #  y, x for numpy
    burst_starts = [int(x.findtext('.//{*}byteOffset')) for x in burst_xmls]
    burst_lengths = burst_starts[1] - burst_starts[0]
    burst_offsets = [utils.Offset(x, x + burst_lengths) for x in burst_starts]
    return burst_shape, burst_offsets


def create_xml_metadata(zipped_safe_path: str, zinfo: zipfile.ZipInfo) -> utils.XmlMetadata:
    """Create object containing information needed to download metadata XML file from
    compressed file directly.

    Args:
        zipped_safe_path: Path to zipped SAFE
        zinfo: ZipInfo object for desired XML

    Returns:
        XmlMetadata object containing offsets needed to download XML
    """
    slc_name = Path(zipped_safe_path).with_suffix('').name
    name = Path(zinfo.filename).name
    offset = utils.get_zip_compressed_offset(zipped_safe_path, zinfo)
    return utils.XmlMetadata(name, slc_name, offset)


def create_burst_name(slc_name: str, swath_name: str, burst_index: str) -> str:
    """Create name for a burst tiff.

    Args:
        slc_name: Name of SLC
        swath_name: Name of swath
        burst_index: Zero-indexed burst number in swath

    Returns:
        Name of burst
    """
    _, swath, _, polarization, *_ = swath_name.split('-')
    all_parts = [slc_name, swath.upper(), polarization.upper(), str(burst_index)]
    return '_'.join(all_parts) + '.tiff'


def create_index(
    zipped_safe_path: str, xml_metadatas: Iterable[utils.XmlMetadata], zinfo: zipfile.ZipInfo, output_json: bool = True
) -> Union[dict, bytes]:
    """Create a burst-specific index containing information needed to download burst tiff from compressed
    SAFE file directly, and remove invalid data.

    Args:
        zipped_safe_path: Path to zipped SAFE
        xml_metadatas: List of XmlMetadata objects for all XML files in SAFE
        zinfo: ZipInfo object for desired XML
        output_json: Whether to output index as json or bytes

    Returns:
        dictionary or bytes object containing information needed to download and remove invalid data
    """
    slc_name = Path(zipped_safe_path).with_suffix('').name
    tiff_name = Path(zinfo.filename).name
    swath_name = Path(zinfo.filename).name.split('-')[1].upper()
    annotation_name = Path(zinfo.filename).with_suffix('.xml').name

    annotation_offset = [item for item in xml_metadatas if item.name == annotation_name][0].offset
    manifest_offset = [item for item in xml_metadatas if item.name == 'manifest.safe'][0].offset
    burst_shape, burst_offsets = get_burst_annotation_data(zipped_safe_path, zinfo.filename)

    bursts = {}
    indexer = utils.ZipIndexer(zipped_safe_path, tiff_name)
    indexer.create_full_dflidx()
    for burst_index, burst_offset in enumerate(burst_offsets):
        burst_name = create_burst_name(slc_name, zinfo.filename, burst_index)

        compressed_offset, uncompressed_offset, modified_index = indexer.subset_dflidx(
            locations=[burst_offset.start], end_location=burst_offset.stop
        )
        dflidx = modified_index.create_index_file()

        index_burst_offset = utils.Offset(
            burst_offset.start - uncompressed_offset.start, burst_offset.stop - uncompressed_offset.start
        )

        burst = utils.BurstMetadata(
            burst_name,
            slc_name,
            swath_name,
            burst_index,
            burst_shape,
            compressed_offset,
            index_burst_offset,
            annotation_offset,
            manifest_offset,
        )

        if output_json:
            bstidx_name = Path(burst_name).with_suffix('.json').name
            burst_dictionary = burst.to_dict()
            burst_dictionary['dflidx_b64'] = base64.b64encode(dflidx).decode('utf-8')
            bursts[bstidx_name] = burst_dictionary
        else:
            bstidx_name = Path(burst_name).with_suffix('.bstidx').name
            bursts[bstidx_name] = burst.to_bytes() + dflidx
    return bursts


def save_xml_metadata_as_json(entries: Iterable[utils.XmlMetadata], out_name: str) -> str:
    """Save a list of XmlMetadata objects as a json.

    Args:
        entries: List of metadata objects to be included
        out_name: Path/name to save json at

    Returns:
        Path/name where json was saved
    """
    slc = entries[0].slc
    metadata_dicts = [entry.to_dict()[slc] for entry in entries]
    combined_dict = {}
    for metadata_dict in metadata_dicts:
        combined_dict.update(metadata_dict)

    with open(out_name, 'w') as json_file:
        json.dump({slc: combined_dict}, json_file)
    return out_name


def get_indexes(zipped_safe_path: Path):
    """Get indexes for XML and bursts in zipped SAFE.

    Args:
        zipped_safe_path: Path to zipped SAFE

    Returns: tuple of lists of XmlMetadata and BurstMetadata objects

    """
    with zipfile.ZipFile(zipped_safe_path) as f:
        tiffs = [x for x in f.infolist() if 'tiff' in Path(x.filename).name]
        xmls = [x for x in f.infolist() if 'xml' in Path(x.filename).name]
        xmls += [x for x in f.infolist() if 'manifest.safe' == Path(x.filename).name]

        non_deflate_tiffs = [tiff for tiff in tiffs if tiff.compress_type != zipfile.ZIP_DEFLATED]
        non_deflate_xmls = [xml for xml in xmls if xml.compress_type != zipfile.ZIP_DEFLATED]
        non_deflate = non_deflate_tiffs + non_deflate_xmls
        if non_deflate:
            raise ValueError(
                'Non-deflate compressed files found in SAFE.'
                'All files must be compressed using the deflate method before indexing.'
            )

    print('Reading XMLs...')
    xml_metadatas = [create_xml_metadata(zipped_safe_path, x) for x in tqdm(xmls)]

    print('Reading Bursts...')
    burst_metadatas = dict(ChainMap(*[create_index(zipped_safe_path, xml_metadatas, x) for x in tqdm(tiffs)]))
    return xml_metadatas, burst_metadatas


def index_safe(slc_name: str, edl_token: str = None, working_dir='.', output_json: bool = True, keep: bool = True):
    """Create the index and other metadata needed to directly download
    and correctly format burst tiffs/metadata Sentinel-1 SAFE zip. Save
    this information in json files. All information for extracting a burst is included in
    the index file.

    Args:
        slc_name: Scene name to index
        edl_token: token for earth data login access
        working_dir: Directory to save metadata and index files
        output_json: Whether to output index as json or bytes
        keep: If False, delete SLC zip after indexing

    Returns:
        No function outputs, but saves a metadata.json, burst indexes to file
    """
    absolute_dir = Path(working_dir).resolve()
    zipped_safe_path = absolute_dir / f'{slc_name}.zip'
    if not zipped_safe_path.exists():
        print('Downloading SLC...')
        utils.download_slc(slc_name, edl_token, working_dir=absolute_dir)
    else:
        print('SLC exists locally, skipping download')

    xml_metadatas, burst_metadatas = get_indexes(zipped_safe_path)

    save_xml_metadata_as_json(xml_metadatas, str(absolute_dir / f'{slc_name}_metadata.json'))

    if output_json:
        for key, value in burst_metadatas.items():
            with open(key, 'w') as json_file:
                json.dump(value, json_file)
    else:
        [(absolute_dir / key).write_bytes(value) for key, value in burst_metadatas.items()]

    if not keep:
        os.remove(zipped_safe_path)


def lambda_handler(event, context):
    print('## ENVIRONMENT VARIABLES')
    print(os.environ)
    print('## EVENT')
    print(event)
    print('## PROCESS BEGIN...')
    s3 = boto3.client('s3')
    bucket_name = os.environ.get('IndexBucketName')
    with tempfile.TemporaryDirectory() as tmpdirname:
        index_safe(event['scene'], event['edl_token'], working_dir=tmpdirname)
        indexes = Path(tmpdirname).glob('*.bstidx')
        [s3.upload_file(str(x), bucket_name, x.name) for x in indexes]
    print('## PROCESS COMPLETE!')


def main():
    """Example Command:

    index_safe.py S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85
    """
    parser = ArgumentParser()
    parser.add_argument('scene')
    args = parser.parse_args()
    index_safe(args.scene)


if __name__ == '__main__':
    main()
