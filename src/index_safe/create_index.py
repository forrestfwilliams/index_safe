import io
import json
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from argparse import ArgumentParser
from collections import ChainMap
from pathlib import Path
from typing import Iterable

import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm


try:
    from index_safe import utils
except ModuleNotFoundError:
    import utils


def compute_valid_window(index: int, burst: ET.Element) -> utils.Window:
    """Written by Jason Ninneman for the ASF I&A team's burst extractor.
    Using the information contained within a SAFE annotation XML burst
    element, identify the window of a burst that contains valid data.

    Args:
        index: zero-indexed burst number to compute the valid window for
        burst: <burst> element of annotation XML for the given index

    Returns:
        Row and column ranges for the valid data within a burst
    """
    # all offsets, even invalid offsets
    offsets_range = utils.Offset(
        np.array([int(val) for val in burst.find('firstValidSample').text.split()]),
        np.array([int(val) for val in burst.find('lastValidSample').text.split()]),
    )

    # returns the indices of lines containing valid data
    lines_with_valid_data = np.flatnonzero(offsets_range.stop - offsets_range.start)

    # get first and last sample with valid data per line
    # x-axis, range
    valid_offsets_range = utils.Offset(
        offsets_range.start[lines_with_valid_data].min(),
        offsets_range.stop[lines_with_valid_data].max(),
    )

    # get the first and last line with valid data
    # y-axis, azimuth
    valid_offsets_azimuth = utils.Offset(
        lines_with_valid_data.min(),
        lines_with_valid_data.max(),
    )

    # x-length
    length_range = valid_offsets_range.stop - valid_offsets_range.start
    # y-length
    length_azimuth = len(lines_with_valid_data)

    valid_window = utils.Window(
        valid_offsets_range.start,
        valid_offsets_range.start + length_range,
        valid_offsets_azimuth.start,
        valid_offsets_azimuth.start + length_azimuth,
    )

    return valid_window


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
        burst_windows: row and column ranges for the valid data within a burst
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
    burst_windows = [compute_valid_window(i, burst_xml) for i, burst_xml in enumerate(burst_xmls)]
    return burst_shape, burst_offsets, burst_windows


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


def create_index(zipped_safe_path: str, zinfo: zipfile.ZipInfo) -> Iterable[bytes]:
    """Create a burst-specificindex containing information needed to download burst tiff from compressed
    SAFE file directly, and remove invalid data.

    Args:
        zipped_safe_path: Path to zipped SAFE
        zinfo: ZipInfo object for desired XML

    Returns:
        byte objects containing information needed to download and remove invalid data
    """
    slc_name = Path(zipped_safe_path).with_suffix('').name
    tiff_name = Path(zinfo.filename).name
    burst_shape, burst_offsets, burst_windows = get_burst_annotation_data(zipped_safe_path, zinfo.filename)

    bursts = {}
    indexer = utils.ZipIndexer(zipped_safe_path, tiff_name)
    indexer.create_full_dflidx()
    for i, (burst_offset, burst_window) in enumerate(zip(burst_offsets, burst_windows)):
        burst_name = create_burst_name(slc_name, zinfo.filename, i)
        bstidx_name = Path(burst_name).with_suffix('.bstidx').name

        compressed_offset, uncompressed_offset, modified_index = indexer.subset_dflidx(
            starts=[burst_offset.start], stops=[burst_offset.stop]
        )
        dflidx = modified_index.create_index_file()

        index_burst_offset = utils.Offset(
            burst_offset.start - uncompressed_offset.start, burst_offset.stop - uncompressed_offset.start
        )

        burst = utils.BurstMetadata(
            burst_name, slc_name, burst_shape, compressed_offset, index_burst_offset, burst_window
        )

        bursts[bstidx_name] = burst.to_bytes() + dflidx
    return bursts


def save_as_csv(entries: Iterable[utils.XmlMetadata | utils.BurstMetadata], out_name: str) -> str:
    """Save a list of metadata objects as a csv.

    Args:
        entries: List of metadata objects to be included
        out_name: Path/name to save csv at

    Returns:
        Path/name where csv was saved
    """
    if isinstance(entries[0], utils.XmlMetadata):
        columns = ['name', 'slc', 'offset_start', 'offset_stop']
    else:
        columns = [
            'name',
            'slc',
            'n_rows',
            'n_columns',
            'index_start',
            'index_stop',
            'offset_start',
            'offset_stop',
            'valid_x_start',
            'valid_x_stop',
            'valid_y_start',
            'valid_y_stop',
        ]

    df = pd.DataFrame([x.to_tuple() for x in entries], columns=columns)
    df.to_csv(out_name, index=False)
    return out_name


def save_as_json(entries: Iterable[utils.XmlMetadata | utils.BurstMetadata], out_name: str) -> str:
    slc = entries[0].slc
    metadata_dicts = [entry.to_dict()[slc] for entry in entries]
    combined_dict = {}
    for metadata_dict in metadata_dicts:
        combined_dict.update(metadata_dict)

    with open(out_name, 'w') as json_file:
        json.dump({slc: combined_dict}, json_file)
    return out_name


def index_safe(slc_name: str, edl_token: str = None, working_dir='.', keep: bool = True):
    """Create the index and other metadata needed to directly download
    and correctly format burst tiffs/metadata Sentinel-1 SAFE zip. Save
    this information in csv files. All information for extracting a burst is included in
    the index file.

    Args:
        slc_name: Scene name to index
        edl_token: token for earth data login access
        keep: If False, delete SLC zip after indexing

    Returns:
        No function outputs, but saves a metadata.csv, burst indexes to file
    """
    absolute_dir = Path(working_dir).resolve()
    zipped_safe_path = absolute_dir / f'{slc_name}.zip'
    if not zipped_safe_path.exists():
        print('Downloading SLC...')
        utils.download_slc(slc_name, edl_token, working_dir=absolute_dir)
    else:
        print('SLC exists locally, skipping download')

    with zipfile.ZipFile(zipped_safe_path) as f:
        tiffs = [x for x in f.infolist() if 'tiff' in Path(x.filename).name]
        xmls = [x for x in f.infolist() if 'xml' in Path(x.filename).name]
        xmls += [x for x in f.infolist() if 'manifest.safe' == Path(x.filename).name]

    print('Reading XMLs...')
    xml_metadatas = [create_xml_metadata(zipped_safe_path, x) for x in tqdm(xmls)]
    save_as_json(xml_metadatas, absolute_dir / 'metadata.json')

    # print('Reading Bursts...')
    # burst_metadatas = dict(ChainMap(*[create_index(zipped_safe_path, x) for x in tqdm(tiffs)]))
    # [(absolute_dir / key).write_bytes(value) for key, value in burst_metadatas.items()]

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
    # index_safe(args.scene, working_dir = 'tmpdir')


if __name__ == '__main__':
    main()
