import io
import os
import struct
import xml.etree.ElementTree as ET
import zipfile
from argparse import ArgumentParser
from gzip import _create_simple_gzip_header
from pathlib import Path
from typing import Iterable

import indexed_gzip as igzip
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import utils

KB = 1024
MB = 1024 * KB
GB = 1024 * MB
MAGIC_NUMBER = 28


def get_compressed_offset(zinfo: zipfile.ZipInfo) -> utils.Offset:
    """Get the byte offset (beginning byte and end byte inclusive)
    for a member of a zip archive. Currently relies on a
    "magic number" due some strangeness in zip archive structure
    """
    file_offset = len(zinfo.FileHeader()) + zinfo.header_offset + MAGIC_NUMBER - len(zinfo.extra)
    file_end = file_offset + zinfo.compress_size
    return utils.Offset(file_offset, file_end)


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


def build_index(file: io.BytesIO) -> np.ndarray:
    """Using the indexed_gzip library, identify
    seek points within an in-memory gzip file
    that are valid locations to decompress from
    using zlib. In ouput array, first column is
    the uncompressed byte location, and the
    second is the compressed byte location.

    Args:
        file: in-memory GZIP-formatted object

    Returns:
        Two-column numpy array containing
        matching seek points. The first column is
        the uncompressed byte location, and
        the second is the compressed byte location.
    """
    with igzip.IndexedGzipFile(file, readbuf_size=2 * GB) as f:
        f.build_full_index()
        seek_points = list(f.seek_points())

    array = np.array(seek_points)
    return array


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
        valid_offsets_azimuth.start,
        valid_offsets_range.start + length_range,
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


def get_extraction_offsets(
    swath_offset: int, index: np.ndarray, uncompressed_offsets: Iterable[utils.Offset]
) -> Iterable[utils.Offset]:
    """Find the compressed location closest to the burst where zlib can begin reading
    from and return the information needed to read the desired data.

    Args:
        swath_offset: byte offset where swath tiff begins in zip
        index: indexed_gzip index of uncompressed and compressed pairs
        uncompressed_offsets: offsets to burst relate to uncompressed swath tiff

    Returns:
        Pair of offsets to compressed data range and uncompressed range relative to the
        compressed range
    """
    first_entry = index[index[:, 0].argmin()]
    header_size = first_entry[1] - first_entry[0]
    compressed_data_offset = swath_offset - header_size

    offset_pairs = []
    # FIXME not always producing valid zlib decompress offsets
    for uncompressed_offset in uncompressed_offsets:
        less_than_start = index[index[:, 0] < uncompressed_offset.start]
        start_pair = less_than_start[less_than_start[:, 0].argmax()]
        start_compress = start_pair[1] + compressed_data_offset

        greater_than_stop = index[index[:, 0] > uncompressed_offset.stop]
        stop_pair = greater_than_stop[greater_than_stop[:, 0].argmin()]
        stop_compress = stop_pair[1] + compressed_data_offset

        decompress_offset = utils.Offset(start_compress, stop_compress)
        data_offset = utils.Offset(
            uncompressed_offset.start - start_pair[0],
            uncompressed_offset.stop - start_pair[0],
        )
        offset_pair = (decompress_offset, data_offset)
        offset_pairs.append(offset_pair)

    return offset_pairs


def create_xml_metadata(zipped_safe_path: str, zinfo: utils.OffsetZipInfo) -> utils.XmlMetadata:
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
    # compressed_offset = get_compressed_offset(zinfo)
    return utils.XmlMetadata(name, slc_name, zinfo.compressed_offset)


def create_burst_name(slc_name: str, swath_name: str, burst_index: str) -> str:
    """Create name for a burst tiff

    Args:
        slc_name: Name of SLC
        swath_name: Name of swath
        burst_index: Zero-indexed burst number in swath

    Returns:
        Name of burst
    """
    slc_parts = slc_name.split('_')[:7]
    _, swath, _, polarization, *_ = swath_name.split('-')
    all_parts = slc_parts + [swath.upper(), polarization.upper(), str(burst_index)]
    return '_'.join(all_parts) + '.tiff'


def create_burst_metadatas(zipped_safe_path: str, zinfo: utils.OffsetZipInfo) -> Iterable[utils.BurstMetadata]:
    """Create objects containing information needed to download burst tiff from compressed file directly,
    and remove invalid data, for a swath tiff.

    Args:
        zipped_safe_path: Path to zipped SAFE
        zinfo: ZipInfo object for desired XML

    Returns:
        BurstMetadata objects containing information needed to download and remove invalid data
    """
    slc_name = Path(zipped_safe_path).with_suffix('').name
    # swath_offset = get_compressed_offset(zinfo)
    swath_offset = zinfo.compressed_offset
    with open(zipped_safe_path, 'rb') as f:
        f.seek(swath_offset.start)
        deflate_content = f.read(swath_offset.stop - swath_offset.start)

    gz_content = wrap_as_gz(deflate_content, zinfo)
    compression_index = build_index(io.BytesIO(gz_content))

    burst_shape, burst_offsets, burst_windows = get_burst_annotation_data(zipped_safe_path, zinfo.filename)
    offset_pairs = get_extraction_offsets(swath_offset.start, compression_index, burst_offsets)

    bursts = []
    for i, (offset_pair, burst_window) in enumerate(zip(offset_pairs, burst_windows)):
        burst_name = create_burst_name(slc_name, zinfo.filename, i)
        burst = utils.BurstMetadata(burst_name, slc_name, burst_shape, offset_pair[0], offset_pair[1], burst_window)
        bursts.append(burst)
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
            'download_start',
            'download_stop',
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


def index_safe(slc_name: str, keep: bool = True):
    """Create the index and other metadata needed to directly download
    and correctly format burst tiffs/metadata Sentinel-1 SAFE zip. Save
    this information in csv files.

    Args:
        slc_name: Scene name to index
        keep: If False, delete SLC zip after indexing

    Returns:
        No function outputs, but saves a metadata.csv and burst.csv to
        the working directory
    """
    zipped_safe_path = f'{slc_name}.zip'
    if not Path(zipped_safe_path).exists():
        print('Downloading SLC...')
        utils.download_slc(slc_name)
    else:
        print('SLC exists locally, skipping download')

    with zipfile.ZipFile(zipped_safe_path) as f:
        tiffs = [x for x in f.infolist() if 'tiff' in Path(x.filename).name]
        xmls = [x for x in f.infolist() if 'xml' in Path(x.filename).name]

    tiffs = [utils.OffsetZipInfo(zipped_safe_path, x) for x in tiffs]
    xmls = [utils.OffsetZipInfo(zipped_safe_path, x) for x in tiffs]

    print('Reading XMLs...')
    xml_metadatas = [create_xml_metadata(slc_name, x) for x in tqdm(xmls)]
    save_as_csv(xml_metadatas, 'metadata.csv')

    print('Reading Bursts...')
    burst_metadatas = []
    for tiff in tqdm(tiffs):
        burst_metadata = create_burst_metadatas(zipped_safe_path, tiff)
        burst_metadatas = burst_metadatas + burst_metadata

    save_as_csv(burst_metadatas, 'bursts.csv')
    if not keep:
        os.remove(zipped_safe_path)


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
