import io
import struct
import xml.etree.ElementTree as ET
import zipfile
from gzip import _create_simple_gzip_header
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

import indexed_gzip as igzip
import numpy as np
import pandas as pd

import data

KB = 1024
MB = 1024 * KB
GB = 1024 * MB
MAGIC_NUMBER = 28

def get_compressed_offset(zinfo: zipfile.ZipInfo):
    file_offset = len(zinfo.FileHeader()) + zinfo.header_offset + MAGIC_NUMBER - len(zinfo.extra)
    file_end = file_offset + zinfo.compress_size - 1
    return data.Offset(file_offset, file_end)


def wrap_as_gz(payload, zinfo: zipfile.ZipInfo):
    header = _create_simple_gzip_header(1)
    trailer = struct.pack("<LL", zinfo.CRC, (zinfo.file_size & 0xFFFFFFFF))
    gz_wrapped = header + payload + trailer
    return gz_wrapped


def build_index(file):
    with igzip.IndexedGzipFile(file, spacing=5*MB, readbuf_size=2*GB) as f:
        f.build_full_index()
        seek_points = list(f.seek_points())

    array = np.array(seek_points)  # first column is uncompressed, second is compressed
    return array


def compute_valid_window(index: int, burst: ET.Element) -> data.Window:
    """Written by Jason Ninneman for the I&A team's burst extractor"""

    # all offsets, even invalid offsets
    offsets_range = data.Offset(
        np.array([int(val) for val in burst.find('firstValidSample').text.split()]),
        np.array([int(val) for val in burst.find('lastValidSample').text.split()]),
    )

    # returns the indices of lines containing valid data
    lines_with_valid_data = np.flatnonzero(offsets_range.stop - offsets_range.start)

    # get first and last sample with valid data per line
    # x-axis, range
    valid_offsets_range = data.Offset(
        offsets_range.start[lines_with_valid_data].min(),
        offsets_range.stop[lines_with_valid_data].max(),
    )

    # get the first and last line with valid data
    # y-axis, azimuth
    valid_offsets_azimuth = data.Offset(
        lines_with_valid_data.min(),
        lines_with_valid_data.max(),
    )

    # x-length
    length_range = valid_offsets_range.stop - valid_offsets_range.start
    # y-length
    length_azimuth = len(lines_with_valid_data)

    valid_window = data.Window(
        valid_offsets_range.start,
        valid_offsets_azimuth.start,
        valid_offsets_range.start + length_range,
        valid_offsets_azimuth.start + length_azimuth,
    )

    return valid_window


def get_burst_annotation_data(zipped_safe_path, swath_path):
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
    burst_offsets = [data.Offset(x, x + burst_lengths) for x in burst_starts]
    burst_windows = [compute_valid_window(i, burst_xml) for i, burst_xml in enumerate(burst_xmls)]
    return burst_shape, burst_offsets, burst_windows


def get_extraction_offsets(swath_offset: int, index: np.ndarray, uncompressed_offsets: Iterable[data.Offset]):
    first_entry = index[index[:, 0].argmin()]
    header_size = first_entry[1] - first_entry[0]
    compressed_data_offset = swath_offset - header_size

    offset_pairs = []
    for uncompressed_offset in uncompressed_offsets:
        less_than_start = index[index[:, 0] < uncompressed_offset.start]
        start_pair = less_than_start[less_than_start[:, 0].argmax()]
        start_compress = start_pair[1] + compressed_data_offset

        greater_than_stop = index[index[:, 0] > uncompressed_offset.stop]
        stop_pair = greater_than_stop[greater_than_stop[:, 0].argmin()]
        stop_compress = stop_pair[1] + compressed_data_offset

        decompress_offset = data.Offset(start_compress, stop_compress)
        data_offset = data.Offset(
            uncompressed_offset.start - start_pair[0],
            uncompressed_offset.stop - start_pair[0],
        )
        offset_pair = (decompress_offset, data_offset)
        offset_pairs.append(offset_pair)

    return offset_pairs


def create_xml_metadata(zipped_safe_path, zinfo):
    slc_name = Path(zipped_safe_path).with_suffix('').name
    name = Path(zinfo.filename).name
    compressed_offset = get_compressed_offset(zinfo)
    return data.XmlMetadata(name, slc_name, compressed_offset)


def create_burst_name(slc_name, swath_name, burst_index):
    slc_parts = slc_name.split('_')[:7]
    _, swath, _, polarization, *_ = swath_name.split('-')
    all_parts = slc_parts + [swath.upper(), polarization.upper(), str(burst_index)]
    return '_'.join(all_parts) + '.tiff'


def create_burst_metadatas(zipped_safe_path, zinfo):
    slc_name = Path(zipped_safe_path).with_suffix('').name
    swath_offset = get_compressed_offset(zinfo)
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
        burst = data.BurstMetadata(burst_name, slc_name, burst_shape, offset_pair[0], offset_pair[1], burst_window)
        bursts.append(burst)
    return bursts


def save_as_csv(entries: Iterable[data.XmlMetadata | data.BurstMetadata], out_name):
    if isinstance(entries[0], data.XmlMetadata):
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


def index_safe(zipped_safe_path):
    slc_name = Path(zipped_safe_path).with_suffix('').name
    with zipfile.ZipFile(zipped_safe_path) as f:
        tiffs = [x for x in f.infolist() if 'tiff' in Path(x.filename).name]
        xmls = [x for x in f.infolist() if 'xml' in Path(x.filename).name]

    print('Reading XMLs...')
    xml_metadatas = [create_xml_metadata(slc_name, x) for x in tqdm(xmls)]
    save_as_csv(xml_metadatas, 'metadata.csv')

    print('Reading Bursts...')
    burst_metadatas = []
    # FIXME 3 fails in index step
    # tiffs = tiffs[0:2] + tiffs[3:]
    for tiff in tqdm(tiffs):
        burst_metadata = create_burst_metadatas(zipped_safe_path, tiff)
        burst_metadatas = burst_metadatas + burst_metadata
    save_as_csv(burst_metadatas, 'bursts.csv')
    return None


if __name__ == '__main__':
    zip_filename = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    out = index_safe(zip_filename)
    print('Done!')
