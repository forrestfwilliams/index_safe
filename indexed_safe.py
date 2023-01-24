import io
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import dataclass
from gzip import _create_simple_gzip_header
from itertools import chain
from pathlib import Path
from typing import Iterable

import indexed_gzip as igzip
import numpy as np
import pandas as pd
from isal import isal_zlib  # noqa

MAGIC_NUMBER = 28


@dataclass(frozen=True)
class CompressedFile:
    path: str
    offset: int
    length: int
    compress_type: int
    crc: int
    uncompressed_size: int


# TODO can also be np.array
@dataclass(frozen=True)
class Offset:
    start: int
    stop: int


@dataclass(frozen=True)
class Window:
    xstart: int
    ystart: int
    xend: int
    yend: int


@dataclass(frozen=True)
class Extraction:
    compressed_offset: Offset
    decompressed_offset: Offset


@dataclass(frozen=True)
class BurstEntry:
    name: str
    slc: str
    n_rows: int
    n_columns: int
    extraction_data: Extraction
    valid_window: Window

    def to_tuple(self):
        tuppled = (
            self.name,
            self.slc,
            self.n_rows,
            self.n_columns,
            self.extraction_data.compressed_offset.start,
            self.extraction_data.compressed_offset.stop,
            self.extraction_data.decompressed_offset.start,
            self.extraction_data.decompressed_offset.stop,
            self.valid_window.xstart,
            self.valid_window.xend,
            self.valid_window.ystart,
            self.valid_window.yend,
        )
        return tuppled


@dataclass(frozen=True)
class MetadataEntry:
    name: str
    slc: str
    offset: Offset

    def to_tuple(self):
        return (self.name, self.slc, self.offset.start, self.offset.stop)


def get_compressed_offset(zinfo: zipfile.ZipInfo):
    file_offset = len(zinfo.FileHeader()) + zinfo.header_offset + MAGIC_NUMBER - len(zinfo.extra)
    file_end = file_offset + zinfo.compress_size - 1
    return Offset(file_offset, file_end)


def wrap_as_gz(payload, zinfo: zipfile.ZipInfo):
    header = _create_simple_gzip_header(1)
    trailer = struct.pack("<LL", zinfo.CRC, (zinfo.file_size & 0xFFFFFFFF))
    gzip_wrapped = header + payload + trailer
    return gzip_wrapped


def build_index(file):
    with igzip.IndexedGzipFile(file) as f:
        f.build_full_index()
        seek_points = list(f.seek_points())

    array = np.array(seek_points)  # first column is uncompressed, second is compressed
    return array


def compute_valid_window(index: int, burst: ET.Element) -> Window:
    """Written by Jason Ninneman for the I&A team's burst extractor"""

    # all offsets, even invalid offsets
    offsets_range = Offset(
        np.array([int(val) for val in burst.find('firstValidSample').text.split()]),
        np.array([int(val) for val in burst.find('lastValidSample').text.split()]),
    )

    # returns the indices of lines containing valid data
    lines_with_valid_data = np.flatnonzero(offsets_range.stop - offsets_range.start)

    # get first and last sample with valid data per line
    # x-axis, range
    valid_offsets_range = Offset(
        offsets_range.start[lines_with_valid_data].min(),
        offsets_range.stop[lines_with_valid_data].max(),
    )

    # get the first and last line with valid data
    # y-axis, azimuth
    valid_offsets_azimuth = Offset(
        lines_with_valid_data.min(),
        lines_with_valid_data.max(),
    )

    # x-length
    length_range = valid_offsets_range.stop - valid_offsets_range.start
    # y-length
    length_azimuth = len(lines_with_valid_data)

    valid_window = Window(
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
    burst_offsets = [Offset(x, x + burst_lengths - 1) for x in burst_starts]
    burst_windows = [compute_valid_window(i, burst_xml) for i, burst_xml in enumerate(burst_xmls)]
    return burst_shape, burst_offsets, burst_windows


def get_extraction_offsets(index: np.ndarray, uncompressed_offsets: Iterable[Offset]):
    first_entry = index[index[:, 0].argmin()]
    header_size = first_entry[1] - first_entry[0]

    extractions = []
    for uncompressed_offset in uncompressed_offsets:
        less_than_start = index[index[:, 0] < uncompressed_offset.start]
        start_pair = less_than_start[less_than_start[:, 0].argmax()]

        greater_than_stop = index[index[:, 0] > uncompressed_offset.stop]
        stop_pair = greater_than_stop[greater_than_stop[:, 0].argmin()]

        decompress_offset = Offset(start_pair[1] - header_size, stop_pair[1] - header_size)
        data_offset = Offset(
            uncompressed_offset.start - start_pair[0],
            uncompressed_offset.stop - start_pair[0],
        )
        extractor = Extraction(decompress_offset, data_offset)
        extractions.append(extractor)

    return extractions


def create_metadata_entry(zipped_safe_path, zinfo):
    slc_name = Path(zipped_safe_path).with_suffix('').name
    name = Path(zinfo.filename).name
    compressed_offset = get_compressed_offset(zinfo)
    return MetadataEntry(name, slc_name, compressed_offset)


def create_burst_entries(zipped_safe_path, zinfo):
    slc_name = Path(zipped_safe_path).with_suffix('').name
    swath_offset = get_compressed_offset(zinfo)
    with open(zipped_safe_path, 'rb') as f:
        f.seek(swath_offset.start)
        deflate_content = f.read(swath_offset.stop - swath_offset.start)

    gz_content = wrap_as_gz(deflate_content, zinfo)
    compression_index = build_index(io.BytesIO(gz_content))

    burst_shape, burst_offsets, burst_windows = get_burst_annotation_data(zipped_safe_path, zinfo.filename)
    extractions = get_extraction_offsets(compression_index, burst_offsets)

    burst_entries = []
    for i, (extraction, burst_window) in enumerate(zip(extractions, burst_windows)):
        burst_entry = BurstEntry(f'{i}.tiff', slc_name, burst_shape[0], burst_shape[0], extraction, burst_window)
        burst_entries.append(burst_entry)
    return burst_entries


def save_as_csv(entries: Iterable[MetadataEntry | BurstEntry], out_name):
    if isinstance(entries[0], MetadataEntry):
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

    metadata_entries = [create_metadata_entry(slc_name, x) for x in xmls]
    save_as_csv(metadata_entries, 'metadata.csv')

    burst_entries = list(chain.from_iterable([create_burst_entries(zipped_safe_path, x) for x in tiffs[0:1]]))
    save_as_csv(burst_entries, 'bursts.csv')
    return None


if __name__ == '__main__':
    zip_filename = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    out = index_safe(zip_filename)

    print('done')
