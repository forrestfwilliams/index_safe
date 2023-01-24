from dataclasses import dataclass


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
