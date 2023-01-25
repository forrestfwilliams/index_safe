from dataclasses import dataclass
from typing import Iterable

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
class BurstMetadata:
    name: str
    slc: str
    shape: Iterable[int]  # n_row, n_column
    compressed_offset: Offset
    decompressed_offset: Offset
    valid_window: Window

    def to_tuple(self):
        tuppled = (
            self.name,
            self.slc,
            self.shape[0],
            self.shape[1],
            self.compressed_offset.start,
            self.compressed_offset.stop,
            self.decompressed_offset.start,
            self.decompressed_offset.stop,
            self.valid_window.xstart,
            self.valid_window.xend,
            self.valid_window.ystart,
            self.valid_window.yend,
        )
        return tuppled


@dataclass(frozen=True)
class XmlMetadata:
    name: str
    slc: str
    offset: Offset

    def to_tuple(self):
        return (self.name, self.slc, self.offset.start, self.offset.stop)
