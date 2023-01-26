from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

import requests
from tqdm import tqdm

KB = 1024
MB = 1024 * KB
GB = 1024 * MB
SENTINEL_DISTRIBUTION_URL = 'https://sentinel1.asf.alaska.edu'


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


def get_download_url(scene: str) -> str:
    """Get download url for Sentinel-1 Scene

    Args:
        scene: scene name
    Returns:
        Scene url
    """
    mission = scene[0] + scene[2]
    product_type = scene[7:10]
    if product_type == 'GRD':
        product_type += '_' + scene[10] + scene[14]
    url = f'{SENTINEL_DISTRIBUTION_URL}/{product_type}/{mission}/{scene}.zip'
    return url


def download_slc(scene: str, chunk_size=10 * MB) -> str:
    """Download a file

    Args:
        url: URL of the file to download
        directory: Directory location to place files into
        chunk_size: Size to chunk the download into
    Returns:
        download_path: The path to the downloaded file
    """
    url = get_download_url(scene)
    download_path = Path(url).name
    session = requests.Session()
    with session.get(url, stream=True) as s:
        s.raise_for_status()
        total = int(s.headers.get('content-length', 0))
        with open(download_path, "wb") as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=total) as pbar:
                for chunk in s.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    session.close()
    return url
