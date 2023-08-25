import os
import random
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
import lxml.etree as ET

SIMPLIFIED_ANNOTATION = b"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <swathTiming>
    <linesPerBurst>1000</linesPerBurst>
    <samplesPerBurst>20000</samplesPerBurst>
    <burstList count="2">
      <burst>
        <azimuthTime>2022-01-01T00:00:00.000000</azimuthTime>
        <azimuthAnxTime>1.00e+03</azimuthAnxTime>
        <sensingTime>2022-00-00T00:00:00.000000</sensingTime>
        <byteOffset>10</byteOffset>
        <firstValidSample count="9">-1 -1 10 10 10 10 10 -1 -1</firstValidSample>
        <lastValidSample count="9">-1 -1 100 100 100 100 100 -1 -1</lastValidSample>
      </burst>
      <burst>
        <azimuthTime>2022-01-01T00:00:01.000000</azimuthTime>
        <azimuthAnxTime>2.00+03</azimuthAnxTime>
        <sensingTime>2022-01-01T00:00:01.000000</sensingTime>
        <byteOffset>110</byteOffset>
        <firstValidSample count="9">-1 -1 10 10 10 10 10 -1 -1</firstValidSample>
        <lastValidSample count="9">-1 -1 100 100 100 100 100 -1 -1</lastValidSample>
      </burst>
    </burstList>
  </swathTiming>
  <geolocationGrid>
    <geolocationGridPointList count="2">
      <geolocationGridPoint>
        <azimuthTime>2022-01-01T00:00:00.000000</azimuthTime>
        <slantRangeTime>5.00e-03</slantRangeTime>
        <line>0</line>
        <pixel>0</pixel>
        <latitude>2.00e+01</latitude>
        <longitude>5.00e+01</longitude>
        <height>1.00e+03</height>
      </geolocationGridPoint>
      <geolocationGridPoint>
        <azimuthTime>2020-06-04T02:22:53.625336</azimuthTime>
        <slantRangeTime>5.5e-03</slantRangeTime>
        <line>0</line>
        <pixel>1273</pixel>
        <latitude>2.50e+01</latitude>
        <longitude>5.50e+01</longitude>
        <height>1.5e+03</height>
      </geolocationGridPoint>
    </geolocationGridPointList>
  </geolocationGrid>
</product>
"""


@pytest.fixture()
def annotation_xml():
    return ET.fromstring(SIMPLIFIED_ANNOTATION)


@pytest.fixture()
def example_credentials():
    expire_time = datetime.now(timezone.utc) + timedelta(hours=1)
    example_credentials = {
        'accessKeyId': 'foo',
        'secretAccessKey': 'bar',
        'sessionToken': 'baz',
        'expiration': expire_time.isoformat(),
    }
    yield example_credentials


@pytest.fixture()
def zip_stored(tmpdir):
    file_path = Path(tmpdir) / 'test_file.txt'
    file_path.write_text('Hello World!')
    zip_path = Path(tmpdir) / 'test_file.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zip_file:
        zip_file.write(file_path)
    yield zip_path, file_path.name
    zip_path.unlink()
    file_path.unlink()


@pytest.fixture(scope='module')
def data():
    # Can't use os.random directly because there needs to be some
    # repitition in order for compression to be effective
    words = [os.urandom(8) for _ in range(1000)]
    out = b''.join([random.choice(words) for _ in range(524288)])
    return out


@pytest.fixture()
def zip_deflated(tmpdir, data):
    file_path = Path(tmpdir) / 'foo-iw2-bar-vv-baz.tiff'
    file_path.write_bytes(data)
    zip_path = Path(tmpdir) / 'SLCNAME.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as zip_file:
        zip_file.write(file_path)
    yield zip_path, file_path.name
    zip_path.unlink()
    file_path.unlink()
