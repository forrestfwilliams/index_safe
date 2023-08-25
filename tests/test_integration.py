import base64
import json
import re
import zipfile

import zran
import numpy as np
from osgeo import gdal
from pathlib import Path

import pytest
import lxml.etree as ET

from index_safe import utils
from index_safe.create_index import index_safe, save_xml_metadata_as_json
from index_safe.extract_metadata import extract_metadata, json_to_metadata_entries
from index_safe.extract_burst import extract_burst, json_to_burst_metadata


def load_geotiff(infile, band=1):
    ds = gdal.Open(infile, gdal.GA_ReadOnly)

    data = ds.GetRasterBand(band).ReadAsArray()
    nodata = ds.GetRasterBand(band).GetNoDataValue()
    projection = ds.GetProjection()
    transform = ds.GetGeoTransform()
    ds = None
    return data, transform, projection, nodata


@pytest.fixture()
def test_data_dir():
    path = Path(__file__).parent / 'test_data'
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture()
def slc_zip_path(test_data_dir):
    slc_name = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85'
    slc_path = test_data_dir / f'{slc_name}.zip'
    if slc_path.exists():
        return slc_path

    print('SLC is not present, downloading SLC')
    in_aws = utils.check_if_in_aws_and_region()
    strategy = 's3' if in_aws else 'http'
    utils.download_slc(slc_name, working_dir=test_data_dir, strategy=strategy)
    return slc_path


@pytest.fixture()
def golden_burst_metadata():
    burst_metadata = utils.BurstMetadata(
        name='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_IW2_VV_7.tiff',
        slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
        swath='IW2',
        burst_index=7,
        shape=(1510, 25448),
        index_offset=utils.Offset(start=2987161209, stop=3087580208),
        uncompressed_offset=utils.Offset(start=1140690, stop=154846610),
        annotation_offset=utils.Offset(start=4735212575, stop=4735444360),
        manifest_offset=utils.Offset(start=4735444504, stop=4735449728),
    )
    return burst_metadata


@pytest.fixture()
def golden_burst_index(test_data_dir):
    index_name = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85_IW2_VV_7.dflidx'
    index_path = test_data_dir / index_name
    index = zran.Index.read_file(str(index_path))
    return index


@pytest.fixture()
def golden_burst(test_data_dir, slc_zip_path, golden_burst_metadata):
    swath, polarization = golden_burst_metadata.name.split('_')[-3:-1]
    regex = re.compile(rf'^s1\w-{swath.lower()}-slc-{polarization.lower()}.*tiff$')
    with zipfile.ZipFile(slc_zip_path) as zip_file:
        zinfo = [zinfo for zinfo in zip_file.infolist() if regex.match(Path(zinfo.filename).name)][0]
        if not (test_data_dir / zinfo.filename).exists():
            zip_file.extract(zinfo, path=test_data_dir)

    golden_array = load_geotiff(str(test_data_dir / 'valid_IW2_VV_7.vrt'))[0]
    yield golden_array


@pytest.fixture()
def golden_xml_metadata():
    xml_metadata = [
        utils.XmlMetadata(
            name='manifest.safe',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4735444504, stop=4735449728),
        ),
        utils.XmlMetadata(
            name='s1a-iw1-slc-vv-20200604t022252-20200604t022317-032861-03ce65-004.xml',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4733372170, stop=4733606318),
        ),
        utils.XmlMetadata(
            name='s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.xml',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4735212575, stop=4735444360),
        ),
        utils.XmlMetadata(
            name='s1a-iw2-slc-vh-20200604t022253-20200604t022318-032861-03ce65-002.xml',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4732740098, stop=4732971601),
        ),
        utils.XmlMetadata(
            name='noise-s1a-iw1-slc-vv-20200604t022252-20200604t022317-032861-03ce65-004.xml',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4733840607, stop=4733877456),
        ),
        utils.XmlMetadata(
            name='noise-s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.xml',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4734619938, stop=4734662287),
        ),
        utils.XmlMetadata(
            name='calibration-s1a-iw1-slc-vv-20200604t022252-20200604t022317-032861-03ce65-004.xml',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4734705983, stop=4734862152),
        ),
        utils.XmlMetadata(
            name='calibration-s1a-iw2-slc-vv-20200604t022253-20200604t022318-032861-03ce65-005.xml',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4734118441, stop=4734329022),
        ),
    ]
    return xml_metadata


@pytest.mark.integration
def test_create_index(slc_zip_path, golden_xml_metadata, golden_burst_metadata):
    safe_name = slc_zip_path.with_suffix('').name
    test_data_dir = slc_zip_path.parent
    index_safe(safe_name, working_dir=test_data_dir)

    files = list(test_data_dir.glob('S1*.json'))
    metadata = [file for file in files if 'metadata' in file.name]
    indexes = [file for file in files if 'metadata' not in file.name]
    assert len(metadata) == 1
    assert len(indexes) == 54

    metadata_path = metadata[0]
    xml_metadatas = json_to_metadata_entries(metadata_path)
    for golden_xml in golden_xml_metadata:
        test_xml = [xml for xml in xml_metadatas if xml.name == golden_xml.name][0]
        assert test_xml == golden_xml

    index_path = [index for index in indexes if 'IW2_VV_7.json' in index.name][0]
    index, burst_metadata = json_to_burst_metadata(index_path)
    assert golden_burst_metadata == burst_metadata

    golden_index_bytes = index_path.with_suffix('.dflidx').read_bytes()
    assert index.create_index_file() == golden_index_bytes


@pytest.mark.integration
def test_extract_metadata(tmpdir, golden_xml_metadata, slc_zip_path):
    tmpdir = Path(tmpdir)
    json_path = tmpdir / 'metadata.json'
    save_xml_metadata_as_json(golden_xml_metadata, str(json_path))

    in_aws = utils.check_if_in_aws_and_region()
    strategy = 's3' if in_aws else 'http'
    extract_metadata(json_path, 'vv', working_dir=tmpdir, strategy=strategy)
    test_xml = ET.parse(tmpdir / 'transformed.xml')
    manifest, metadata = [child for child in test_xml.getroot()]
    sub_xmls = {child.attrib['source_filename']: child.find('content') for child in metadata}

    with zipfile.ZipFile(slc_zip_path) as zip_file:
        zinfos = zip_file.infolist()
        golden_xmls = []
        for xml_name in sub_xmls:
            zip_path = [zinfo for zinfo in zinfos if xml_name == Path(zinfo.filename).name][0]
            with zip_file.open(zip_path, 'r') as xml_file:
                golden_xmls.append(ET.parse(xml_file).getroot())

    for xml_name, golden_xml in zip(sub_xmls, golden_xmls):
        golden_xml.tag = 'content'
        golden_string = ET.tostring(golden_xml, encoding='unicode')
        golden_string = golden_string.replace(' ', '').replace('\n', '')
        test_string = ET.tostring(sub_xmls[xml_name], encoding='unicode')
        test_string = test_string.replace(' ', '').replace('\n', '')
        assert len(test_string) == len(golden_string)
        assert test_string == golden_string


@pytest.mark.integration
def test_golden_by_burst(tmpdir, golden_burst_metadata, golden_burst_index, golden_burst):
    tmpdir = Path(tmpdir)
    burst_dictionary = golden_burst_metadata.to_dict()
    dflidx = golden_burst_index.create_index_file()
    burst_dictionary['dflidx_b64'] = base64.b64encode(dflidx).decode('utf-8')

    burst_index_path = tmpdir / 'burst_index.json'
    with open(burst_index_path, 'w') as json_file:
        json.dump(burst_dictionary, json_file)

    in_aws = utils.check_if_in_aws_and_region()
    strategy = 's3' if in_aws else 'http'
    burst_path = extract_burst(burst_index_path, strategy=strategy, working_dir=tmpdir)

    test_burst = load_geotiff(str(burst_path))[0]

    equal = np.isclose(golden_burst, test_burst)
    assert np.all(equal)
