from pathlib import Path

import pytest

from index_safe import utils
from index_safe.create_index import index_safe
from index_safe.extract_metadata import json_to_metadata_entries
from index_safe.extract_burst import json_to_burst_metadata


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
            name='noise-s1a-iw2-slc-vh-20200604t022253-20200604t022318-032861-03ce65-002.xml',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4734577564, stop=4734619710),
        ),
        utils.XmlMetadata(
            name='calibration-s1a-iw3-slc-vv-20200604t022251-20200604t022317-032861-03ce65-006.xml',
            slc='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
            offset=utils.Offset(start=4735015380, stop=4735212365),
        ),
    ]
    return xml_metadata


@pytest.fixture()
def test_data_dir():
    path = Path(__file__).parent / 'test_data'
    path.mkdir(exist_ok=True)
    print(path)
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

@pytest.mark.skip(reason='Integration test')
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
