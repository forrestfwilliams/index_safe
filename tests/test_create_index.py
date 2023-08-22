from index_safe import create_index, utils
import zipfile
from unittest.mock import patch


def test_get_burst_annotation_data(annotation_xml):
    burst_shape, burst_offsets = create_index.get_burst_annotation_data(annotation_xml)
    assert burst_shape == (1000, 20000)
    assert len(burst_offsets) == 2
    assert burst_offsets[0] == utils.Offset(10, 110)
    assert burst_offsets[1] == utils.Offset(110, 210)


def test_create_xml_metadata():
    with patch('index_safe.utils.get_zip_compressed_offset', return_value=utils.Offset(0, 100)):
        metadata = create_index.create_xml_metadata('test.zip', zipfile.ZipInfo('test.xml'))

    assert metadata.slc == 'test'
    assert metadata.name == 'test.xml'
    assert metadata.offset == utils.Offset(0, 100)


def test_create_burst_name():
    test_name = 'SLCNAME'
    test_swath = 'foo-iw2-bar-vv-baz.tiff'
    test_burst_index = 1
    burst_name = create_index.create_burst_name(test_name, test_swath, test_burst_index)
    assert burst_name == 'SLCNAME_IW2_VV_1.tiff'


def test_create_burst_dflidx(zip_deflated):
    zip_path, member_name = zip_deflated
    indexer = utils.ZipIndexer(zip_path, member_name, spacing=2**18)
    indexer.create_full_dflidx()

    burst_offset = utils.Offset(5 * 2**20, 10 * 2**20)
    compressed_offset, index_burst_offset, dflidx = create_index.create_burst_dflidx(indexer, burst_offset)
    assert index_burst_offset.start != 0
    assert index_burst_offset.start < burst_offset.start
    assert index_burst_offset.stop < burst_offset.stop

    burst_length = burst_offset.stop - burst_offset.start
    index_burst_length = index_burst_offset.stop - index_burst_offset.start
    assert index_burst_length >= burst_length


def test_create_index(zip_deflated, annotation_xml):
    zip_path, member_name = zip_deflated
    annotation_offset = utils.Offset(0, 100)
    manifest_offset = utils.Offset(100, 200)
    with patch('index_safe.create_index.load_annotation_data', return_value=annotation_xml):
        test_burst_metadata = create_index.create_index(zip_path, member_name, annotation_offset, manifest_offset)

    keys = list(test_burst_metadata.keys())
    first_burst = test_burst_metadata[keys[0]]
    golden_burst = utils.BurstMetadata(
        name='SLCNAME_IW2_VV_0.tiff',
        slc='SLCNAME',
        swath='IW2',
        burst_index=0,
        shape=(1000, 20000),
        index_offset=utils.Offset(0, 0),
        uncompressed_offset=utils.Offset(0, 0),
        annotation_offset=utils.Offset(0, 0),
        manifest_offset=utils.Offset(0, 0),
    )
    assert first_burst['name'] == golden_burst.name
    assert first_burst['slc'] == golden_burst.slc
    assert first_burst['swath'] == golden_burst.swath
    assert first_burst['burst_index'] == golden_burst.burst_index
