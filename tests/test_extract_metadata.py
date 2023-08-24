import json
from pathlib import Path

import lxml.etree as ET
import pytest
from unittest.mock import patch

from index_safe import extract_metadata, utils


def create_minimal_xml(output_path, name):
    root = ET.Element('root')
    name_element = ET.Element('name')
    name_element.text = name
    root.append(name_element)

    tree = ET.ElementTree(root)
    xml_string = ET.tostring(tree, pretty_print=True, encoding='utf-8', xml_declaration=True)
    output_path.write_bytes(xml_string)
    return output_path


def test_extract_metadata_xml_bytes():
    equal_offset = utils.Offset(100, 99)
    negative_offset = utils.Offset(-100, 100)
    with patch('index_safe.utils.http_range_get', return_value=b''):
        client, range_get = utils.setup_download_client(strategy='http')

        with pytest.raises(ValueError, match='offset stop must be greater than offset start'):
            extract_metadata.extract_metadata_xml('https://foo.com', equal_offset, client, range_get)

        with pytest.raises(ValueError, match='offset stop and offset start must be greater than 0'):
            extract_metadata.extract_metadata_xml('https://foo.com', negative_offset, client, range_get)


def test_json_to_metadata_entries(tmpdir):
    test_json = {
        'SLCNAME': {
            'manifest.safe': {'offset_start': 0, 'offset_stop': 100},
            'annotation.xml': {'offset_start': 100, 'offset_stop': 200},
        }
    }
    json_path = Path(tmpdir) / 'test.json'
    json_path.write_text(json.dumps(test_json))
    metadata_entries = extract_metadata.json_to_metadata_entries(json_path)
    assert len(metadata_entries) == 2

    manifest = utils.XmlMetadata('manifest.safe', 'SLCNAME', utils.Offset(0, 100))
    assert metadata_entries[0] == manifest

    annotation = utils.XmlMetadata('annotation.xml', 'SLCNAME', utils.Offset(100, 200))
    assert metadata_entries[1] == annotation


def test_make_input_xml():
    paths = [Path('annotation.xml'), Path('noise.xml'), Path('calibration.xml')]
    manifest = Path('manifest.safe')
    input_xml = extract_metadata.make_input_xml(paths, manifest)
    assert input_xml.tag == 'files'
    assert input_xml.attrib['manifest'] == str(manifest)
    assert len(input_xml) == 3
    for child, path in zip(input_xml, paths):
        assert child.tag == 'file'
        assert child.attrib['name'] == str(path)
        assert child.attrib['label'] in str(paths)


def test_xlst_render(tmpdir):
    tmpdir = Path(tmpdir)
    manifest_path = create_minimal_xml(tmpdir / 'manifest.safe', 'manifest')
    paths = []
    for name in ['annotation', 'calibration', 'noise']:
        paths.append(create_minimal_xml(tmpdir / f'{name}.xml', name))

    renderer = extract_metadata.XsltRenderer(Path(__file__).parent.parent / 'src' / 'index_safe' / 'burst.xsl')
    renderer.render_to(tmpdir / 'burst.xml', paths, manifest_path)

    burst_xml = ET.parse(tmpdir / 'burst.xml')
    child_xmls = burst_xml.findall('.//{*}name')
    assert len(child_xmls) == 4
    assert child_xmls[0].text == 'manifest'
    assert child_xmls[1].text == 'annotation'
    assert child_xmls[2].text == 'calibration'
    assert child_xmls[3].text == 'noise'


def test_select_and_reorder_metadatas():
    unordered_names = [
        's1a-iw2-slc-vh-NULL-NULL-NULL-NULL-NULL.xml',
        's1a-iw1-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        's1a-iw2-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        'manifest.safe',
        'noise-s1a-iw1-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        'calibration-s1a-iw2-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        'noise-s1a-iw2-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        'calibration-s1a-iw1-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
    ]

    ordered_names = [
        's1a-iw1-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        'noise-s1a-iw1-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        'calibration-s1a-iw1-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        's1a-iw2-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        'noise-s1a-iw2-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        'calibration-s1a-iw2-slc-vv-NULL-NULL-NULL-NULL-NULL.xml',
        'manifest.safe',
    ]

    unordered_metadatas = [utils.XmlMetadata(name, 'SLCNAME', utils.Offset(0, 100)) for name in unordered_names]
    ordered_metadatas = extract_metadata.select_and_reorder_metadatas(unordered_metadatas, 'vv')
    test_names = [metadata.name for metadata in ordered_metadatas]
    assert 's1a-iw2-slc-vh-NULL-NULL-NULL-NULL-NULL.xml' not in test_names
    assert test_names == ordered_names
