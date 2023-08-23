import json
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import botocore
import requests
import zran

from index_safe import utils


def test_calculate_range_parameters():
    params = utils.calculate_range_parameters(100, 50, 10)
    assert int(params[-1][-3:]) == 149
    assert int(params[0][6:8]) == 50
    assert len(params) == 10

    spacing = int(params[1][-2:]) - int(params[1][6:8])
    spacing += 1  # range reqests are inclusive
    assert spacing == 10


def test_get_credentials(tmpdir, example_credentials):
    credential_file = Path(tmpdir) / 'credentials.json'
    credential_file.write_text(json.dumps(example_credentials))
    output_file = utils.get_credentials(working_dir=Path(tmpdir))
    assert output_file['accessKeyId'] == 'foo'

    example_credentials['expiration'] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    credential_file.write_text(json.dumps(example_credentials))
    with patch('index_safe.utils.get_tmp_access_keys', return_value={}) as mock_func:
        utils.get_credentials(working_dir=Path(tmpdir))
        mock_func.assert_called_once()

    credential_file.unlink()
    with patch('index_safe.utils.get_tmp_access_keys', return_value={}) as mock_func:
        utils.get_credentials(working_dir=Path(tmpdir))
        mock_func.assert_called_once()


def test_get_download_url():
    safe_name = 'S1A_IW_SLC__1SDV_20230812T001358_20230812T001424_049835_05FE51_CAB0'
    real_url = f'https://sentinel1.asf.alaska.edu/SLC/SA/{safe_name}.zip'
    created_url = utils.get_download_url(safe_name)
    assert real_url == created_url


def test_setup_download_client(example_credentials):
    client, range_get = utils.setup_download_client(strategy='http')
    assert isinstance(client, requests.sessions.Session)
    assert range_get is utils.http_range_get

    with patch('index_safe.utils.get_tmp_access_keys', return_value=example_credentials):
        client, range_get = utils.setup_download_client(strategy='s3')
        assert isinstance(client, botocore.client.BaseClient)
        assert range_get is utils.s3_range_get


def test_get_zip_compressed_offset(zip_stored):
    zip_path, member_name = zip_stored
    with zipfile.ZipFile(zip_path) as f:
        zinfo = [x for x in f.infolist() if member_name in x.filename][0]

    offset = utils.get_zip_compressed_offset(zip_path, zinfo)
    with open(zip_path, 'rb') as f:
        f.seek(offset.start)
        output = f.read(offset.stop - offset.start)

    assert b'Hello World!' == output


def test_get_zip_indexer():
    index = utils.ZipIndexer('test.zip', 'test.txt')
    assert index.zip_path == 'test.zip'
    assert index.member_name == 'test.txt'
    assert index.spacing == 2**20
    assert index.index is None


def test_create_full_dflidx(zip_deflated):
    zip_path, member_name = zip_deflated
    indexer = utils.ZipIndexer(zip_path, member_name)
    indexer.create_full_dflidx()
    assert isinstance(indexer.index, zran.Index)
    assert indexer.index.have > 0
    first_point = indexer.index.points[0]
    assert first_point.outloc == 0
    assert first_point.inloc == 0
    assert first_point.bits == 0


def test_subset_dflidx(zip_deflated, data):
    zip_path, member_name = zip_deflated
    indexer = utils.ZipIndexer(zip_path, member_name, spacing=2**18)
    indexer.create_full_dflidx()
    step = 2**18
    compressed_offset, uncompressed_offset, modified_index = indexer.subset_dflidx([2 * step, 4 * step], 6 * step)
    assert modified_index.have == 2

    first_point = modified_index.points[0]
    assert first_point.outloc == 0
    assert first_point.inloc == 1  # to account for non-zero point.bits

    with open(zip_path, 'rb') as f:
        f.seek(compressed_offset.start)
        compressed_test_data = f.read(compressed_offset.stop - compressed_offset.start)
    test_data = zran.decompress(compressed_test_data, modified_index, 0, modified_index.uncompressed_size)
    assert test_data == data[uncompressed_offset.start : uncompressed_offset.stop]
