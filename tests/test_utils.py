import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import botocore
import pytest
import requests
from index_safe import utils

@pytest.fixture(scope='module')
def example_credentials():
    expire_time = datetime.now(timezone.utc) + timedelta(hours=1)
    example_credentials = {
        'accessKeyId': 'foo',
        'secretAccessKey': 'bar',
        'sessionToken': 'baz',
        'expiration': expire_time.isoformat(),
    }
    yield example_credentials


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


