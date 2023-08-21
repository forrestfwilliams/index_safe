import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from index_safe import utils


def test_calculate_range_parameters():
    params = utils.calculate_range_parameters(100, 50, 10)
    assert int(params[-1][-3:]) == 149
    assert int(params[0][6:8]) == 50
    assert len(params) == 10

    spacing = int(params[1][-2:]) - int(params[1][6:8])
    spacing += 1  # range reqests are inclusive
    assert spacing == 10


def test_get_credentials(tmpdir):
    creation_time = datetime.now(timezone.utc)
    example_credentials = {
        'accessKeyId': 'foo',
        'secretAccessKey': 'bar',
        'sessionToken': 'baz',
        'expiration': (creation_time + timedelta(hours=1)).isoformat(),
    }

    credential_file = Path(tmpdir) / 'credentials.json'
    credential_file.write_text(json.dumps(example_credentials))
    output_file = utils.get_credentials(working_dir=Path(tmpdir))
    assert output_file['accessKeyId'] == 'foo'
