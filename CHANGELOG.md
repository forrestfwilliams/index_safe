# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1]

### Added
* `xml_metadata_as_json` function that returns XML metadata JSON string.

### Changed
* `save_xml_metadata_as_json` function so that it calls the new `xml_metadata_as_json` function.

## [0.2.0]

### Added
* Tests for utils.py
* Tests for create_index.py
* Tests for extract_metadata.py
* Tests for extract_burst.py
* Integration tests that use a downloaded S1 SLC

### Changed
* Standardize all xml operations so that they use lxml
* `extract_metadata` polarization option is now required

### Removed
* Original (limited) test suite dependent on S1 SLC
* Burst index byte format removed in favor of json format

## [0.1.0]

### Added
* Geographic control point information to indexes
* Insertion of geographic control points into the output burst tiffs
* Add check for deflate compression before indexing
* Workflows for PyPI publishing

### Changed
* `create_index` and `extract_burst` so that valid window and GCP calculations happen during `extract_burst`
* `utils.BurstMetadata` so that it contains annotation and manifest offsets, but not valid window or GCP data
* Created centralized range get request functionality in `utils`

## [0.0.1]

### Added
* Initial version of project (with no changes), to merge a non-zero version into main

## [0.0.0]

### Added
* Initial version of project

