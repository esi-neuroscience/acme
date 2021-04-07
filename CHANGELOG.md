# Changelog of ACME
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### NEW
- New keyword `write_pickle` can be used to override HDF5 as default storage
  format in favor of pickle

### FIXED
- If auto-saving to HDF5 fails, a new "emergency pickling" mechanic kicks in and
  attempts to pickle the offending return values instead

## [v0.1b] - 2020-01-15
### NEW
- This CHANGELOG file

### CHANGED
- Modified dependencies to not include Jupyter-related packages

### FIXED
- Fixed markdown syntax and URLs
- Fixed CI pipelines and repaired `h5py` version mismatch in dependencies
- Pin ACME to Python 3.8.x due to various packages not working properly
  (yet) in Python 3.9

## [v0.1a] - 2020-12-30
### NEW
- Initial ACME pre-release on PyPI

### CHANGED
- Made ACME GitHub repository public
