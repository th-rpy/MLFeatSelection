"""Top-level package for MLFeatSelection."""

__author__ = """Saraei Thamer"""
__email__ = 'thamer.saraei@polymtl.ca'
__version__ = '0.1.0'

import sys

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()