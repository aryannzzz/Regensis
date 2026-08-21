"""Regenesis conditional-mutual-information synthetic validation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("regenesis-cmi")
except PackageNotFoundError:  # Source-tree execution.
    __version__ = "0.1.0"

__all__ = ["__version__"]

