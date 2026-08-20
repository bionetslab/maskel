"""Public package API for maskel."""

from importlib.metadata import version

from . import thin

__version__ = version("maskel")
__all__ = ["thin"]
