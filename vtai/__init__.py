"""
VT.ai - Package initialization
"""

try:
    from importlib.metadata import version

    __version__ = version("vtai")
except Exception:
    __version__ = "0.7.7"

__all__ = ["__version__"]
