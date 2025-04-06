"""
VT.ai - Package initialization
"""

try:
    from importlib.metadata import version

    __version__ = version("vtai")
except:
    __version__ = "0.1.0"
