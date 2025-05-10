from importlib import metadata

try:
    __version__ = metadata.version("grasp")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
