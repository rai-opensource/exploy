# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
