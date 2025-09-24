"""
scrna_functions: A collection of functions for single-cell RNA analysis.

This package provides functions for quality control, normalization, and visualization
of single-cell RNA sequencing data.
"""

from .core import *
from .scanpy_gpu_helper import pick_backend, gpu_session
from .celltypemarkers import CellTypeMarkers
from .hgtc_classes import *

__version__ = "0.1.0"
__author__ = "Your Name"
