"""
scrna_functions: A collection of functions for single-cell RNA analysis.

This package provides functions for quality control, normalization, and visualization
of single-cell RNA sequencing data.
"""

from . import functions
from . import scanpy_gpu_helper
from . import celltypemarkers
# from . import hgtc_classes  # don't do by default

__version__ = "0.1.0"
__author__ = "drgmk"
