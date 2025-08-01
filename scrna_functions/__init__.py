"""
scrna_functions: A collection of functions for single-cell RNA analysis.

This package provides functions for quality control, normalization, and visualization
of single-cell RNA sequencing data.
"""

from .core import (
    trim_outliers,
    plot_gene_counts,
    clr_normalize_each_cell,
    normalisation_kernel_density_plot,
    normalisation_check,
    normalisation_plots,
    pca_heatmap,
    load_marker_genes,
    filter_genes_in_adata,
    load_cell_cycle_genes,
)

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "trim_outliers",
    "plot_gene_counts", 
    "clr_normalize_each_cell",
    "normalisation_kernel_density_plot",
    "normalisation_check",
    "normalisation_plots",
    "pca_heatmap",
    "load_marker_genes",
    "filter_genes_in_adata",
    "load_cell_cycle_genes",
]
