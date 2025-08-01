# scrna-functions

A collection of functions for single-cell RNA analysis, including quality control, normalization, and visualization tools.

## Installation

You can install this package using pip:

```bash
pip install -e .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Usage

```python
import scrna_functions as sf

# Example usage
mask = sf.trim_outliers(x, y, pct=95)
sf.plot_gene_counts(adata, hue='sample')
adata_normalized = sf.clr_normalize_each_cell(adata)

# Load marker genes
marker_genes = sf.load_marker_genes()  # Original case
marker_genes_human = sf.load_marker_genes(case='upper')  # For human data
marker_genes_mouse = sf.load_marker_genes(case='title')  # For mouse data

# Filter genes present in your data (use appropriate case when loading)
filtered_genes = sf.filter_genes_in_adata(marker_genes_mouse, adata)

# Load cell cycle genes
cc_genes = sf.load_cell_cycle_genes(case='upper')  # For human data
s_genes = cc_genes['s_genes']
g2m_genes = cc_genes['g2m_genes']
```

## Functions

- `trim_outliers`: Fit a line in log space and trim outliers
- `plot_gene_counts`: Plot gene counts and mitochondrial fraction for each sample
- `clr_normalize_each_cell`: CLR (Centered Log-Ratio) normalize each cell
- `normalisation_kernel_density_plot`: Plot subset of data with kernel density estimate
- `normalisation_check`: Check normalization of RNA data
- `normalisation_plots`: Combined plots to check normalization
- `pca_heatmap`: Plot heatmap of PCA scores for variable genes
- `load_marker_genes`: Load predefined marker gene sets with optional case conversion
- `filter_genes_in_adata`: Filter marker genes to only include those present in AnnData
- `load_cell_cycle_genes`: Load cell cycle genes from Tirosh et al. file with optional case conversion

## Requirements

- numpy
- scipy
- matplotlib
- pandas
- seaborn
- anndata
- scanpy

## License

MIT License
