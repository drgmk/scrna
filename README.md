# scrna-functions

A collection of functions for single-cell RNA analysis, including quality control, normalization, and visualization tools.

## Installation

Do it in your notebook like so:

```python
!pip install git+https://github.com/drgmk/scrna.git

import scrna_functions as scfunc
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

- `get_plot_list`: List available QC plots for AnnData
- `do_qc`: Calculate QC metrics for RNA data
- `trim_outliers`: Fit a line in log space and trim outliers
- `plot_gene_counts`: Plot gene counts and mitochondrial fraction for each sample
- `plot_top_genes`: Plot top N most highly expressed genes for each sample
- `plot_umaps`: Plot UMAPs for each sample and all samples combined
- `plot_cell_counts`: Plot heatmap of cell numbers per sample and cell type
- `_seurat_clr`: Centered Log-Ratio normalization for a count vector
- `clr_normalize_each_cell`: CLR normalize each cell in AnnData
- `normalisation_kernel_density_plot`: Plot subset of data with kernel density estimate
- `normalisation_check`: Check normalization of RNA data
- `normalisation_plots`: Combined plots to check normalization
- `pca_heatmap`: Seurat DimHeatmap equivalent for PCA components
- `load_cell_cycle_genes`: Load cell cycle genes from Tirosh et al. file
- `remove_doublet_clusters`: Remove clusters identified as doublets
- `get_vmax`: Get vmax values for marker genes
- `rank_genes_groups_to_df`: Convert sc.tl.rank_genes_groups results to DataFrame
- `get_pseudobulk`: Create pseudobulk data and perform PCA
- `do_deg`: Differential expression analysis using DESeq2 via decoupler
- `celltypist_annotate`: Annotate cell types using CellTypist
- `cellphonedb_prepare`: Export data and run CellPhoneDB analysis

## License

MIT License
