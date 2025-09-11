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

# wrap some common tasks
import anndata as ad
adata = ad.read_h5ad('your_data.h5ad')
sf.do_qc(adata)
sf.plot_gene_counts(adata, hue='sample')

# load marker genes, e.g. for use with decoupler
markers = scfunc.CellTypeMarkers('human')
markers.filter_genes(adata.var_names, verbose=True)
marker_genes = markers.to_dict(include_secondary=False)
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
