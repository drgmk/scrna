# scrna-functions

A collection of functions for single-cell RNA analysis, including quality control, normalization, and visualization tools.

## Installation

Do it in your notebook like so:

```python
!pip install git+https://github.com/drgmk/scrna.git

import scrna_functions as sf
```

## Usage

```python
import scrna_functions as sf

# wrap some common tasks
import anndata as ad
adata = ad.read_h5ad('your_data.h5ad')
sf.compute_qc_metrics(adata)
sf.plot_gene_counts(adata, hue='sample')

# load marker genes, e.g. for use with decoupler
markers = sf.CellTypeMarkers('human')
markers.filter_genes(adata.var_names, verbose=True)
marker_genes = markers.to_dict(include_secondary=False)
```

## Auto GPU/CPU selection

A generic attempt to automatically use the GPU via `rapids-singlecell` 
if available. Just call `pick_backend()` at the start of your notebook/script,
setting this to `sc` (or whatever you usually call scanpy).
Then proceed as normal.

Arguments are passed straight through to `scanpy` or `rsc`, which will result
in problems when there are differences in the `rsc` and `sc` APIs,
for example `sc.pp.log1p` has `base` as a keyword, while `rsc.pp.log1p` does not.

```python
import scrna_functions as sf

# set backend to GPU if available
sc = sf.pick_backend()

# proceed as normal
adata = sc.read_h5ad('your_data.h5ad')
sc.pp.normalize_total(adata)
```
