import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse
import seaborn as sns
import anndata as ad
import scanpy as sc
import json
import os


def trim_outliers(x, y, groups=None, extra_mask=None, pct=100):
    """Function to fit a line in log space, trim outliers, and return boolean mask.
    Parameters
    ----------
    x : array-like
        Independent variable.
    y : array-like
        Dependent variable.
    groups : array-like, optional
        Group names, to trim outliers per-group.
    extra_mask : array-like, optional
        Extra mask to apply before trimming outliers.
    pct : int, optional
        Percentile to use for trimming outliers, default is 100 (no trimming).
    """
    x_ = np.log10(x)
    y_ = np.log10(y)
    mask = np.ones(x_.shape[0], dtype=bool)
    if extra_mask is None:
        extra_mask = np.ones(x_.shape[0], dtype=bool)

    if groups is not None:
        for g in groups.unique():
            mask_g = groups == g
            mask[mask_g] = trim_outliers(x_[mask_g], y_[mask_g], extra_mask=extra_mask[mask_g], pct=pct)
        return mask
    
    fit = scipy.stats.linregress(x_[extra_mask], y_[extra_mask])
    y_fit = fit.intercept + fit.slope * x_
    resid = y_ - y_fit
    thresh = np.percentile(resid, [100-pct, pct])
    mask = np.logical_and(resid > thresh[0], resid < thresh[1])
    return np.logical_and(mask, extra_mask)


def plot_gene_counts(rna, hue=None, mask=None,
                     colour_by='pct_counts_in_top_1_genes',
                     size_by='pct_counts_mt',):
    """Plot gene counts and mitochondrial fraction for each sample.
   
    Parameters
    ----------
    rna : AnnData
        Annotated data matrix containing RNA expression data.
    hue : str, optional
        Column name in `rna.obs` to use for multiple panels.
    mask : array-like, optional
        Boolean mask to filter the data before plotting.
        This could come from `trim_outliers`.
    colour_by : str, optional
        Column name in `rna.obs` to use for coloring the points.
    size_by : str, optional
        Column name in `rna.obs` to use for the size of the points.
    """

    if mask is None:
        mask = np.ones(rna.shape[0], dtype=bool)

    vmax = np.max(rna[mask].obs[colour_by]) if colour_by in rna.obs.columns else None

    npanel2 = int(np.ceil(len(rna.obs[hue].unique())/2))
    fig, ax = plt.subplots(2, npanel2,
                           sharey=True, sharex=True,
                           figsize=(npanel2*2.5, 6))
    for i, s in enumerate(rna.obs[hue].unique()):
        tmp = rna[(rna.obs[hue] == s) & mask, :]
        a = ax.flatten()[i]
        _ = a.scatter(tmp.obs.total_counts, tmp.obs.n_genes_by_counts,
                    s=tmp.obs[size_by]/4, c=tmp.obs[colour_by],
                    vmin=0, vmax=vmax)
        
        a.set_title(s)

    [a.set_visible(False) for a in ax.flatten()[i+1:]]
    ax[1,0].set_ylabel('n_genes_by_counts')
    ax[1,0].set_xlabel('total_counts')
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')

    cb = fig.colorbar(_)
    cb.set_label(colour_by)
    # turn grid on for all axes
    for a in ax.flatten():
        a.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)
        a.set_axisbelow(True)
    fig.tight_layout()


def _seurat_clr(x, length=None):
    """Centered Log-Ratio (CLR) normalization for a count vector.
    
    Parameters
    ----------
    x : array-like
        Count vector to normalize.
    length : int, optional
        Length of the vector to assume for averaging. If None, uses the length of `x`.
        Useful when `x` is from a sparse matrix and zeros should be included in the mean.
    """
    s = np.sum(np.log1p(x[x > 0]))
    if length is None:
        length = len(x)
    exp = np.exp(s / length)
    return np.log1p(x / exp)


def clr_normalize_each_cell(adata, inplace=False):
    """CLR (Centered Log-Ratio) normalize each cell (row of .X).

    https://github.com/scverse/scanpy/issues/1208
    
    The mean is over all entries (zeros included), not just non-zero entries.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    inplace : bool, optional
        If True, modifies the input AnnData object in place.
    
    Returns
    -------
    AnnData
        Normalized AnnData object (or modified in place).
    """
    if not inplace:
        adata = adata.copy()

    X = adata.X.astype(float)  # likely counts as int 

    if scipy.sparse.issparse(X):
        n_genes = X.shape[1]

        for i in range(X.shape[0]):
            start, end = X.indptr[i], X.indptr[i + 1]
            row_data = X.data[start:end]
            X.data[start:end] = _seurat_clr(row_data, length=n_genes)

        adata.X = X

    else:
        print("Warning: Input data is dense.")
        adata.X = np.apply_along_axis(_seurat_clr, 1, adata.X)

    return adata



def normalisation_kernel_density_plot(rna, n_sample=500, ax=None, hue=None):
    """Plot subset of data with kernel density estimate.
    
    Parameters
    ----------
    rna : AnnData
        Annotated data matrix containing RNA expression data.
    n_sample : int, optional
        Number of random samples to plot. Default is 500.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, creates a new figure.
    """
    x = np.linspace(0, 2.5, 100)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # create colours for each sample
    if hue is not None and hue in rna.obs.columns:
        hue = rna.obs[hue].unique()
        colours = sns.color_palette("husl", len(hue))
        hue_dict = dict(zip(hue, colours))
        hue_offset = dict(zip(rna.obs['sample'].unique(), np.arange(len(rna.obs['sample'].unique()))))
    else:
        hue_dict = None

    for i in np.random.randint(low=0, high=rna.shape[0], size=n_sample):
        if scipy.sparse.issparse(rna.X):
            tmp =rna.X[i,:].toarray()[0]
        else:
            tmp = rna.X[i,:]
        kern = scipy.stats.gaussian_kde(tmp)
        ax.plot(x, kern(x) + 0.2*hue_offset[rna.obs['sample'].iloc[i]] if hue_dict else 0, 
                linewidth=1, color=hue_dict[rna.obs['sample'].iloc[i]] if hue_dict else 'black')
        
    for k in hue_offset.keys():
        ax.text(2.4, 0.2*hue_offset[k]+0.05, k, color=hue_dict[k] if hue_dict else 'black')
    ax.set_ylim(0, 1+0.2*len(rna.obs['sample'].unique()))
    ax.set_xlabel('normalised expression')
    if ax is None:
        fig.tight_layout()
        return fig
    else:
        return ax


def normalisation_check(rna, percentile=95, ax=None, hue=None, n_sample=None):
    """Check normalisation of RNA data.
    
    This function checks the normalisation of RNA data by plotting the mean and
    a high percentile (e.g., 95th) of the expression values for each cell.
    It expects that with good normalisation, the high percentile and mean are not
    too correlated.
    
    Parameters
    ----------
    rna : AnnData
        Annotated data matrix containing RNA expression data.
    percentile : float, optional
        Percentile at which to plot each gene.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, creates a new figure.
    n_sample : int, optional
        If provided, plot only a random subset of this many rows from rna.
    """

    n_cells = rna.shape[0]
    if n_sample is not None and n_sample < n_cells:
        idx = np.random.choice(n_cells, n_sample, replace=False)
    else:
        idx = np.arange(n_cells)

    pc = np.zeros(len(idx))
    mn = pc.copy()
    for i, j in enumerate(idx):
        if scipy.sparse.issparse(rna.X):
            row = rna.X[j,:].toarray()
        else:
            row = rna.X[j,:]
        ok = row > 0
        mn[i] = np.sum(row[ok])/len(row)
        pc[i] = np.percentile(row, percentile)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if hue is not None and hue in rna.obs.columns:
        hue = rna.obs['sample'].iloc[idx]
    sns.scatterplot(x=mn, y=pc, hue=hue, ax=ax)
    ax.set_xlabel('mean normalised expression')
    ax.set_ylabel(f'{percentile}th percentile normalised expression')
    if ax is None:
        fig.tight_layout()
        return fig
    else:
        return ax
    

def normalisation_plots(rna, n_sample=1000, hue=None):
    """Plots to check normalisation of RNA data.

    Parameters
    ----------
    rna : AnnData
        Annotated data matrix containing RNA expression data.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    _ = normalisation_kernel_density_plot(rna, ax=ax[0], hue=hue, n_sample=n_sample)
    _ = normalisation_check(rna, ax=ax[1], n_sample=n_sample, hue=hue)

    if hue is not None and hue in rna.obs.columns:
        for i, g in enumerate(rna.obs[hue].unique()):
            ax[2].hist(rna[rna.obs[hue] == g].X.sum(1), bins=100, label=g, alpha=0.5)
    else:
        ax[2].hist(rna.X.sum(1), bins=100)
    ax[2].set_xlabel('normalised counts')
    fig.tight_layout()
    return fig


def pca_heatmap(adata, component, layer=None):
    """Plot heatmap of PCA scores for the top and bottom 20 variable genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing PCA results in `varm['PCs']`.
    component : int
        The PCA component to visualize.
    layer : str, optional
        The layer of the AnnData object to use for the heatmap. If None, uses the main data layer.
    """
    attr = 'varm'
    keys = 'PCs'
    scores = getattr(adata, attr)[keys][:, component]
    dd = pd.DataFrame(scores, index=adata.var_names)
    var_names_pos = dd.sort_values(0, ascending=False).index[:20]

    var_names_neg = dd.sort_values(0, ascending=True).index[:20]

    pd2 = pd.DataFrame(adata.obsm['X_pca'][:, component], index=adata.obs.index)

    bottom_cells = pd2.sort_values(0).index[:300].tolist()
    top_cells = pd2.sort_values(0, ascending=False).index[:300].tolist()

    sc.pl.heatmap(adata[top_cells+bottom_cells], list(var_names_pos) + list(var_names_neg), 'il21',
                  show_gene_labels=True, figsize=(4,4),
                  swap_axes=True, cmap='inferno',
                  use_raw=False, layer=layer, vmin=-1, vmax=3)


def load_marker_genes(case=None):
    """Load marker genes from the JSON file included with the package.
    
    Parameters
    ----------
    case : str, optional
        Convert gene names to specific case. Options:
        - 'upper': Convert to uppercase (typical for human genes)
        - 'lower': Convert to lowercase  
        - 'title': Convert to title case (typical for mouse genes, e.g., 'Bcl6')
        - None: Keep original case
    
    Returns
    -------
    dict
        Dictionary with cell type/gene set names as keys and lists of gene names as values.
    """
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    marker_genes_path = os.path.join(module_dir, 'marker_genes.json')
    
    try:
        with open(marker_genes_path, 'r') as f:
            marker_genes = json.load(f)
        
        # Apply case conversion if specified
        if case is not None:
            converted_genes = {}
            for gene_set, genes in marker_genes.items():
                if case.lower() == 'upper':
                    converted_genes[gene_set] = [gene.upper() for gene in genes]
                elif case.lower() == 'lower':
                    converted_genes[gene_set] = [gene.lower() for gene in genes]
                elif case.lower() == 'title':
                    converted_genes[gene_set] = [gene.capitalize() for gene in genes]
                else:
                    print(f"Warning: Unknown case option '{case}'. Valid options are 'upper', 'lower', 'title', or None.")
                    converted_genes[gene_set] = genes
            return converted_genes
        
        return marker_genes
    except FileNotFoundError:
        print(f"Marker genes file not found at {marker_genes_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error reading marker genes file at {marker_genes_path}")
        return {}


def filter_genes_in_adata(marker_genes, adata):
    """Filter marker genes to only include those present in the AnnData object.
    
    Parameters
    ----------
    marker_genes : dict
        Dictionary with gene set names as keys and lists of gene names as values.
    adata : AnnData
        Annotated data matrix containing gene names in var_names.
    
    Returns
    -------
    dict
        Filtered dictionary with only genes present in adata.var_names.
    """
    filtered_genes = {}
    for gene_set, genes in marker_genes.items():
        filtered_genes[gene_set] = [gene for gene in genes if gene in adata.var_names]
        missing_genes = [gene for gene in genes if gene not in adata.var_names]
        if missing_genes:
            print(f"Missing genes in {gene_set}: {missing_genes}")
    
    return filtered_genes


def load_cell_cycle_genes(case=None):
    """Load cell cycle genes from the Tirosh et al. file included with the package.
    
    Parameters
    ----------
    case : str, optional
        Convert gene names to specific case. Options:
        - 'upper': Convert to uppercase (typical for human genes)
        - 'lower': Convert to lowercase  
        - 'title': Convert to title case (typical for mouse genes, e.g., 'Bcl6')
        - None: Keep original case
    
    Returns
    -------
    dict
        Dictionary with 's_genes' and 'g2m_genes' as keys and lists of gene names as values.
        Also includes 'all_genes' key with all cell cycle genes.
    """
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    tirosh_file_path = os.path.join(module_dir, 'tirosh15_regev_lab_cell_cycle_genes.txt')
    
    try:
        with open(tirosh_file_path, 'r') as f:
            cell_cycle_genes = [line.strip() for line in f.readlines() if line.strip()]
        
        # Split into S and G2M phases (first 43 are S genes, rest are G2M)
        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]
        
        # Apply case conversion if specified
        if case is not None:
            if case.lower() == 'upper':
                s_genes = [gene.upper() for gene in s_genes]
                g2m_genes = [gene.upper() for gene in g2m_genes]
                all_genes = [gene.upper() for gene in cell_cycle_genes]
            elif case.lower() == 'lower':
                s_genes = [gene.lower() for gene in s_genes]
                g2m_genes = [gene.lower() for gene in g2m_genes]
                all_genes = [gene.lower() for gene in cell_cycle_genes]
            elif case.lower() == 'title':
                s_genes = [gene.capitalize() for gene in s_genes]
                g2m_genes = [gene.capitalize() for gene in g2m_genes]
                all_genes = [gene.capitalize() for gene in cell_cycle_genes]
            else:
                print(f"Warning: Unknown case option '{case}'. Valid options are 'upper', 'lower', 'title', or None.")
                all_genes = cell_cycle_genes
        else:
            all_genes = cell_cycle_genes
        
        return {
            's_genes': s_genes,
            'g2m_genes': g2m_genes,
            'all_genes': all_genes
        }
        
    except FileNotFoundError:
        print(f"Cell cycle genes file not found at {tirosh_file_path}")
        return {'s_genes': [], 'g2m_genes': [], 'all_genes': []}
    except Exception as e:
        print(f"Error reading cell cycle genes file: {e}")
        return {'s_genes': [], 'g2m_genes': [], 'all_genes': []}
