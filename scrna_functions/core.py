import os
import shutil
from pathlib import Path
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse
import seaborn as sns
import anndata as ad
import scanpy as sc
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
import celltypist


def get_plot_list(adata=None):
    """Get list of QC plots that can be made based on available columns in adata.obs.

    Parameters
    ----------
    adata : AnnData, optional
        Annotated data matrix containing RNA expression data. If None, returns all possible plots.
    """

    plot_list = [['n_genes_by_counts', 'total_counts', 'pct_counts_in_top_1_genes'],
                 ['pct_counts_ribosomal', 'pct_counts_malat', 'pct_counts_mt']]

    if adata is None:
        return plot_list
    
    tmp = []
    for p in plot_list:
        tmp.append([pp for pp in p if pp in adata.obs.columns.tolist()])
    return tmp


def do_qc(adata, organism, extra_genes=[]):
    """Calculate QC metrics for RNA data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing RNA expression data.
    organism : str
        Target organism to determine gene name case. Options:
        - 'human': uppercase genes (e.g., 'FOXP3')
        - 'mouse': title case genes (e.g., 'Foxp3')
    extra_genes : list, optional
        List of extra gene prefixes to calculate percent counts for.
    """

    pct_counts = {'mt': ['MT-'],
              'ribosomal': ['RPL', 'RPS'],
              'malat': ['MALAT'],
    }
    for g in extra_genes:
        pct_counts[g] = [g]

    if organism.lower() == 'mouse':
        for k, v in pct_counts.items():
            pct_counts[k] = [s.capitalize() for s in v]
    elif organism.lower() == 'human':
        for k, v in pct_counts.items():
            pct_counts[k] = [s.upper() for s in v]

    for k in pct_counts.keys():
        adata.var[k] = adata.var_names.str.startswith(pct_counts[k][0])
        if len(pct_counts[k]) > 1:
            for s in pct_counts[k][1:]:
                adata.var[k] = np.logical_or(adata.var[k], adata.var_names.str.startswith(s))
            
        sc.pp.calculate_qc_metrics(adata, qc_vars=[k], percent_top=None, log1p=False, inplace=True)

    adata.var['nomalat'] = np.invert(adata.var['malat'])
    sc.pp.calculate_qc_metrics(adata, qc_vars=['nomalat'], percent_top=[1], log1p=False, inplace=True)


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


def plot_gene_counts(adata, hue='sample', mask=None, order=None,
                     colour_by='pct_counts_in_top_1_genes',
                     size_by='pct_counts_mt',):
    """Plot gene counts and mitochondrial fraction for each sample.
   
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing RNA expression data.
    hue : str, optional
        Column name in `adata.obs` to use for multiple panels.
    mask : array-like, optional
        Boolean mask to filter the data before plotting.
        This could come from `trim_outliers`.
    colour_by : str, optional
        Column name in `adata.obs` to use for coloring the points.
    size_by : str, optional
        Column name in `adata.obs` to use for the size of the points.
    """

    if mask is None:
        mask = np.ones(adata.shape[0], dtype=bool)

    if order is None:
        order = adata.obs[hue].unique()

    vmax = np.max(adata[mask].obs[colour_by]) if colour_by in adata.obs.columns else None

    npanel2 = int(np.ceil(len(order)/2))
    fig, ax = plt.subplots(2, npanel2,
                           sharey=True, sharex=True,
                           figsize=(npanel2*2.5, 6))
    ax = ax.reshape(2, npanel2)
    
    for i, s in enumerate(order):
        tmp = adata[(adata.obs[hue] == s) & mask, :]
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
    return fig


def plot_top_genes(adata, hue='sample', n_top=10, order=None):
    """Plot the top 10 most highly expressed genes for each sample.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing RNA expression data.
    hue : str, optional
        Column name in `adata.obs` to use for multiple panels.
    order : list, optional
        List of group names in the order to plot. If None, uses the order in `adata.obs[hue].unique()`.
    """
    if order is None:
        order = adata.obs[hue].unique()
    npanel2 = int(np.ceil(len(order)/2))
    fig, ax = plt.subplots(2, npanel2, figsize=(18, 10), sharex=True, sharey=True)
    ax = ax.reshape(2, npanel2)
    for i, s in enumerate(order):
        sc.pl.highest_expr_genes(adata[adata.obs[hue] == s].copy(), n_top=n_top, log=True, ax=ax[i//npanel2, i%npanel2], show=False)
        ax[i//npanel2, i%npanel2].axvline(x=1, alpha=0.5)
        ax[i//npanel2, i%npanel2].set_title(s)

    fig.tight_layout()
    return fig


def plot_umaps(adata, hue='sample', order=None):
    """Plot UMAPs for each sample and for all samples combined.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing RNA expression data.
    hue : str, optional
        Column name in `adata.obs` to use for multiple panels.
    order : list, optional
        List of group names in the order to plot. If None, uses the order in `adata.obs[hue].unique()`.
    """
    if order is None:
        order = adata.obs[hue].unique()
    npanel2 = int(np.ceil((len(order)+1)/2))
    fig, ax = plt.subplots(2, npanel2, figsize=(15,8), sharex=True, sharey=True)
    for i, s in enumerate(order):
        a = ax.flatten()[i]
        sc.pl.umap(adata[adata.obs[hue] == s], ax=a, show=False, size=10)
        a.set_title(s)

    sc.pl.umap(adata, ax=ax.flatten()[-1], show=False, size=10)
    ax.flatten()[-1].set_title('all')
    fig.tight_layout()
    return fig


def plot_cell_counts(adata, x='sample', y='celltype', x_order=None, y_order=None):
    """Plot heatmap of cell numbers per x (e.g. sample) and y (e.g. cell type).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing RNA expression data.
    x : str, optional
        Column name in `adata.obs` to use for x-axis.
    y : str, optional
        Column name in `adata.obs` to use for y-axis.
    x_order : list, optional
        List of group names in the order to plot for x. If None, uses the order in `adata.obs[x].unique()`.
    y_order : list, optional
        List of group names in the order to plot for y. If None, uses the order in `adata.obs[y].unique()`.
    """

    if x_order is None:
        x_order = adata.obs[x].unique()
    if y_order is None:
        y_order = adata.obs[y].unique()

    ct = pd.crosstab(adata.obs[x], adata.obs[y])
    ct = ct[y_order].transpose()
    ct = ct[x_order]
    ct['total'] = ct.sum(axis=1)
    ct.loc['total'] = ct.sum(axis=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(ct, annot=True, cmap='viridis', cbar_kws={'label': 'Number of cells'}, ax=ax, fmt='g', norm='log')
    fig.tight_layout()
    return fig


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



def normalisation_kernel_density_plot(adata, n_sample=500, ax=None, hue=None):
    """Plot subset of data with kernel density estimate.
    
    Parameters
    ----------
    adata : AnnData
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
    if hue is not None and hue in adata.obs.columns:
        hue_ = adata.obs[hue].unique()
        colours = sns.color_palette("husl", len(hue_))
        hue_dict = dict(zip(hue_, colours))
        hue_offset = dict(zip(adata.obs[hue].unique(), np.arange(len(adata.obs[hue].unique()))))
    else:
        hue_dict = None

    # print(hue_dict, hue_offset)

    for i in np.random.randint(low=0, high=adata.shape[0], size=n_sample):
        if scipy.sparse.issparse(adata.X):
            tmp =adata.X[i,:].toarray()[0]
        else:
            tmp = adata.X[i,:]
        kern = scipy.stats.gaussian_kde(tmp)
        ax.plot(x, kern(x) + (0.2*hue_offset[adata.obs[hue].iloc[i]] if hue_dict else 0), 
                linewidth=1, color=hue_dict[adata.obs[hue].iloc[i]] if hue_dict else 'black')

    if hue_dict is not None:
        for k in hue_offset.keys():
            ax.text(2.4, 0.2*hue_offset[k]+0.05, k, color=hue_dict[k] if hue_dict else 'black')
        ax.set_ylim(0, 1+0.2*len(adata.obs[hue].unique()))
    ax.set_xlabel('normalised expression')
    if ax is None:
        fig.tight_layout()
        return fig
    else:
        return ax


def normalisation_check(adata, percentile=95, ax=None, hue=None, n_sample=None):
    """Check normalisation of RNA data.
    
    This function checks the normalisation of RNA data by plotting the mean and
    a high percentile (e.g., 95th) of the expression values for each cell.
    It expects that with good normalisation, the high percentile and mean are not
    too correlated.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing RNA expression data.
    percentile : float, optional
        Percentile at which to plot each gene.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, creates a new figure.
    n_sample : int, optional
        If provided, plot only a random subset of this many rows from adata.
    """

    n_cells = adata.shape[0]
    if n_sample is not None and n_sample < n_cells:
        idx = np.random.choice(n_cells, n_sample, replace=False)
    else:
        idx = np.arange(n_cells)

    pc = np.zeros(len(idx))
    mn = pc.copy()
    for i, j in enumerate(idx):
        if scipy.sparse.issparse(adata.X):
            row = adata.X[j,:].toarray()
        else:
            row = adata.X[j,:]
        ok = row > 0
        mn[i] = np.sum(row[ok])/len(row)
        pc[i] = np.percentile(row, percentile)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if hue is not None and hue in adata.obs.columns:
        hue = adata.obs['sample'].iloc[idx]
    sns.scatterplot(x=mn, y=pc, hue=hue, ax=ax)
    ax.set_xlabel('mean normalised expression')
    ax.set_ylabel(f'{percentile}th percentile normalised expression')
    if ax is None:
        fig.tight_layout()
        return fig
    else:
        return ax
    

def normalisation_plots(adata, n_sample=1000, hue=None):
    """Plots to check normalisation of RNA data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing RNA expression data.
    n_sample : int, optional
        Number of random samples to plot in the kernel density plot and normalisation check. Default is 1000.
    hue : str, optional
        Column name in `adata.obs` to use for coloring the kernel density plot and histogram.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    _ = normalisation_kernel_density_plot(adata, ax=ax[0], hue=hue, n_sample=n_sample)
    _ = normalisation_check(adata, ax=ax[1], n_sample=n_sample, hue=hue)

    if hue is not None and hue in adata.obs.columns:
        for i, g in enumerate(adata.obs[hue].unique()):
            ax[2].hist(adata[adata.obs[hue] == g].X.sum(1), bins=100, label=g, alpha=0.5)
    else:
        ax[2].hist(adata.X.sum(1), bins=100)
    ax[2].set_xlabel('normalised counts')
    fig.tight_layout()
    return fig


def pca_heatmap(adata, component, n_genes=30, layer=None):
    """Plot heatmap of PCA scores for the top and bottom 20 variable genes.

    Seurat DimHeatmap equivalent: top N genes by absolute loading, cells ordered by PC score.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing PCA results in `varm['PCs']`.
    component : int
        The PCA component to visualize.
    layer : str, optional
        The layer of the AnnData object to use for the heatmap. If None, uses the main data layer.
    """

    # Get PC loadings for the selected component
    pc_loadings = adata.varm['PCs'][:, component]
    gene_names = adata.var_names
    # Top N genes by absolute loading
    top_idx = np.argsort(np.abs(pc_loadings))[::-1][:n_genes]
    top_genes = gene_names[top_idx]

    # Get cell scores for the component and order cells
    cell_scores = adata.obsm['X_pca'][:, component]
    cell_order = np.argsort(cell_scores)[::-1]  # descending order
    ordered_cells = adata.obs_names[cell_order]

    # Get expression matrix for top genes and ordered cells
    if layer is not None:
        expr = adata[ordered_cells, top_genes].layers[layer]
    else:
        expr = adata[ordered_cells, top_genes].X
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()

    # Optionally scale expression (z-score per gene)
    expr = (expr - expr.mean(axis=0)) / (expr.std(axis=0) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(expr.T, cmap='inferno', center=0, cbar_kws={'label': 'Scaled Expression'}, ax=ax)
    ax.set_yticks(np.arange(len(top_genes)) + 0.5)
    ax.set_yticklabels(top_genes, fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel('Cells (ordered by PC score)')
    ax.set_ylabel('Top genes (by PC loading)')
    ax.set_title(f'DimHeatmap: PC{component+1}')
    fig.tight_layout()
    return fig


def load_cell_cycle_genes(organism, gene_list=None):
    """Load cell cycle genes from the Tirosh et al. file included with the package.
    
    Parameters
    ----------
    organism : str
        Target organism to determine gene name case. Options:
        - 'human': uppercase genes (e.g., 'FOXP3')
        - 'mouse': title case genes (e.g., 'Foxp3')
    gene_list : list, optional
        List of gene names to filter against (e.g., adata.var_names).
        If provided, only genes present in this list will be returned.
    
    Returns
    -------
    dict
        Dictionary with 's_genes' and 'g2m_genes' as keys and lists of gene names as values.
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
        
        # Apply case conversion based on organism
        if organism.lower() == 'human':
            s_genes = [gene.upper() for gene in s_genes]
            g2m_genes = [gene.upper() for gene in g2m_genes]
        elif organism.lower() == 'mouse':
            s_genes = [gene.capitalize() for gene in s_genes]
            g2m_genes = [gene.capitalize() for gene in g2m_genes]
        else:
            raise ValueError(f"Unknown organism '{organism}'. Valid options are 'human' or 'mouse'.")
        
        # Filter genes if gene_list is provided
        if gene_list is not None:
            gene_set = set(gene_list)
            s_genes_filtered = [gene for gene in s_genes if gene in gene_set]
            g2m_genes_filtered = [gene for gene in g2m_genes if gene in gene_set]
            
            # Report missing genes
            missing_s = [gene for gene in s_genes if gene not in gene_set]
            missing_g2m = [gene for gene in g2m_genes if gene not in gene_set]
            
            if missing_s:
                print(f"Missing S phase genes: {missing_s}")
            if missing_g2m:
                print(f"Missing G2M phase genes: {missing_g2m}")
            
            return {
                's_genes': s_genes_filtered,
                'g2m_genes': g2m_genes_filtered,
            }
        
        return {
            's_genes': s_genes,
            'g2m_genes': g2m_genes,
        }
        
    except FileNotFoundError:
        print(f"Cell cycle genes file not found at {tirosh_file_path}")
        return {'s_genes': [], 'g2m_genes': []}
    except Exception as e:
        print(f"Error reading cell cycle genes file: {e}")
        return {'s_genes': [], 'g2m_genes': []}


def remove_doublet_clusters(adata, groupby='leiden'):
    """Remove groups identified as majority doublets by Scrublet."""
    tmp = adata.obs.groupby(groupby)['predicted_doublet'].agg(pd.Series.mode).reset_index()
    
    if tmp['predicted_doublet'].sum() == 0:
        print('no doublet clusters found')
        return

    remove = []
    for i in tmp.loc[tmp['predicted_doublet'] == True, groupby]:
        remove.append(i)

    print(f'doublet clusters removed: {remove}')
    return adata[~adata.obs[groupby].isin(remove)]


def get_vmax(adata, markers, percentile=95, min_vmax=0.1):
    """Get vmax values for a list of marker genes.
    
    Parameters
    ----------
    adata : AnnData
        The RNA data.
    markers : list
        List of marker genes.
    """
    vmax = [np.percentile(adata[:, adata.var_names.isin([g])].X.toarray(), percentile, axis=0) for g in markers]
    vmax = [v if v > min_vmax else min_vmax for v in vmax]
    return vmax


def rank_genes_groups_to_df(adata, key='rank_genes_groups'):
    """Convert the results from pairwise sc.tl.rank_genes_groups to a pandas DataFrame.

    Parameters
    ----------
    adata : AnnData
        The RNA data with results from sc.tl.rank_genes_groups.
    key : str, optional
        The key in adata.uns where the rank_genes_groups results are stored (default is 'rank_genes_groups').
    """
    result = adata.uns[key]
    groups = result['names'].dtype.names
    dfs = []
    for g in groups:
        df_dict = {}
        df_dict['group'] = [g] * len(result['names'][g])
        df_dict['reference'] = result['params']['reference']
        for k in result.keys():
            if k != 'params':
                df_dict[k] = result[k][g]
        df = pd.DataFrame(df_dict)
        df.set_index('names', inplace=True)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=False)


def celltypist_annotate(adata, recluster=False):
    """Quick annotation of cell types using CellTypist.

    Parameters
    ----------
    adata : AnnData
        The RNA data.
    recluster : bool, optional
        Whether to recluster the data before annotation (default is False).
    """
    if recluster:
        sc.pp.highly_variable_genes(adata)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=20)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=0.8)

    rna_tmp = adata.copy()
    rna_tmp.X = rna_tmp.layers['norm_1e4']

    predictions_subtypes = celltypist.annotate(rna_tmp, model = 'Immune_All_Low.pkl', majority_voting = True)
    rna_tmp = predictions_subtypes.to_adata(prefix='subtypes_')
    predictions_maintypes = celltypist.annotate(rna_tmp, model = 'Immune_All_High.pkl', majority_voting = True)
    rna_tmp = predictions_maintypes.to_adata(prefix='maintypes_')

    for col in rna_tmp.obs.columns:
        if col.startswith('subtypes_') or col.startswith('maintypes_'):
            adata.obs[col] = rna_tmp.obs[col].copy()


def cellphonedb_prepare(adata, annotation, outdir, layer=None,
                        viz_dir=os.path.expanduser('~/Documents/scRNA/cellphonedbviz'),
                        cpdb_file_path=os.path.expanduser('~/Documents/scRNA/cellphonedb-v5/cellphonedb.zip')
                        ):
    """Export data for CellPhoneDB analysis and run it with statistical method.

    Parameters
    ----------
    adata : AnnData
        The RNA data.
    annotation : str
        The annotation to use for cell types.
    outdir : str
        The output directory. This will be used for the cellphonedb run name, so be descriptive.
    layer : str, optional
        The layer key to use for the expression data (default is adata.X).
    viz_dir : str, optional
        The directory for visualization outputs (default is ~/Documents/scRNA/cellphonedbviz).
    cpdb_file_path : str, optional
        The path to the CellPhoneDB database zip file (default is ~/Documents/scRNA/cellphonedb-v5/cellphonedb.zip).
    """
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_name = outdir.name
    meta_file_path = outdir / 'metadata.tsv'
    counts_file_path = outdir / 'normalised_log_counts.h5ad'
    microenvs_file_path = outdir / 'microenvironment.tsv'
    out_suffix = ''

    meta = pd.DataFrame(adata.obs[[annotation]])
    meta.rename(columns={annotation: 'cell_type'}, inplace=True)
    meta.to_csv(meta_file_path, sep='\t', index=True)

    microenv = pd.DataFrame({'cell_type': adata.obs[annotation].unique(), 'microenvironment': 'Env1'})
    microenv.to_csv(microenvs_file_path, sep='\t', index=False)

    if layer is None:
        adata_export = ad.AnnData(X=adata.X, obs=meta, var=pd.DataFrame(index=adata.var_names))
    else:
        adata_export = ad.AnnData(X=adata.layers[layer], obs=meta, var=pd.DataFrame(index=adata.var_names))

    adata_export.write(counts_file_path, compression='gzip')

    cpdb_results = cpdb_statistical_analysis_method.call(
        cpdb_file_path = str(cpdb_file_path),
        meta_file_path = str(meta_file_path),
        counts_file_path = str(counts_file_path),
        counts_data = 'hgnc_symbol',
        microenvs_file_path = str(microenvs_file_path),
        score_interactions = True,
        iterations = 1000,
        threshold = 0.1,
        threads = 5,
        debug_seed = 42,
        result_precision = 3,
        pvalue = 0.05,
        subsampling = False,
        subsampling_log = False,
        subsampling_num_pc = 100,
        subsampling_num_cells = 1000,
        separator = '|',
        debug = False,
        output_path = str(outdir),
        output_suffix = out_suffix
    )

    out_files = {
        'deconvoluted_result': outdir / f'statistical_analysis_deconvoluted_{out_suffix}.txt',
        'deconvoluted_percents': outdir / f'statistical_analysis_deconvoluted_percents_{out_suffix}.txt',
        'analysis_means': outdir / f'statistical_analysis_means_{out_suffix}.txt',
        'pvalues': outdir / f'statistical_analysis_pvalues_{out_suffix}.txt',
        'relevant_interactions': outdir / f'statistical_analysis_significant_means_{out_suffix}.txt',
        'interaction_scores': outdir / f'statistical_analysis_interaction_scores_{out_suffix}.txt',
    }

    viz_dir = Path(viz_dir).resolve()
    viz_data_dir = viz_dir / 'data' / run_name
    viz_data_dir.mkdir(parents=True, exist_ok=True)

    for k, v in out_files.items():
        shutil.copy(v, viz_data_dir / v.name)

    shutil.copy(microenvs_file_path, viz_data_dir / microenvs_file_path.name)
    shutil.copy(cpdb_file_path, viz_data_dir / Path(cpdb_file_path).name)

    with open(viz_data_dir / 'config.yml', 'w') as f:
        f.write(f"separator: '|' \n"
                f"title: {run_name} \n"
                f"microenvironments: microenvironment.tsv \n"
                f"cellphonedb: cellphonedb.zip \n"
        )
        for k, v in out_files.items():
            f.write(f"{k}: {v.name} \n")

    print(f'\nCellPhoneDB analysis complete. Results saved to {viz_data_dir}/')
    print('\n Now go there and run "docker-compose build"')
