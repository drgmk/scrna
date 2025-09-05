import os
import shutil
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse
import seaborn as sns
import anndata as ad
import scanpy as sc
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method


def do_qc(rna, organism, extra_genes=[]):

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
        rna.var[k] = rna.var_names.str.startswith(pct_counts[k][0])
        if len(pct_counts[k]) > 1:
            for s in pct_counts[k][1:]:
                rna.var[k] = np.logical_or(rna.var[k], rna.var_names.str.startswith(s))
            
        sc.pp.calculate_qc_metrics(rna, qc_vars=[k], percent_top=None, log1p=False, inplace=True)

    rna.var['nomalat'] = np.invert(rna.var['malat'])
    sc.pp.calculate_qc_metrics(rna, qc_vars=['nomalat'], percent_top=[1], log1p=False, inplace=True)


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


def remove_doublet_clusters(rna):
    """Remove clusters identified as majority doublets by Scrublet."""
    tmp = rna.obs.groupby('leiden')['predicted_doublet'].agg(pd.Series.mode).reset_index()
    i_doublet = tmp.loc[tmp['predicted_doublet'] == True, 'leiden'].iloc[0]
    print(f'removing cluster {i_doublet} as doublets')

    rna = rna[~rna.obs['leiden'].isin([i_doublet])].copy()

def get_vmax(rna, markers, percentile=95, min_vmax=0.1):
    """Get vmax values for a list of marker genes.
    
    Parameters
    ----------
    rna : AnnData
        The RNA data.
    markers : list
        List of marker genes.
    """
    vmax = [np.percentile(rna[:, rna.var_names.isin([g])].X.toarray(), percentile, axis=0) for g in markers]
    vmax = [v if v > min_vmax else min_vmax for v in vmax]
    return vmax


def cellphonedb_prepare(rna, annotation, outdir, layer=None,
                        viz_dir=os.path.expanduser('~/Documents/scRNA/cellphonedbviz'),
                        cpdb_file_path=os.path.expanduser('~/Documents/scRNA/cellphonedb-v5/cellphonedb.zip')
                        ):
    """Export data for CellPhoneDB analysis and run it with statistical method.

    Parameters
    ----------
    rna : AnnData
        The RNA data.
    annotation : str
        The annotation to use for cell types.
    outdir : str
        The output directory. This will be used for the cellphonedb run name, so be descriptive.
    layer : str, optional
        The layer key to use for the expression data (default is rna.X).
    viz_dir : str, optional
        The directory for visualization outputs (default is ~/Documents/scRNA/cellphonedbviz).
    cpdb_file_path : str, optional
        The path to the CellPhoneDB database zip file (default is ~/Documents/scRNA/cellphonedb-v5/cellphonedb.zip).
    """

    outdir = outdir.rstrip('/') + '/'
    os.makedirs(outdir, exist_ok=True)

    run_name = os.path.basename(outdir.rstrip('/'))
    meta_file_path = f'{outdir}metadata.tsv'
    counts_file_path = f'{outdir}normalised_log_counts.h5ad'
    microenvs_file_path = f'{outdir}microenvironment.tsv'
    out_suffix = ''

    meta = pd.DataFrame(rna.obs[[annotation]])
    meta.rename(columns={
        annotation: 'cell_type'
        }, inplace=True)
    meta.to_csv(meta_file_path, sep='\t', index=True)

    microenv = pd.DataFrame({'cell_type': rna.obs[annotation].unique(), 'microenvironment': 'Env1'})
    microenv.to_csv(microenvs_file_path, sep='\t', index=False)

    if layer is None:
        rna_export = ad.AnnData(X=rna.X, obs=meta, var=pd.DataFrame(index=rna.var_names))
    else:
        rna_export = ad.AnnData(X=rna.layers[layer], obs=meta, var=pd.DataFrame(index=rna.var_names))

    rna_export.write(counts_file_path, compression='gzip')

    # now run cellphonedb
    cpdb_results = cpdb_statistical_analysis_method.call(
        cpdb_file_path = cpdb_file_path,                 # mandatory: CellphoneDB database zip file.
        meta_file_path = meta_file_path,                 # mandatory: tsv file defining barcodes to cell label.
        counts_file_path = counts_file_path,             # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
        counts_data = 'hgnc_symbol',                     # defines the gene annotation in counts matrix.
        # active_tfs_file_path = active_tf_path,           # optional: defines cell types and their active TFs.
        microenvs_file_path = microenvs_file_path,       # optional (default: None): defines cells per microenvironment.
        score_interactions = True,                       # optional: whether to score interactions or not. 
        iterations = 1000,                               # denotes the number of shufflings performed in the analysis.
        threshold = 0.1,                                 # defines the min % of cells expressing a gene for this to be employed in the analysis.
        threads = 5,                                     # number of threads to use in the analysis.
        debug_seed = 42,                                 # debug randome seed. To disable >=0.
        result_precision = 3,                            # Sets the rounding for the mean values in significan_means.
        pvalue = 0.05,                                   # P-value threshold to employ for significance.
        subsampling = False,                             # To enable subsampling the data (geometri sketching).
        subsampling_log = False,                         # (mandatory) enable subsampling log1p for non log-transformed data inputs.
        subsampling_num_pc = 100,                        # Number of componets to subsample via geometric skectching (dafault: 100).
        subsampling_num_cells = 1000,                    # Number of cells to subsample (integer) (default: 1/3 of the dataset).
        separator = '|',                                 # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
        debug = False,                                   # Saves all intermediate tables employed during the analysis in pkl format.
        output_path = outdir,                            # Path to save results.
        output_suffix = out_suffix                       # Replaces the timestamp in the output files by a user defined string in the  (default: None).
    )

    # copy files to cellphonedbviz folder
    out_files = {
        'deconvoluted_result': os.path.join(outdir, f'statistical_analysis_deconvoluted_{out_suffix}.txt'),
        'deconvoluted_percents': os.path.join(outdir, f'statistical_analysis_deconvoluted_percents_{out_suffix}.txt'),
        'analysis_means': os.path.join(outdir, f'statistical_analysis_means_{out_suffix}.txt'),
        'pvalues': os.path.join(outdir, f'statistical_analysis_pvalues_{out_suffix}.txt'),
        'relevant_interactions': os.path.join(outdir, f'statistical_analysis_significant_means_{out_suffix}.txt'),
        'interaction_scores': os.path.join(outdir, f'statistical_analysis_interaction_scores_{out_suffix}.txt'),
    }
    
    os.makedirs(os.path.join(viz_dir, 'data', run_name), exist_ok=True)
    for k in out_files.keys():
        shutil.copy(out_files[k], os.path.join(viz_dir, 'data', run_name, os.path.basename(out_files[k])))

    shutil.copy(microenvs_file_path, os.path.join(viz_dir, 'data', run_name, os.path.basename(microenvs_file_path)))
    shutil.copy(cpdb_file_path, os.path.join(viz_dir, 'data', run_name, os.path.basename(cpdb_file_path)))

    # add files to yaml file
    with open(os.path.join(viz_dir, 'data', run_name, 'config.yml'), 'w') as f:
        f.write(f"separator: '|' \n"
                f"title: {run_name} \n"
                f"microenvironments: microenvironment.tsv \n"
                f"cellphonedb: cellphonedb.zip \n"
        )
        for k, v in out_files.items():
            f.write(f"{k}: {os.path.basename(v)} \n")

    # help message
    print(f'\nCellPhoneDB analysis complete. Results saved to {viz_dir}/data/{run_name}/')
    print('\n Now go there and run "docker-compose build"')
