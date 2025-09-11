import pytest
import scanpy as sc
from scrna_functions import core
from scrna_functions import CellTypeMarkers

@pytest.fixture(scope="module")
def pbmc():
    adata = sc.datasets.pbmc68k_reduced()
    # add some random sample group annotations for testing
    adata.obs['sample'] = ['A'] * (adata.n_obs // 2) + ['B'] * (adata.n_obs - adata.n_obs // 2)
    return adata

def test_get_plot_list(pbmc):
    result = core.get_plot_list(pbmc)
    assert isinstance(result, list)

def test_do_qc(pbmc):
    core.do_qc(pbmc)
    # No return, just check no error

def test_trim_outliers(pbmc):
    x = pbmc.obs['n_genes_by_counts']
    y = pbmc.obs['total_counts']
    mask = core.trim_outliers(x, y)
    assert len(mask) == len(x)

def test_plot_gene_counts(pbmc):
    core.do_qc(pbmc)
    fig = core.plot_gene_counts(pbmc)
    assert fig is not None

def test_plot_top_genes(pbmc):
    fig = core.plot_top_genes(pbmc)
    assert fig is not None

def test_plot_umaps(pbmc):
    fig = core.plot_umaps(pbmc)
    assert fig is not None

def test_plot_cell_counts(pbmc):
    fig = core.plot_cell_counts(pbmc, y='bulk_labels')
    assert fig is not None

def test__seurat_clr(pbmc):
    x = pbmc.X[0].A1 if hasattr(pbmc.X[0], 'A1') else pbmc.X[0]
    result = core._seurat_clr(x)
    assert result.shape == x.shape

def test_clr_normalize_each_cell(pbmc):
    adata_norm = core.clr_normalize_each_cell(pbmc)
    assert adata_norm.shape == pbmc.shape

def test_normalisation_kernel_density_plot(pbmc):
    fig = core.normalisation_kernel_density_plot(pbmc)
    assert fig is not None

def test_normalisation_check(pbmc):
    fig = core.normalisation_check(pbmc)
    assert fig is not None

def test_normalisation_plots(pbmc):
    fig = core.normalisation_plots(pbmc, hue='sample')
    assert fig is not None

def test_pca_heatmap(pbmc):
    fig = core.pca_heatmap(pbmc, component=0)
    assert fig is not None

def test_load_cell_cycle_genes():
    result = core.load_cell_cycle_genes('human')
    assert 's_genes' in result and 'g2m_genes' in result

def test_remove_doublet_clusters(pbmc):
    pbmc.obs['predicted_doublet'] = [False] * (pbmc.n_obs - 10) + [True] * 10
    core.remove_doublet_clusters(pbmc, groupby='sample')
    # No return, just check no error

def test_get_vmax(pbmc):
    markers = CellTypeMarkers('human')
    markers.filter_genes(gene_names=pbmc.var_names.tolist())
    for k in markers.keys():
        vmax = core.get_vmax(pbmc, markers=markers[k])
        assert isinstance(vmax, list)

def test_cellphonedb_prepare(pbmc):
    pass


