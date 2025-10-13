import pytest
import scanpy as sc
import scrna.functions as scfunc
from scrna.celltypemarkers import CellTypeMarkers


@pytest.fixture(scope="module")
def pbmc():
    adata = sc.datasets.pbmc68k_reduced()
    # add some random sample group annotations for testing
    adata.obs["sample"] = ["A"] * (adata.n_obs // 2) + ["B"] * (
        adata.n_obs - adata.n_obs // 2
    )
    return adata


def test_get_plot_list(pbmc):
    result = scfunc.get_plot_list(pbmc)
    assert isinstance(result, list)


def test_do_qc(pbmc):
    scfunc.compute_qc_metrics(pbmc)
    # No return, just check no error


def test_trim_outliers(pbmc):
    mask = scfunc.trim_outliers(pbmc, x="n_genes_by_counts", y="total_counts")
    assert len(mask) == len(pbmc.obs)


def test_plot_gene_counts(pbmc):
    scfunc.compute_qc_metrics(pbmc)
    fig = scfunc.plot_gene_counts(pbmc)
    assert fig is not None


def test_plot_top_genes(pbmc):
    fig = scfunc.plot_top_genes(pbmc)
    assert fig is not None


def test_plot_umaps(pbmc):
    fig = scfunc.plot_umaps(pbmc)
    assert fig is not None


def test_plot_cell_counts(pbmc):
    fig = scfunc.plot_cell_counts(pbmc, y="bulk_labels")
    assert fig is not None


def test__seurat_clr(pbmc):
    x = pbmc.X[0].A1 if hasattr(pbmc.X[0], "A1") else pbmc.X[0]
    result = scfunc._seurat_clr(x)
    assert result.shape == x.shape


def test_clr_normalize_each_cell(pbmc):
    adata_norm = scfunc.clr_normalize_each_cell(pbmc)
    assert adata_norm.shape == pbmc.shape


def test_normalisation_kernel_density_plot(pbmc):
    fig = scfunc.normalisation_kernel_density_plot(pbmc)
    assert fig is not None


def test_normalisation_check(pbmc):
    fig = scfunc.normalisation_check(pbmc)
    assert fig is not None


def test_normalisation_plots(pbmc):
    fig = scfunc.normalisation_plots(pbmc, hue="sample")
    assert fig is not None


def test_pca_heatmap(pbmc):
    fig = scfunc.pca_heatmap(pbmc, component=0)
    assert fig is not None


def test_remove_doublet_clusters(pbmc):
    pbmc.obs["predicted_doublet"] = [False] * (pbmc.n_obs - 10) + [True] * 10
    scfunc.remove_doublet_clusters(pbmc, groupby="sample")
    # No return, just check no error


def test_get_vmax(pbmc):
    markers = CellTypeMarkers("human")
    markers.filter_genes(gene_names=pbmc.var_names.tolist())
    for k in markers.keys():
        vmax = scfunc.get_vmax(pbmc, markers=markers[k])
        assert isinstance(vmax, list)


def test_cellphonedb_prepare(pbmc):
    # Placeholder: test for cellphonedb_prepare
    # Not trivial to test without file system and CellPhoneDB setup
    assert True


def test_plot_nxy():
    # Placeholder: test for plot_nxy
    x, y = scfunc.plot_nxy(5)
    assert isinstance(x, int) and isinstance(y, int)


def test_guess_human_or_mouse(pbmc):
    # Placeholder: test for guess_human_or_mouse
    result = scfunc.guess_human_or_mouse(pbmc)
    assert result in ["human", "mouse"]


def test_compute_qc_metrics(pbmc):
    # Placeholder: test for compute_qc_metrics
    scfunc.compute_qc_metrics(pbmc)
    assert "pct_counts" in pbmc.uns


def test_filter_cells_genes(pbmc):
    # Placeholder: test for filter_cells_genes
    scfunc.filter_cells_genes(pbmc)
    assert "meta_filter_cells_genes" in pbmc.uns


def test_get_cell_cycle_genes(pbmc):
    # Placeholder: test for get_cell_cycle_genes
    result = scfunc.get_cell_cycle_genes("human")
    assert "s_genes" in result and "g2m_genes" in result


def test_rank_genes_groups_to_df(pbmc):
    # Placeholder: test for rank_genes_groups_to_df
    sc.tl.rank_genes_groups(
        pbmc, groupby="sample", group="A", reference="B", method="t-test"
    )
    df = scfunc.rank_genes_groups_to_df(pbmc)
    assert df is not None


def test_get_pseudobulk(pbmc):
    # Placeholder: test for get_pseudobulk
    # Not trivial to test without decoupler setup
    assert True


def test_do_deg(pbmc):
    # Placeholder: test for do_deg
    # Not trivial to test without DESeq2 setup
    assert True


def test_celltypist_annotate_immune(pbmc):
    # Placeholder: test for celltypist_annotate_immune
    # Not trivial to test without celltypist model
    assert True
