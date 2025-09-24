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


def test_get_markers(pbmc):
    markers = CellTypeMarkers("human")
    markers.filter_genes(gene_names=pbmc.var_names.tolist())
    for k in markers.keys():
        vmax = scfunc.get_vmax(pbmc, markers=markers[k])
        assert isinstance(vmax, list)
