import pytest
import scanpy as sc
import scrna


@pytest.fixture(scope="module")
def pbmc():
    adata = sc.datasets.pbmc68k_reduced()
    # add some random sample group annotations for testing
    adata.obs["sample"] = ["A"] * (adata.n_obs // 2) + ["B"] * (
        adata.n_obs - adata.n_obs // 2
    )
    return adata


def test_pick_backend_load(pbmc):
    b = scrna.scanpy_gpu_helper.pick_backend()
    b.pp.normalize_total(pbmc, target_sum=1e4)


def test_backend_passthrough():
    b = scrna.scanpy_gpu_helper.pick_backend()
    rna = b.datasets.pbmc68k_reduced()
    b.pp.normalize_total(rna)
