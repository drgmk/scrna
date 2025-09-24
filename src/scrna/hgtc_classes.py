"""
Classes to incorporate functions from `core.py` into HGTC workflow.

todo: any actual code here, e.g. from `firstlook.py`, should probably
 be put into functions in `core.py` or somewhere else. This file should
 ideally be close to 100% structural class details.

todo: `core.py` will use `scanpy_gpu_helper.pick_backend()` to automatically
 set whether we use `rapids-singlecell` or not. This is not very robust
 as there are some differences in some `sc` vs `rsc` functions. So should
 think about whether there is (a need for) a smarter way to set this.
"""

import os

import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.stats import norm
import struct

import matplotlib.pyplot as plt
import seaborn as sns
import dataframe_image as dfi

import scanpy as sc

# import scanpy.external as sce
# import rapids_singlecell as rsc
import decoupler as dc

import matplotlib.pyplot as plt

# import leidenalg
# import scvi
# import pyclustree as clt
# import doubletdetection

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple

# from hgtc_toolkit.src.scrna.scrna_functions import core as scfunc
from . import functions as scfunc

# from hgtc_toolkit.src.datascience.tools_mixtureModels import mixture_models as mix
# from hgtc_toolkit.src.datascience.orchestration.BaseClasses import (
# BaseMethod,
# BaseOperation,
# )


# placeholders until I can import hgtc_toolkit
class BaseOperation(ABC):
    pass


class BaseMethod(ABC):
    pass


class inspect(BaseOperation):
    class metrics(BaseMethod):
        pars = {
            "raw": {
                "type": "folderPath",
                "display": True,
                "required": True,
                "description": "File must be established prior to tool use.",
                "parentParam": {"master": "user"},
            },
            "metrics_folder": {
                "type": "folderPath",
                "display": True,
                "required": False,
                "description": "Folder must be established prior to tool use.",
                "parentParam": "raw",
            },
        }

        def parameterDerivatives(CONFIG, UID, **kwargs):
            CONFIG, pars = super().parameterDerivatives(UID, **kwargs)

            mpars = CONFIG["master"]["parameters"]
            file_keys = ["raw", "metrics"]
            for file in file_keys:
                if file in pars:
                    pars.update(
                        {
                            file: os.path.join(mpars["user"], pars[file]),
                        }
                    )

            return CONFIG, pars

        def main(adata: Any, pars: Dict[str, Any], **kwargs) -> Any:
            if "metrics_files" in adata.uns:
                metrics = []
                # migrate this to for file in pars['metrics_folder']
                for s, f in adata.uns["metrics_files"].items():
                    metrics.append(pd.read_csv(file_path / f))

                    metrics[-1]["index"] = s
                    for c in metrics[-1].columns:
                        try:
                            metrics[-1][c] = (
                                metrics[-1][c]
                                .str.replace(",", "", regex=False)
                                .str.replace("%", "", regex=False)
                                .astype(float)
                            )
                        except:
                            pass
                metrics = pd.concat(metrics)
                metrics["index"] = metrics["index"].str.split("/")
                metrics["index"] = list(map(lambda x: x[-2], metrics["index"]))
                metrics.set_index("index", inplace=True)
                keep_cols = [
                    "Estimated Number of Cells",
                    "Mean Reads per Cell",
                    "Median Genes per Cell",
                    "Total Genes Detected",
                    "Valid Barcodes",
                    "Reads Mapped Confidently to Genome",
                    "Sequencing Saturation",
                ]
                metrics_table = metrics[keep_cols].T

                fig, ax = plt.subplots()
                sns.scatterplot(
                    metrics,
                    x="Estimated Number of Cells",
                    y="Mean Reads per Cell",
                    size="Median Genes per Cell",
                    hue=metrics.index,
                    legend=True,
                    ax=ax,
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
                fig.tight_layout()
                fig.savefig(str(figs_path / "metrics_scatter.png"))
            else:
                metrics_table = pd.DataFrame(
                    {
                        "sample": sample_order,
                        "Estimated Number of Cells": rna.obs[sample_col]
                        .value_counts()
                        .reindex(sample_order)
                        .values,
                        "Total Genes Detected": [
                            rna[rna.obs[sample_col] == s].n_vars for s in sample_order
                        ],
                    }
                )
                metrics_table.set_index("sample", inplace=True)
                metrics_table = metrics_table.T

            metrics_img_path = f"{os.getcwd()}/figures/metrics_table.png"
            with open(metrics_img_path, "wb") as f:
                dfi.export(metrics_table, f)


class doublets(BaseOperation):
    class scrublet(BaseMethod):
        """
        scrublet:
        - package: rapids-singlecell
        - source: https://rapids-singlecell.readthedocs.io/en/latest/api/generated/rapids_singlecell.pp.scrublet.html
        """

        parameters = {}

        def main(adata: Any, pars: Dict[str, Any], **kwargs) -> Any:
            adata = rsc.pp.scrublet(adata, copy=True)
            sc.pl.scrublet_score_distribution(
                adata, frameon=False, save=f"doublets.png"
            )

            doublets = adata.obs["predicted_doublet"]
            doubletString = f"Doublets Detected: {len(doublets)}/{len(adata.obs)}={np.round(len(doublets)/len(adata)*100, 4)}%"

            if len(doublets) == len(adata.obs):
                print("Warning: all cells identified as doublets.")
                doubletString = r"!!: 100% doublets;" + doubletString

            return {"primary": adata, "comment": doubletString}


class filtering(BaseOperation):
    @classmethod
    def preamble(cls, adata, pars, **kwargs):
        # convert from dense to sparse matrix to reduce memory used
        adata.X = csr_matrix(adata.X).copy()
        # Save raw data into counts in preparation for integration
        adata.layers["counts"] = adata.X.copy()

        # Add qc metrics to adata.vars
        adata = cls.generate_feature_mask(adata, pars)
        qcargs = {
            "adata": adata,
            "qc_vars": pars["features"],
            "percent_top": [1],
            "log1p": True,
            "inplace": True,
        }
        sc.pp.calculate_qc_metrics(**qcargs)

        # nFeature by nCount plot, coloured by doublets
        if "doublet" in adata.obs.keys():

            # save_to = f"doublets.png"
            # if 'sampleID' in pars: save_to = f"{pars['sampleID']}/{save_to}"
            save_to = f"{pars['sampleID']}_doublets.png"

            vargs = {
                "adata": adata,
                "x": "total_counts",
                "y": "n_genes_by_counts",
                "color": ["doublet"],
                "save": save_to,
            }

            sc.pl.scatter(**vargs)
            # Filter out doublets
            adata = adata[~adata.obs["doublet"]].copy()
        return adata

    def generate_feature_mask(adata, pars):
        # Get all features as a flat list
        features = pars.get("features", [])
        if not isinstance(features, list):
            features = [features]
        if "feature" in pars:
            features.append(pars["feature"])

        for feature in features:
            if not feature or feature not in pars.get("gene_id", {}):
                continue

            value = pars["gene_id"][feature]
            if isinstance(value, list):
                adata.var[feature] = adata.var.index.str.startswith(tuple(value))
            elif isinstance(value, str):
                if "http://" in value:
                    genes = pd.read_table(value, skiprows=2, header=None)
                    adata.var[feature] = adata.var_names.isin(genes[0].values)
                else:
                    adata.var[feature] = adata.var.index.str.startswith(value)

        return adata

    class pct_counts_linear(BaseMethod):
        description = "2D-Filtering Alog c/o GKennedy; leveraging pct_counts_feature vs pct_counts_in_top_1_genes"
        parameters = {
            "n_gene_min": {
                "description": "",
                "type": "integer",
                "display": True,
                "required": True,
                "default": 0,
            },
            "n_gene_max": {
                "description": "",
                "type": "integer",
                "display": True,
                "required": True,
                "default": 1e6,
            },
            "feature": {
                "type": "list",
                "display": True,
                "required": True,
                "description": "Only: mt, ribo, hemo, log1p, & top_1",
                "multi_select": False,
                "options": [
                    "mitochondrial",
                    "ribosomal",
                    "hemoglobin",
                ],
                "unimplemented": [
                    "malat",
                    "log1p_total_counts",
                    "pct_counts_in_top_1_genes",
                    "log1p_n_genes_by_counts",
                ],
                "default": [
                    "mitochondrial",
                ],
            },
            "target_feature_id": {
                "type": "dict",
                "display": False,
                "default": {
                    "mitochondrial": "pct_counts_mitochondrial",
                    "ribosomal": "pct_counts_ribosomal",
                    "hemoglobin": "pct_counts_hemoglobin",
                    "malat": "pct_counts_malat",
                    "log1p_total_counts": "log1p_total_counts",
                    "log1p_n_genes_by_counts": "log1p_n_genes_by_counts",
                    "pct_counts_in_top_1_genes": "pct_counts_in_top_1_genes",
                },
            },
            "confidence_interval": {
                "description": "p-value for truncating mito distribution.",
                "type": "float",
                "display": True,
                "required": True,
                "default": 0.97,
            },
            "human": {
                "description": "Human gene-name library:",
                "type": "dict",
                "display": False,
                "required": True,
                "value": {
                    "minGeneCount": 500,
                    "gene_id": {
                        "mitochondrial": "MT-",
                        "ribosomal": "http://software.broadinstitute.org/gsea/msigdb/download_geneset.jsp?geneSetName=KEGG_RIBOSOME&fileType=txt",
                        "hemoglobin": "^HB[^(P)]",
                        "malat": "MALAT",
                    },
                },
            },
            "mouse": {
                "description": "Mouse gene-name library",
                "type": "dict",
                "display": False,
                "required": True,
                "value": {
                    "minGeneCount": 300,
                    "mitochondrial": "Mt-",
                    "ribosomal": "http://software.broadinstitute.org/gsea/msigdb/mouse/download_geneset.jsp?geneSetName=GOCC_RIBOSOMAL_SUBUNIT&fileType=txt",
                },
            },
        }

        @classmethod
        def parameterDerivatives(cls, CONFIG, UID, **kwargs) -> Any:
            # Call the parent method to get the base functionality
            """
            #cls.parameters.update(**super.assets)
            Desire this to collect operation-default parameters
            """
            CONFIG, pars = super().parameterDerivatives(CONFIG, UID, **kwargs)
            species = pars["config"]["species"].lower()
            if not species in ["human", "mouse"]:
                print(
                    "InputError: <experimentData.species> must be 'human' or 'mouse'."
                )
                return None
            else:
                for key, value in pars[species].items():
                    pars.update({key: value})
                del pars["human"]
                del pars["mouse"]

            return CONFIG, pars

        @classmethod
        def main(cls, adata: Any, pars: Dict[str, Any], **kwargs) -> Any:
            # Ensure required args
            adata = filtering.preamble(adata, pars)

            # Preplot {mt X pct_counts} linear scatter
            ftID = pars["target_feature_id"][pars["feature"]]
            fig = sns.relplot(
                x="pct_counts_in_top_1_genes",
                y=ftID,
                col="sampleID",
                data=adata.obs,
                size=0.1,
                col_wrap=4,
            )
            fig.set(xscale="log", yscale="log")
            fig.savefig(f"figures/{ftID}_vs_top1gene.png")

            scfunc.filter_cells_genes(adata)
            scfunc.compute_qc_metrics(adata)

            max_mt_pct = 20  # pct_mito mask limit could be inferred from PKM
            max_top1_pct = 15
            # filter off-linear reads
            mask = scfunc.trim_outliers(
                adata,
                groupby="sampleID",
                pct=pars["confidence_interval"],
                extra_mask={
                    "pct_counts_mt": [max_mt_pct, "max"],
                    "pct_counts_in_top_1_genes": [max_top1_pct, "max"],
                },
            )

            # Plot pre-filtered {nGenesbyCountsXtotal_counts}
            gc_raw = scfunc.plot_gene_counts(adata, hue="sampleID")
            gc_raw.savefig(f"figures/geneCount_vs_totalCount_raw.png")
            # Plot masked {nGenesbyCountsXtotal_counts}
            gc_mask = scfunc.plot_gene_counts(adata, hue="sampleID", mask=mask)
            gc_mask.savefig(f"figures/geneCount_vs_totalCount_mask.png")
            return {"adata": adata}


class annotation(BaseOperation):
    class cellTypist(BaseMethod):
        parameters = {}

        @classmethod
        def parameterDerivatives(cls, CONFIG, UID, **kwargs) -> Any:
            # Call the parent method to get the base functionality
            CONFIG, pars = super().parameterDerivatives(CONFIG, UID, **kwargs)

            species = CONFIG["master"]["parameters"]["species"].lower()
            pars.update({"species": species})

            return CONFIG, pars

        @classmethod
        def main(cls, adata: Any, pars: Dict[str, Any], **kwargs) -> Any:
            markers = dc.op.resource("PanglaoDB", organism=pars["species"])
            markers = markers[
                markers[pars["species"]].astype(bool)
                & markers["canonical_marker"].astype(bool)
                & (markers[f"{pars['species']}_sensitivity"].astype(float) > 0.5)
            ]
            markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]
            markers = markers.rename(
                columns={
                    "cell_type": "source",
                    "genesymbol": "target",
                    f"{pars['species']}_sensitivity": "weight",
                }
            )
            markers = markers[["source", "target", "weight"]]
            dc.mt.ulm(adata, markers, verbose=False)
            score = dc.pp.get_obsm(adata, key="score_ulm")
            df = dc.tl.rankby_group(
                adata=score,
                groupby="leiden",
                reference="rest",
                method="t-test_overestim_var",
            )
            df = df[df["stat"] > 0]
            dict_ann = (
                df[df["stat"] > 0]
                .groupby("group")
                .head(1)
                .set_index("group")["name"]
                .to_dict()
            )
            adata.obs["celltype_panglao"] = adata.obs["leiden"].map(dict_ann)

            scfunc.celltypist_annotate_immune(adata)

            return {"adata": adata}
