"""
First look.

Aim is to use scrna-functions to make a first pass and give an idea
of data quality and content.

```shell
scrna_firstlook -f path/to/data.h5ad
```

There are some assumptions:

- that there has been a little bit of organising already,
  i.e. combining data into one h5ad file with sample and perhaps groups
  columns in `adata.obs`.

- if there is a helpful order for samples, these will be listed in
  `adata.uns['sample_order']`.

- that any metrics metadata, e.g. from cellranger, will have been saved as
  dataframes, with a per-sample dictionary in `adata.uns['metrics_summary']`.

The results can be saved in another h5ad file if required. Most results
will have been saved in `adata.obs`, `adata.obsm`, or `adata.uns`.

todo: downsampling for plots

todo: think about whether VDJ and other sequencing output can/should
 be included.

todo: can probably save more memory by only keeping highly variable genes
 after normalisation. but need to restore afterwards.

todo: simple differential expression between groups if group_col given.
"""

import os
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd
import scanpy
import decoupler as dc
import seaborn as sns
import matplotlib.pyplot as plt
import scrna.functions as scfunc
import sklearn.metrics
import argparse
import gc

from fpdf import FPDF
from pdf2image import convert_from_path
from fpdf import FPDF
import dataframe_image as dfi

import scrna
import scrna.functions as scfunc


# subsampled rna (and vectors in also) for plotting
def rna_pl(rna, also=[], n=20_000):
    """Subset rna (and vectors in also) to len(n) for plotting."""
    if rna.n_obs > n:
        keep = np.random.choice(rna.n_obs, size=n, replace=False)
        rna_pl = rna[keep].copy()
        also_pl = [a[keep].copy() for a in also]
    else:
        rna_pl = rna.copy()
        also_pl = also

    if len(also) > 0:
        return rna_pl, also_pl
    else:
        return rna_pl


def main():
    # Default values for CLI
    sample_col = "sample"
    group_col = "group"
    use_raw = False
    max_mt_pct = 20.0
    max_top1_pct = 15.0
    min_genes = 200
    min_cells = 3
    pct_outlier_cutoff = 99.0
    n_neighbours = 20
    leiden_res = 0.8
    min_umap_dist = 0.3

    parser = argparse.ArgumentParser(
        description="First look at single-cell RNA-seq data"
    )
    parser.add_argument(
        "--file_path", "-f", type=str, help="Path to the input .h5ad file"
    )
    parser.add_argument(
        "--figs_path",
        type=str,
        metavar="path/datafile/datafile_firstlook",
        help="Path relative to the input file to save output figures",
    )
    parser.add_argument(
        "--sample_col",
        type=str,
        default=sample_col,
        metavar=sample_col,
        help="Column name for independent samples",
    )
    parser.add_argument(
        "--group_col",
        type=str,
        default=group_col,
        metavar=group_col,
        help="Column name for grouping samples",
    )
    parser.add_argument(
        "--use_raw", action="store_true", default=use_raw, help="Use raw data if set"
    )
    parser.add_argument(
        "--max_mt_pct",
        type=float,
        default=max_mt_pct,
        metavar=str(max_mt_pct),
        help="Max mitochondrial percentage for QC",
    )
    parser.add_argument(
        "--max_top1_pct",
        type=float,
        default=max_top1_pct,
        metavar=str(max_top1_pct),
        help="Max top 1 gene percentage for QC",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=min_genes,
        metavar=str(min_genes),
        help="Min genes per cell for QC",
    )
    parser.add_argument(
        "--min_cells",
        type=int,
        default=min_cells,
        metavar=str(min_cells),
        help="Min cells per gene for QC",
    )
    parser.add_argument(
        "--pct_outlier_cutoff",
        type=float,
        default=pct_outlier_cutoff,
        metavar=str(pct_outlier_cutoff),
        help="Percentile cutoff for outlier detection",
    )
    parser.add_argument(
        "--zero_center",
        action="store_true",
        default=False,
        help="Center data for PCA (densifies X)",
    )
    parser.add_argument(
        "--n_neighbours",
        type=int,
        default=n_neighbours,
        metavar=str(n_neighbours),
        help="Number of neighbours for clustering",
    )
    parser.add_argument(
        "--min_umap_dist",
        type=float,
        default=min_umap_dist,
        metavar=str(min_umap_dist),
        help="Minimum UMAP distance",
    )
    parser.add_argument(
        "--leiden_res",
        type=float,
        default=leiden_res,
        metavar=str(leiden_res),
        help="Leiden clustering resolution",
    )
    parser.add_argument(
        "--mem_mgmt",
        action="store_true",
        default=False,
        help="Enable GPU memory management",
    )
    parser.add_argument(
        "--save", "-s", action="store_true", help="Save the processed AnnData object"
    )

    args = parser.parse_args()

    file_path = Path(args.file_path)
    if args.figs_path:
        figs_path = file_path.parent / args.figs_path
    else:
        figs_path = file_path.parent / f"{file_path.stem}_firstlook"
    sample_col = args.sample_col
    group_col = args.group_col
    use_raw = args.use_raw
    max_mt_pct = args.max_mt_pct
    max_top1_pct = args.max_top1_pct
    min_genes = args.min_genes
    min_cells = args.min_cells
    pct_outlier_cutoff = args.pct_outlier_cutoff
    n_neighbours = args.n_neighbours
    leiden_res = args.leiden_res
    min_umap_dist = args.min_umap_dist
    save = args.save

    # Setup
    os.makedirs(figs_path, exist_ok=True)

    # Read in data
    rna = scanpy.read_h5ad(file_path)
    rna.var_names_make_unique()
    rna.obs_names_make_unique()
    print(f"reading: {file_path}")
    print(rna)

    if use_raw:
        rna.X = rna.raw.X
    else:
        if "counts" in rna.layers:
            rna.X = rna.layers["counts"]

    # cut down the object to save memory
    print(f"memory in original adata: {rna.__sizeof__() // 1_000_000} MB")
    rna.layers = None
    # rna.raw = None
    rna.obsm = None
    rna.varm = None
    rna.X = rna.X.astype(np.float32)
    print(f"             cut down to: {rna.__sizeof__() // 1_000_000} MB")

    # gpu helper, 4GB is about the limit for 16BG GPU (scrublet)
    sc = scrna.scanpy_gpu_helper.pick_backend()
    print(f"using backend: {'GPU' if sc.is_gpu else 'CPU'}")
    if args.mem_mgmt:
        sc.enable_memory_manager()
    print(f"memory management enabled: {sc._memory_manager_enabled}")

    # sample details
    if "sample_order" in rna.uns:
        sample_order = rna.uns["sample_order"]
    else:
        sample_order = sorted(rna.obs[sample_col].unique().tolist())

    sample_numbers = {s: i + 1 for i, s in enumerate(sample_order)}
    rna.obs["samp_no"] = pd.Categorical(rna.obs[sample_col].map(sample_numbers))
    sample_number_order = [sample_numbers[s] for s in sample_order]

    # assume each sample belongs to a unique group (e.g. infected vs control)
    if group_col != "":
        groups = rna.obs[group_col].unique().tolist()
        sample_group_mapping = {}
        for g in groups:
            samples_in_group = (
                rna.obs.loc[rna.obs[group_col] == g, sample_col].unique().tolist()
            )
            for s in samples_in_group:
                if s in sample_group_mapping.keys():
                    raise ValueError(f"Sample {s} appears in multiple groups!")
                sample_group_mapping[s] = g

    # Guess whether human or mouse
    organism = scfunc.guess_human_or_mouse(rna)
    print(f"assuming organism: {organism}")

    # compile/generate metrics table and plot
    if "metrics_summary" in rna.uns:
        metrics = []
        # for s, df in rna.uns['metrics_summary'].items():
        for s in sample_order:
            df = rna.uns["metrics_summary"][s]
            metrics.append(df)
            metrics[-1]["sample"] = s
            metrics[-1]["samp_no"] = [sample_numbers[s]]
            metrics[-1][group_col] = sample_group_mapping[s] if group_col != "" else ""
            # convert to float if possible
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
        metrics.set_index("samp_no", inplace=True)

        keep_cols = [
            "sample",
            "Estimated Number of Cells",
            "Mean Reads per Cell",
            "Median Genes per Cell",
            "Total Genes Detected",
            "Valid Barcodes",
            "Reads Mapped Confidently to Genome",
            "Sequencing Saturation",
        ]
        if group_col != "":
            keep_cols = keep_cols + [group_col]

        metrics_table = metrics[keep_cols].copy()

        # diagnostic plot for outliers
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            metrics_table,
            x="Estimated Number of Cells",
            y="Mean Reads per Cell",
            size="Median Genes per Cell",
            hue=metrics_table.index,
            style=group_col if group_col != "" else None,
            legend=True,
            ax=ax,
        )

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
        metrics_table["samp_no"] = metrics_table["sample"].map(sample_numbers)
        metrics_table["group"] = (
            metrics_table["sample"].map(sample_group_mapping) if group_col != "" else ""
        )
        metrics_table.set_index("samp_no", inplace=True)

        # diagnostic plot for outliers
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            metrics_table,
            x="Estimated Number of Cells",
            y="Total Genes Detected",
            hue=metrics_table.index,
            style="group",
            legend=True,
            ax=ax,
        )

    ax.set_title(
        f"metrics overview {'(basic)' if 'metrics_summary' not in rna.uns else ''}"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(str(figs_path / "metrics_scatter.pdf"))

    # keep metrics table
    rna.uns["meta_metrics_table"] = metrics_table

    # Quality control
    mask_cells, _ = scanpy.pp.filter_cells(rna, min_genes=min_genes, inplace=False)
    mask_genes, _ = scanpy.pp.filter_genes(rna, min_cells=min_cells, inplace=False)
    scfunc.compute_qc_metrics(rna)

    mask = scfunc.trim_outliers(
        rna,
        groupby=sample_col,
        extra_mask={
            "pct_counts_mt": [max_mt_pct, "max"],
            "pct_counts_in_top_1_genes": [max_top1_pct, "max"],
        },
        extra_mask_boolean=mask_cells,
        pct=pct_outlier_cutoff,
    )

    rna_toplot, mask_toplot = rna_pl(rna, also=[mask])
    fig = scfunc.plot_gene_counts(
        rna_toplot,
        hue="samp_no",
        order=sample_number_order,
        mask=mask_toplot[0],
        show_masked=True,
    )
    fig.savefig(str(figs_path / "gene_counts_per_sample.pdf"))

    # save and apply mask
    rna.uns["meta_qc_mask_cells"] = mask
    rna.uns["meta_qc_mask_genes"] = mask_genes
    rna = rna[mask, mask_genes].copy()

    # save this to compute log1p(norm) later
    rna.raw = rna.copy()

    # Doublets
    sc.pp.scrublet(rna, batch_key=sample_col)

    # Cell cycle
    cell_cycle_genes = scfunc.get_cell_cycle_genes(organism, gene_list=rna.var_names)
    sc.tl.score_genes_cell_cycle(
        rna,
        s_genes=cell_cycle_genes["s_genes"],
        g2m_genes=cell_cycle_genes["g2m_genes"],
    )

    # UMAPs
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna, zero_center=args.zero_center, max_value=10)
    sc.pp.highly_variable_genes(rna)
    sc.tl.pca(rna)
    sc.external.pp.harmony_integrate(rna, key=sample_col)
    sc.pp.neighbors(rna, n_neighbors=n_neighbours, use_rep="X_pca_harmony")
    sc.tl.umap(rna, min_dist=min_umap_dist, random_state=42)
    sc.tl.leiden(rna, resolution=leiden_res)

    # extra umaps with different min_dist
    sc.tl.umap(
        rna,
        min_dist=min_umap_dist / 2,
        random_state=42,
        key_added=f"X_umap_half_{min_umap_dist}_",
    )
    sc.tl.umap(
        rna,
        min_dist=min_umap_dist * 2,
        random_state=42,
        key_added=f"X_umap_twice_{min_umap_dist}_",
    )

    # the expensive processing is largely done, free up GPU memory
    sc.to_cpu(rna)
    gc.collect()

    # Batch correction metric
    batch_s_original = []
    batch_s_corrected = []
    for s in sample_order:
        tmp = rna[rna.obs[sample_col] == s].copy()
        # can't compute silhouette with only one cluster (e.g. small no. of cells)
        if len(tmp.obs["leiden"].unique()) < 2:
            continue
        scores = 1 - np.abs(
            sklearn.metrics.silhouette_samples(tmp.obsm["X_pca"], tmp.obs["leiden"])
        )
        batch_s_original.append(scores.sum() / len(scores))
        scores = 1 - np.abs(
            sklearn.metrics.silhouette_samples(
                tmp.obsm["X_pca_harmony"], tmp.obs["leiden"]
            )
        )
        batch_s_corrected.append(scores.sum() / len(scores))

    asw_original = np.mean(batch_s_original)
    asw_corrected = np.mean(batch_s_corrected)
    print(f"ASW original: {asw_original:.4f}, ASW corrected: {asw_corrected:.4f}")

    # fix this (again)
    rna.obs["samp_no"] = pd.Categorical(rna.obs["samp_no"])

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for i, x in enumerate(
        zip(
            ["samp_no", "predicted_doublet", "S_score", "G2M_score"],
            [False, False, True, True],
        )
    ):
        col, vminmax = x
        vmin, vmax = None, None
        if vminmax:
            vmin, vmax = np.percentile(rna.obs[col], (1, 99))
        sc.pl.umap(
            rna_pl(rna),
            color=col,
            vmin=vmin,
            vmax=vmax,
            ax=ax[i // 2, i % 2],
            show=False,
        )
        if i == 0:
            ax[i // 2, i % 2].set_title(
                f"samp_no (batch ASW {asw_original:.3f} -> {asw_corrected:.3f})"
            )
    fig.tight_layout()
    fig.savefig(str(figs_path / "umap_overview.pdf"))

    # umaps with different min_dist
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for i, min_dist in enumerate(["half", "twice"]):
        sc.pl.embedding(
            rna_pl(rna),
            f"X_umap_{min_dist}_{min_umap_dist}_",
            color="leiden",
            ax=ax[i],
            show=False,
        )
        ax[i].set_title(
            f'leiden (min_dist={min_umap_dist/2 if min_dist=="half" else min_umap_dist*2})'
        )

    fig.tight_layout()
    fig.savefig(str(figs_path / f"umap_leiden_min_dist.pdf"))

    fig.tight_layout()
    fig.savefig(str(figs_path / f"umap_leiden_min_dist.pdf"))

    # cell types
    markers = dc.op.resource("PanglaoDB", organism=organism)
    markers = markers[
        markers[organism].astype(bool)
        & markers["canonical_marker"].astype(bool)
        & (markers[f"{organism}_sensitivity"].astype(float) > 0.5)
    ]
    markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]
    markers = markers.rename(
        columns={
            "cell_type": "source",
            "genesymbol": "target",
            f"{organism}_sensitivity": "weight",
        }
    )
    markers = markers[["source", "target", "weight"]]
    if sc._using_rsc:
        sc.to_gpu(rna)
        sc._rsc.dcg.ulm(rna, markers, verbose=False)
        sc.to_cpu(rna)
    else:
        dc.mt.ulm(rna, markers, verbose=False)

    score = dc.pp.get_obsm(rna, key="score_ulm")
    df = dc.tl.rankby_group(
        adata=score, groupby="leiden", reference="rest", method="t-test_overestim_var"
    )
    df = df[df["stat"] > 0]
    dict_ann = (
        df[df["stat"] > 0].groupby("group").head(1).set_index("group")["name"].to_dict()
    )
    rna.obs["celltype_panglao"] = rna.obs["leiden"].map(dict_ann)

    # celltypist, which expects log1p(norm), just use scanpy (CPU) for now
    # otherwise we need to manage moving data between CPU and GPU
    rna.layers['log1p_1e4'] = rna.raw.X.copy()
    scanpy.pp.normalize_total(rna, target_sum=1e4, layer='log1p_1e4')
    scanpy.pp.log1p(rna, layer='log1p_1e4')
    scfunc.celltypist_annotate_immune(rna, layer_key='log1p_1e4')

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for i, x in enumerate(
        zip(
            ["leiden", "celltype_panglao", "maintypes_immune", "subtypes_immune"],
            [True, True, True, True],
        )
    ):
        col, show_legend = x
        sc.pl.umap(
            rna_pl(rna),
            color=col,
            ncols=2,
            legend_loc="on data" if show_legend else True,
            legend_fontsize=8,
            legend_fontoutline=2,
            legend_fontweight="normal",
            ax=ax[i // 2, i % 2],
            show=False,
        )
    fig.tight_layout()
    fig.savefig(str(figs_path / "umap_celltypes.pdf"))

    # marker genes, we want the most specific one for each of T, TfH, B, GC B, Stromal, FDC
    marker_genes = {
        "T": {"genes": ["CD3E", "CD3D", "TRAC"]},
        "TfH": {"genes": ["S1PR2", "CXCR5", "PDCD1"]},
        "B": {"genes": ["MS4A1", "CD79A", "CD19"]},
        "GC B": {"genes": ["AICDA", "S1PR2", "PCNA"]},
        "Stromal": {"genes": ["COL1A2", "PDGFRA", "VIM"]},
        "FDC": {"genes": ["CR2", "FDCSP", "CXCL13"]},
    }
    markers = scrna.celltypemarkers.CellTypeMarkers(
        organism=organism, data=marker_genes
    )
    markers.filter_genes(rna.var_names)
    marker_genes = markers.to_dict()

    fig, ax = plt.subplots(2, 3, figsize=(20, 7))
    for i, k in enumerate(marker_genes.keys()):
        a = ax.flatten()[i]
        gene = marker_genes[k][0]
        vmax = scfunc.get_vmax(rna, [gene], percentile=99)
        sc.pl.umap(rna_pl(rna), color=gene, vmax=vmax, ax=a, show=False)
        a.set_title(f"{k}: {gene}")

    fig.tight_layout()
    fig.savefig(figs_path / "umap_markers.pdf")

    # save the processed object, restoring some of the original data
    if save:
        rna_orig = sc.read_h5ad(file_path)
        rna_orig.var_names_make_unique()
        rna_orig.obs_names_make_unique()
        # original counts
        if "counts" in rna_orig.layers:
            rna.layers["counts"] = rna_orig.layers["counts"][mask, :][:, mask_genes]
        else:
            rna.layers["counts"] = rna_orig.X[mask, :][:, mask_genes]
        # apply mask to original obsm and varm
        for k in rna_orig.obsm.keys():
            rna.obsm[f"{k}_original"] = rna_orig.obsm[k][mask]
        for k in rna_orig.varm.keys():
            rna.varm[f"{k}_original"] = rna_orig.varm[k][mask_genes]

        rna.write_h5ad(figs_path / f"{file_path.stem}_firstlook.h5ad")

    # Save metrics_table as an image
    metrics_table_path = figs_path / "metrics_table.png"
    with open(metrics_table_path, "wb") as f:
        try:
            dfi.export(metrics_table, f)
        except:
            dfi.export(metrics_table, f, table_conversion="matplotlib")

    # Prepare file paths for images
    pdfs = [
        "metrics_scatter",
        "gene_counts_per_sample",
        "umap_overview",
        "umap_celltypes",
        "umap_markers",
        "umap_leiden_min_dist",
    ]
    pdf_paths = {}
    png_paths = {}
    for pdf in pdfs:
        pdf_paths[pdf] = figs_path / f"{pdf}.pdf"
        png_paths[pdf] = figs_path / f"{pdf}.png"

    # Convert PDFs to PNGs for FPDF (if needed)
    def pdf_to_png(pdf_path, out_path):
        images = convert_from_path(str(pdf_path), dpi=150)
        images[0].save(out_path, "PNG")

    # convert PDFs to PNGs using multiprocessing
    # with mp.Pool() as pool:
    #     pool.starmap(
    #         pdf_to_png,
    #         [(pdf_paths[k], png_paths[k]) for k in pdf_paths.keys()],
    #     )
    # convert one-by-one
    for k in pdf_paths.keys():
        pdf_to_png(pdf_paths[k], png_paths[k])

    # Create PDF report
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()

    page_w, page_h = pdf.w, pdf.h
    margin_x, margin_y = 1, 1

    # Add title and subtitle to the PDF page
    title = file_path.name
    subtitle = str(file_path.resolve())

    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(margin_x)
    pdf.cell(page_w - 2 * margin_x, 2, subtitle, align="C", new_y="NEXT")
    title_h = pdf.get_y() + 2

    # Panel positions and sizes
    panel_w, panel_h = (page_w - 2 * margin_x) / 2, (
        page_h - 2 * margin_y - title_h
    ) / 2

    pdf.image(
        str(metrics_table_path),
        x=margin_x,
        y=margin_y + title_h,
        w=panel_w,
        h=panel_h / 2,
        keep_aspect_ratio=True,
    )
    pdf.image(
        str(png_paths["metrics_scatter"]),
        x=margin_x,
        y=margin_y + title_h + panel_h / 2,
        w=panel_w / 2,
        h=panel_h / 2,
        keep_aspect_ratio=True,
    )

    pdf.image(
        str(png_paths["gene_counts_per_sample"]),
        x=margin_x,
        y=margin_y + title_h + panel_h,
        w=panel_w,
        h=panel_h,
    )
    pdf.image(
        str(png_paths["umap_overview"]),
        x=margin_x + panel_w,
        y=margin_y + title_h,
        w=panel_w,
        h=panel_h,
    )
    pdf.image(
        str(png_paths["umap_celltypes"]),
        x=margin_x + panel_w,
        y=margin_y + panel_h + title_h,
        w=panel_w,
        h=panel_h,
    )

    # second page with more specific stuff
    pdf.add_page()
    pdf.image(
        str(png_paths["umap_markers"]), x=margin_x, y=margin_y, w=panel_w * 2, h=panel_h
    )
    pdf.image(
        str(png_paths["umap_leiden_min_dist"]),
        x=margin_x,
        y=margin_y + panel_h,
        w=panel_w,
        h=panel_h / 2,
    )

    report_path = figs_path / "firstlook_report.pdf"
    pdf.output(str(report_path))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
