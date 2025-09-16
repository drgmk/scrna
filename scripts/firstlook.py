
# First look
# Automated first look analysis of single cell RNA-seq data.
# - Aim is to use scrna-functions to make a first pass and give an idea of data quality and content.
# - We will assume that there has been a little bit of organising, i.e. combining data into one h5ad file with sample and groups in `adata.obs`.
# - We will also assume that any metrics metadata will be pointed to, relative to `adata.h5ad`, with a per-sample dictionaryin `adata.uns['metrics_files']`.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
import seaborn as sns
import matplotlib.pyplot as plt
import scrna_functions as scfunc
import argparse

from fpdf import FPDF
from pdf2image import convert_from_path
from fpdf import FPDF
import dataframe_image as dfi


def main():
    # Default values for CLI
    figs_path = './figures/firstlook/'
    sample_col = 'sample'
    # group_col = ''
    use_raw = False
    max_mt_pct = 20.
    max_top1_pct = 15.
    min_genes = 200
    min_cells = 3
    pct_outlier_cutoff = 99.
    n_neighbours = 20
    leiden_res = 0.8

    parser = argparse.ArgumentParser(description="First look at single-cell RNA-seq data")
    parser.add_argument("--file_path", "-f", type=str, help="Path to the input .h5ad file")
    parser.add_argument("--figs_path", type=str, default=figs_path, metavar=figs_path, help="Path to save figures")
    parser.add_argument("--sample_col", type=str, default=sample_col, metavar=sample_col, help="Column name for independent samples")
    # parser.add_argument("--group_col", type=str, default=group_col, metavar=group_col, help="Column name for grouping samples")
    parser.add_argument("--use_raw", action='store_true', default=use_raw, help="Use raw data if set")
    parser.add_argument("--max_mt_pct", type=float, default=max_mt_pct, metavar=max_mt_pct, help="Max mitochondrial percentage for QC")
    parser.add_argument("--max_top1_pct", type=float, default=max_top1_pct, metavar=max_top1_pct, help="Max top 1 gene percentage for QC")
    parser.add_argument("--min_genes", type=int, default=min_genes, metavar=min_genes, help="Min genes per cell for QC")
    parser.add_argument("--min_cells", type=int, default=min_cells, metavar=min_cells, help="Min cells per gene for QC")
    parser.add_argument("--pct_outlier_cutoff", type=float, default=pct_outlier_cutoff, metavar=pct_outlier_cutoff, help="Percentile cutoff for outlier detection")
    parser.add_argument("--n_neighbours", type=int, default=n_neighbours, metavar=n_neighbours, help="Number of neighbours for clustering")
    parser.add_argument("--leiden_res", type=float, default=leiden_res, metavar=leiden_res, help="Leiden clustering resolution")

    args = parser.parse_args()

    file_path = Path(args.file_path)
    figs_path = Path(args.figs_path)
    sample_col = args.sample_col
    # group_col = args.group_col
    use_raw = args.use_raw
    max_mt_pct = args.max_mt_pct
    max_top1_pct = args.max_top1_pct
    min_genes = args.min_genes
    min_cells = args.min_cells
    pct_outlier_cutoff = args.pct_outlier_cutoff
    n_neighbours = args.n_neighbours
    leiden_res = args.leiden_res

    # Setup
    os.makedirs(figs_path, exist_ok=True)

    # Read in data
    rna = sc.read_h5ad(file_path)
    rna.var_names_make_unique()
    rna.obs_names_make_unique()
    print(f'reading: {file_path}')
    print(rna)

    # todo: check if there is also VDJ or other data

    if use_raw:
        rna.X = rna.raw.X
    else:
        rna.raw = rna.copy()

    if 'sample_order' in rna.uns:
        sample_order = rna.uns['sample_order']
    else:
        sample_order = sorted(rna.obs[sample_col].unique().tolist())

    # Guess whether human or mouse
    organism = scfunc.guess_human_or_mouse(rna)
    print(f'assuming organism: {organism}')

    # Metrics
    if 'metrics_files' in rna.uns:
        metrics = []
        for s, f in rna.uns['metrics_files'].items():
            metrics.append(pd.read_csv(file_path / f))
            metrics[-1]['index'] = s
            for c in metrics[-1].columns:
                try:
                    metrics[-1][c] = metrics[-1][c].str.replace(',', '', regex=False).str.replace('%', '', regex=False).astype(float)
                except:
                    pass
        metrics = pd.concat(metrics)
        metrics['index'] = metrics['index'].str.split('/')
        metrics['index'] = list(map(lambda x: x[-2], metrics['index']))
        metrics.set_index('index', inplace=True)
        keep_cols = ['Estimated Number of Cells', 'Mean Reads per Cell', 'Median Genes per Cell', 'Total Genes Detected',
                    'Valid Barcodes', 'Reads Mapped Confidently to Genome', 'Sequencing Saturation']
        metrics_table = metrics[keep_cols].T
        fig, ax = plt.subplots()
        sns.scatterplot(metrics, x='Estimated Number of Cells', y='Mean Reads per Cell', size='Median Genes per Cell',
                        hue=metrics.index, legend=True, ax=ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        fig.tight_layout()
        fig.savefig(str(figs_path / 'metrics_scatter.png'))
    else:
        metrics_table = pd.DataFrame({'sample': sample_order, 
                                      'Estimated Number of Cells': rna.obs[sample_col].value_counts().reindex(sample_order).values,
                                      'Total Genes Detected': [rna[rna.obs[sample_col] == s].n_vars for s in sample_order]})
        metrics_table.set_index('sample', inplace=True)
        metrics_table = metrics_table.T

    # Quality control
    scfunc.filter_cells_genes(rna)
    scfunc.compute_qc_metrics(rna)

    mask = scfunc.trim_outliers(rna, groupby=sample_col,
                                extra_mask={'pct_counts_mt': [max_mt_pct, 'max'],
                                            'pct_counts_in_top_1_genes': [max_top1_pct, 'max']},
                                pct=pct_outlier_cutoff,
                                )

    fig = scfunc.plot_gene_counts(rna, hue=sample_col, order=sample_order, mask=mask, show_masked=True)
    fig.savefig(str(figs_path / 'gene_counts_per_sample.pdf'))

    # apply mask
    rna = rna[mask, :].copy()

    # Doublets
    sc.pp.scrublet(rna, batch_key=sample_col)

    # Cell cycle
    cell_cycle_genes = scfunc.get_cell_cycle_genes(organism, gene_list=rna.var_names)
    sc.tl.score_genes_cell_cycle(rna, s_genes=cell_cycle_genes['s_genes'], g2m_genes=cell_cycle_genes['g2m_genes'])

    # UMAPs
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna)
    sc.tl.pca(rna)
    sc.external.pp.harmony_integrate(rna, key=sample_col)
    sc.pp.neighbors(rna, n_neighbors=n_neighbours, use_rep="X_pca_harmony")
    sc.tl.umap(rna, random_state=42)
    sc.tl.leiden(rna, resolution=leiden_res)

    fig , ax = plt.subplots(2, 2, figsize=(10,7))
    for i, x in enumerate(zip([sample_col, 'predicted_doublet', 'S_score', 'G2M_score'],
                              [False, False, True, True])):
        col, vminmax = x
        vmin, vmax = None, None
        if vminmax:
            vmin, vmax = np.percentile(rna.obs[col], (1, 99))
        sc.pl.umap(rna, color=col, vmin=vmin, vmax=vmax,
                   ax=ax[i//2, i%2], show=False)
    fig.tight_layout()
    fig.savefig(str(figs_path / 'umap_overview.pdf'))

    # cell types
    markers = dc.op.resource("PanglaoDB", organism=organism)
    markers = markers[
        markers[organism].astype(bool)
        & markers["canonical_marker"].astype(bool)
        & (markers[f"{organism}_sensitivity"].astype(float) > 0.5)
    ]
    markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]
    markers = markers.rename(columns={"cell_type": "source", "genesymbol": "target", f"{organism}_sensitivity": "weight"})
    markers = markers[["source", "target", "weight"]]
    dc.mt.ulm(rna, markers, verbose=False)
    score = dc.pp.get_obsm(rna, key="score_ulm")
    df = dc.tl.rankby_group(adata=score, groupby="leiden", reference="rest", method="t-test_overestim_var")
    df = df[df["stat"] > 0]
    dict_ann = df[df["stat"] > 0].groupby("group").head(1).set_index("group")["name"].to_dict()
    rna.obs['celltype_panglao'] = rna.obs["leiden"].map(dict_ann)

    scfunc.celltypist_annotate_immune(rna)

    fig , ax = plt.subplots(2, 2, figsize=(10,7))
    for i, x in enumerate(zip(['leiden', 'celltype_panglao', 'maintypes_immune', 'subtypes_immune'],
                              [True, True, True, True])):
        col, show_legend = x
        sc.pl.umap(rna, color=col, ncols=2,
                   legend_loc='on data' if show_legend else True, legend_fontsize=8, legend_fontoutline=2, legend_fontweight='normal',
                   ax=ax[i//2, i%2], show=False)
    fig.tight_layout()
    fig.savefig(str(figs_path / 'umap_celltypes.pdf'))

    # Generate summary
    metrics_img_path = figs_path / "metrics_table.png"
    with open(metrics_img_path, 'wb') as f:
        dfi.export(metrics_table, f)

    gene_counts_img = figs_path / "gene_counts_per_sample.pdf"
    umap_overview_img = figs_path / "umap_overview.pdf"
    umap_celltypes_img = figs_path / "umap_celltypes.pdf"

    def pdf_to_png(pdf_path, out_path):
        images = convert_from_path(str(pdf_path), dpi=150)
        images[0].save(out_path, 'PNG')

    gene_counts_png = figs_path / "gene_counts_per_sample.png"
    umap_overview_png = figs_path / "umap_overview.png"
    umap_celltypes_png = figs_path / "umap_celltypes.png"

    pdf_to_png(gene_counts_img, gene_counts_png)
    pdf_to_png(umap_overview_img, umap_overview_png)
    pdf_to_png(umap_celltypes_img, umap_celltypes_png)

    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    page_w, page_h = pdf.w, pdf.h
    margin_x, margin_y = 1, 1
    title = file_path.name
    subtitle = str(file_path.resolve())
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(margin_x)
    pdf.cell(page_w - 2 * margin_x, 2, subtitle, align="C", new_y='NEXT')
    title_h = pdf.get_y() + 2
    panel_w, panel_h = (page_w - 2 * margin_x) / 2, (page_h - 2 * margin_y - title_h) / 2
    pdf.image(str(metrics_img_path), x=margin_x, y=margin_y + title_h)
    pdf.image(str(gene_counts_png), x=margin_x, y=margin_y + title_h + panel_h, w=panel_w, h=panel_h)
    pdf.image(str(umap_overview_png), x=margin_x + panel_w, y=margin_y + title_h, w=panel_w, h=panel_h)
    pdf.image(str(umap_celltypes_png), x=margin_x + panel_w, y=margin_y + panel_h + title_h, w=panel_w, h=panel_h)
    report_path = figs_path / "firstlook_report.pdf"
    pdf.output(str(report_path))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
