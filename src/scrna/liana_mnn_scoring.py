"""Small MultiNicheNet-inspired scoring helpers for LIANA LR results.

This module adds an optional, transparent prioritisation layer on top of a
LIANA ligand-receptor (LR) candidate table. It is intended for the situation
where we want LIANA's Python workflow and plotting ecosystem, but we also want
the ranked LR list to resemble the broad logic of MultiNicheNet.

The core idea is deliberately simple:

1. Use LIANA to define candidate LR pairs and source/target cell types.
2. Add receiver/sender differential-expression evidence for ligand and
   receptor genes.
3. Add a ligand-activity score computed elsewhere, typically from a compact
   Python approximation of NicheNet ligand activity.
4. Add condition/cell-type expression specificity for the ligand and receptor.
5. Add donor-level support: in how many samples are both ligand and receptor
   detectably expressed?
6. Combine those components with the same six conceptual weights used by
   MultiNicheNet's "regular" scenario.

How this relates to multinichenetr
----------------------------------
The relevant MultiNicheNet implementation is
``multinichenetr::generate_prioritization_tables``. In the regular scenario it
combines six criteria:

* ligand DE
* receptor DE
* ligand activity
* ligand expression specificity
* receptor expression specificity
* fraction of samples where ligand and receptor are both expressed

For ligand and receptor DE, MultiNicheNet gives half-weight to the logFC rank
and half-weight to a signed p-value rank. Its final score is approximately:

```
(0.5 * ligand_lfc + 0.5 * ligand_signed_p
 + 0.5 * receptor_lfc + 0.5 * receptor_signed_p
 + ligand_activity + ligand_expression + receptor_expression
 + sample_lr_fraction) / 6
```

This module mirrors that formula, but it does not port the whole R package.
There are several intentional differences:

* LIANA defines the LR candidate table; MultiNicheNet builds its own sender/
  receiver table and applies its own expression filters.
* This module uses percentile ranks via pandas; MultiNicheNet often uses
  ``rank(...) / max(rank(...))`` for DE and ``nichenetr::scale_quantile`` for
  activity/expression components.
* This module computes expression from ``adata.X`` as sample means/fractions
  and then ranks condition/cell-type combinations per gene. MultiNicheNet uses
  precomputed ``sender_receiver_info`` tables, including pseudobulk expression
  summaries that may include package-specific filtering and batch handling.
* Ligand activity is passed in as a table. In the current notebook this is a
  small Python AUPR approximation. MultiNicheNet uses
  ``nichenetr::predict_ligand_activities`` followed by z-score and quantile
  scaling.
* Gene-name harmonisation here only rescues obvious R ``make.names`` changes
  such as ``HLA.A`` to ``HLA-A``. MultiNicheNet also applies alias conversion
  to the expression data and resources.

These differences are useful rather than accidental: they keep the Python
analysis explainable and reproducible with standard objects, while making the
remaining disagreement with MultiNicheNet interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


__all__ = [
    "LIANA_FACTOR_COLUMNS",
    "REGULAR_WEIGHTS",
    "SHADOW_FACTOR_COLUMNS",
    "ResourceHarmonisation",
    "compute_ligand_activity",
    "compute_ligand_activity_aupr",
    "compute_ligand_activity_ulm",
    "diagnose_ligand_activity_inputs",
    "group_gene_expression",
    "harmonize_lr_resource",
    "make_liana_lr_factor_plot",
    "mnn_shadow_prioritise",
    "pseudobulk_deseq_by_celltype",
    "read_ligand_target_prior",
    "run_liana_lr_candidates",
    "sample_gene_expression",
    "top_overlap_summary",
]


REGULAR_WEIGHTS = {
    "de_ligand": 1.0,
    "de_receptor": 1.0,
    "activity_scaled": 1.0,
    "exprs_ligand": 1.0,
    "exprs_receptor": 1.0,
    "frac_exprs_ligand_receptor": 1.0,
}
"""Conceptual MultiNicheNet regular-scenario weights.

The denominator of the final score is the sum of these six weights. The two
ligand DE subcomponents and two receptor DE subcomponents each receive half of
their parent criterion weight, matching the MultiNicheNet regular formula.
"""

SHADOW_FACTOR_COLUMNS = [
    ("shadow_scaled_lfc_ligand", "Ligand\nLFC"),
    ("shadow_scaled_p_val_ligand_adapted", "Ligand\nDE p"),
    ("shadow_scaled_lfc_receptor", "Receptor\nLFC"),
    ("shadow_scaled_p_val_receptor_adapted", "Receptor\nDE p"),
    ("shadow_max_scaled_activity", "Ligand\nactivity"),
    ("shadow_scaled_pb_ligand", "Ligand\nexpr."),
    ("shadow_scaled_pb_receptor", "Receptor\nexpr."),
    ("shadow_fraction_expressing_ligand_receptor", "Donor LR\nsupport"),
]
"""Default component labels for plotting the shadow score factors."""

LIANA_FACTOR_COLUMNS = [
    ("ligand_activity_score", "Ligand\nactivity"),
    ("sample_lr_fraction_score", "Donor LR\nsupport"),
    ("interaction_props_score", "Cell expr\nprop."),
    ("ligand_lfc_score", "Ligand\nLFC"),
    ("ligand_de_p_score", "Ligand\nDE p"),
    ("receptor_lfc_score", "Receptor\nLFC"),
    ("receptor_de_p_score", "Receptor\nDE p"),
    ("ligand_specificity_score", "Ligand\nspecificity"),
    ("receptor_specificity_score", "Receptor\nspecificity"),
    ("ligand_expr_score", "Ligand\nexpr."),
    ("receptor_expr_score", "Receptor\nexpr."),
]
"""Component labels for the earlier, broader LIANA prioritisation score."""


@dataclass
class ResourceHarmonisation:
    """Summary of LR-resource gene-name rescue.

    Attributes
    ----------
    n_rows:
        Number of LR rows in the input resource.
    n_rows_both_present_before:
        Number of LR rows where both ligand and receptor were found in
        ``adata.var_names`` before rescue.
    n_rows_both_present_after:
        Number of LR rows where both genes were found after simple rescue.
    n_entities:
        Number of unique ligand/receptor gene names in the resource.
    n_entities_present_before:
        Number of unique resource genes found in ``adata.var_names`` before
        rescue.
    n_entities_present_after:
        Number of unique resource genes found after simple rescue.
    rescued_entities:
        Resource gene names that were changed by the rescue rule.

    Notes
    -----
    MultiNicheNet's R workflow often uses ``make.names`` on gene symbols,
    turning symbols such as ``HLA-A`` into ``HLA.A``. The rescue tracked here is
    intentionally conservative and only reverses dot-to-hyphen changes when the
    hyphenated symbol exists in the AnnData object.
    """

    n_rows: int
    n_rows_both_present_before: int
    n_rows_both_present_after: int
    n_entities: int
    n_entities_present_before: int
    n_entities_present_after: int
    rescued_entities: list[str]

    def as_frame(self) -> pd.DataFrame:
        """Return the summary as a one-row DataFrame for notebook display."""
        return pd.DataFrame([self.__dict__])


def _unique_str(values: Iterable[object]) -> pd.Index:
    """Return unique non-missing values as strings, preserving first appearance."""
    return pd.Index(pd.Series(values).dropna().astype(str).drop_duplicates())


def _rank01(values: Iterable[object]) -> pd.Series:
    """Percentile-rank scale finite values to 0..1, with missing as 0.

    This is the main scaling primitive used in the shadow score. It is close to
    the DE scaling in MultiNicheNet, which uses ``rank(...) / max(rank(...))``.
    It is not identical to ``nichenetr::scale_quantile``, which MultiNicheNet
    uses for activity and expression components. The rank transform is chosen
    here because it is simple, robust to outliers, and easy to explain.
    """
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    out = pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    finite = np.isfinite(s)
    if finite.any():
        out.loc[finite] = s.loc[finite].rank(method="average", pct=True)
    return out


def _signed_logp(lfc: Iterable[object], pvalue: Iterable[object], floor: float = 1e-300) -> pd.Series:
    """Convert logFC and p-value into a signed significance score.

    MultiNicheNet uses ``-log10(p_value) * sign(logFC)`` for its adapted
    p-value rankings. This keeps strongly significant up-regulated genes high,
    significant down-regulated genes low, and non-significant genes in between.
    """
    lfc = pd.to_numeric(pd.Series(lfc), errors="coerce")
    pvalue = pd.to_numeric(pd.Series(pvalue), errors="coerce").clip(lower=floor)
    return -np.log10(pvalue) * np.sign(lfc)


def _mean_vector(x) -> np.ndarray:
    """Column means for dense or sparse matrices."""
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x).mean(axis=0).ravel()


def _fraction_vector(x) -> np.ndarray:
    """Column-wise fraction of cells with expression greater than zero."""
    if sparse.issparse(x):
        return np.asarray((x > 0).mean(axis=0)).ravel()
    return (np.asarray(x) > 0).mean(axis=0).ravel()


def _normalise_lr_columns(lr_df: pd.DataFrame) -> pd.DataFrame:
    """Return an LR table with canonical ``source/target/ligand/receptor`` columns.

    LIANA result tables often contain ``ligand_complex`` and
    ``receptor_complex``. The shadow scorer is currently gene-level, so those
    columns are copied into ``ligand`` and ``receptor`` when needed. Complex
    handling is therefore inherited from LIANA's preprocessing rather than
    reimplemented here.
    """
    lr = lr_df.copy().reset_index(drop=True)
    if "ligand" not in lr.columns and "ligand_complex" in lr.columns:
        lr["ligand"] = lr["ligand_complex"]
    if "receptor" not in lr.columns and "receptor_complex" in lr.columns:
        lr["receptor"] = lr["receptor_complex"]
    required = {"source", "target", "ligand", "receptor"}
    missing = sorted(required - set(lr.columns))
    if missing:
        raise ValueError(f"LR table is missing required columns: {missing}")
    for col in required:
        lr[col] = lr[col].astype(str)
    return lr


def harmonize_lr_resource(
    lr_resource: pd.DataFrame,
    var_names: Iterable[str],
    ligand_col: str = "ligand",
    receptor_col: str = "receptor",
) -> tuple[pd.DataFrame, ResourceHarmonisation]:
    """Map R ``make.names``-style resource names back to h5ad gene names.

    Parameters
    ----------
    lr_resource:
        DataFrame containing at least ligand and receptor columns.
    var_names:
        Gene names from the AnnData object that will be analysed with LIANA.
    ligand_col, receptor_col:
        Column names in ``lr_resource``.

    Returns
    -------
    tuple
        ``(harmonised_resource, summary)``. The resource has duplicated
        ligand/receptor rows removed after rescue.

    Notes
    -----
    MultiNicheNet's R workflow applies alias conversion and ``make.names`` to
    expression genes and resources. The resulting LR resources can contain
    names such as ``HLA.A`` where the Python AnnData still contains ``HLA-A``.
    This helper only rescues names when:

    * the exact resource name is absent from ``adata.var_names``; and
    * replacing dots with hyphens creates a gene that is present.

    It does not attempt broad alias conversion. That keeps the rule auditable
    and avoids accidentally rewriting genuine dot-containing symbols or
    transcript IDs.
    """
    genes = set(pd.Index(var_names).astype(str))
    lr = lr_resource.copy()
    for col in (ligand_col, receptor_col):
        lr[col] = lr[col].astype(str)

    entities = _unique_str(pd.concat([lr[ligand_col], lr[receptor_col]]))
    present_before = entities.isin(genes)

    def rescue(gene: str) -> str:
        if gene in genes:
            return gene
        candidate = gene.replace(".", "-")
        return candidate if candidate in genes else gene

    rescued_entities = pd.Index([rescue(gene) for gene in entities])
    present_after = rescued_entities.isin(genes)
    rescued_names = entities[(~present_before) & present_after].tolist()

    ligand_before = lr[ligand_col].isin(genes)
    receptor_before = lr[receptor_col].isin(genes)
    out = lr.copy()
    out[ligand_col] = out[ligand_col].map(rescue)
    out[receptor_col] = out[receptor_col].map(rescue)
    ligand_after = out[ligand_col].isin(genes)
    receptor_after = out[receptor_col].isin(genes)

    summary = ResourceHarmonisation(
        n_rows=len(lr),
        n_rows_both_present_before=int((ligand_before & receptor_before).sum()),
        n_rows_both_present_after=int((ligand_after & receptor_after).sum()),
        n_entities=len(entities),
        n_entities_present_before=int(present_before.sum()),
        n_entities_present_after=int(present_after.sum()),
        rescued_entities=rescued_names,
    )
    return out.drop_duplicates([ligand_col, receptor_col]).reset_index(drop=True), summary


def pseudobulk_deseq_by_celltype(
    adata,
    condition_value: str,
    reference_value: str,
    celltype_col: str = "label",
    condition_col: str = "tonsillitis",
    sample_col: str = "donor_id",
    celltypes: Iterable[str] | None = None,
    counts_layer: str = "counts",
    min_cells: int = 10,
    min_counts: int = 1000,
    min_count: int = 10,
    min_total_count: int = 15,
    large_n: int = 10,
    min_prop: float = 0.7,
    min_sample_prop: float = 0.1,
    min_samples: int = 2,
    n_cpus: int = 8,
) -> pd.DataFrame:
    """Run sample-aware pseudobulk DE separately for each cell type.

    Parameters
    ----------
    adata:
        AnnData containing raw counts in ``counts_layer`` and the requested
        observation columns.
    condition_value, reference_value:
        Numerator and denominator levels for the DESeq2 contrast.
    celltype_col, condition_col, sample_col:
        Observation columns defining cell type, biological condition, and
        independent sample/donor.
    celltypes:
        Cell types to analyse. By default all values in ``celltype_col`` are
        used in first-appearance order.
    counts_layer:
        AnnData layer containing integer-like raw counts.
    min_cells, min_counts:
        Sample-level filters applied after pseudobulk aggregation.
    min_count, min_total_count, large_n, min_prop:
        Parameters passed to ``decoupler.pp.filter_by_expr``.
    min_sample_prop, min_samples:
        Parameters passed to ``decoupler.pp.filter_by_prop``.
    n_cpus:
        Number of CPUs used by pydeseq2's default inference backend.

    Returns
    -------
    pandas.DataFrame
        Concatenated pydeseq2 result tables. Gene symbols form the index and
        ``celltype_col`` identifies the cell type for each result row.

    Notes
    -----
    This package-level helper delegates pseudobulk construction, filtering, and
    DESeq2 fitting to :func:`scrna.functions.dc_get_pseudobulk` and
    :func:`scrna.functions.dc_deseq_deg`. Keeping that behaviour in one place
    means changes to the lab's standard pseudobulk workflow are inherited by
    this scoring workflow.

    MultiNicheNet normally obtains cell-type DE through its muscat-based R
    workflow. Pseudobulk DESeq2 is conceptually comparable but not numerically
    identical; different gene filtering, dispersion estimation, and hypothesis
    testing can change the DE ranks used by the final score.
    """
    from .functions import dc_deseq_deg, dc_get_pseudobulk

    required_obs = {celltype_col, condition_col, sample_col}
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    if missing_obs:
        raise ValueError(f"adata.obs is missing required columns: {missing_obs}")
    if counts_layer not in adata.layers:
        raise ValueError(f"adata.layers does not contain counts layer {counts_layer!r}")

    if celltypes is None:
        celltypes = _unique_str(adata.obs[celltype_col])
    else:
        celltypes = _unique_str(celltypes)

    results = []
    for celltype in celltypes:
        subset = adata[adata.obs[celltype_col].astype(str) == str(celltype)].copy()
        observed_conditions = set(subset.obs[condition_col].astype(str))
        required_conditions = {str(condition_value), str(reference_value)}
        if not required_conditions.issubset(observed_conditions):
            missing = sorted(required_conditions - observed_conditions)
            raise ValueError(f"Cell type {celltype!r} is missing condition levels: {missing}")

        pdata, _ = dc_get_pseudobulk(
            subset,
            min_cells=min_cells,
            sample=sample_col,
            group=condition_col,
            obsm_plot=False,
            counts_layer=counts_layer,
            min_counts=min_counts,
            filter_min_count=min_count,
            filter_min_total_count=min_total_count,
            filter_large_n=large_n,
            filter_min_prop=min_prop,
            filter_by_prop_min_prop=min_sample_prop,
            filter_by_prop_min_samples=min_samples,
            diagnostic_plots=False,
            compute_pca=False,
        )
        result = dc_deseq_deg(
            pdata,
            design=f"~{condition_col}",
            contrast=[condition_col, condition_value, reference_value],
            n_cpus=n_cpus,
        ).copy()
        result[celltype_col] = str(celltype)
        results.append(result)

    if not results:
        raise ValueError("No cell types were supplied for pseudobulk DE")
    combined = pd.concat(results, axis=0)
    combined.index = combined.index.astype(str)
    combined.index.name = "gene"
    return combined


def run_liana_lr_candidates(
    adata,
    dea_df: pd.DataFrame,
    resource: pd.DataFrame,
    groupby: str = "label",
    receiver: str | None = None,
    senders: Iterable[str] | None = None,
    expr_prop: float = 0.05,
    stat_keys: tuple[str, ...] = ("stat", "pvalue", "padj"),
    use_raw: bool = False,
    complex_col: str = "stat",
    return_all_lrs: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate a LIANA LR candidate table from expression and DE results.

    ``adata`` should normally be restricted to the condition being prioritised,
    while ``dea_df`` should describe that condition versus its reference. The
    resource must contain ``ligand`` and ``receptor`` columns; use
    :func:`harmonize_lr_resource` first when the table came from an R
    MultiNicheNet export.

    ``receiver`` and ``senders`` are optional convenience filters applied after
    LIANA returns its table. Passing ``receiver='FDC'`` and
    ``senders=('TFH', 'GCB')`` reproduces the analysis used in the example
    notebook.

    This function intentionally leaves candidate generation to LIANA rather
    than porting MultiNicheNet's sender/receiver construction and expression
    filtering. Consequently, the candidate universes can differ even when the
    same LR database is supplied.
    """
    import liana as li

    required = {"ligand", "receptor"}
    missing = sorted(required - set(resource.columns))
    if missing:
        raise ValueError(f"LR resource is missing required columns: {missing}")

    result = li.multi.df_to_lr(
        adata,
        dea_df=dea_df,
        resource=resource[["ligand", "receptor"]].drop_duplicates(),
        expr_prop=expr_prop,
        groupby=groupby,
        stat_keys=list(stat_keys),
        use_raw=use_raw,
        complex_col=complex_col,
        verbose=verbose,
        return_all_lrs=return_all_lrs,
    )
    if receiver is not None:
        result = result[result["target"].astype(str) == str(receiver)]
    if senders is not None:
        sender_values = set(_unique_str(senders))
        result = result[result["source"].astype(str).isin(sender_values)]
    return result.copy()


def sample_gene_expression(
    adata,
    genes: Iterable[str],
    groupby: str = "label",
    condition_col: str = "tonsillitis",
    sample_col: str = "donor_id",
    min_cells: int = 10,
) -> pd.DataFrame:
    """Compute sample-level expression summaries for LR genes.

    Parameters
    ----------
    adata:
        AnnData object. ``adata.X`` should contain the expression values to use
        for scoring; in the notebook this is log1p-normalised counts.
    genes:
        Genes to summarise, usually the union of LR ligands and receptors.
    groupby:
        Cell-type column, for example ``label``.
    condition_col:
        Condition/group column, for example ``tonsillitis``.
    sample_col:
        Donor/sample column.
    min_cells:
        Minimum number of cells required for a sample-condition-celltype group.

    Returns
    -------
    pandas.DataFrame
        Long table with one row per sample, condition, cell type, and gene. It
        contains ``mean_expr`` and ``frac_expr``.

    MultiNicheNet comparison
    ------------------------
    MultiNicheNet constructs several ``sender_receiver_info`` tables, including
    sample-level average expression, pseudobulk expression, and expression
    fraction. This function provides the analogous Python ingredients from
    ``adata.X`` directly. It is less elaborate than MultiNicheNet's full
    preprocessing, but it is transparent and keeps sample-level donor support
    explicit.
    """
    var_names = pd.Index(adata.var_names.astype(str))
    genes_present = [gene for gene in _unique_str(genes) if gene in set(var_names)]
    if not genes_present:
        raise ValueError("None of the requested LR genes are present in adata.var_names")

    gene_idx = var_names.get_indexer(genes_present)
    obs = adata.obs[[condition_col, groupby, sample_col]].copy()
    obs = obs.astype(str)
    obs["_row"] = np.arange(adata.n_obs)

    frames = []
    for (condition, celltype, sample), sub_obs in obs.groupby(
        [condition_col, groupby, sample_col],
        observed=True,
        sort=False,
    ):
        row_idx = sub_obs["_row"].to_numpy()
        if len(row_idx) < min_cells:
            continue
        x = adata.X[row_idx][:, gene_idx]
        frames.append(
            pd.DataFrame(
                {
                    condition_col: condition,
                    groupby: celltype,
                    sample_col: sample,
                    "gene": genes_present,
                    "mean_expr": _mean_vector(x),
                    "frac_expr": _fraction_vector(x),
                    "n_cells": len(row_idx),
                }
            )
        )

    if not frames:
        raise ValueError("No sample/cell-type groups passed min_cells")
    return pd.concat(frames, ignore_index=True)


def group_gene_expression(
    sample_expr: pd.DataFrame,
    groupby: str = "label",
    condition_col: str = "tonsillitis",
    sample_col: str = "donor_id",
) -> pd.DataFrame:
    """Collapse sample-level expression to condition/cell-type expression scores.

    The returned table contains mean expression and mean expression fraction
    across samples for each condition, cell type, and gene. It also adds
    per-gene percentile ranks:

    * ``group_expr_score`` ranks condition/cell-type mean expression.
    * ``group_frac_score`` ranks condition/cell-type expression fraction.

    The shadow score currently uses ``group_expr_score`` as the analogue of
    MultiNicheNet's ``scaled_pb_ligand`` and ``scaled_pb_receptor``. We retain
    the fraction score for diagnostics, but MultiNicheNet's regular final score
    does not include it separately; sample-level LR support covers the fraction
    criterion.
    """
    grouped = (
        sample_expr.groupby([condition_col, groupby, "gene"], observed=True)
        .agg(
            group_expr=("mean_expr", "mean"),
            group_frac=("frac_expr", "mean"),
            n_samples=(sample_col, "nunique"),
            n_cells=("n_cells", "sum"),
        )
        .reset_index()
    )
    grouped["group_expr_score"] = (
        grouped.groupby("gene", observed=True)["group_expr"]
        .transform(lambda x: _rank01(x).to_numpy())
        .fillna(0)
    )
    grouped["group_frac_score"] = (
        grouped.groupby("gene", observed=True)["group_frac"]
        .transform(lambda x: _rank01(x).to_numpy())
        .fillna(0)
    )
    return grouped


def _series_looks_like_gene_names(values: Iterable[object], n: int = 1000) -> bool:
    """Heuristically distinguish gene-name strings from numeric matrix values."""
    sample = pd.Series(values).dropna().astype(str).head(n)
    if sample.empty:
        return False
    numeric_fraction = pd.to_numeric(sample, errors="coerce").notna().mean()
    return numeric_fraction < 0.5


def _index_looks_like_gene_names(index: pd.Index, n: int = 1000) -> bool:
    """Return whether a non-default DataFrame index plausibly contains genes."""
    if isinstance(index, pd.RangeIndex):
        return False
    values = pd.Index(index).to_series(index=None)
    return _series_looks_like_gene_names(values, n=n)


def _wide_ligand_target_to_long(
    raw: pd.DataFrame,
    target_values: Iterable[str],
    top_n_targets: int | None = 250,
    min_weight: float | None = 0,
) -> pd.DataFrame:
    """Convert a target-by-ligand matrix to source/target/weight long form."""
    target_values = pd.Index(target_values).astype(str)
    rows = []
    for ligand in raw.columns.astype(str):
        weights = pd.to_numeric(raw[ligand], errors="coerce")
        weights.index = target_values
        weights = weights.dropna()
        if min_weight is not None:
            weights = weights[weights > min_weight]
        if top_n_targets is not None:
            weights = weights.nlargest(top_n_targets)
        if not weights.empty:
            rows.append(
                pd.DataFrame(
                    {
                        "source": ligand,
                        "target": weights.index.astype(str),
                        "weight": weights.to_numpy(),
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=["source", "target", "weight"])
    return pd.concat(rows, ignore_index=True)


def read_ligand_target_prior(
    path: str | Path | None = None,
    top_n_targets: int | None = 250,
    min_weight: float | None = 0,
) -> pd.DataFrame:
    """Read a NicheNet-style ligand-target prior in long or wide format.

    Parameters
    ----------
    path:
        Pickle, CSV, or TSV file. If omitted, common filenames are searched in
        the current directory and ``data/``.
    top_n_targets:
        Maximum target links retained per ligand. Set to ``None`` to retain all
        links.
    min_weight:
        Minimum positive regulatory-potential weight. Set to ``None`` to avoid
        weight filtering.

    Returns
    -------
    pandas.DataFrame
        Long-form network with ``source`` (ligand), ``target`` (target gene),
        and numeric ``weight`` columns.

    Notes
    -----
    Wide matrices must preserve target genes either as the DataFrame index or
    as their first column. A default integer RangeIndex is rejected because it
    usually means target names were lost when an R matrix was exported to a
    pandas pickle. MultiNicheNet uses the full NicheNet ligand-target matrix;
    retaining the top 250 targets per ligand follows its common
    ``top_n_target`` setting while keeping the Python representation compact.
    """
    if path is None:
        candidates = [
            Path("data/ligand_target_matrix.pkl"),
            Path("data/ligand_target_matrix_long.csv"),
            Path("data/ligand_target_matrix_long.tsv"),
            Path("ligand_target_matrix.pkl"),
            Path("ligand_target_matrix_long.csv"),
            Path("ligand_target_matrix_long.tsv"),
            Path("ligand_target_prior.csv"),
            Path("ligand_target_prior.tsv"),
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "No ligand-target prior found; pass a pickle, CSV, or TSV path"
            )
    path = Path(path)

    if path.suffix.lower() in {".pkl", ".pickle"}:
        raw = pd.read_pickle(path)
    else:
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        raw = pd.read_csv(path, sep=separator)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"Ligand-target prior must be a DataFrame, got {type(raw)}")

    raw = raw.copy()
    raw.columns = [str(column).strip() for column in raw.columns]
    lower = {column.lower(): column for column in raw.columns}

    if {"ligand", "target"}.issubset(lower) or {"source", "target"}.issubset(lower):
        source_col = lower.get("ligand") or lower.get("source")
        target_col = lower["target"]
        weight_col = (
            lower.get("weight")
            or lower.get("regulatory_potential")
            or lower.get("score")
        )
        network = raw[[source_col, target_col]].copy()
        network["weight"] = raw[weight_col] if weight_col is not None else 1.0
        network = network.rename(columns={source_col: "source", target_col: "target"})
    else:
        first_col = raw.columns[0]
        if _series_looks_like_gene_names(raw[first_col]):
            target_values = raw[first_col].astype(str)
            wide = raw.drop(columns=[first_col])
        elif _index_looks_like_gene_names(raw.index):
            target_values = raw.index.astype(str)
            wide = raw
        else:
            raise ValueError(
                "The ligand-target matrix has ligand columns but no target gene names. "
                "Recreate it with target genes as the DataFrame index or a target column."
            )
        network = _wide_ligand_target_to_long(
            wide,
            target_values=target_values,
            top_n_targets=top_n_targets,
            min_weight=min_weight,
        )

    network["source"] = network["source"].astype(str)
    network["target"] = network["target"].astype(str)
    network["weight"] = pd.to_numeric(network["weight"], errors="coerce")
    network = network.dropna(subset=["source", "target", "weight"])
    if min_weight is not None:
        network = network[network["weight"] > min_weight]
    network = network.sort_values("weight", ascending=False)
    network = network.drop_duplicates(["source", "target"])
    if top_n_targets is not None:
        network = network.groupby("source", group_keys=False).head(top_n_targets)
    return network.reset_index(drop=True)


def diagnose_ligand_activity_inputs(
    dea_df: pd.DataFrame,
    ligand_target_prior: pd.DataFrame,
    lr_df: pd.DataFrame,
    receiver: str = "FDC",
    celltype_col: str = "label",
) -> pd.DataFrame:
    """Report gene/ligand overlap before computing ligand activity."""
    receiver_de = dea_df[dea_df[celltype_col].astype(str) == str(receiver)].copy()
    receiver_genes = pd.Index(receiver_de.index.astype(str))
    prior_targets = pd.Index(ligand_target_prior["target"].astype(str).unique())
    prior_ligands = pd.Index(ligand_target_prior["source"].astype(str).unique())
    lr_ligands = _unique_str(lr_df["ligand"])
    summary = pd.DataFrame(
        [
            {
                "receiver": receiver,
                "receiver_de_genes": len(receiver_genes),
                "prior_targets": len(prior_targets),
                "target_overlap": len(receiver_genes.intersection(prior_targets)),
                "lr_ligands": len(lr_ligands),
                "prior_ligands": len(prior_ligands),
                "ligand_overlap": len(lr_ligands.intersection(prior_ligands)),
            }
        ]
    )
    return summary


def _average_precision_from_scores(y_true: Iterable[int], scores: Iterable[float]) -> float:
    """Compute average precision without adding a scikit-learn dependency."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_positive = y_true.sum()
    if n_positive == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    precision_at_k = np.cumsum(y_sorted) / (np.arange(y_sorted.size) + 1)
    return float(precision_at_k[y_sorted == 1].sum() / n_positive)


def compute_ligand_activity_ulm(
    dea_df: pd.DataFrame,
    ligand_target_prior: pd.DataFrame,
    receiver: str = "FDC",
    celltype_col: str = "label",
    stat: str = "stat",
    restrict_ligands: Iterable[str] | None = None,
    tmin: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """Score ligands by ULM enrichment of target weights against receiver DE.

    This optional decoupler method treats each ligand's weighted target set as
    a regulator network and tests enrichment against a signed receiver DE
    statistic. It is useful as a fully Python alternative, but it is less
    similar to MultiNicheNet than :func:`compute_ligand_activity_aupr`, because
    MultiNicheNet uses NicheNet's corrected AUPR.
    """
    import decoupler as dc

    de = dea_df[dea_df[celltype_col].astype(str) == str(receiver)].copy()
    de.index = de.index.astype(str)
    de = de[~de.index.duplicated(keep="first")]
    response = pd.to_numeric(de[stat], errors="coerce").dropna()
    response = response[response.index.isin(ligand_target_prior["target"].astype(str))]

    network = ligand_target_prior.copy()
    if restrict_ligands is not None:
        network = network[network["source"].isin(_unique_str(restrict_ligands))]
    if response.empty or network.empty:
        return pd.DataFrame(
            columns=["ligand", "ligand_activity", "ligand_activity_padj", "n_activity_targets"]
        )

    response_df = response.to_frame(name=f"{receiver}_{stat}").T
    scores, padj = dc.mt.ulm(response_df, network, tmin=tmin, verbose=verbose)
    activity = scores.T.rename(columns={scores.index[0]: "ligand_activity"}).reset_index()
    activity = activity.rename(columns={"index": "ligand"})
    if padj is not None:
        padj_df = padj.T.rename(columns={padj.index[0]: "ligand_activity_padj"}).reset_index()
        activity = activity.merge(padj_df.rename(columns={"index": "ligand"}), on="ligand", how="left")

    n_targets = network[network["target"].isin(response.index)].groupby("source")["target"].nunique()
    activity["n_activity_targets"] = activity["ligand"].map(n_targets).fillna(0).astype(int)
    activity["ligand_activity_method"] = "ulm"
    return activity.sort_values("ligand_activity", ascending=False).reset_index(drop=True)


def compute_ligand_activity_aupr(
    dea_df: pd.DataFrame,
    ligand_target_prior: pd.DataFrame,
    receiver: str = "FDC",
    celltype_col: str = "label",
    pvalue_col: str = "pvalue",
    pvalue_threshold: float = 0.05,
    logfc_col: str = "log2FoldChange",
    logfc_threshold: float = 0.25,
    fallback_stat: str = "stat",
    fallback_top_n: int | None = 100,
    restrict_ligands: Iterable[str] | None = None,
    min_targets: int = 10,
) -> pd.DataFrame:
    """Approximate NicheNet ligand activity as baseline-corrected AUPR.

    The receiver gene set is defined by positive logFC and p-value thresholds.
    Each ligand's prior target weights rank background genes, and average
    precision is calculated for recovery of the receiver gene set. The random
    prevalence baseline is subtracted, matching the meaning of NicheNet's
    ``aupr_corrected`` output.

    MultiNicheNet calls ``nichenetr::predict_ligand_activities`` rather than
    this compact implementation. Both use corrected AUPR and the same conceptual
    foreground/background setup, but details of ties, matrix preprocessing,
    and NicheNet internals can produce different values. The optional fallback
    keeps the Python analysis usable when strict thresholds select fewer than
    five receiver genes; its use is recorded in ``activity_definition``.
    """
    de = dea_df[dea_df[celltype_col].astype(str) == str(receiver)].copy()
    de.index = de.index.astype(str)
    de = de[~de.index.duplicated(keep="first")]

    network = ligand_target_prior.copy()
    if restrict_ligands is not None:
        network = network[network["source"].isin(_unique_str(restrict_ligands))]
    if network.empty:
        return pd.DataFrame(columns=["ligand", "ligand_activity", "n_activity_targets"])

    background = pd.Index(de.index.intersection(pd.Index(network["target"].astype(str).unique())))
    if background.empty:
        raise ValueError(
            "No overlap between ligand-target prior targets and receiver DE genes; "
            "check gene names and the ligand-target matrix export"
        )

    lfc = pd.to_numeric(de.reindex(background)[logfc_col], errors="coerce")
    pvalue = pd.to_numeric(de.reindex(background)[pvalue_col], errors="coerce")
    geneset = background[(lfc >= logfc_threshold) & (pvalue <= pvalue_threshold)]
    definition = (
        f"{receiver} {logfc_col}>={logfc_threshold} and "
        f"{pvalue_col}<={pvalue_threshold}"
    )
    if len(geneset) < 5 and fallback_top_n is not None and fallback_stat in de.columns:
        fallback = pd.to_numeric(de.reindex(background)[fallback_stat], errors="coerce")
        fallback = fallback[fallback > 0].sort_values(ascending=False)
        geneset = pd.Index(fallback.head(fallback_top_n).index)
        definition = f"{receiver} top {len(geneset)} positive {fallback_stat} fallback"
    if len(geneset) == 0:
        raise ValueError(
            "No receiver genes were selected for ligand activity; relax the "
            "p-value/logFC thresholds or inspect the receiver DE table"
        )

    y_true = background.isin(geneset).astype(int)
    baseline = float(y_true.mean())
    target_position = {gene: i for i, gene in enumerate(background)}
    rows = []
    for ligand, links in network.groupby("source", sort=False):
        links = links[links["target"].isin(target_position)]
        if links["target"].nunique() < min_targets:
            continue
        scores = np.zeros(len(background), dtype=float)
        for link in links.itertuples(index=False):
            position = target_position[str(link.target)]
            scores[position] = max(scores[position], float(link.weight))
        aupr = _average_precision_from_scores(y_true, scores)
        rows.append(
            {
                "ligand": ligand,
                "ligand_activity": aupr - baseline,
                "ligand_activity_aupr": aupr,
                "ligand_activity_baseline": baseline,
                "ligand_activity_padj": np.nan,
                "n_activity_targets": links["target"].nunique(),
                "activity_geneset_size": int(y_true.sum()),
                "activity_background_size": len(background),
                "ligand_activity_method": "aupr_corrected",
                "activity_definition": definition,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ligand", "ligand_activity", "n_activity_targets"])
    return pd.DataFrame(rows).sort_values("ligand_activity", ascending=False).reset_index(drop=True)


def compute_ligand_activity(
    dea_df: pd.DataFrame,
    ligand_target_prior: pd.DataFrame,
    receiver: str = "FDC",
    method: str = "aupr",
    restrict_ligands: Iterable[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Dispatch to the corrected-AUPR or ULM ligand-activity method."""
    if method == "aupr":
        return compute_ligand_activity_aupr(
            dea_df,
            ligand_target_prior,
            receiver=receiver,
            restrict_ligands=restrict_ligands,
            **kwargs,
        )
    if method == "ulm":
        return compute_ligand_activity_ulm(
            dea_df,
            ligand_target_prior,
            receiver=receiver,
            restrict_ligands=restrict_ligands,
            **kwargs,
        )
    raise ValueError("method must be 'aupr' or 'ulm'")


def _de_lookup(dea_df: pd.DataFrame, celltype_col: str = "label") -> pd.DataFrame:
    """Index a DE table by cell type and gene.

    ``dea_df`` is expected to come from the LIANA notebook's pseudobulk DE
    workflow. It should contain a cell-type column, ``log2FoldChange``, and
    ``pvalue``. If gene names live in the index, they are promoted to a ``gene``
    column.

    MultiNicheNet difference
    ------------------------
    MultiNicheNet typically uses muscat-derived DE tables. This helper does not
    reproduce muscat; it simply standardises whichever Python DE table is
    supplied, usually from pydeseq2/decoupler in the notebook.
    """
    de = dea_df.copy()
    if "gene" not in de.columns:
        de = de.reset_index()
        gene_col = de.index.name or "index"
        if gene_col not in de.columns:
            gene_col = de.columns[0]
        de = de.rename(columns={gene_col: "gene"})
    de["gene"] = de["gene"].astype(str)
    de[celltype_col] = de[celltype_col].astype(str)
    de = de.drop_duplicates([celltype_col, "gene"], keep="first")
    return de.set_index([celltype_col, "gene"])


def _lookup_de_values(
    lr: pd.DataFrame,
    de: pd.DataFrame,
    celltype_col: str,
    gene_col: str,
    value_col: str,
) -> np.ndarray:
    """Look up one DE value for each LR row.

    ``celltype_col`` is either ``source`` for ligands or ``target`` for
    receptors. ``gene_col`` is either ``ligand`` or ``receptor``. Missing DE
    values are returned as NaN and later contribute zero after ranking/fill.
    """
    idx = pd.MultiIndex.from_arrays(
        [lr[celltype_col].astype(str), lr[gene_col].astype(str)],
        names=de.index.names,
    )
    if value_col not in de.columns:
        return np.full(len(lr), np.nan)
    return pd.to_numeric(de[value_col].reindex(idx), errors="coerce").to_numpy()


def _lookup_group_values(
    lr: pd.DataFrame,
    group_expr: pd.DataFrame,
    condition_value: str,
    groupby: str,
    condition_col: str,
    celltype_col: str,
    gene_col: str,
    value_col: str,
) -> np.ndarray:
    """Look up one condition/cell-type/gene expression value for each LR row."""
    table = group_expr.set_index([condition_col, groupby, "gene"])
    idx = pd.MultiIndex.from_arrays(
        [
            pd.Series(condition_value, index=lr.index).astype(str),
            lr[celltype_col].astype(str),
            lr[gene_col].astype(str),
        ],
        names=[condition_col, groupby, "gene"],
    )
    return pd.to_numeric(table[value_col].reindex(idx), errors="coerce").to_numpy()


def _sample_lr_support(
    lr: pd.DataFrame,
    sample_expr: pd.DataFrame,
    condition_value: str,
    groupby: str,
    condition_col: str,
    sample_col: str,
    fraction_cutoff: float,
) -> pd.DataFrame:
    """Calculate donor-level support for each LR pair.

    For each LR row, this checks samples in ``condition_value`` and counts how
    many donors have:

    * ligand fraction in the sender cell type greater than ``fraction_cutoff``;
      and
    * receptor fraction in the receiver cell type greater than
      ``fraction_cutoff``.

    The output fraction is
    ``shadow_n_expressing_lr_donors / shadow_n_eligible_donors``.

    MultiNicheNet comparison
    ------------------------
    This mirrors MultiNicheNet's
    ``fraction_expressing_ligand_receptor`` criterion, which uses the same idea
    with ``fraction_cutoff`` on sample-level ligand and receptor fractions.
    """
    frac = sample_expr[sample_expr[condition_col].astype(str) == str(condition_value)]
    frac = frac[[sample_col, groupby, "gene", "frac_expr"]].copy()

    lig = lr[["_lr_id", "source", "ligand"]].merge(
        frac.rename(columns={groupby: "source", "gene": "ligand", "frac_expr": "fraction_ligand"}),
        on=["source", "ligand"],
        how="left",
    )
    rec = lr[["_lr_id", "target", "receptor"]].merge(
        frac.rename(columns={groupby: "target", "gene": "receptor", "frac_expr": "fraction_receptor"}),
        on=["target", "receptor"],
        how="left",
    )
    paired = lig.merge(rec, on=["_lr_id", sample_col], how="inner")
    if paired.empty:
        return pd.DataFrame(
            {
                "_lr_id": lr["_lr_id"],
                "shadow_fraction_expressing_ligand_receptor": 0.0,
                "shadow_n_expressing_lr_donors": 0,
                "shadow_n_eligible_donors": 0,
            }
        )

    paired["expressing_lr"] = (
        (paired["fraction_ligand"] > fraction_cutoff)
        & (paired["fraction_receptor"] > fraction_cutoff)
    )
    support = (
        paired.groupby("_lr_id", observed=True)
        .agg(
            shadow_n_eligible_donors=(sample_col, "nunique"),
            shadow_n_expressing_lr_donors=("expressing_lr", "sum"),
        )
        .reset_index()
    )
    support["shadow_fraction_expressing_ligand_receptor"] = (
        support["shadow_n_expressing_lr_donors"] / support["shadow_n_eligible_donors"]
    )
    return lr[["_lr_id"]].merge(support, on="_lr_id", how="left").fillna(
        {
            "shadow_fraction_expressing_ligand_receptor": 0.0,
            "shadow_n_expressing_lr_donors": 0,
            "shadow_n_eligible_donors": 0,
        }
    )


def _prepare_ligand_activity(ligand_activity: pd.DataFrame | None) -> pd.DataFrame | None:
    """Prepare externally computed ligand activity for merging into LR rows.

    Parameters
    ----------
    ligand_activity:
        Table with a ``ligand`` column and either ``ligand_activity`` or
        ``activity``. Extra diagnostic columns such as ``n_activity_targets``
        are preserved when present.

    Returns
    -------
    pandas.DataFrame or None
        One row per ligand with ``shadow_ligand_activity`` and
        ``shadow_max_scaled_activity``.

    MultiNicheNet comparison
    ------------------------
    MultiNicheNet computes ligand activity internally with
    ``nichenetr::predict_ligand_activities``. It then z-score scales activity
    within receiver/contrast and applies quantile scaling during
    prioritisation. Here we accept any ligand-activity table and use a simple
    percentile rank. This keeps the scorer independent from the activity
    method, while making the approximation explicit.
    """
    if ligand_activity is None or len(ligand_activity) == 0:
        return None
    activity = ligand_activity.copy()
    if "ligand" not in activity.columns:
        raise ValueError("ligand_activity must contain a 'ligand' column")
    value_col = "ligand_activity" if "ligand_activity" in activity.columns else "activity"
    if value_col not in activity.columns:
        raise ValueError("ligand_activity must contain 'ligand_activity' or 'activity'")
    activity["ligand"] = activity["ligand"].astype(str)
    activity[value_col] = pd.to_numeric(activity[value_col], errors="coerce")
    activity = activity.sort_values(value_col, ascending=False).drop_duplicates("ligand")
    activity = activity.rename(columns={value_col: "shadow_ligand_activity"})
    activity["shadow_max_scaled_activity"] = _rank01(activity["shadow_ligand_activity"]).to_numpy()
    keep_cols = ["ligand", "shadow_ligand_activity", "shadow_max_scaled_activity"]
    for col in ("ligand_activity_padj", "n_activity_targets", "activity_definition"):
        if col in activity.columns:
            keep_cols.append(col)
    return activity[keep_cols]


def mnn_shadow_prioritise(
    lr_df: pd.DataFrame,
    adata,
    dea_df: pd.DataFrame,
    condition_value: str = "yes",
    condition_col: str = "tonsillitis",
    groupby: str = "label",
    sample_col: str = "donor_id",
    ligand_activity: pd.DataFrame | None = None,
    fraction_cutoff: float = 0.05,
    min_cells: int = 10,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add a compact MultiNicheNet-like prioritisation score to LIANA LR pairs.

    Parameters
    ----------
    lr_df:
        LIANA result table. It must identify sender/source, receiver/target,
        ligand, and receptor. Tables with ``ligand_complex`` and
        ``receptor_complex`` are accepted.
    adata:
        AnnData object containing all conditions. This matters: expression
        specificity is ranked across condition/cell-type combinations, so the
        function should receive the full object rather than only the condition
        of interest.
    dea_df:
        Differential-expression table with cell type labels and gene-level
        ``log2FoldChange`` and ``pvalue``.
    condition_value:
        Condition to prioritise, for example ``"yes"`` for tonsillitis.
    condition_col:
        Column in ``adata.obs`` containing the condition labels.
    groupby:
        Cell-type column in ``adata.obs`` and ``dea_df``.
    sample_col:
        Donor/sample column in ``adata.obs``.
    ligand_activity:
        Optional ligand-activity table. If omitted, the activity criterion is
        given zero weight by setting ``activity_scaled`` to 0.
    fraction_cutoff:
        Per-sample expression-fraction cutoff used for donor LR support.
    min_cells:
        Minimum cells required for a sample-condition-celltype group to
        contribute expression summaries.
    weights:
        Optional overrides for ``REGULAR_WEIGHTS``.

    Returns
    -------
    pandas.DataFrame
        The LR table with added ``shadow_*`` component columns,
        ``mnn_shadow_score``, and ``mnn_shadow_rank``.

    Formula
    -------
    The default score mirrors the regular MultiNicheNet weighting:

    ``mnn_shadow_score = (``
    ``0.5 * ligand_lfc_rank + 0.5 * ligand_signed_p_rank +``
    ``0.5 * receptor_lfc_rank + 0.5 * receptor_signed_p_rank +``
    ``ligand_activity_rank + ligand_expression_rank +``
    ``receptor_expression_rank + donor_lr_fraction``
    ``) / 6``

    where each conceptual criterion has weight 1 by default. The DE criterion
    is split into logFC and signed-p-value halves, as in MultiNicheNet.

    What is intentionally different from MultiNicheNet?
    --------------------------------------------------
    This function does not reproduce all internals of
    ``multinichenetr::generate_prioritization_tables``. In particular:

    * LR candidates come from LIANA, not MultiNicheNet's own filtering.
    * DE values come from the provided Python DE table, not muscat.
    * Expression scores are percentile ranks of sample-averaged ``adata.X``
      summaries, not ``nichenetr::scale_quantile`` of MultiNicheNet's exact
      pseudobulk tables.
    * Ligand activity is whatever table is supplied by the caller; the notebook
      uses a Python AUPR approximation rather than NicheNet's exact R function.

    These differences are expected sources of residual rank disagreement, and
    they are useful to mention when interpreting or reporting the method.
    """
    lr = _normalise_lr_columns(lr_df)
    lr["_lr_id"] = np.arange(len(lr))
    weights = {**REGULAR_WEIGHTS, **(weights or {})}

    genes = _unique_str(pd.concat([lr["ligand"], lr["receptor"]]))
    sample_expr = sample_gene_expression(
        adata,
        genes,
        groupby=groupby,
        condition_col=condition_col,
        sample_col=sample_col,
        min_cells=min_cells,
    )
    group_expr = group_gene_expression(
        sample_expr,
        groupby=groupby,
        condition_col=condition_col,
        sample_col=sample_col,
    )

    de = _de_lookup(dea_df, celltype_col=groupby)
    lr["shadow_lfc_ligand"] = _lookup_de_values(lr, de, "source", "ligand", "log2FoldChange")
    lr["shadow_lfc_receptor"] = _lookup_de_values(lr, de, "target", "receptor", "log2FoldChange")
    lr["shadow_p_val_ligand"] = _lookup_de_values(lr, de, "source", "ligand", "pvalue")
    lr["shadow_p_val_receptor"] = _lookup_de_values(lr, de, "target", "receptor", "pvalue")
    lr["shadow_p_val_ligand_adapted_raw"] = _signed_logp(
        lr["shadow_lfc_ligand"],
        lr["shadow_p_val_ligand"],
    )
    lr["shadow_p_val_receptor_adapted_raw"] = _signed_logp(
        lr["shadow_lfc_receptor"],
        lr["shadow_p_val_receptor"],
    )

    lr["shadow_scaled_lfc_ligand"] = _rank01(lr["shadow_lfc_ligand"]).to_numpy()
    lr["shadow_scaled_lfc_receptor"] = _rank01(lr["shadow_lfc_receptor"]).to_numpy()
    lr["shadow_scaled_p_val_ligand_adapted"] = _rank01(
        lr["shadow_p_val_ligand_adapted_raw"]
    ).to_numpy()
    lr["shadow_scaled_p_val_receptor_adapted"] = _rank01(
        lr["shadow_p_val_receptor_adapted_raw"]
    ).to_numpy()

    lr["shadow_pb_ligand_group"] = _lookup_group_values(
        lr,
        group_expr,
        condition_value,
        groupby,
        condition_col,
        "source",
        "ligand",
        "group_expr",
    )
    lr["shadow_pb_receptor_group"] = _lookup_group_values(
        lr,
        group_expr,
        condition_value,
        groupby,
        condition_col,
        "target",
        "receptor",
        "group_expr",
    )
    lr["shadow_fraction_ligand_group"] = _lookup_group_values(
        lr,
        group_expr,
        condition_value,
        groupby,
        condition_col,
        "source",
        "ligand",
        "group_frac",
    )
    lr["shadow_fraction_receptor_group"] = _lookup_group_values(
        lr,
        group_expr,
        condition_value,
        groupby,
        condition_col,
        "target",
        "receptor",
        "group_frac",
    )
    lr["shadow_scaled_pb_ligand"] = _lookup_group_values(
        lr,
        group_expr,
        condition_value,
        groupby,
        condition_col,
        "source",
        "ligand",
        "group_expr_score",
    )
    lr["shadow_scaled_pb_receptor"] = _lookup_group_values(
        lr,
        group_expr,
        condition_value,
        groupby,
        condition_col,
        "target",
        "receptor",
        "group_expr_score",
    )

    support = _sample_lr_support(
        lr,
        sample_expr,
        condition_value=condition_value,
        groupby=groupby,
        condition_col=condition_col,
        sample_col=sample_col,
        fraction_cutoff=fraction_cutoff,
    )
    lr = lr.merge(support, on="_lr_id", how="left")

    activity = _prepare_ligand_activity(ligand_activity)
    if activity is not None:
        lr = lr.merge(activity, on="ligand", how="left")
    else:
        weights["activity_scaled"] = 0.0
        lr["shadow_ligand_activity"] = np.nan
        lr["shadow_max_scaled_activity"] = 0.0
    lr["shadow_max_scaled_activity"] = lr["shadow_max_scaled_activity"].fillna(0)

    numerator = (
        0.5 * weights["de_ligand"] * lr["shadow_scaled_lfc_ligand"].fillna(0)
        + 0.5 * weights["de_ligand"] * lr["shadow_scaled_p_val_ligand_adapted"].fillna(0)
        + 0.5 * weights["de_receptor"] * lr["shadow_scaled_lfc_receptor"].fillna(0)
        + 0.5 * weights["de_receptor"] * lr["shadow_scaled_p_val_receptor_adapted"].fillna(0)
        + weights["activity_scaled"] * lr["shadow_max_scaled_activity"].fillna(0)
        + weights["exprs_ligand"] * lr["shadow_scaled_pb_ligand"].fillna(0)
        + weights["exprs_receptor"] * lr["shadow_scaled_pb_receptor"].fillna(0)
        + weights["frac_exprs_ligand_receptor"]
        * lr["shadow_fraction_expressing_ligand_receptor"].fillna(0)
    )
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("At least one score weight must be positive")

    lr["mnn_shadow_score"] = numerator / denominator
    lr["mnn_shadow_rank"] = lr["mnn_shadow_score"].rank(ascending=False, method="dense").astype(int)
    return lr.drop(columns=["_lr_id"]).sort_values("mnn_shadow_score", ascending=False).reset_index(drop=True)


def _score_factor_columns(
    frame: pd.DataFrame,
    factor_cols: Iterable[tuple[str, str]] | dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Normalise plot factor specifications and retain available columns."""
    if factor_cols is None:
        factor_cols = SHADOW_FACTOR_COLUMNS
    if isinstance(factor_cols, dict):
        factor_cols = list(factor_cols.items())
    return [(column, label) for column, label in factor_cols if column in frame.columns]


def _lr_row_labels(frame: pd.DataFrame) -> pd.Series:
    """Build readable sender/receiver and ligand/receptor row labels."""
    ligand_col = "ligand_complex" if "ligand_complex" in frame.columns else "ligand"
    receptor_col = "receptor_complex" if "receptor_complex" in frame.columns else "receptor"
    return (
        frame["source"].astype(str)
        + " -> "
        + frame["target"].astype(str)
        + " | "
        + frame[ligand_col].astype(str)
        + " - "
        + frame[receptor_col].astype(str)
    )


def make_liana_lr_factor_plot(
    priority_res: pd.DataFrame,
    top_n: int | None = 30,
    sort_by: str | Iterable[str] = "mnn_shadow_score",
    ascending: bool | Iterable[bool] = False,
    top_n_by: str | None = None,
    factor_cols: Iterable[tuple[str, str]] | dict[str, str] | None = None,
    score_col: str = "mnn_shadow_score",
    support_col: str = "shadow_fraction_expressing_ligand_receptor",
    n_expressing_col: str = "shadow_n_expressing_lr_donors",
    n_eligible_col: str = "shadow_n_eligible_donors",
    title: str = "LIANA / MultiNicheNet-shadow LR prioritisation factors",
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
):
    """Plot normalised factors, total score, and donor support for LR pairs.

    Parameters
    ----------
    priority_res:
        Ranked LR table returned by :func:`mnn_shadow_prioritise` or another
        scorer with compatible component columns.
    top_n:
        Number of interactions to display. Set to ``None`` for all rows.
    sort_by:
        One column or a sequence of display-sort columns.
    ascending:
        One boolean or a sequence matching ``sort_by``.
    top_n_by:
        Column used to select the top rows before display sorting. By default
        the final ``sort_by`` column is used. Thus sorting by
        ``('source', 'target', 'mnn_shadow_score')`` still selects the overall
        top-scoring rows before grouping them by source and target.
    factor_cols:
        ``(column, label)`` pairs for the factor heatmap. Defaults to
        :data:`SHADOW_FACTOR_COLUMNS`.
    score_col, support_col, n_expressing_col, n_eligible_col:
        Columns used for the score and donor-support bar panels.
    title, cmap, figsize:
        Standard display options.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with component heatmap, total-score bars, and donor-support bars.

    Notes
    -----
    This is an explanatory plot rather than a MultiNicheNet plotting port. It
    is analogous to ``make_sample_lr_prod_activity_plots`` in that it exposes
    the factors influencing prioritisation, but it displays the explicit
    shadow-score components in one compact heatmap.
    """
    import matplotlib.pyplot as plt

    if isinstance(sort_by, str):
        sort_cols = [sort_by]
    else:
        sort_cols = list(sort_by)
    if not sort_cols:
        raise ValueError("sort_by must contain at least one column")
    missing = [column for column in sort_cols if column not in priority_res.columns]
    if missing:
        raise ValueError(f"sort_by columns are not in priority_res: {missing}")

    if isinstance(ascending, (bool, np.bool_)):
        sort_ascending = [bool(ascending)] * len(sort_cols)
    else:
        sort_ascending = list(ascending)
        if len(sort_ascending) != len(sort_cols):
            raise ValueError("ascending must be a boolean or match the length of sort_by")

    top_n_by = sort_cols[-1] if top_n_by is None else top_n_by
    if top_n_by not in priority_res.columns:
        raise ValueError(f"top_n_by={top_n_by!r} is not in priority_res columns")
    top_n_ascending = (
        sort_ascending[sort_cols.index(top_n_by)] if top_n_by in sort_cols else False
    )

    frame = priority_res.copy()
    if top_n is not None:
        frame = frame.sort_values(
            top_n_by,
            ascending=top_n_ascending,
            kind="stable",
        ).head(top_n)
    frame = frame.sort_values(
        sort_cols,
        ascending=sort_ascending,
        kind="stable",
    ).reset_index(drop=True)

    factors = _score_factor_columns(frame, factor_cols=factor_cols)
    if not factors:
        raise ValueError("No requested score factor columns were found in priority_res")
    for column in (score_col, support_col):
        if column not in frame.columns:
            raise ValueError(f"Required plot column {column!r} is missing")

    factor_data = pd.DataFrame(
        {
            label: pd.to_numeric(frame[column], errors="coerce").fillna(0).clip(0, 1)
            for column, label in factors
        }
    )
    row_labels = _lr_row_labels(frame)
    y = np.arange(frame.shape[0])
    if figsize is None:
        figsize = (
            max(11, 0.55 * len(factors) + 8),
            max(6, 0.36 * frame.shape[0] + 1.6),
        )

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[max(4, 0.55 * len(factors)), 1.25, 1.65],
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_score = fig.add_subplot(grid[0, 1], sharey=ax_heat)
    ax_support = fig.add_subplot(grid[0, 2], sharey=ax_heat)

    image = ax_heat.imshow(
        factor_data.to_numpy(),
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap=cmap,
    )
    ax_heat.set_yticks(y)
    ax_heat.set_yticklabels(row_labels, fontsize=8 if frame.shape[0] <= 25 else 7)
    ax_heat.set_xticks(np.arange(len(factors)))
    ax_heat.set_xticklabels(factor_data.columns, rotation=45, ha="right", fontsize=9)
    ax_heat.set_title("Normalised component scores")
    ax_heat.set_xlim(-0.5, len(factors) - 0.5)
    ax_heat.set_ylim(frame.shape[0] - 0.5, -0.5)
    ax_heat.set_xticks(np.arange(-0.5, len(factors), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, frame.shape[0], 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.7)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    score = pd.to_numeric(frame[score_col], errors="coerce").fillna(0).clip(lower=0)
    ax_score.barh(y, score, color="#4269a6", height=0.78)
    ax_score.set_title("Total\nscore")
    ax_score.set_xlim(0, max(1, score.max() * 1.15 if score.max() > 0 else 1))
    ax_score.set_yticks(y)
    ax_score.tick_params(axis="y", left=False, labelleft=False)
    ax_score.grid(axis="x", alpha=0.25)
    ax_score.set_ylim(frame.shape[0] - 0.5, -0.5)
    for yi, value in zip(y, score, strict=False):
        ax_score.text(value + 0.01, yi, f"{value:.2f}", va="center", fontsize=7)

    support = pd.to_numeric(frame[support_col], errors="coerce").fillna(0).clip(0, 1)
    ax_support.barh(y, support, color="#4c956c", height=0.78)
    ax_support.set_title("Donor\nsupport")
    ax_support.set_xlim(0, 1.05)
    ax_support.set_yticks(y)
    ax_support.tick_params(axis="y", left=False, labelleft=False)
    ax_support.grid(axis="x", alpha=0.25)
    ax_support.set_ylim(frame.shape[0] - 0.5, -0.5)

    n_expressing = pd.to_numeric(frame.get(n_expressing_col, np.nan), errors="coerce")
    n_eligible = pd.to_numeric(frame.get(n_eligible_col, np.nan), errors="coerce")
    for yi, fraction, n_on, n_all in zip(
        y,
        support,
        n_expressing,
        n_eligible,
        strict=False,
    ):
        label = (
            f"{int(n_on)}/{int(n_all)}"
            if pd.notna(n_on) and pd.notna(n_all)
            else f"{fraction:.2f}"
        )
        ax_support.text(min(fraction + 0.02, 1.01), yi, label, va="center", fontsize=7)

    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.015)
    colorbar.set_label("Component score")
    fig.suptitle(title, fontsize=13)
    return fig


def top_overlap_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_rank_col: str,
    right_rank_col: str,
    left_cols: tuple[str, str, str, str] = ("source", "target", "ligand", "receptor"),
    right_cols: tuple[str, str, str, str] = ("source", "target", "ligand", "receptor"),
    top_ns: tuple[int, ...] = (10, 20, 30, 50, 100),
) -> pd.DataFrame:
    """Summarise top-N LR overlap between two ranked LR tables.

    This is a small diagnostic helper for comparing the shadow score with
    another ranking, usually the saved MultiNicheNet output. It builds LR keys
    from source, target, ligand, and receptor columns, then reports overlap at
    each requested top-N threshold.

    Parameters
    ----------
    left, right:
        Ranked LR tables to compare.
    left_rank_col, right_rank_col:
        Rank columns where lower is better and rank 1 is the top interaction.
    left_cols, right_cols:
        Column names used to identify source, target, ligand, and receptor in
        each table.
    top_ns:
        Top-N thresholds to report.

    Returns
    -------
    pandas.DataFrame
        One row per top-N threshold with overlap counts and the fraction of the
        left top-N recovered in the right top-N.
    """
    left_key = (
        left[left_cols[0]].astype(str)
        + "|"
        + left[left_cols[1]].astype(str)
        + "|"
        + left[left_cols[2]].astype(str)
        + "|"
        + left[left_cols[3]].astype(str)
    )
    right_key = (
        right[right_cols[0]].astype(str)
        + "|"
        + right[right_cols[1]].astype(str)
        + "|"
        + right[right_cols[2]].astype(str)
        + "|"
        + right[right_cols[3]].astype(str)
    )
    left_tmp = left.assign(_key=left_key)
    right_tmp = right.assign(_key=right_key)
    rows = []
    for n in top_ns:
        left_top = set(left_tmp.loc[left_tmp[left_rank_col] <= n, "_key"])
        right_top = set(right_tmp.loc[right_tmp[right_rank_col] <= n, "_key"])
        rows.append(
            {
                "top_n": n,
                "left_n": len(left_top),
                "right_n": len(right_top),
                "overlap_n": len(left_top & right_top),
                "overlap_fraction_left": len(left_top & right_top) / max(len(left_top), 1),
            }
        )
    return pd.DataFrame(rows)
