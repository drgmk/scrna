"""
Plotting utilities for long-form single-cell summaries.

This module bundles a house "bio_journal" matplotlib style and helpers to annotate
simple grouped plots with p-values.
"""

from __future__ import annotations

import importlib.resources as importlib_resources
from dataclasses import dataclass
from contextlib import nullcontext
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


@dataclass
class StatTestResult:
    """Result of a single pairwise comparison drawn on a grouped plot."""

    group: Optional[object]
    group_a: object
    group_b: object
    statistic: float
    pvalue: float
    test: str
    n_a: int
    n_b: int
    x1: float
    x2: float


def get_bio_journal_style_path() -> str:
    """Return the filesystem path to the bundled bio_journal.mplstyle file."""

    try:
        return str(
            importlib_resources.files("scrna.data").joinpath("bio_journal.mplstyle")
        )
    except Exception as exc:  # pragma: no cover - unexpected importlib failure
        raise FileNotFoundError(
            "bio_journal.mplstyle not found in scrna.data"
        ) from exc


def format_pvalue(p: float, style: str = "p", stars: bool = False) -> str:
    """Format a p-value as text or significance stars."""

    if stars:
        if p < 1e-4:
            return "****"
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return "ns"

    if style == "p<":
        if p < 1e-3:
            return "p < 0.001"
        if p < 1e-2:
            return "p < 0.01"
        if p < 5e-2:
            return "p < 0.05"
        return f"p = {p:.2f}"

    if style == "numeric":
        if p < 1e-3:
            return f"{p:.1e}"
        return f"{p:.3f}".rstrip("0").rstrip(".")

    if p < 1e-3:
        return f"p = {p:.1e}"

    return f"p = {p:.3f}".rstrip("0").rstrip(".")


def add_pvalue_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    p: float,
    y: Optional[float] = None,
    h: Optional[float] = None,
    text: Optional[str] = None,
    text_offset: float = 0.0,
    linewidth: float = 1.0,
    fontsize: Optional[float] = None,
    bracket_width: float = 0.15,
    stars: bool = False,
    p_style: str = "p<",
    cmap: Optional[object] = None,
) -> None:
    """Draw a staple bracket between x1 and x2 with p-value text."""

    texts = [child for child in ax.get_children() if isinstance(child, mpl.text.Text)]
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    if h is None:
        h = 0.02 * yr
    if y is None:
        y = ymax - 0.10 * yr
        candidates: List[float] = []

        for line in ax.lines:
            ydata = line.get_ydata()
            if len(ydata):
                candidates.append(np.nanmax(ydata))

        for patch in ax.patches:
            try:
                candidates.append(patch.get_y() + patch.get_height())
            except Exception:
                pass

        # for coll in ax.collections:
        #     offsets = getattr(coll, "get_offsets", None)
        #     if offsets is not None:
        #         off = coll.get_offsets()
        #         if len(off):
        #             candidates.append(np.nanmax(off[:, 1]))

        # for text_artist in texts:
        #     try:
        #         candidates.append(text_artist.get_position()[1])
        #     except Exception:
        #         pass

        if candidates:
            y = np.nanmax(candidates) + 0.07 * yr

    if text is None:
        text = format_pvalue(p, style=p_style, stars=stars)

    bracket_color = None
    if cmap is not None:
        resolved_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        bracket_color = resolved_cmap(np.clip(1.0 - float(p), 0.0, 1.0))

    x1, x2 = float(x1), float(x2)
    if x2 < x1:
        x1, x2 = x2, x1

    dx = x2 - x1
    stub = bracket_width * dx

    xs = [x1, x1, x1 + stub, x2 - stub, x2, x2]
    ys = [y, y + h, y + h, y + h, y + h, y]
    ax.plot(xs, ys, lw=linewidth, clip_on=False, color=bracket_color)

    ax.text(
        (x1 + x2) / 2,
        y + h + text_offset * yr,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        clip_on=False,
        color=bracket_color,
    )

    top_needed = y + h + (text_offset * yr) + 0.05 * yr
    if top_needed > ymax:
        ax.set_ylim(ymin, top_needed)


def _ordered_levels(
    df: pd.DataFrame,
    column: str,
    order: Optional[Sequence[object]] = None,
) -> List[object]:
    """Return observed levels, preserving categorical order when possible."""

    if order is not None:
        return [level for level in order if level in set(df[column].dropna())]

    series = df[column].dropna()
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = set(series.unique())
        return [level for level in series.cat.categories if level in observed]
    return list(series.unique())


def _pairwise_levels(
    levels: Sequence[object],
    pairs: Optional[Sequence[Tuple[object, object]]] = None,
) -> List[Tuple[object, object]]:
    if pairs is not None:
        return [tuple(pair) for pair in pairs]
    return list(combinations(levels, 2))


def _subset_values(
    df: pd.DataFrame,
    *,
    x: Optional[str],
    x_level: Optional[object],
    y: str,
    hue: Optional[str],
    hue_level: Optional[object],
) -> np.ndarray:
    mask = pd.Series(True, index=df.index)
    if x is not None and x_level is not None:
        mask &= df[x] == x_level
    if hue is not None and hue_level is not None:
        mask &= df[hue] == hue_level
    return df.loc[mask, y].dropna().to_numpy()


def _run_pairwise_test(
    a: np.ndarray,
    b: np.ndarray,
    test: str = "mannwhitney",
) -> Tuple[str, float, float]:
    """Run a two-group test and return test name, statistic, and p-value."""

    test_key = test.lower()
    if test_key in {"t_test", "ttest", "t-test"}:
        stat, pvalue = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return "t_test", float(stat), float(pvalue)
    if test_key in {"mannwhitney", "mann_whitney", "u_test", "wilcoxon_rank_sum"}:
        stat, pvalue = stats.mannwhitneyu(a, b, alternative="two-sided")
        return "mannwhitney", float(stat), float(pvalue)
    raise ValueError(f"Unknown test: {test}")


def _apply_pvalue_correction(
    pvalues: Sequence[float],
    method: Optional[str] = None,
) -> List[float]:
    if method is None:
        return list(pvalues)

    method_key = method.lower()
    if method_key != "bonferroni":
        raise ValueError(f"Unknown p-value correction: {method}")

    n_tests = len(pvalues)
    return [min(float(p) * n_tests, 1.0) for p in pvalues]


def _hue_offsets(n_hue: int, dodge_width: float = 0.8) -> np.ndarray:
    if n_hue <= 1:
        return np.array([0.0])
    step = dodge_width / n_hue
    start = -dodge_width / 2 + step / 2
    stop = dodge_width / 2 - step / 2
    return np.linspace(start, stop, n_hue)


def compute_stat_tests(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[Sequence[object]] = None,
    hue_order: Optional[Sequence[object]] = None,
    pairs: Optional[Sequence[Tuple[object, object]]] = None,
    test: str = "mannwhitney",
    correction: Optional[str] = None,
    min_n: int = 2,
) -> List[StatTestResult]:
    """Compute pairwise grouped tests for seaborn-style categorical plots.

    When ``hue`` is given, pairwise tests are run among hue levels within each
    ``x`` category. When ``hue`` is omitted, pairwise tests are run among ``x``
    levels across the full dataset.
    """

    required = [x, y] + ([hue] if hue else [])
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Missing columns in data: {', '.join(missing)}")

    frame = data.loc[:, required].dropna(subset=[x, y]).copy()
    if frame.empty:
        return []

    results: List[StatTestResult] = []

    if hue is None:
        x_levels = _ordered_levels(frame, x, order=order)
        comparisons = _pairwise_levels(x_levels, pairs=pairs)
        x_lookup = {level: idx for idx, level in enumerate(x_levels)}
        for group_a, group_b in comparisons:
            a = _subset_values(frame, x=x, x_level=group_a, y=y, hue=None, hue_level=None)
            b = _subset_values(frame, x=x, x_level=group_b, y=y, hue=None, hue_level=None)
            if len(a) < min_n or len(b) < min_n:
                continue
            test_name, statistic, pvalue = _run_pairwise_test(a, b, test=test)
            results.append(
                StatTestResult(
                    group=None,
                    group_a=group_a,
                    group_b=group_b,
                    statistic=statistic,
                    pvalue=pvalue,
                    test=test_name,
                    n_a=len(a),
                    n_b=len(b),
                    x1=float(x_lookup[group_a]),
                    x2=float(x_lookup[group_b]),
                )
            )
    else:
        frame = frame.dropna(subset=[hue])
        if frame.empty:
            return []
        x_levels = _ordered_levels(frame, x, order=order)
        hue_levels = _ordered_levels(frame, hue, order=hue_order)
        hue_lookup = {level: idx for idx, level in enumerate(hue_levels)}
        offsets = _hue_offsets(len(hue_levels))
        comparisons = _pairwise_levels(hue_levels, pairs=pairs)
        for x_idx, x_level in enumerate(x_levels):
            group_frame = frame.loc[frame[x] == x_level]
            observed_hue_levels = [
                level for level in hue_levels if level in set(group_frame[hue].unique())
            ]
            group_comparisons = [
                (group_a, group_b)
                for group_a, group_b in comparisons
                if group_a in observed_hue_levels and group_b in observed_hue_levels
            ]
            for group_a, group_b in group_comparisons:
                a = _subset_values(
                    group_frame,
                    x=None,
                    x_level=None,
                    y=y,
                    hue=hue,
                    hue_level=group_a,
                )
                b = _subset_values(
                    group_frame,
                    x=None,
                    x_level=None,
                    y=y,
                    hue=hue,
                    hue_level=group_b,
                )
                if len(a) < min_n or len(b) < min_n:
                    continue
                test_name, statistic, pvalue = _run_pairwise_test(a, b, test=test)
                results.append(
                    StatTestResult(
                        group=x_level,
                        group_a=group_a,
                        group_b=group_b,
                        statistic=statistic,
                        pvalue=pvalue,
                        test=test_name,
                        n_a=len(a),
                        n_b=len(b),
                        x1=float(x_idx + offsets[hue_lookup[group_a]]),
                        x2=float(x_idx + offsets[hue_lookup[group_b]]),
                    )
                )

    corrected = _apply_pvalue_correction(
        [result.pvalue for result in results],
        method=correction,
    )
    for result, corrected_pvalue in zip(results, corrected):
        result.pvalue = corrected_pvalue
    return results


def add_stat_tests(
    *,
    x: str,
    y: str,
    data: pd.DataFrame,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    order: Optional[Sequence[object]] = None,
    hue_order: Optional[Sequence[object]] = None,
    pairs: Optional[Sequence[Tuple[object, object]]] = None,
    test: str = "mannwhitney",
    correction: Optional[str] = None,
    min_n: int = 2,
    line_height: float = 0.02,
    line_offset: float = 0.06,
    stack_step: float = 0.06,
    text_offset: float = 0.01,
    linewidth: float = 1.0,
    fontsize: Optional[float] = None,
    bracket_width: float = 0.15,
    stars: bool = False,
    p_style: str = "p<",
    hide_ns: bool = False,
    cmap: Optional[object] = "viridis_r",
) -> List[StatTestResult]:
    """Annotate an existing seaborn categorical plot with pairwise stats."""

    if ax is None:
        ax = plt.gca()

    results = compute_stat_tests(
        data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        pairs=pairs,
        test=test,
        correction=correction,
        min_n=min_n,
    )
    if not results:
        return results

    frame = data.copy()
    required = [x, y] + ([hue] if hue else [])
    frame = frame.loc[:, required].dropna(subset=[x, y])
    if hue is not None:
        frame = frame.dropna(subset=[hue])

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin if ymax > ymin else 1.0
    per_group_counts: Dict[object, int] = {}
    top_needed = ymax

    for result in results:
        if hide_ns and result.pvalue >= 0.05:
            continue

        if hue is None:
            values_a = _subset_values(frame, x=x, x_level=result.group_a, y=y, hue=None, hue_level=None)
            values_b = _subset_values(frame, x=x, x_level=result.group_b, y=y, hue=None, hue_level=None)
            stack_key = "all"
        else:
            group_frame = frame.loc[frame[x] == result.group]
            values_a = _subset_values(
                group_frame,
                x=None,
                x_level=None,
                y=y,
                hue=hue,
                hue_level=result.group_a,
            )
            values_b = _subset_values(
                group_frame,
                x=None,
                x_level=None,
                y=y,
                hue=hue,
                hue_level=result.group_b,
            )
            stack_key = result.group

        data_top = np.nanmax(np.concatenate([values_a, values_b]))
        stack_idx = per_group_counts.get(stack_key, 0)
        bracket_y = data_top + y_range * (line_offset + stack_idx * stack_step)
        per_group_counts[stack_key] = stack_idx + 1

        add_pvalue_bracket(
            ax=ax,
            x1=result.x1,
            x2=result.x2,
            p=result.pvalue,
            y=bracket_y,
            h=y_range * line_height,
            text_offset=text_offset,
            linewidth=linewidth,
            fontsize=fontsize,
            bracket_width=bracket_width,
            stars=stars,
            p_style=p_style,
            cmap=cmap,
        )
        top_needed = max(
            top_needed,
            bracket_y + y_range * (line_height + text_offset + 0.08),
        )

    if top_needed > ax.get_ylim()[1]:
        ax.set_ylim(ax.get_ylim()[0], top_needed)

    return results
