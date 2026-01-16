"""
Plotting utilities for long-form single-cell summaries.

This module bundles a house "bio_journal" matplotlib style and helpers to annotate
simple grouped plots with p-values.
"""

from __future__ import annotations

import importlib.resources as importlib_resources
from contextlib import nullcontext
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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
    color: Optional[str] = None,
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

    x1, x2 = float(x1), float(x2)
    if x2 < x1:
        x1, x2 = x2, x1

    dx = x2 - x1
    stub = bracket_width * dx

    xs = [x1, x1, x1 + stub, x2 - stub, x2, x2]
    ys = [y, y + h, y + h, y + h, y + h, y]
    ax.plot(xs, ys, lw=linewidth, clip_on=False, color=color)

    ax.text(
        (x1 + x2) / 2,
        y + h + text_offset * yr,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        clip_on=False,
        color=color,
    )

    top_needed = y + h + (text_offset * yr) + 0.05 * yr
    if top_needed > ymax:
        ax.set_ylim(ymin, top_needed)
