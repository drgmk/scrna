import matplotlib
matplotlib.use("Agg")

import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "scrna" / "plotting.py"
SPEC = importlib.util.spec_from_file_location("scrna_plotting", MODULE_PATH)
plotting = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = plotting
SPEC.loader.exec_module(plotting)

add_stat_tests = plotting.add_stat_tests
compute_stat_tests = plotting.compute_stat_tests


def test_compute_stat_tests_by_hue_within_x_groups():
    df = pd.DataFrame(
        {
            "cell_type": ["B", "B", "B", "B", "T", "T", "T", "T"],
            "age": ["young", "young", "old", "old", "young", "young", "old", "old"],
            "histones_frac": [1.0, 1.2, 2.5, 2.7, 0.5, 0.6, 1.4, 1.6],
        }
    )

    results = compute_stat_tests(
        df,
        x="cell_type",
        y="histones_frac",
        hue="age",
        test="mannwhitney",
    )

    assert len(results) == 2
    assert {result.group for result in results} == {"B", "T"}
    assert {(result.group_a, result.group_b) for result in results} == {
        ("young", "old")
    }
    assert all(result.pvalue <= 1 for result in results)
    assert all(result.x1 != result.x2 for result in results)


def test_compute_stat_tests_pairwise_x_without_hue():
    df = pd.DataFrame(
        {
            "cell_type": ["B", "B", "Mono", "Mono", "T", "T"],
            "histones_frac": [1.0, 1.1, 1.8, 2.0, 3.0, 3.1],
        }
    )

    results = compute_stat_tests(
        df,
        x="cell_type",
        y="histones_frac",
        test="t_test",
    )

    assert len(results) == 3
    assert {result.group for result in results} == {None}
    assert {(result.group_a, result.group_b) for result in results} == {
        ("B", "Mono"),
        ("B", "T"),
        ("Mono", "T"),
    }


def test_add_stat_tests_adds_annotations_to_existing_seaborn_axis():
    df = pd.DataFrame(
        {
            "cell_type": ["B", "B", "B", "B", "T", "T", "T", "T"],
            "age": ["young", "young", "old", "old", "young", "young", "old", "old"],
            "histones_frac": [1.0, 1.2, 2.5, 2.7, 0.5, 0.6, 1.4, 1.6],
        }
    )

    fig, ax = plt.subplots()
    sns.violinplot(
        data=df,
        x="cell_type",
        y="histones_frac",
        hue="age",
        ax=ax,
    )

    results = add_stat_tests(
        x="cell_type",
        y="histones_frac",
        hue="age",
        data=df,
        ax=ax,
    )

    assert len(results) == 2
    assert len(ax.texts) >= 2
    plt.close(fig)


def test_add_stat_tests_colors_brackets_from_cmap():
    df = pd.DataFrame(
        {
            "cell_type": ["B", "B", "B", "B"],
            "age": ["young", "young", "old", "old"],
            "histones_frac": [1.0, 1.1, 2.5, 2.6],
        }
    )

    fig, ax = plt.subplots()
    sns.violinplot(
        data=df,
        x="cell_type",
        y="histones_frac",
        hue="age",
        ax=ax,
    )

    results = add_stat_tests(
        x="cell_type",
        y="histones_frac",
        hue="age",
        data=df,
        ax=ax,
        cmap="viridis_r",
    )

    expected_color = plt.get_cmap("viridis_r")(1.0 - results[0].pvalue)
    line_color = ax.lines[-1].get_color()

    assert np.allclose(line_color, expected_color)
    plt.close(fig)
