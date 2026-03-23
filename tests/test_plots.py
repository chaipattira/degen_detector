# ABOUTME: Smoke tests for diagnostic plot functions.
# ABOUTME: Verifies each plot function returns a matplotlib Figure without error.

"""Smoke tests for degen_detector.diagnostics.plots."""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')

from degen_detector.implicit_fit import ImplicitFit
from degen_detector.diagnostics.analyzer import FitAnalyzer
from degen_detector.diagnostics.plots import (
    plot_corner,
    plot_components,
    plot_true_vs_predicted,
    plot_residuals,
    plot_manifold_2d,
    plot_mi_matrix,
)
from degen_detector.analysis import MIResult


def _make_2d_fit():
    return ImplicitFit(
        component_exprs=[sp.Symbol('x'), sp.Symbol('y')],
        constant=1.0,
        param_names=['x', 'y'],
        residual_std=0.01,
        orthogonal_r2=0.99,
        equation_str='x + y = 1.0000',
        complexity=2,
    )


def _make_samples_2d():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = 1 - x + rng.normal(0, 0.01, 50)
    return np.column_stack([x, y])


def test_plot_corner_returns_figure():
    import matplotlib.pyplot as plt
    fig = plot_corner(_make_samples_2d(), ['x', 'y'])
    assert hasattr(fig, 'savefig')
    plt.close(fig)


def test_plot_components_returns_figure():
    import matplotlib.pyplot as plt
    analyzer = FitAnalyzer(_make_2d_fit())
    fig = plot_components(analyzer, _make_samples_2d(), ['x', 'y'])
    assert hasattr(fig, 'savefig')
    plt.close(fig)


def test_plot_true_vs_predicted_returns_figure():
    import matplotlib.pyplot as plt
    analyzer = FitAnalyzer(_make_2d_fit())
    fig = plot_true_vs_predicted(analyzer, _make_samples_2d(), ['x', 'y'])
    assert hasattr(fig, 'savefig')
    plt.close(fig)


def test_plot_residuals_returns_figure():
    import matplotlib.pyplot as plt
    analyzer = FitAnalyzer(_make_2d_fit())
    fig = plot_residuals(analyzer, _make_samples_2d(), ['x', 'y'])
    assert hasattr(fig, 'savefig')
    plt.close(fig)


def test_plot_manifold_2d_returns_figure():
    import matplotlib.pyplot as plt
    analyzer = FitAnalyzer(_make_2d_fit())
    fig = plot_manifold_2d(analyzer, _make_samples_2d(), ['x', 'y'])
    assert hasattr(fig, 'savefig')
    plt.close(fig)


def test_plot_mi_matrix_returns_figure():
    import matplotlib.pyplot as plt
    mi_result = MIResult(
        mi_matrix=np.array([[0, 0.5], [0.5, 0]]),
        param_names=['x', 'y'],
    )
    fig = plot_mi_matrix(mi_result)
    assert hasattr(fig, 'savefig')
    plt.close(fig)
