# ABOUTME: Tests for FitAnalyzer class from the diagnostics.analyzer module.
# ABOUTME: Verifies evaluation, prediction, solving, and metric computation.

"""Tests for degen_detector.diagnostics.analyzer."""

import numpy as np
import sympy as sp

from degen_detector.implicit_fit import ImplicitFit
from degen_detector.diagnostics.analyzer import FitAnalyzer


def _make_linear_fit():
    """Create a simple linear fit: x + y = 1."""
    return ImplicitFit(
        component_exprs=[sp.Symbol('x'), sp.Symbol('y')],
        constant=1.0,
        param_names=['x', 'y'],
        residual_std=0.01,
        orthogonal_r2=0.99,
        equation_str='x + y = 1.0000',
        complexity=2,
    )


def _make_quadratic_fit():
    """Create a quadratic fit: x**2 + y = 1."""
    x = sp.Symbol('x')
    return ImplicitFit(
        component_exprs=[x**2, sp.Symbol('y')],
        constant=1.0,
        param_names=['x', 'y'],
        residual_std=0.01,
        orthogonal_r2=0.98,
        equation_str='x**2 + y = 1.0000',
        complexity=3,
    )


def test_g_evaluation():
    """FitAnalyzer.g() evaluates component functions correctly."""
    analyzer = FitAnalyzer(_make_linear_fit())
    values = np.array([1.0, 2.0, 3.0])
    result = analyzer.g('x', values)
    np.testing.assert_array_almost_equal(result, values)


def test_constraint_residual():
    """constraint_residual returns near-zero for points on the surface."""
    analyzer = FitAnalyzer(_make_linear_fit())
    samples = np.array([[0.3, 0.7], [0.5, 0.5], [1.0, 0.0]])
    residuals = analyzer.constraint_residual(samples, ['x', 'y'])
    np.testing.assert_array_almost_equal(residuals, 0.0)


def test_predict_component():
    """predict_component returns matching true/predicted values for perfect fit."""
    analyzer = FitAnalyzer(_make_linear_fit())
    samples = np.array([[0.3, 0.7], [0.5, 0.5]])
    true_g, pred_g = analyzer.predict_component('x', samples, ['x', 'y'])
    np.testing.assert_array_almost_equal(true_g, pred_g)


def test_is_linear():
    """is_linear detects linear components."""
    analyzer = FitAnalyzer(_make_linear_fit())
    assert analyzer.is_linear('x')
    assert analyzer.is_linear('y')


def test_is_quadratic():
    """is_quadratic detects quadratic components."""
    analyzer = FitAnalyzer(_make_quadratic_fit())
    assert analyzer.is_quadratic('x')
    assert not analyzer.is_quadratic('y')


def test_solve_for_linear_param():
    """solve_for_param solves linear components analytically."""
    analyzer = FitAnalyzer(_make_linear_fit())
    result = analyzer.solve_for_param('x', {'y': np.array([0.3])})
    np.testing.assert_array_almost_equal(result, [0.7])


def test_solve_for_quadratic_param():
    """solve_for_param solves quadratic components analytically."""
    analyzer = FitAnalyzer(_make_quadratic_fit())
    result = analyzer.solve_for_param('x', {'y': np.array([0.0])}, bounds=(0, 2))
    np.testing.assert_array_almost_equal(result, [1.0])


def test_get_metrics():
    """get_metrics returns expected keys."""
    analyzer = FitAnalyzer(_make_linear_fit())
    metrics = analyzer.get_metrics()
    assert 'equation' in metrics
    assert 'r2_ortho' in metrics
    assert metrics['n_params'] == 2
