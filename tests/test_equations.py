# ABOUTME: Tests for equation formatting and output functions.
# ABOUTME: Verifies _make_form_string replaces floats and format_all_equations produces valid text.

"""Tests for degen_detector.diagnostics.equations."""

import numpy as np
import sympy as sp

from degen_detector.implicit_fit import ImplicitFit
from degen_detector.diagnostics.equations import _make_form_string, format_all_equations


def _make_fit_with_floats():
    """Create a fit with float constants in equation_str."""
    x = sp.Symbol('x')
    return ImplicitFit(
        component_exprs=[2.5 * x**2, sp.Symbol('y')],
        constant=1.3,
        param_names=['x', 'y'],
        residual_std=0.01,
        orthogonal_r2=0.98,
        equation_str='2.5*x**2 + y = 1.3000',
        complexity=4,
    )


def test_make_form_string_replaces_floats():
    """_make_form_string replaces float literals with c_1, c_2, etc."""
    fit = _make_fit_with_floats()
    form = _make_form_string(fit)
    assert '2.5' not in form
    assert '1.3' not in form
    assert 'c_' in form
    assert '2' in form  # the exponent should be preserved


def test_make_form_string_preserves_integers():
    """_make_form_string keeps integer exponents as-is."""
    fit = ImplicitFit(
        component_exprs=[sp.Symbol('x')**3],
        constant=0.0,
        param_names=['x'],
        residual_std=0.0,
        orthogonal_r2=1.0,
        equation_str='x**3 = 0',
        complexity=2,
    )
    form = _make_form_string(fit)
    assert '3' in form


def test_format_all_equations_with_ground_truth():
    """format_all_equations includes ground truth when provided."""
    from degen_detector.core import CouplingFit, CouplingSearchResult
    from degen_detector.analysis import MIResult

    cf = CouplingFit(
        param_names=['x', 'y'],
        param_indices=[0, 1],
        mi_score=0.5,
        fits=[_make_fit_with_floats()],
        fit_order=0,
    )
    result = CouplingSearchResult(
        fits=[cf],
        n_fits_attempted=1,
        n_tuples_total=1,
        mi_result=MIResult(mi_matrix=np.array([[0, 0.5], [0.5, 0]]), param_names=['x', 'y']),
        selected_params=['x', 'y'],
    )
    ground_truth = {'equation': 'x + y = 1'}

    text = format_all_equations(result, ground_truth)
    assert 'x + y = 1' in text
    assert 'DEGENERACY DETECTION RESULTS' in text
    assert 'R\u00b2_ortho' in text
