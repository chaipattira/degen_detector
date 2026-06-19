# ABOUTME: Tests for DegenLogMode coordinate-transform wrapper.
# ABOUTME: Covers transform application, param renaming, back-substitution, edge cases, and params translation.

import numpy as np
import pytest
import sympy

from degen_detector.implicit_fit import ImplicitFit
from degen_detector.transforms import (
    LOG_TRANSFORM,
    ParameterTransform,
    DegenLogMode,
    _back_transform_fit,
    _is_already_log,
)


# ---------------------------------------------------------------------------
# _is_already_log
# ---------------------------------------------------------------------------


def test_is_already_log_detects_log_prefix():
    assert _is_already_log("logA")
    assert _is_already_log("log_A")
    assert _is_already_log("LOG_SIGMA8")
    assert _is_already_log("lnAs")
    assert _is_already_log("ln_omega")


def test_is_already_log_passes_normal_params():
    assert not _is_already_log("sigma8")
    assert not _is_already_log("Omega_m")
    assert not _is_already_log("H0")
    assert not _is_already_log("ns")
    assert not _is_already_log("As")


# ---------------------------------------------------------------------------
# ParameterTransform basics
# ---------------------------------------------------------------------------


def test_log_transform_apply():
    """LOG_TRANSFORM.apply maps to np.log values."""
    x = np.array([1.0, 2.0, np.e])
    result = LOG_TRANSFORM.apply(x)
    np.testing.assert_allclose(result, np.log(x))


def test_log_transform_sympy_fn():
    """LOG_TRANSFORM.sympy_fn applied to a Symbol gives sympy.log."""
    sym = sympy.Symbol("x")
    expr = LOG_TRANSFORM.sympy_fn(sym)
    assert expr == sympy.log(sym)


def test_log_transform_name_prefix():
    """LOG_TRANSFORM.name_prefix is 'log_'."""
    assert LOG_TRANSFORM.name_prefix == "log_"


# ---------------------------------------------------------------------------
# DegenLogMode construction
# ---------------------------------------------------------------------------


def _make_samples(n=200, seed=42):
    rng = np.random.RandomState(seed)
    a = rng.uniform(0.1, 0.5, n)
    b = rng.uniform(0.5, 1.5, n)
    return np.column_stack([a, b]), ["Omega_m", "h"]


def test_degen_log_mode_transforms_all_samples():
    """Default (None transforms) applies log to every column."""
    samples, names = _make_samples()
    detector = DegenLogMode(samples, names)
    expected = np.log(samples)
    np.testing.assert_allclose(detector._transformed_samples, expected)


def test_degen_log_mode_renames_all_params():
    """Default transform renames 'X' → 'log_X' for all params."""
    samples, names = _make_samples()
    detector = DegenLogMode(samples, names)
    assert detector._transformed_names == ["log_Omega_m", "log_h"]


def test_degen_log_mode_partial_transform():
    """Only specified parameters are transformed; others pass through."""
    samples, names = _make_samples()
    detector = DegenLogMode(samples, names, transforms={"Omega_m": LOG_TRANSFORM})
    # First column should be log-transformed
    np.testing.assert_allclose(
        detector._transformed_samples[:, 0], np.log(samples[:, 0])
    )
    # Second column should be unchanged
    np.testing.assert_allclose(
        detector._transformed_samples[:, 1], samples[:, 1]
    )


def test_degen_log_mode_partial_renames_only_transformed():
    """Only transformed params get renamed; others keep original names."""
    samples, names = _make_samples()
    detector = DegenLogMode(samples, names, transforms={"Omega_m": LOG_TRANSFORM})
    assert detector._transformed_names == ["log_Omega_m", "h"]


# ---------------------------------------------------------------------------
# _back_transform_fit
# ---------------------------------------------------------------------------


def _make_log_space_fit():
    """ImplicitFit as if returned from pipeline in log space.

    Represents: log_Omega_m + 2*log_h = 1.5
    (i.e., g1(z0) = z0, g2(z1) = 2*z1 in normalized space, already
    back-substituted to log_Omega_m / log_h symbols by fit_separable_implicit).
    """
    log_Om = sympy.Symbol("log_Omega_m")
    log_h = sympy.Symbol("log_h")
    return ImplicitFit(
        component_exprs=[log_Om, 2 * log_h],
        constant=1.5,
        param_names=["log_Omega_m", "log_h"],
        residual_std=0.01,
        orthogonal_r2=0.99,
        equation_str="log_Omega_m + 2*log_h = 1.5000",
        complexity=2,
    )


def test_back_transform_fit_restores_param_names():
    """Back-transformed fit uses original param names."""
    fit = _make_log_space_fit()
    transform_map = {"Omega_m": LOG_TRANSFORM, "h": LOG_TRANSFORM}
    result = _back_transform_fit(fit, transform_map)
    assert result.param_names == ["Omega_m", "h"]


def test_back_transform_fit_expr_component0():
    """Back-transformed component 0 is log(Omega_m)."""
    fit = _make_log_space_fit()
    transform_map = {"Omega_m": LOG_TRANSFORM, "h": LOG_TRANSFORM}
    result = _back_transform_fit(fit, transform_map)
    Om = sympy.Symbol("Omega_m")
    assert sympy.simplify(result.component_exprs[0] - sympy.log(Om)) == 0


def test_back_transform_fit_expr_component1():
    """Back-transformed component 1 is 2*log(h)."""
    fit = _make_log_space_fit()
    transform_map = {"Omega_m": LOG_TRANSFORM, "h": LOG_TRANSFORM}
    result = _back_transform_fit(fit, transform_map)
    h = sympy.Symbol("h")
    assert sympy.simplify(result.component_exprs[1] - 2 * sympy.log(h)) == 0


def test_back_transform_fit_equation_str_no_log_prefix():
    """Equation string does not contain 'log_Omega_m' or 'log_h'."""
    fit = _make_log_space_fit()
    transform_map = {"Omega_m": LOG_TRANSFORM, "h": LOG_TRANSFORM}
    result = _back_transform_fit(fit, transform_map)
    assert "log_Omega_m" not in result.equation_str
    assert "log_h" not in result.equation_str


def test_back_transform_fit_preserves_metrics():
    """Back-transformation preserves orthogonal_r2, residual_std, constant."""
    fit = _make_log_space_fit()
    transform_map = {"Omega_m": LOG_TRANSFORM, "h": LOG_TRANSFORM}
    result = _back_transform_fit(fit, transform_map)
    assert result.orthogonal_r2 == fit.orthogonal_r2
    assert result.residual_std == fit.residual_std
    assert result.constant == fit.constant


def test_back_transform_fit_untransformed_param_unchanged():
    """Params not in transform_map keep their expression and name."""
    sigma8 = sympy.Symbol("sigma8")
    log_Om = sympy.Symbol("log_Omega_m")
    fit = ImplicitFit(
        component_exprs=[log_Om, sigma8 ** 2],
        constant=2.0,
        param_names=["log_Omega_m", "sigma8"],
        residual_std=0.02,
        orthogonal_r2=0.95,
        equation_str="log_Omega_m + sigma8**2 = 2.0000",
        complexity=3,
    )
    transform_map = {"Omega_m": LOG_TRANSFORM}
    result = _back_transform_fit(fit, transform_map)
    # sigma8 component unchanged
    assert result.param_names[1] == "sigma8"
    assert sympy.simplify(result.component_exprs[1] - sigma8 ** 2) == 0


# ---------------------------------------------------------------------------
# DegenLogMode.evaluate round-trip (no PySR — uses constructed fit)
# ---------------------------------------------------------------------------


def test_evaluate_on_original_samples_after_back_transform():
    """ImplicitFit.evaluate() works correctly on original samples after back-transform."""
    fit = _make_log_space_fit()
    transform_map = {"Omega_m": LOG_TRANSFORM, "h": LOG_TRANSFORM}
    result = _back_transform_fit(fit, transform_map)

    # Construct samples satisfying log(Omega_m) + 2*log(h) = 1.5 exactly
    rng = np.random.RandomState(0)
    Om = rng.uniform(0.1, 0.5, 100)
    # log(Om) + 2*log(h) = 1.5 => log(h) = (1.5 - log(Om)) / 2
    h = np.exp((1.5 - np.log(Om)) / 2)
    samples = np.column_stack([Om, h])

    residuals = result.evaluate(samples)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases: non-finite values after transform
# ---------------------------------------------------------------------------


def test_log_of_zero_filters_row():
    """Log transform on samples containing zero drops the bad row."""
    samples = np.array([[0.0, 1.0], [0.5, 2.0]])
    detector = DegenLogMode(samples, ["a", "b"])
    # Row 0 (contains zero) is dropped; only 1 sample should remain.
    assert len(detector.samples) == 1
    np.testing.assert_allclose(detector.samples[0], [0.5, 2.0])


def test_log_of_negative_filters_row():
    """Log transform on samples containing negative values drops the bad row."""
    samples = np.array([[-0.1, 1.0], [0.5, 2.0]])
    detector = DegenLogMode(samples, ["a", "b"])
    assert len(detector.samples) == 1
    np.testing.assert_allclose(detector.samples[0], [0.5, 2.0])


def test_log_of_positive_does_not_raise():
    """Log transform on strictly positive samples is fine."""
    samples = np.array([[0.1, 1.0], [0.5, 2.0]])
    DegenLogMode(samples, ["a", "b"])  # no exception


def test_already_log_param_not_transformed():
    """Parameters with 'log'/'ln' prefix are passed through unchanged."""
    rng = np.random.RandomState(0)
    logA = rng.uniform(2.5, 3.5, 100)   # already in log space, can be negative
    sigma8 = rng.uniform(0.6, 1.0, 100)
    samples = np.column_stack([logA, sigma8])
    detector = DegenLogMode(samples, ["logA", "sigma8"])
    # logA column should be unchanged
    np.testing.assert_allclose(detector._transformed_samples[:, 0], logA)
    # sigma8 column should be log-transformed
    np.testing.assert_allclose(detector._transformed_samples[:, 1], np.log(sigma8))
    # names: logA stays as-is, sigma8 becomes log_sigma8
    assert detector._transformed_names == ["logA", "log_sigma8"]


def test_untransformed_param_with_zeros_does_not_raise():
    """Zero values are fine if that param is not being log-transformed."""
    samples = np.array([[0.0, 1.0], [0.0, 2.0]])
    # Only transform "b", not "a"
    DegenLogMode(samples, ["a", "b"], transforms={"b": LOG_TRANSFORM})  # no exception


# ---------------------------------------------------------------------------
# params translation: original names → transformed names
# ---------------------------------------------------------------------------


def test_translate_params_list_of_strings():
    """_translate_params converts original param names to transformed names."""
    samples, names = _make_samples()
    detector = DegenLogMode(samples, names)
    translated = detector._translate_params(["Omega_m", "h"])
    assert translated == ["log_Omega_m", "log_h"]


def test_translate_params_untransformed_name_unchanged():
    """Untransformed params pass through _translate_params unchanged."""
    samples = np.column_stack([
        np.random.uniform(0.1, 0.5, 200),
        np.random.uniform(0.5, 1.5, 200),
        np.random.uniform(0.6, 0.9, 200),
    ])
    detector = DegenLogMode(
        samples, ["Omega_m", "h", "sigma8"],
        transforms={"Omega_m": LOG_TRANSFORM, "h": LOG_TRANSFORM},
    )
    translated = detector._translate_params(["Omega_m", "sigma8"])
    assert translated == ["log_Omega_m", "sigma8"]


def test_translate_params_none_passthrough():
    """_translate_params returns None unchanged (means 'all params')."""
    samples, names = _make_samples()
    detector = DegenLogMode(samples, names)
    assert detector._translate_params(None) is None


def test_translate_params_int_passthrough():
    """_translate_params returns int unchanged (means 'top N by MI')."""
    samples, names = _make_samples()
    detector = DegenLogMode(samples, names)
    assert detector._translate_params(5) == 5
