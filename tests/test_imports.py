# ABOUTME: Verifies all public imports from degen_detector package work correctly.
# ABOUTME: Guards against regressions during restructuring.

"""Import verification tests for degen_detector package."""


def test_top_level_imports():
    """All public names importable from degen_detector."""
    from degen_detector import (
        DegenDetector,
        CouplingFit,
        CouplingSearchResult,
        MIResult,
        mutual_info_matrix,
        select_params_by_mi,
        RankedTuple,
        generate_ranked_tuples,
        ImplicitFit,
        fit_separable_implicit,
        compute_orthogonal_loss,
        compute_orthogonal_r2,
        generate_banana_degeneracy,
        generate_cubic_degeneracy,
        generate_trig_separable,
        generate_scurve_separable,
    )


def test_diagnostics_imports():
    """All public names importable from degen_detector.diagnostics."""
    from degen_detector.diagnostics import (
        FitAnalyzer,
        DiagnosticsRunner,
        plot_corner,
        plot_components,
        plot_true_vs_predicted,
        plot_residuals,
        plot_manifold_2d,
        plot_manifold_3d,
        plot_projections_3d,
        plot_manifold,
        plot_mi_matrix,
        format_all_equations,
        save_equations,
    )


def test_diagnostics_private_imports():
    """Internal helper still importable for backward compat."""
    from degen_detector.diagnostics import _make_form_string


def test_implicit_fit_private_imports():
    """Private helpers used by diagnostics.equations module."""
    from degen_detector.implicit_fit import _functional_form_key, _rank_by_consensus
