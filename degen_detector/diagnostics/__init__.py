# ABOUTME: Public API for the diagnostics subpackage.
# ABOUTME: Re-exports all public names for backward-compatible imports.

from degen_detector.diagnostics.analyzer import FitAnalyzer as FitAnalyzer

from degen_detector.diagnostics.plots import (
    plot_corner as plot_corner,
    plot_components as plot_components,
    plot_true_vs_predicted as plot_true_vs_predicted,
    plot_residuals as plot_residuals,
    plot_manifold_2d as plot_manifold_2d,
    plot_manifold_3d as plot_manifold_3d,
    plot_projections_3d as plot_projections_3d,
    plot_manifold as plot_manifold,
    plot_mi_matrix as plot_mi_matrix,
)

from degen_detector.diagnostics.equations import (
    _make_form_string as _make_form_string,
    format_all_equations as format_all_equations,
    save_equations as save_equations,
)

from degen_detector.diagnostics.runner import DiagnosticsRunner as DiagnosticsRunner
