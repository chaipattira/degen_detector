# ABOUTME: Public API exports for degen_detector package.

from degen_detector.core import DegenDetector, CouplingFit, CouplingSearchResult
from degen_detector.analysis import MIResult, mutual_info_matrix, select_params_by_mi
from degen_detector.groups import RankedTuple, generate_ranked_tuples
from degen_detector.implicit_fit import ImplicitFit, fit_separable_implicit
from degen_detector.loss import compute_orthogonal_loss, compute_orthogonal_r2

from degen_detector.synthetic import (
    generate_polynomial_separable,
    generate_nonlinear_mixed,
    generate_trig_separable,
    generate_scurve_separable,
)
