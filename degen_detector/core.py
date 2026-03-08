"""Main API for implicit separable degeneracy detection."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from degen_detector.analysis import MIResult, mutual_info_matrix, select_params_by_mi
from degen_detector.groups import generate_ranked_tuples
from degen_detector.implicit_fit import ImplicitFit, fit_separable_implicit


@dataclass
class CouplingFit:
    """Result of fitting one parameter tuple."""
    param_names: list
    param_indices: list
    mi_score: float
    fit: 'ImplicitFit'
    fit_order: int


@dataclass
class CouplingSearchResult:
    """Results from implicit coupling search."""
    fits: list
    best_fit: CouplingFit
    stopped_early: bool
    n_fits_attempted: int
    n_fits_total: int
    mi_result: 'MIResult'
    selected_params: list


class DegenDetector:
    """Implicit separable degeneracy search in Bayesian posterior samples."""

    def __init__(self, samples, param_names):
        """Initialize the detector with posterior samples.

        Parameters
        ----------
        samples : array-like, shape (n_samples, n_params)
            Posterior samples, one row per sample.
        param_names : list[str]
            Names of each parameter column.
        """
        if samples.shape[1] != len(param_names):
            raise ValueError(
                f"samples has {samples.shape[1]} columns but "
                f"{len(param_names)} param names given"
            )
        self.samples = np.asarray(samples)
        self.param_names = list(param_names)

    def search_couplings(
        self,
        params: Optional[Union[list, int]] = None,
        coupling_depth: int = 2,
        r2_threshold: float = 0.95,
        max_fits: Optional[int] = None,
        mi_rank_method: str = "sum",
        max_complexity: int = 15,
        niterations: int = 40,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
        verbose: bool = True,
    ) -> CouplingSearchResult:
        """Search for implicit separable degeneracies.

        Parameters
        ----------
        params : list[str] | int | None
            Which parameters to consider:
            - list[str]: Explicit parameter names to use
            - int: Auto-select top N parameters by total MI
            - None: Use all parameters
        coupling_depth : int
            Size of parameter tuples to test (2 for pairs, 3 for triplets, etc.)
        r2_threshold : float
            Stop searching when a fit achieves orthogonal R² >= this value.
        max_fits : int | None
            Maximum number of symbolic fits to attempt. None means try all.
        mi_rank_method : str
            How to rank tuples by MI: "min", "avg", "sum", or "geometric".
        max_complexity : int
            Maximum equation complexity for PySR.
        niterations : int
            Number of PySR evolution iterations.
        max_iterations : int
            Maximum iterations for alternating optimization.
        convergence_threshold : float
            Convergence threshold for alternating optimization.
        verbose : bool
            Print progress information.

        Returns
        -------
        CouplingSearchResult
            Contains all attempted fits, best fit, and metadata.
        """
        if verbose:
            print("Computing mutual information matrix...")
        mi_result = mutual_info_matrix(self.samples, self.param_names)

        if params is None:
            selected_indices = list(range(len(self.param_names)))
            selected_names = self.param_names
        elif isinstance(params, int):
            selected_indices, selected_names = select_params_by_mi(mi_result, params)
        else:
            selected_names = list(params)
            selected_indices = [self.param_names.index(p) for p in selected_names]

        if verbose:
            print(f"Selected {len(selected_names)} parameters: {selected_names}")

        ranked_tuples = generate_ranked_tuples(
            mi_result, selected_indices, coupling_depth, mi_rank_method
        )
        n_total = len(ranked_tuples)

        if verbose:
            print(f"Generated {n_total} {coupling_depth}-tuples, ranked by {mi_rank_method} MI")

        fits = []
        best_fit = None
        stopped_early = False

        for i, rt in enumerate(ranked_tuples):
            if max_fits is not None and i >= max_fits:
                break

            if verbose:
                print(f"Fitting {i+1}/{n_total}: {rt.param_names} (MI={rt.mi_score:.4f})")

            tuple_samples = self.samples[:, rt.param_indices]

            try:
                fit = fit_separable_implicit(
                    tuple_samples,
                    rt.param_names,
                    max_complexity=max_complexity,
                    niterations=niterations,
                    max_iterations=max_iterations,
                    convergence_threshold=convergence_threshold,
                    verbose=False,
                )
            except Exception as e:
                if verbose:
                    print(f"  -> Fit failed: {e}")
                fit = None

            coupling_fit = CouplingFit(
                param_names=rt.param_names,
                param_indices=rt.param_indices,
                mi_score=rt.mi_score,
                fit=fit,
                fit_order=i,
            )
            fits.append(coupling_fit)

            if fit is not None:
                if verbose:
                    print(f"  -> R²_ortho = {fit.orthogonal_r2:.4f}: {fit.equation_str}")

                if best_fit is None or fit.orthogonal_r2 > best_fit.fit.orthogonal_r2:
                    best_fit = coupling_fit

                if fit.orthogonal_r2 >= r2_threshold:
                    if verbose:
                        print(f"Early stop: R²_ortho >= {r2_threshold}")
                    stopped_early = True
                    break
            else:
                if verbose:
                    print("  -> Fit failed")

        return CouplingSearchResult(
            fits=fits,
            best_fit=best_fit,
            stopped_early=stopped_early,
            n_fits_attempted=len(fits),
            n_fits_total=n_total,
            mi_result=mi_result,
            selected_params=selected_names,
        )
