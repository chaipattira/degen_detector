# ABOUTME: Main API for the degen_detector package — DegenDetector class and result types.
# ABOUTME: Orchestrates MI ranking and symbolic fitting to find separable parameter degeneracies.

"""Main API for implicit separable degeneracy detection."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from degen_detector.analysis import MIResult, mutual_info_matrix, select_params_by_mi
from degen_detector.groups import generate_ranked_tuples
from degen_detector.implicit_fit import fit_separable_implicit


@dataclass
class CouplingFit:
    """Result of fitting one parameter tuple.

    Attributes
    ----------
    param_names : list
        Names of parameters in this tuple.
    param_indices : list
        Indices of parameters in this tuple.
    mi_score : float
        Mutual information score for this tuple.
    fits : list of ImplicitFit
        Top candidate fits ranked by orthogonal_r2 (descending).
        First element is the best fit.
    fit_order : int
        Order in which this tuple was fitted (by MI ranking).
    """
    param_names: list
    param_indices: list
    mi_score: float
    fits: list  # list of ImplicitFit, ranked by orthogonal_r2 (descending)
    fit_order: int

    @property
    def fit(self):
        """Best fit by orthogonal_r2 (first in ranked list)."""
        return self.fits[0] if self.fits else None


@dataclass
class CouplingSearchResult:
    """Results from implicit coupling search."""
    fits: list  # All fits ranked by MI (descending)
    n_fits_attempted: int
    n_tuples_total: int
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
        max_fits: Optional[int] = None,
        max_complexity: int = 15,
        niterations: int = 40,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
        n_candidates: int = 5,
        batch_size: int = 50,
        log_mode: bool = False,
        verbose: bool = True,
    ) -> CouplingSearchResult:
        """Search for implicit separable degeneracies.

        Ranks parameter combinations by MI (mutual information) and fits
        symbolic equations to each. Results are ranked by MI score - higher
        MI indicates stronger statistical dependency (more likely true
        degeneracy).

        Parameters
        ----------
        params : list[str] | int | None
            Which parameters to consider:
            - list[str]: Explicit parameter names to use
            - int: Auto-select top N parameters by total MI
            - None: Use all parameters
        coupling_depth : int
            Size of parameter tuples to test (2 for pairs, 3 for triplets, etc.)
        max_fits : int | None
            Maximum number of tuples to fit. Tuples are processed in MI-ranked
            order, so this fits the top N most promising combinations.
            None means fit all.
        max_complexity : int
            Maximum equation complexity for PySR.
        niterations : int
            Number of PySR evolution iterations.
        max_iterations : int
            Maximum iterations for alternating optimization.
        convergence_threshold : float
            Convergence threshold for alternating optimization.
        n_candidates : int
            Number of candidate equations to save per tuple (from PySR hall of fame).
        batch_size : int
            PySR batch size. Increase for large datasets (e.g. 500–2000 for
            n > 10k) to reduce fitness evaluation noise.
        log_mode : bool
            If True, exclude exp from PySR unary operators. Set automatically
            by DegenLogMode; manual use when samples are already in log space.
        verbose : bool
            Print progress information.

        Returns
        -------
        CouplingSearchResult
            Contains all fits ranked by MI (descending), with metadata.
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
            mi_result, selected_indices, coupling_depth
        )
        n_total = len(ranked_tuples)

        # Limit number of fits if requested
        tuples_to_fit = ranked_tuples[:max_fits] if max_fits else ranked_tuples
        n_to_fit = len(tuples_to_fit)

        if verbose:
            print(f"Generated {n_total} {coupling_depth}-tuples, ranked by sum MI")
            if max_fits and max_fits < n_total:
                print(f"Fitting top {n_to_fit} tuples by MI")

        fits = []

        for i, rt in enumerate(tuples_to_fit):
            if verbose:
                print(f"Fitting {i+1}/{n_to_fit}: {rt.param_names} (MI={rt.mi_score:.4f})")

            tuple_samples = self.samples[:, rt.param_indices]

            try:
                candidate_fits = fit_separable_implicit(
                    tuple_samples,
                    rt.param_names,
                    max_complexity=max_complexity,
                    niterations=niterations,
                    max_iterations=max_iterations,
                    convergence_threshold=convergence_threshold,
                    n_candidates=n_candidates,
                    batch_size=batch_size,
                    log_mode=log_mode,
                    verbose=True,
                )
            except Exception as e:
                if verbose:
                    print(f"  -> Fit failed: {e}")
                candidate_fits = []

            coupling_fit = CouplingFit(
                param_names=rt.param_names,
                param_indices=rt.param_indices,
                mi_score=rt.mi_score,
                fits=candidate_fits,
                fit_order=i,
            )
            fits.append(coupling_fit)

            if candidate_fits:
                best_fit = candidate_fits[0]
                if verbose:
                    n_success = len(candidate_fits)
                    print(f"  -> {n_success} candidates, best R²_ortho = {best_fit.orthogonal_r2:.4f}: {best_fit.equation_str}")
            else:
                if verbose:
                    print("  -> Fit failed")

        # Fits are already in MI-ranked order (descending)
        return CouplingSearchResult(
            fits=fits,
            n_fits_attempted=n_to_fit,
            n_tuples_total=n_total,
            mi_result=mi_result,
            selected_params=selected_names,
        )