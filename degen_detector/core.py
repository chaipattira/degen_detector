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
    fits: list  # list of ImplicitFit, ranked by BIC ascending (lower is better)
    fit_order: int

    @property
    def fit(self):
        """Best fit by BIC (first in ranked list)."""
        return self.fits[0] if self.fits else None


@dataclass
class RankingResult:
    """Output of rank_couplings — parameter tuples ranked by MI, before fitting.

    Attributes
    ----------
    tuples : list[RankedTuple]
        All coupling_depth-tuples sorted by aggregated MI score (descending).
    mi_result : MIResult
        Full pairwise MI matrix for all selected parameters.
    selected_params : list[str]
        Parameter names that were screened.
    coupling_depth : int
        Tuple size used.
    """
    tuples: list
    mi_result: 'MIResult'
    selected_params: list
    coupling_depth: int


@dataclass
class CouplingSearchResult:
    """Results from implicit coupling search."""
    fits: list  # All fits ranked by MI (descending)
    n_fits_attempted: int
    n_tuples_total: int
    mi_result: 'MIResult'
    selected_params: list


def rank_couplings(
    samples: np.ndarray,
    param_names: list,
    coupling_depth: int = 2,
    params: Optional[Union[list, int]] = None,
    verbose: bool = True,
) -> list:
    """Rank parameter tuples by mutual information — fast screening stage.

    Computes pairwise MI between all parameters and ranks every coupling_depth-
    tuple by its aggregated MI score. No symbolic fitting is performed; this is
    cheap and returns quickly even for large parameter spaces.

    Use this to inspect which parameter combinations are most likely degenerate
    before committing to the expensive PySR fitting stage (fit_couplings).

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_params)
        Posterior samples.
    param_names : list[str]
        Names matching columns of samples.
    coupling_depth : int
        Size of parameter tuples to rank (2 for pairs, 3 for triplets, etc.)
    params : list[str] | int | None
        Which parameters to consider:
        - list[str]: Explicit parameter names
        - int: Auto-select top N parameters by total MI
        - None: Use all parameters
    verbose : bool
        Print progress information.

    Returns
    -------
    ranked_tuples : list[RankedTuple]
        All coupling_depth-tuples sorted by aggregated MI score (descending).
        Inspect .mi_score and .param_names to decide what to fit.
    """
    samples = np.asarray(samples)
    if verbose:
        print("Computing mutual information matrix...")
    mi_result = mutual_info_matrix(samples, param_names)

    if params is None:
        selected_indices = list(range(len(param_names)))
        selected_names = param_names
    elif isinstance(params, int):
        selected_indices, selected_names = select_params_by_mi(mi_result, params)
    else:
        selected_names = list(params)
        selected_indices = [param_names.index(p) for p in selected_names]

    if verbose:
        print(f"Selected {len(selected_names)} parameters: {selected_names}")

    ranked_tuples = generate_ranked_tuples(mi_result, selected_indices, coupling_depth)

    if verbose:
        print(f"Generated {len(ranked_tuples)} {coupling_depth}-tuples, ranked by sum MI")

    return RankingResult(
        tuples=ranked_tuples,
        mi_result=mi_result,
        selected_params=selected_names,
        coupling_depth=coupling_depth,
    )


def fit_couplings(
    samples: np.ndarray,
    ranked_tuples: "Union[RankingResult, list]",
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
    """Fit symbolic degeneracy equations to pre-ranked parameter tuples.

    Takes the output of rank_couplings and runs alternating symbolic regression
    (PySR) on the top tuples. This is the expensive stage — budget ~minutes per
    tuple depending on niterations and coupling_depth.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_params)
        Posterior samples (same array passed to rank_couplings).
    ranked_tuples : RankingResult or list[RankedTuple]
        Output of rank_couplings (preferred), or a plain list of RankedTuple.
        Passing a RankingResult preserves the MI matrix in the output.
    max_fits : int | None
        Maximum number of tuples to fit. None means fit all.
    max_complexity : int
        Maximum equation complexity for PySR.
    niterations : int
        Number of PySR evolution iterations per component.
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
        If True, exclude exp from PySR unary operators. Use when samples are
        already in log space (e.g. from DegenLogMode).
    verbose : bool
        Print progress information.

    Returns
    -------
    CouplingSearchResult
        Contains all fits ranked by MI (descending), with metadata.
    """
    samples = np.asarray(samples)

    if isinstance(ranked_tuples, RankingResult):
        mi_result = ranked_tuples.mi_result
        selected_params = ranked_tuples.selected_params
        tuples_list = ranked_tuples.tuples
    else:
        mi_result = None
        selected_params = list({n for rt in ranked_tuples for n in rt.param_names})
        tuples_list = ranked_tuples

    tuples_to_fit = tuples_list[:max_fits] if max_fits else tuples_list
    n_to_fit = len(tuples_to_fit)
    n_total = len(tuples_list)

    if verbose and max_fits and max_fits < n_total:
        print(f"Fitting top {n_to_fit} of {n_total} tuples by MI")

    fits = []
    for i, rt in enumerate(tuples_to_fit):  # type: ignore[union-attr]
        if verbose:
            print(f"Fitting {i+1}/{n_to_fit}: {rt.param_names} (MI={rt.mi_score:.4f})")

        tuple_samples = samples[:, rt.param_indices]

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
                verbose=verbose,
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

        if candidate_fits and verbose:
            best = candidate_fits[0]
            print(f"  -> {len(candidate_fits)} candidates, best BIC = {best.bic:.2f} (AIC={best.aic:.2f}, R²_ortho={best.orthogonal_r2:.4f}): {best.equation_str}")
        elif verbose:
            print("  -> Fit failed")

    return CouplingSearchResult(
        fits=fits,
        n_fits_attempted=n_to_fit,
        n_tuples_total=n_total,
        mi_result=mi_result,
        selected_params=selected_params,
    )


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

    def rank_couplings(
        self,
        params: Optional[Union[list, int]] = None,
        coupling_depth: int = 2,
        verbose: bool = True,
    ) -> RankingResult:
        """Rank parameter tuples by MI — fast screening stage.

        See module-level rank_couplings for full documentation.
        """
        return rank_couplings(
            self.samples, self.param_names,
            coupling_depth=coupling_depth, params=params, verbose=verbose,
        )

    def fit_couplings(
        self,
        ranked_tuples: list,
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
        """Fit symbolic equations to pre-ranked tuples — expensive fitting stage.

        See module-level fit_couplings for full documentation.
        """
        return fit_couplings(
            self.samples, ranked_tuples,
            max_fits=max_fits,
            max_complexity=max_complexity,
            niterations=niterations,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            n_candidates=n_candidates,
            batch_size=batch_size,
            log_mode=log_mode,
            verbose=verbose,
        )

