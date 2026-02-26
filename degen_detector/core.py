# ABOUTME: Main API for user-controlled degeneracy search in posterior samples.
# ABOUTME: Orchestrates MI analysis, tuple generation, and progressive symbolic regression.

from dataclasses import dataclass

import numpy as np

from degen_detector.analysis import MIResult, mutual_info_matrix, select_params_by_mi
from degen_detector.groups import generate_ranked_tuples
from degen_detector.regression import MultiSymbolicFit, fit_tuple


@dataclass
class CouplingFit:
    """Result of fitting one parameter tuple."""
    param_names: list
    param_indices: list
    mi_score: float
    fit: MultiSymbolicFit  # None if fit failed
    fit_order: int  # Position in MI-ranked search order


@dataclass
class CouplingSearchResult:
    """Results from user-controlled coupling search."""
    fits: list  # list[CouplingFit], in MI-ranked order
    best_fit: CouplingFit  # Highest R² among successful fits (None if none succeeded)
    stopped_early: bool  # True if r2_threshold was reached
    n_fits_attempted: int
    n_fits_total: int  # Total C(N,k) combinations
    mi_result: MIResult
    selected_params: list  # Parameters that were searched


class DegenDetector:
    """User-controlled degeneracy search in Bayesian posterior samples.

    Finds which parameters are degenerate (via mutual information ranking)
    and what the functional form is (via symbolic regression).
    """

    def __init__(self, samples, param_names):
        if samples.shape[1] != len(param_names):
            raise ValueError(
                f"samples has {samples.shape[1]} columns but "
                f"{len(param_names)} param names given"
            )
        self.samples = np.asarray(samples)
        self.param_names = list(param_names)

    def search_couplings(
        self,
        params=None,
        coupling_depth=2,
        r2_threshold=0.95,
        max_fits=None,
        mi_rank_method="min",
        max_complexity=25,
        niterations=60,
        batching=False,
        verbose=True,
        output_dir=None,
    ):
        """Search for degeneracies with user-controlled depth and parameters.

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
            Stop searching when a fit achieves R² >= this value.
        max_fits : int | None
            Maximum number of symbolic fits to attempt. None means try all.
        mi_rank_method : str
            How to rank tuples by MI:
            - "min": minimum pairwise MI (most conservative)
            - "avg": average pairwise MI
            - "sum": sum of all pairwise MI
            - "geometric": geometric mean
        max_complexity : int
            Maximum equation complexity for PySR.
        niterations : int
            Number of PySR evolution iterations.
        batching : bool
            Enable batching for large datasets.
        verbose : bool
            Print progress information.
        output_dir : str or Path, optional
            Directory for PySR output files. Creates descriptive subfolders
            named after the parameter tuples being fit.

        Returns
        -------
        result : CouplingSearchResult
            Contains all attempted fits, best fit, and metadata.
        """
        # Step 1: Compute MI matrix (always on all params for selection)
        if verbose:
            print("Computing mutual information matrix...")
        mi_result = mutual_info_matrix(self.samples, self.param_names)

        # Step 2: Parameter selection
        if params is None:
            selected_indices = list(range(len(self.param_names)))
            selected_names = self.param_names
        elif isinstance(params, int):
            selected_indices, selected_names = select_params_by_mi(mi_result, params)
        else:
            # Explicit list of names
            selected_names = list(params)
            selected_indices = [self.param_names.index(p) for p in selected_names]

        if verbose:
            print(f"Selected {len(selected_names)} parameters: {selected_names}")

        # Step 3: Generate MI-ranked tuples
        ranked_tuples = generate_ranked_tuples(
            mi_result, selected_indices, coupling_depth, mi_rank_method
        )
        n_total = len(ranked_tuples)

        if verbose:
            print(f"Generated {n_total} {coupling_depth}-tuples, ranked by {mi_rank_method} MI")
            if n_total > 100:
                print(f"  (Consider reducing params or using max_fits for faster search)")

        # Step 4: Progressive fitting
        fits = []
        best_fit = None
        stopped_early = False

        for i, rt in enumerate(ranked_tuples):
            if max_fits is not None and i >= max_fits:
                break

            if verbose:
                print(f"Fitting {i+1}/{n_total}: {rt.param_names} (MI={rt.mi_score:.4f})")

            fit = fit_tuple(
                self.samples, self.param_names, rt.param_indices,
                max_complexity, niterations, batching, output_dir
            )

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
                    print(f"  -> R² = {fit.r_squared:.4f}: {fit.equation_str}")

                if best_fit is None or fit.r_squared > best_fit.fit.r_squared:
                    best_fit = coupling_fit

                if fit.r_squared >= r2_threshold:
                    if verbose:
                        print(f"Early stop: R² >= {r2_threshold}")
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
