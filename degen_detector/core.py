# ABOUTME: Main API for automatic degeneracy detection in posterior samples.
# ABOUTME: Orchestrates MI analysis, group discovery, and symbolic regression.

from dataclasses import dataclass

import numpy as np

from degen_detector.analysis import mutual_info_matrix
from degen_detector.groups import GroupingResult, find_degenerate_groups
from degen_detector.regression import MultiSymbolicFit, fit_group_all_targets


@dataclass
class MultiDegeneracy:
    """One discovered degeneracy relationship between parameters."""
    param_names: list
    fit: MultiSymbolicFit


@dataclass
class MultiDegenResults:
    """Full results from the degeneracy detection pipeline."""
    multi_degeneracies: list  # list[MultiDegeneracy], ranked by R²
    grouping: GroupingResult
    param_names: list
    samples: np.ndarray


class DegenDetector:
    """Automatic degeneracy detection in Bayesian posterior samples.

    Finds which parameters are degenerate (via mutual information) and
    what the functional form is (via symbolic regression).
    """

    def __init__(self, samples, param_names):
        if samples.shape[1] != len(param_names):
            raise ValueError(
                f"samples has {samples.shape[1]} columns but "
                f"{len(param_names)} param names given"
            )
        self.samples = np.asarray(samples)
        self.param_names = list(param_names)

    def analyze_multi(self, mi_threshold=0.1, min_group_size=2,
                      max_group_size=5, max_complexity=25, niterations=60):
        """Run the full degeneracy detection pipeline.

        Steps:
        1. Compute mutual information matrix
        2. Discover degenerate groups
        3. Fit symbolic equations for each group
        """
        # Step 1: MI matrix
        mi_result = mutual_info_matrix(self.samples, self.param_names)

        # Step 2: Group discovery
        grouping = find_degenerate_groups(
            mi_result,
            mi_threshold=mi_threshold,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            samples=self.samples,
        )

        # Step 3: Symbolic regression for each group
        degeneracies = []
        for group in grouping.groups:
            fit = fit_group_all_targets(
                group, self.samples, self.param_names,
                max_complexity=max_complexity,
                niterations=niterations,
            )
            if fit is not None:
                degeneracies.append(MultiDegeneracy(
                    param_names=group.param_names,
                    fit=fit,
                ))

        # Rank by R² descending
        degeneracies.sort(key=lambda d: d.fit.r_squared, reverse=True)

        return MultiDegenResults(
            multi_degeneracies=degeneracies,
            grouping=grouping,
            param_names=self.param_names,
            samples=self.samples,
        )
