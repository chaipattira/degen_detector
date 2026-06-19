# ABOUTME: Generates and ranks parameter tuples for degeneracy search.
# ABOUTME: Replaces graph-based group discovery with user-controlled tuple enumeration.

from dataclasses import dataclass
from itertools import combinations

@dataclass
class RankedTuple:
    """A parameter tuple with its MI-based ranking score."""
    param_indices: list
    param_names: list
    mi_score: float
    pairwise_mi: dict  # {(i, j): mi_value} for inspection


def compute_tuple_mi_score(indices, mi_matrix):
    """Compute MI score for a parameter tuple as the sum of all pairwise MI.

    Parameters
    ----------
    indices : list[int]
        Parameter indices in the tuple.
    mi_matrix : ndarray
        Mutual information matrix.

    Returns
    -------
    score : float
        Sum of pairwise MI values.
    pairwise_mi : dict
        Individual pairwise MI values.
    """
    pairs = list(combinations(indices, 2))
    pairwise_mi = {(i, j): mi_matrix[i, j] for i, j in pairs}
    values = list(pairwise_mi.values())

    if not values:
        return 0.0, pairwise_mi

    score = sum(values)
    return score, pairwise_mi


def generate_ranked_tuples(mi_result, param_indices, coupling_depth):
    """Generate all k-tuples and rank by MI score descending.

    Parameters
    ----------
    mi_result : MIResult
        Precomputed mutual information matrix.
    param_indices : list[int]
        Which parameter indices to consider.
    coupling_depth : int
        Size of tuples (k). Must be >= 2.

    Returns
    -------
    ranked_tuples : list[RankedTuple]
        All C(N,k) tuples, sorted by mi_score descending.
    """
    if coupling_depth < 2:
        raise ValueError("coupling_depth must be >= 2")
    if coupling_depth > len(param_indices):
        raise ValueError(
            f"coupling_depth ({coupling_depth}) cannot exceed "
            f"number of parameters ({len(param_indices)})"
        )

    tuples = []
    for indices in combinations(param_indices, coupling_depth):
        indices_list = list(indices)
        score, pairwise = compute_tuple_mi_score(
            indices_list, mi_result.mi_matrix
        )
        tuples.append(RankedTuple(
            param_indices=indices_list,
            param_names=[mi_result.param_names[i] for i in indices_list],
            mi_score=score,
            pairwise_mi=pairwise,
        ))

    # Sort descending by MI score (highest = most likely degenerate)
    tuples.sort(key=lambda t: t.mi_score, reverse=True)
    return tuples
