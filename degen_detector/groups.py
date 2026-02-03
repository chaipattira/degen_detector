# ABOUTME: Discovers groups of degenerate parameters from mutual information structure.
# ABOUTME: Uses graph-based connected components with validation and constraint counting.

from collections import deque
from dataclasses import dataclass

import numpy as np

from degen_detector.analysis import MIResult, local_pca_intrinsic_dim


@dataclass
class DegeneracyGroup:
    """A group of jointly degenerate parameters."""
    param_names: list
    param_indices: list
    n_constraints: int
    avg_mi: float


@dataclass
class GroupingResult:
    """All discovered degenerate groups plus the MI matrix used."""
    groups: list  # list[DegeneracyGroup]
    mi_result: MIResult


def _bfs(start, adjacency):
    """Return the connected component containing start."""
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def find_degenerate_groups(mi_result, mi_threshold=0.1, min_group_size=2,
                           max_group_size=5, samples=None):
    """Identify groups of degenerate parameters from an MI matrix.

    Steps:
    1. Build adjacency graph from MI threshold
    2. Find connected components via BFS
    3. Validate: remove params with weak connections
    4. Cap group sizes
    5. Filter small groups
    6. Count constraints via local PCA (if samples provided)
    """
    mi = mi_result.mi_matrix
    n_params = len(mi_result.param_names)

    # 1. Build adjacency
    adjacency = {i: set() for i in range(n_params)}
    for i in range(n_params):
        for j in range(i + 1, n_params):
            if mi[i, j] > mi_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # 2. BFS connected components
    visited = set()
    components = []
    for node in range(n_params):
        if node not in visited:
            component = _bfs(node, adjacency)
            visited.update(component)
            components.append(component)

    # 3. Validate: remove params where max MI to others in group < threshold
    validated = []
    for comp in components:
        kept = set()
        for p in comp:
            others = comp - {p}
            if others:
                max_mi = max(mi[p, q] for q in others)
                if max_mi > mi_threshold:
                    kept.add(p)
        validated.append(kept)

    # 4. Cap group sizes
    capped = []
    for comp in validated:
        if len(comp) > max_group_size:
            # Keep params with highest total MI within the group
            total_mi = {
                p: sum(mi[p, q] for q in comp if q != p) for p in comp
            }
            sorted_params = sorted(total_mi, key=total_mi.get, reverse=True)
            comp = set(sorted_params[:max_group_size])
        capped.append(comp)

    # 5. Filter small groups
    filtered = [comp for comp in capped if len(comp) >= min_group_size]

    # 6. Build DegeneracyGroup objects
    groups = []
    for comp in filtered:
        indices = sorted(comp)
        names = [mi_result.param_names[i] for i in indices]

        # Constraint counting
        n_constraints = 0
        if samples is not None:
            group_samples = samples[:, indices]
            intrinsic_dim = local_pca_intrinsic_dim(group_samples)
            n_constraints = len(indices) - intrinsic_dim

        # Average MI: mean of all unique pairwise MI values in the group
        mi_values = []
        for i_idx, i in enumerate(indices):
            for j in indices[i_idx + 1:]:
                mi_values.append(mi[i, j])
        avg_mi = float(np.mean(mi_values)) if mi_values else 0.0

        groups.append(DegeneracyGroup(
            param_names=names,
            param_indices=indices,
            n_constraints=n_constraints,
            avg_mi=avg_mi,
        ))

    return GroupingResult(groups=groups, mi_result=mi_result)
