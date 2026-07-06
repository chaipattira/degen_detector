# ABOUTME: Coordinate-transform wrapper for DegenDetector (DegenLogMode).
# ABOUTME: Applies per-parameter transforms (e.g., log), runs the pipeline, reports in original coords.

"""Coordinate-transform mode for degeneracy detection.

Useful for multiplicative degeneracies like Omega_m * h^2 ≈ const, which
become additive in log space: log(Omega_m) + 2*log(h) = const, and are
therefore trivially separable by the existing implicit fitter.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import sympy

from degen_detector.core import CouplingFit, CouplingSearchResult, DegenDetector
from degen_detector.implicit_fit import ImplicitFit


@dataclass
class ParameterTransform:
    """Specification of a coordinate transform for one parameter.

    Attributes
    ----------
    apply : callable
        Numpy function to apply to sample arrays, e.g. np.log.
    sympy_fn : callable
        Sympy function used for back-substitution into original coords,
        e.g. sympy.log.
    name_prefix : str
        Prefix prepended to the parameter name in transformed space,
        e.g. "log_" turns "Omega_m" into "log_Omega_m".
    """
    apply: Callable[[np.ndarray], np.ndarray]
    sympy_fn: Callable
    name_prefix: str


#: Built-in log transform. Use for multiplicative degeneracies like Omega_m * h^2 = const.
LOG_TRANSFORM = ParameterTransform(
    apply=np.log,
    sympy_fn=sympy.log,
    name_prefix="log_",
)


def _transformed_name(orig_name: str, t: ParameterTransform) -> str:
    return t.name_prefix + orig_name


def _is_already_log(name: str) -> bool:
    """Return True if the parameter name suggests it is already log-transformed.

    Matches names starting with ``log`` or ``ln`` (case-insensitive), with an
    optional separator (underscore, dash, space) or immediately followed by an
    uppercase/digit. Examples: ``logA``, ``log_A``, ``lnAs``, ``LOG_SIGMA8``.
    """
    low = name.lower()
    return low.startswith("log") or low.startswith("ln")


def _back_transform_fit(
    fit: ImplicitFit,
    transform_map: Dict[str, ParameterTransform],
) -> ImplicitFit:
    """Substitute transformed symbols back to original parameterization.

    Each component expression uses sympy.Symbol(transformed_name), e.g.
    Symbol("log_Omega_m"). This function substitutes those symbols back:
    Symbol("log_Omega_m") → log(Symbol("Omega_m")), so the returned fit
    is expressed in the original parameter space.

    Parameters
    ----------
    fit : ImplicitFit
        Fit produced by running the pipeline on transformed samples.
        ``fit.param_names`` are the transformed names.
    transform_map : dict
        Maps original parameter name → ParameterTransform. Only parameters
        present here were transformed; others are passed through unchanged.

    Returns
    -------
    ImplicitFit
        Same fit with expressions back-substituted into original coordinates.
        ``evaluate(original_samples)`` works correctly on the result.
    """
    new_exprs = []
    new_param_names = []

    for expr, tname in zip(fit.component_exprs, fit.param_names):
        orig_name = None
        transform = None
        for oname, t in transform_map.items():
            if tname == _transformed_name(oname, t):
                orig_name = oname
                transform = t
                break

        if orig_name is not None:
            # e.g. Symbol("log_Omega_m") → log(Symbol("Omega_m"))
            transformed_sym = sympy.Symbol(tname)
            orig_sym = sympy.Symbol(orig_name)
            new_expr = expr.subs(transformed_sym, transform.sympy_fn(orig_sym))
            new_exprs.append(new_expr)
            new_param_names.append(orig_name)
        else:
            new_exprs.append(expr)
            new_param_names.append(tname)

    # Rebuild equation string
    terms = []
    for expr in new_exprs:
        term_str = str(expr)
        terms.append(f"({term_str})" if term_str.startswith("-") else term_str)
    equation_str = " + ".join(terms) + f" = {fit.constant:.4f}"

    return ImplicitFit(
        component_exprs=new_exprs,
        constant=fit.constant,
        param_names=new_param_names,
        residual_std=fit.residual_std,
        orthogonal_r2=fit.orthogonal_r2,
        equation_str=equation_str,
        complexity=fit.complexity,
        aic=fit.aic,
        bic=fit.bic,
    )


def _back_transform_coupling_fit(
    coupling_fit: CouplingFit,
    transform_map: Dict[str, ParameterTransform],
) -> CouplingFit:
    """Back-transform all ImplicitFit objects in a CouplingFit."""
    new_fits = [_back_transform_fit(f, transform_map) for f in coupling_fit.fits]

    # Recover original param names for the coupling tuple
    new_pnames = []
    for tname in coupling_fit.param_names:
        orig = tname
        for oname, t in transform_map.items():
            if tname == _transformed_name(oname, t):
                orig = oname
                break
        new_pnames.append(orig)

    return CouplingFit(
        param_names=new_pnames,
        param_indices=coupling_fit.param_indices,
        mi_score=coupling_fit.mi_score,
        fits=new_fits,
        fit_order=coupling_fit.fit_order,
    )


class DegenLogMode:
    """Degeneracy detection in coordinate-transformed parameter space.

    Applies per-parameter coordinate transforms (default: log), runs the
    full MI + symbolic-regression pipeline on the transformed samples, then
    reports all results back in the original parameterization.

    Typical use: multiplicative degeneracies like Omega_m * h^2 ≈ const
    become linear in log space (log(Omega_m) + 2*log(h) = const) and are
    trivially recoverable by the separable-implicit fitter.

    The returned ``ImplicitFit`` objects are expressed in original coordinates:
    ``evaluate(original_samples)`` works correctly, and ``equation_str``
    contains e.g. ``log(Omega_m) + 2*log(h) = 1.5000``.

    The ``orthogonal_r2`` and ``residual_std`` values are computed in
    transformed space (i.e., how tightly the log-space constraint is satisfied).

    Parameters
    ----------
    samples : array-like, shape (n_samples, n_params)
        Posterior samples in original coordinates.
    param_names : list[str]
        Names of each parameter column.
    transforms : dict[str, ParameterTransform] | None
        Per-parameter transforms. Keys are original parameter names.
        If None, ``LOG_TRANSFORM`` is applied to every parameter.
        Parameters not in this dict are left untransformed.

    Examples
    --------
    >>> detector = DegenLogMode(
    ...     samples,
    ...     ["Omega_m", "h", "sigma8"],
    ...     transforms={"Omega_m": LOG_TRANSFORM, "h": LOG_TRANSFORM},
    ... )
    >>> ranking = detector.rank_couplings(coupling_depth=2)
    >>> result = detector.fit_couplings(ranking)
    >>> for cf in result.fits:
    ...     if cf.fit:
    ...         print(cf.fit.equation_str)  # e.g. "log(Omega_m) + 2*log(h) = 1.5000"
    """

    def __init__(
        self,
        samples,
        param_names: list,
        transforms: Optional[Dict[str, ParameterTransform]] = None,
    ):
        self.samples = np.asarray(samples)
        self.param_names = list(param_names)

        if transforms is None:
            skipped = [n for n in param_names if _is_already_log(n)]
            if skipped:
                print(
                    f"  DegenLogMode: skipping log transform for already-log parameters: {skipped}"
                )
            self._transform_map = {
                name: LOG_TRANSFORM
                for name in param_names
                if not _is_already_log(name)
            }
        else:
            self._transform_map = dict(transforms)

        self._transformed_samples, self._transformed_names = self._apply_transforms()

    def _apply_transforms(self):
        """Apply all transforms to produce the transformed data matrix.

        Samples where any transformed column is non-finite (e.g., log of zero
        or negative) are silently dropped. A warning is printed if any rows are
        removed. This is preferable to skipping the transform for an entire
        parameter when only a handful of boundary samples are problematic.
        """
        # Apply all transforms up-front so we can identify bad rows in one pass.
        transformed_cols = []
        transformed_names = []

        for j, name in enumerate(self.param_names):
            col = self.samples[:, j]
            if name in self._transform_map:
                t = self._transform_map[name]
                transformed_cols.append(t.apply(col))
                transformed_names.append(_transformed_name(name, t))
            else:
                transformed_cols.append(col.copy())
                transformed_names.append(name)

        # Build valid-row mask: keep rows where every column is finite.
        valid = np.ones(len(self.samples), dtype=bool)
        for col_t in transformed_cols:
            valid &= np.isfinite(col_t)

        n_dropped = int(np.sum(~valid))
        if n_dropped > 0:
            print(
                f"  DegenLogMode: dropping {n_dropped}/{len(self.samples)} samples "
                f"with non-finite values after transform (zeros or negatives)."
            )
            if not np.any(valid):
                # Identify which parameters caused every row to be dropped
                bad_params = []
                for j, (col_t, name) in enumerate(zip(transformed_cols, transformed_names)):
                    if not np.all(np.isfinite(col_t)):
                        n_bad = int(np.sum(~np.isfinite(col_t)))
                        bad_params.append(f"{name} ({n_bad} non-finite)")
                raise ValueError(
                    f"DegenLogMode: all {len(self.samples)} samples dropped after transform — "
                    f"no valid rows remain.\n"
                    f"Parameters with non-finite values after transform: {bad_params}\n"
                    f"Likely cause: log-transforming parameters with non-positive values. "
                    f"Pass explicit `transforms={{}}` to skip log for those parameters, "
                    f"or check that column indices are correct."
                )
            self.samples = self.samples[valid]
            transformed_cols = [col[valid] for col in transformed_cols]

        return np.column_stack(transformed_cols), transformed_names

    def _translate_params(self, params):
        """Translate original parameter names to their transformed counterparts.

        Allows callers to specify ``params`` using original names (e.g.,
        ``["Omega_m", "h"]``) even though the internal detector operates on
        transformed names (``["log_Omega_m", "log_h"]``).

        Parameters
        ----------
        params : list[str] | int | None
            As accepted by ``rank_couplings()``.
            - list[str]: each name is mapped to its transformed name if it was
              transformed, or left unchanged otherwise.
            - int or None: returned as-is.

        Returns
        -------
        list[str] | int | None
            Translated params ready to pass to the internal detector.
        """
        if not isinstance(params, list):
            return params
        translated = []
        for name in params:
            if name in self._transform_map:
                translated.append(_transformed_name(name, self._transform_map[name]))
            else:
                translated.append(name)
        return translated

    def rank_couplings(
        self,
        params=None,
        coupling_depth: int = 2,
        verbose: bool = True,
    ):
        """Rank parameter tuples by MI in transformed space — fast screening stage.

        The ``params`` argument accepts original parameter names (e.g.,
        ``["Omega_m", "h"]``) — they are translated to transformed names
        automatically. Returns a RankingResult whose tuples reference the
        transformed parameter names.
        """
        from degen_detector.core import rank_couplings as _rank_couplings
        translated_params = self._translate_params(params)
        return _rank_couplings(
            self._transformed_samples,
            self._transformed_names,
            coupling_depth=coupling_depth,
            params=translated_params,
            verbose=verbose,
        )

    def fit_couplings(self, ranked_tuples, **kwargs) -> CouplingSearchResult:
        """Fit symbolic equations in transformed space, report in original coordinates.

        Takes the output of rank_couplings and runs symbolic fitting. All
        ImplicitFit expressions are back-substituted into original parameter
        coordinates before returning.

        Keyword arguments are forwarded to the module-level fit_couplings.
        """
        from degen_detector.core import fit_couplings as _fit_couplings
        kwargs.setdefault("log_mode", True)
        result = _fit_couplings(self._transformed_samples, ranked_tuples, **kwargs)

        new_fits = [
            _back_transform_coupling_fit(cf, self._transform_map)
            for cf in result.fits
        ]

        return CouplingSearchResult(
            fits=new_fits,
            n_fits_attempted=result.n_fits_attempted,
            n_tuples_total=result.n_tuples_total,
            mi_result=result.mi_result,
            selected_params=result.selected_params,
        )
