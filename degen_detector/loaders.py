"""Sample loaders for degen_detector pipeline.

Each loader returns (samples, param_names) where samples is an ndarray of
shape (n_samples, n_params) and param_names is a list of strings.
"""

import warnings
from pathlib import Path

import numpy as np


def load_posterior(path, params=None, **kwargs):
    """Load posterior samples from any supported file format.

    The format is auto-detected from the file extension and/or content.
    Supported formats: ArviZ NetCDF (.nc), CSV, NumPy (.npy/.npz),
    emcee HDF5 (.h5/.hdf5), and GetDist/CosmoMC chain stems.

    Parameters
    ----------
    path : path-like
        File path or GetDist chain stem.
    params : list[str] or None
        Parameter names to extract. Required for numpy and getdist formats.
        Optional for all others (None means load all).
    **kwargs
        Format-specific options passed through to the underlying loader:
        - emcee: burn_in (int), thin (int)
        - getdist: ignore_rows (float)
        - arviz: group (str, default "posterior")

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
    param_names : list[str]

    Examples
    --------
    >>> samples, names = load_posterior("trace.nc")
    >>> samples, names = load_posterior("posterior.csv", params=["alpha", "beta"])
    >>> samples, names = load_posterior("chains.h5", burn_in=200, thin=5)
    >>> samples, names = load_posterior("data/planck/base_plik", params=["omegam", "H0"])
    """
    fmt = detect_format(path)

    if fmt == "numpy":
        if params is None:
            raise ValueError(
                "params (column names) is required for numpy files — "
                "there is no metadata to infer names from."
            )
        return load_numpy(path, params)
    elif fmt == "arviz":
        return load_arviz(path, params=params, **kwargs)
    elif fmt == "csv":
        return load_csv(path, params=params)
    elif fmt == "emcee":
        return load_emcee(path, params=params, **kwargs)
    else:  # getdist
        if params is None:
            raise ValueError(
                "params is required for GetDist chains — "
                "pass the parameter names you want to extract."
            )
        return load_getdist(path, params, **kwargs)


def detect_format(path):
    """Auto-detect the posterior sample format from a file path.

    Parameters
    ----------
    path : path-like
        File path or getdist chain stem.

    Returns
    -------
    str
        One of: 'numpy', 'csv', 'arviz', 'emcee', 'getdist'

    Raises
    ------
    ValueError
        If the format cannot be determined.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".npy", ".npz"):
        return "numpy"
    if suffix == ".csv":
        return "csv"
    if suffix in (".nc", ".netcdf"):
        return "arviz"
    if suffix in (".h5", ".hdf5"):
        return _detect_hdf5(path)
    if _is_getdist_stem(path):
        return "getdist"

    raise ValueError(
        f"Cannot auto-detect format for '{path}'. "
        f"Pass --format explicitly: getdist, emcee, numpy, arviz, or csv."
    )


def _is_getdist_stem(path):
    return (
        Path(str(path) + ".paramnames").exists()
        or Path(str(path) + "_1.txt").exists()
    )


def _detect_hdf5(path):
    try:
        import h5py
        with h5py.File(str(path), "r") as f:
            if "mcmc" in f and "chain" in f["mcmc"]:
                return "emcee"
            if "posterior" in f:
                return "arviz"
    except Exception:
        pass
    raise ValueError(
        f"Cannot determine format of HDF5 file '{path}'. "
        f"Pass --format emcee or --format arviz explicitly."
    )


def load_numpy(array_or_path, param_names):
    """Load samples from a numpy array or .npy/.npz file.

    Parameters
    ----------
    array_or_path : ndarray or path-like
        Array directly, or path to a .npy or .npz file.
        For .npz files, loads the array named "samples" if present,
        otherwise the first array (with a UserWarning).
    param_names : list[str]
        Names for each column. Required — no metadata to infer from.

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
    param_names : list[str]
    """
    if isinstance(array_or_path, np.ndarray):
        samples = array_or_path
    else:
        path = Path(array_or_path)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if path.suffix == ".npz":
            data = np.load(path)
            if "samples" in data:
                samples = data["samples"]
            else:
                keys = list(data.keys())
                warnings.warn(
                    f"No 'samples' key in {path.name}; loading first array '{keys[0]}'. "
                    f"Available keys: {keys}",
                    UserWarning,
                    stacklevel=2,
                )
                samples = data[keys[0]]
        else:
            samples = np.load(path)

    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array, got shape {samples.shape}. "
            f"Samples must have shape (n_samples, n_params)."
        )
    if samples.shape[1] != len(param_names):
        raise ValueError(
            f"Shape mismatch: array has {samples.shape[1]} columns but "
            f"param_names has {len(param_names)} entries: {param_names}"
        )
    return samples, list(param_names)


def load_getdist(chain_root, params, ignore_rows=0.3):
    """Load samples from a getdist/CosmoMC chain.

    Parameters
    ----------
    chain_root : str or path-like
        Path stem for the chain files (without _1.txt, _2.txt, etc.).
        Same convention used by getdist.loadMCSamples.
    params : list[str]
        Parameter names to extract. Required — no default.
    ignore_rows : float
        Fraction of chain to discard as burn-in (default 0.3).

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
    param_names : list[str]
    """
    try:
        from getdist import loadMCSamples
    except ImportError as exc:
        raise ImportError(
            "getdist is required for load_getdist. "
            "Install it with: pip install getdist"
        ) from exc

    mc = loadMCSamples(str(chain_root), settings={"ignore_rows": ignore_rows})
    p = mc.getParams()

    cols, available = [], []
    for name in params:
        arr = getattr(p, name, None)
        if arr is None:
            warnings.warn(
                f"Parameter '{name}' not found in chain at {chain_root}; skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        cols.append(arr)
        available.append(name)

    if not available:
        raise ValueError(
            f"No params could be loaded from {chain_root}. "
            f"Requested: {params}. Check the chain's .paramnames file."
        )

    return np.column_stack(cols), available


def load_arviz(idata_or_path, params=None, group="posterior"):
    """Load samples from an ArviZ InferenceData object or NetCDF file.

    Covers PyMC, NumPyro, Stan (via arviz), Bambi, Blackjax, and any sampler
    that writes ArviZ-compatible output.

    Parameters
    ----------
    idata_or_path : InferenceData or path-like
        An in-memory arviz.InferenceData object, or path to a .nc/.netcdf file.
    params : list[str] or None
        Parameter names to extract. If None, uses all scalar variables in the
        group (variables with no extra dimensions beyond chain/draw).
    group : str
        Which group to read (default: "posterior").

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
    param_names : list[str]
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "arviz is required for load_arviz. "
            "Install it with: pip install arviz"
        ) from exc

    if not hasattr(idata_or_path, "posterior"):
        idata = az.from_netcdf(str(idata_or_path))
    else:
        idata = idata_or_path

    ds = getattr(idata, group)

    if params is None:
        # Keep only scalar variables (no dimensions beyond chain/draw)
        params = [
            v for v in ds.data_vars
            if ds[v].dims == ("chain", "draw")
        ]
        if not params:
            raise ValueError(
                f"No scalar variables found in group '{group}'. "
                f"Available variables: {list(ds.data_vars)}. "
                f"Pass params=[...] to select vector-valued variables by name."
            )

    cols, available = [], []
    for name in params:
        if name not in ds:
            warnings.warn(
                f"Parameter '{name}' not found in group '{group}'; skipping. "
                f"Available: {list(ds.data_vars)}",
                UserWarning,
                stacklevel=2,
            )
            continue
        arr = ds[name].values  # shape: (chain, draw) or (chain, draw, ...)
        if arr.ndim == 2:
            cols.append(arr.reshape(-1))
            available.append(name)
        else:
            # Vector param: flatten each element to its own column
            n_chains, n_draws = arr.shape[:2]
            flat = arr.reshape(n_chains * n_draws, -1)
            for i in range(flat.shape[1]):
                cols.append(flat[:, i])
                available.append(f"{name}[{i}]")

    if not available:
        raise ValueError(f"No params could be loaded from group '{group}'.")

    return np.column_stack(cols), available


def load_weighted(samples, weights, n_samples=None, seed=None):
    """Resample importance-weighted samples to produce equal-weight output.

    Use this after loading chains from nested samplers (PolyChord, MultiNest,
    dynesty) which output samples with unequal importance weights.

    Parameters
    ----------
    samples : ndarray, shape (n, d)
    weights : array-like, shape (n,)
        Importance weights (need not be normalised; negative values are clipped).
    n_samples : int or None
        Number of equal-weight samples to draw. Defaults to the effective
        sample size floor(1 / sum(w²)).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (n_samples, d)
    """
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    w /= w.sum()
    eff_n = int(1.0 / np.sum(w ** 2))
    n_draw = min(n_samples, eff_n) if n_samples is not None else eff_n
    idx = np.random.default_rng(seed).choice(len(samples), size=n_draw, replace=True, p=w)
    return samples[idx]


def load_csv(path, params=None):
    """Load samples from a CSV file.

    Parameters
    ----------
    path : path-like
        Path to a CSV file. The first row must be a header with column names.
    params : list[str] or None
        Subset of column names to extract. If None, uses all columns.

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
    param_names : list[str]
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for load_csv. "
            "Install it with: pip install pandas"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    df = pd.read_csv(path)

    if params is not None:
        missing = [p for p in params if p not in df.columns]
        if missing:
            raise ValueError(
                f"Columns not found in {path.name}: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        df = df[params]

    return df.to_numpy(dtype=float), list(df.columns)


def load_emcee(h5_path, params=None, burn_in=0, thin=1):
    """Load samples from an emcee HDFBackend HDF5 file.

    Parameters
    ----------
    h5_path : path-like
        Path to the emcee HDFBackend HDF5 file.
    params : list[str] or None
        Parameter names. If None, tries to read from the file's
        "param_names" attribute; falls back to theta_0, theta_1, …
        with a UserWarning if no labels are stored.
    burn_in : int
        Number of steps to discard as burn-in before flattening.
    thin : int
        Keep every `thin`-th step before flattening.

    Returns
    -------
    samples : ndarray, shape (n_samples, n_params)
    param_names : list[str]
    """
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "emcee is required for load_emcee. "
            "Install it with: pip install emcee"
        ) from exc

    path = Path(h5_path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    backend = emcee.backends.HDFBackend(str(path), read_only=True)
    chain = backend.get_chain(discard=burn_in, thin=thin, flat=True)
    # chain shape: (n_samples, n_dim)

    n_dim = chain.shape[1]

    if params is not None:
        if len(params) != n_dim:
            raise ValueError(
                f"params has {len(params)} entries but chain has {n_dim} dimensions."
            )
        param_names = list(params)
    else:
        # Try to read labels stored as a file attribute
        try:
            import h5py
            with h5py.File(str(path), "r") as f:
                stored = f.attrs.get("param_names", None)
            if stored is not None:
                param_names = list(stored)
            else:
                raise KeyError("no labels")
        except Exception:
            param_names = [f"theta_{i}" for i in range(n_dim)]
            warnings.warn(
                f"No parameter labels found in {path.name}; "
                f"using auto-generated names: {param_names}. "
                f"Pass params=[...] explicitly to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )

    return chain, param_names
