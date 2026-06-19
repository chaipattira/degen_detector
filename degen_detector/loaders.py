"""Sample loaders for degen_detector pipeline.

Each loader returns (samples, param_names) where samples is an ndarray of
shape (n_samples, n_params) and param_names is a list of strings.
"""

import warnings
from pathlib import Path

import numpy as np


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
