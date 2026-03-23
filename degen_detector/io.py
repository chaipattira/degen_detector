# ABOUTME: Shared I/O utilities for saving/loading results and managing output directories.
# ABOUTME: Provides pickle serialization (with dill fallback) and timestamped directory creation.

"""Shared I/O utilities for degen_detector."""

from datetime import datetime
from pathlib import Path

try:
    import dill as pickle
except ImportError:
    import pickle


def save_pickle(data, path):
    """Save data to a pickle file.

    Uses dill if available (handles sympy lambdas), falls back to stdlib pickle.

    Parameters
    ----------
    data : object
        Data to serialize.
    path : Path or str
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    """Load data from a pickle file.

    Uses dill if available, falls back to stdlib pickle.

    Parameters
    ----------
    path : Path or str
        Input file path.

    Returns
    -------
    data : object
        Deserialized data.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def create_output_dir(base_dir):
    """Create a timestamped output directory under base_dir.

    Creates ``base_dir/YYYYMMDD_HHMMSS/`` and returns the path.

    Parameters
    ----------
    base_dir : Path or str
        Parent directory for timestamped output.

    Returns
    -------
    output_dir : Path
        The created timestamped directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
