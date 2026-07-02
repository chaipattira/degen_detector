"""run_detector: point at samples, get diagnostics."""

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from degen_detector.core import DegenDetector
from degen_detector.diagnostics.runner import DiagnosticsRunner
from degen_detector.io import save_pickle
from degen_detector.transforms import DegenLogMode


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


@contextmanager
def _tee_stdout(path):
    f = open(path, "w")
    old = sys.stdout
    sys.stdout = _Tee(old, f)
    try:
        yield
    finally:
        sys.stdout = old
        f.close()


def run_detector(
    samples,
    param_names,
    output_dir,
    *,
    log_mode=False,
    coupling_depth=2,
    max_fits=2,
    niterations=200,
    max_complexity=15,
    batch_size=1000,
):
    """Run the full degeneracy detection pipeline and save all outputs.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_params)
        Posterior samples.
    param_names : list[str]
        Names matching columns of samples.
    output_dir : path-like
        Directory to write outputs. Created if it does not exist.
        No timestamp subdirectory is added.
    log_mode : bool
        If True, use DegenLogMode (log-transforms all params not already
        named log_* / ln_*, drops rows with non-finite values after transform).
        If False, use DegenDetector.
    coupling_depth : int
        2 = search pairs, 3 = search triplets.
    max_fits : int
        Number of MI-ranked tuples to attempt fitting.
    niterations : int
        PySR iterations per component.
    max_complexity : int
        PySR complexity cap.
    batch_size : int
        PySR batch size.

    Returns
    -------
    CouplingSearchResult
    """
    samples = np.asarray(samples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with _tee_stdout(output_dir / "summary.txt"):
        return _run_detector_inner(
            samples, param_names, output_dir,
            log_mode=log_mode,
            coupling_depth=coupling_depth,
            max_fits=max_fits,
            niterations=niterations,
            max_complexity=max_complexity,
            batch_size=batch_size,
        )


def _run_detector_inner(
    samples,
    param_names,
    output_dir,
    *,
    log_mode,
    coupling_depth,
    max_fits,
    niterations,
    max_complexity,
    batch_size,
):
    # 1. Run detection
    if log_mode:
        detector = DegenLogMode(samples, param_names)
    else:
        detector = DegenDetector(samples, param_names)

    ranking = detector.rank_couplings(coupling_depth=coupling_depth)
    result = detector.fit_couplings(
        ranking,
        niterations=niterations,
        max_complexity=max_complexity,
        max_fits=max_fits,
        batch_size=batch_size,
    )

    # 2. Save pkl
    save_pickle(
        {"samples": samples, "param_names": param_names, "result": result},
        output_dir / "result.pkl",
    )

    # 3. Print summary (tee captures it to summary.txt)
    _write_summary(result)

    # 4. Run diagnostics (DiagnosticsRunner takes the pkl path, not the result object)
    try:
        runner = DiagnosticsRunner(output_dir / "result.pkl")
        runner.run(output_dir=output_dir / "diagnostics")
    except Exception as e:
        print(f"Warning: Diagnostics failed: {e}", file=sys.stderr)

    return result


def _write_summary(result):
    header = f"\n{'='*80}"
    col_header = f"{'Params':<30} {'MI':>8} {'BIC':>10} {'R²_ortho':>10}  Equation"
    sep = "=" * 80

    lines = [header, col_header, sep]
    for cf in result.fits:
        if cf.fit:
            row = (
                f"{str(cf.param_names):<30} {cf.mi_score:>8.4f} "
                f"{cf.fit.bic:>10.2f} {cf.fit.orthogonal_r2:>10.4f}  {cf.fit.equation_str}"
            )
        else:
            row = f"{str(cf.param_names):<30} {cf.mi_score:>8.4f} {'N/A':>10} {'N/A':>10}  (fit failed)"
        lines.append(row)
    lines.append(sep)

    print("\n".join(lines))
