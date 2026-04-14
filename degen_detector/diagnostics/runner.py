# ABOUTME: Orchestrates diagnostic analysis on pickle result files.
# ABOUTME: DiagnosticsRunner auto-detects result format and generates all plots and equations.

from pathlib import Path

from degen_detector.diagnostics.analyzer import FitAnalyzer
from degen_detector.diagnostics.plots import (
    plot_corner, plot_components, plot_true_vs_predicted, plot_residuals,
    plot_manifold_2d, plot_manifold_3d, plot_projections_3d, plot_mi_matrix,
)
from degen_detector.diagnostics.equations import save_equations
from degen_detector.io import load_pickle


class DiagnosticsRunner:
    """Run diagnostics on pkl files containing fit results.

    Auto-detects pkl structure and generates appropriate plots.

    Parameters
    ----------
    pkl_path : Path or str
        Path to pkl file or directory containing pkl files.

    Examples
    --------
    >>> runner = DiagnosticsRunner("outputs/results.pkl")
    >>> runner.run(output_dir="outputs/plots")
    """

    def __init__(self, pkl_path):
        self.pkl_path = Path(pkl_path)
        self.data = None
        self._load()

    def _load(self):
        """Load and parse the pkl file."""
        if self.pkl_path.is_dir():
            # Load all pkl files in directory
            pkl_files = list(self.pkl_path.glob("*.pkl"))
            if not pkl_files:
                raise FileNotFoundError(f"No pkl files in {self.pkl_path}")
            self.data = {}
            for pf in pkl_files:
                self.data[pf.stem] = load_pickle(pf)
        else:
            self.data = load_pickle(self.pkl_path)

    def _extract_fit_data(self, data):
        """Extract samples, param_names, result, ground_truth from various formats.

        Returns
        -------
        dict with keys: samples, param_names, result, ground_truth (optional), name
        """
        # Format 1: Direct dict with samples, result, etc.
        if isinstance(data, dict) and 'samples' in data and 'result' in data:
            return {
                'samples': data['samples'],
                'param_names': data['param_names'],
                'result': data['result'],
                'ground_truth': data.get('ground_truth'),
                'name': data.get('name', 'unnamed'),
            }

        # Format 2: Just a CouplingSearchResult
        from degen_detector.core import CouplingSearchResult
        if isinstance(data, CouplingSearchResult):
            return {
                'samples': None,
                'param_names': data.selected_params,
                'result': data,
                'ground_truth': None,
                'name': 'result',
            }

        # Format 3: Dict of experiments
        if isinstance(data, dict):
            # Check if any value looks like experiment data
            for key, val in data.items():
                if isinstance(val, dict) and 'samples' in val:
                    # This is a multi-experiment dict, return as-is
                    return None  # Signal to iterate over dict

        raise ValueError(f"Unknown pkl format: {type(data)}")

    def run(self, output_dir=None):
        """Run diagnostics and generate plots.

        Generates one subdirectory per valid coupling fit (ranked by MI).
        Each subdirectory contains component, residual, and manifold plots.

        Parameters
        ----------
        output_dir : Path or str, optional
            Directory to save plots. Defaults to pkl_path parent / 'diagnostics'.
        """
        import matplotlib.pyplot as plt

        if output_dir is None:
            if self.pkl_path.is_dir():
                output_dir = self.pkl_path / 'diagnostics'
            else:
                output_dir = self.pkl_path.parent / 'diagnostics'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Handle different data formats
        if isinstance(self.data, dict):
            # Check if it's a single experiment or multiple
            extracted = self._extract_fit_data(self.data)
            if extracted is not None:
                # Single experiment
                self._run_single(extracted, output_dir)
            else:
                # Multiple experiments
                for name, exp_data in self.data.items():
                    try:
                        extracted = self._extract_fit_data(exp_data)
                        if extracted is not None:
                            extracted['name'] = name
                            exp_dir = output_dir / name
                            exp_dir.mkdir(parents=True, exist_ok=True)
                            self._run_single(extracted, exp_dir)
                    except Exception as e:
                        print(f"Warning: Could not process {name}: {e}")
        else:
            extracted = self._extract_fit_data(self.data)
            self._run_single(extracted, output_dir)

        print(f"\nDiagnostics saved to: {output_dir}")

    def _run_single(self, data, output_dir):
        """Run diagnostics for a single experiment."""
        import matplotlib.pyplot as plt

        name = data['name']
        samples = data['samples']
        param_names = data['param_names']
        result = data['result']
        ground_truth = data.get('ground_truth')

        print(f"\nProcessing: {name}")

        valid_fits = [cf for cf in result.fits if cf.fit is not None]
        if not valid_fits:
            print(f"  Warning: No valid fit for {name}, skipping plots")
            return

        # 1. Save all equations
        print("  - Equations (all fits)")
        save_equations(result, output_dir / "equations.txt", ground_truth)

        # 2. MI matrix
        print("  - MI matrix")
        fig = plot_mi_matrix(result.mi_result)
        fig.savefig(output_dir / "mi_matrix.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        if samples is None:
            print("  - No samples available, skipping plots")
            return

        # 3. Corner plot (once, over all samples)
        print("  - Corner plot")
        fig = plot_corner(samples, param_names)
        fig.savefig(output_dir / "corner.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 4-7. Per-fit plots — one subdirectory per coupling tuple
        for cf in valid_fits:
            fit = cf.fit
            analyzer = FitAnalyzer(fit)
            fit_dir = output_dir / "_".join(cf.param_names)
            fit_dir.mkdir(parents=True, exist_ok=True)
            label = str(cf.param_names)
            print(f"  [{label}]")

            print(f"    - Component functions")
            fig = plot_components(analyzer, samples, param_names)
            fig.savefig(fit_dir / "components.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"    - True vs predicted")
            fig = plot_true_vs_predicted(analyzer, samples, param_names)
            fig.savefig(fit_dir / "true_vs_predicted.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"    - Residuals")
            fig = plot_residuals(analyzer, samples, param_names)
            fig.savefig(fit_dir / "residuals.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            n_fit_params = len(fit.param_names)
            if n_fit_params == 2:
                print(f"    - 2D manifold")
                fig = plot_manifold_2d(analyzer, samples, param_names)
                fig.savefig(fit_dir / "manifold_2d.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            elif n_fit_params == 3:
                print(f"    - 3D manifold")
                fig = plot_manifold_3d(analyzer, samples, param_names)
                fig.savefig(fit_dir / "manifold_3d.png", dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"    - 2D projections")
                fig = plot_projections_3d(analyzer, samples, param_names)
                fig.savefig(fit_dir / "projections_2d.png", dpi=150, bbox_inches='tight')
                plt.close(fig)

        print(f"  Done: {output_dir}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for degeneracy detection results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on a single pkl file
    python -m degen_detector.diagnostics results.pkl

    # Run on a directory of pkl files
    python -m degen_detector.diagnostics outputs/synthetic_15710222/20260315_091812

    # Specify output directory
    python -m degen_detector.diagnostics results.pkl -o my_plots/
        """
    )
    parser.add_argument(
        "pkl_path",
        type=Path,
        help="Path to pkl file or directory containing pkl files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for plots (default: <pkl_path>/diagnostics)",
    )

    args = parser.parse_args()

    if not args.pkl_path.exists():
        print(f"Error: {args.pkl_path} does not exist")
        return 1

    runner = DiagnosticsRunner(args.pkl_path)
    runner.run(output_dir=args.output)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
