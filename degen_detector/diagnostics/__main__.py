# ABOUTME: CLI entry point for running diagnostics via python -m degen_detector.diagnostics.
# ABOUTME: Delegates to runner.main() for backward compatibility.

import sys
from degen_detector.diagnostics.runner import main

sys.exit(main() or 0)
