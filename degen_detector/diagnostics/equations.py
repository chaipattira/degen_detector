# ABOUTME: Equation formatting and text output for degeneracy detection results.
# ABOUTME: Converts ImplicitFit objects to human-readable equation strings and text files.


def _make_form_string(fit):
    """Return equation string with numeric constants replaced by c_1, c_2, ...

    Integer values (e.g. exponents like 2 in x**2) are kept as-is; only
    floating-point literals are replaced.  The same numeric value always gets
    the same c_i label.
    """
    import re

    # Use the pre-built equation_str so we don't need to re-assemble it.
    eq_str = fit.equation_str

    seen = {}
    counter = [1]

    def _replace(m):
        num = m.group(0)
        if num not in seen:
            seen[num] = f"c_{counter[0]}"
            counter[0] += 1
        return seen[num]

    # Match decimal numbers like 1.23, 1.23e-4, .23 – but NOT bare integers.
    form_str = re.sub(r"\d*\.\d+(?:[eE][+-]?\d+)?", _replace, eq_str)
    return form_str


def format_all_equations(coupling_search_result, ground_truth=None):
    """Format all fitted equations as text.

    Parameters
    ----------
    coupling_search_result : CouplingSearchResult
        Result from DegenDetector.search_couplings().
    ground_truth : dict, optional
        Ground truth info with 'equation' and 'component_functions' keys.

    Returns
    -------
    text : str
        Formatted text with all equations.
    """
    lines = []
    lines.append("DEGENERACY DETECTION RESULTS")

    # Ground truth (if available)
    if ground_truth:
        lines.append("\nGround truth:")
        if 'equation' in ground_truth:
            lines.append(f"  {ground_truth['equation']}")
        if 'component_functions' in ground_truth:
            lines.append("  Component functions:")
            for comp in ground_truth['component_functions']:
                lines.append(f"    {comp}")
        lines.append("")

    # All fitted equations
    lines.append("\nFitted equations (ranked by MI score):")

    n_valid = 0
    for i, cf in enumerate(coupling_search_result.fits):
        rank = i + 1
        params_str = ", ".join(cf.param_names)

        lines.append(f"\n[{rank}] Parameters: ({params_str})")
        lines.append(f"    MI score: {cf.mi_score:.4f}")

        if not cf.fits:
            lines.append("    Fit: FAILED")
            continue

        n_valid += 1

        # Count distinct functional forms among candidates for the header note
        from degen_detector.implicit_fit import _rank_by_consensus
        _, consensus_count, n_forms = _rank_by_consensus(cf.fits)

        lines.append(
            f"    Candidates ({len(cf.fits)} total, "
            f"ranked by functional form consensus then R\u00b2_ortho; "
            f"top form: {consensus_count}/{len(cf.fits)} agree, {n_forms} distinct form(s)):"
        )
        lines.append(f"    Top form: {_make_form_string(cf.fits[0])}")
        for k, fit in enumerate(cf.fits):
            lines.append(f"\n    [{k+1}] {fit.equation_str}")
            lines.append(f"        R\u00b2_ortho:    {fit.orthogonal_r2:.4f}")
            lines.append(f"        Residual std: {fit.residual_std:.6f}")
            lines.append(f"        Complexity:   {fit.complexity}")
            lines.append("        Components:")
            for j, (expr, pname) in enumerate(zip(fit.component_exprs, fit.param_names)):
                lines.append(f"          g{j+1}({pname}) = {expr}")

    lines.append(f"Total: {len(coupling_search_result.fits)} fits attempted, {n_valid} successful")
    lines.append("-" * 80)

    return "\n".join(lines)


def save_equations(coupling_search_result, output_path, ground_truth=None):
    """Save all equations to a text file.

    Parameters
    ----------
    coupling_search_result : CouplingSearchResult
        Result from DegenDetector.search_couplings().
    output_path : Path or str
        Path to save the text file.
    ground_truth : dict, optional
        Ground truth info.
    """
    text = format_all_equations(coupling_search_result, ground_truth)
    with open(output_path, 'w') as f:
        f.write(text)
