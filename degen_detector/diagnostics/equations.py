# ABOUTME: Equation formatting and text output for degeneracy detection results.
# ABOUTME: Converts ImplicitFit objects to human-readable equation strings and text files.


def _simplified_equation_parts(fit):
    """Return (sym_str, combined_const) for the RHS-zero form.

    Sums all component_exprs symbolically, extracts the numeric constant, and
    also absorbs fit.constant so the equation reads:
        sym_str [± |combined_const|] = 0
    """
    import sympy

    total = sum(fit.component_exprs, sympy.Integer(0))
    free_syms = total.free_symbols

    if free_syms:
        const_part, var_part = total.as_independent(*free_syms, as_Add=True)
        combined_const = float(const_part) - fit.constant
    else:
        var_part = sympy.Integer(0)
        combined_const = float(total) - fit.constant

    sym_str = str(var_part)
    return ("" if sym_str == "0" else sym_str), combined_const


def _fmt_simplified_eq(sym_str, combined_const, prec=4):
    """Build 'sym_str [± |combined_const|] = 0.0000'."""
    if not sym_str:
        lhs = f"{combined_const:.{prec}f}"
    elif abs(combined_const) < 1e-12:
        lhs = sym_str
    elif combined_const < 0:
        lhs = f"{sym_str} - {abs(combined_const):.{prec}f}"
    else:
        lhs = f"{sym_str} + {combined_const:.{prec}f}"
    return f"{lhs} = 0.0000"


def _make_form_string(fit):
    """Return equation string with numeric constants replaced by c_1, c_2, ...

    Constants are first combined (including fit.constant) so the form reflects
    the simplified structure with RHS = 0. Integer values (e.g. exponents like
    2 in x**2) are kept as-is; only floating-point literals are replaced. The
    same numeric value always gets the same c_i label.
    """
    import re

    sym_str, combined_const = _simplified_equation_parts(fit)

    # Build the LHS only (regex operates here); RHS is always the literal "= 0.0000"
    if not sym_str:
        lhs = f"{combined_const:.10f}"
    elif abs(combined_const) < 1e-12:
        lhs = sym_str
    elif combined_const < 0:
        lhs = f"{sym_str} - {abs(combined_const):.10f}"
    else:
        lhs = f"{sym_str} + {combined_const:.10f}"

    seen = {}
    counter = [1]

    def _replace(m):
        num = m.group(0)
        if num not in seen:
            seen[num] = f"c_{counter[0]}"
            counter[0] += 1
        return seen[num]

    form_lhs = re.sub(r"\d*\.\d+(?:[eE][+-]?\d+)?", _replace, lhs)
    return f"{form_lhs} = 0.0000"


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

        from degen_detector.implicit_fit import _functional_form_key

        # Group fits by functional form, preserving order (consensus form first, then R\u00b2_ortho)
        form_groups = {}
        for fit in cf.fits:
            fk = tuple(
                _functional_form_key(expr, pname)
                for expr, pname in zip(fit.component_exprs, fit.param_names)
            )
            if fk not in form_groups:
                form_groups[fk] = []
            form_groups[fk].append(fit)

        n_forms = len(form_groups)
        lines.append(f"    Candidates ({len(cf.fits)} total, {n_forms} distinct form(s)):")

        for form_num, group in enumerate(form_groups.values(), 1):
            lines.append(f"\n    #{form_num} top form ({len(group)}/{len(cf.fits)} agree):")
            lines.append(f"      {_make_form_string(group[0])}")
            for k, fit in enumerate(group, 1):
                sym_str, combined_const = _simplified_equation_parts(fit)
                lines.append(f"\n      [{k}] {_fmt_simplified_eq(sym_str, combined_const, prec=4)}")
                lines.append(f"          R\u00b2_ortho:    {fit.orthogonal_r2:.4f}")
                lines.append(f"          Residual std: {fit.residual_std:.6f}")
                lines.append(f"          Complexity:   {fit.complexity}")
                lines.append("          Components:")
                for j, (expr, pname) in enumerate(zip(fit.component_exprs, fit.param_names)):
                    lines.append(f"            g{j+1}({pname}) = {expr}")

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
