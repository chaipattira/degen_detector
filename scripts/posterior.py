# ABOUTME: Generate power spectrum posterior samples with known degeneracies
# Cosmology forward model and MCMC helpers (adapted from TailedUniform repo)

import numpy as np
from typing import Tuple, Optional
import emcee
import symbolic_pofk.syren_new as syren_new

# Global constants for power spectrum computation
_L, _N = 1000, 64
_kf = 2 * np.pi / _L
_knyq = np.pi * _N / _L
_kedges = np.arange(0, _knyq, _kf)
_kcenters = (_kedges[:-1] + _kedges[1:]) / 2
_a = 1.0

# Parameter ranges for Om and h
_PARAM_RANGES = {
    'Om': (0.27, 0.37),
    'h': (0.63, 0.71)
}

def _forward_model_deterministic(theta: np.ndarray) -> np.ndarray:
    """
    Deterministic forward model: theta -> P_theory(k)
    Uses syren emulator to compute theoretical power spectrum.

    Parameters
    ----------
    theta : np.ndarray
        [Om, h] cosmological parameters

    Returns
    -------
    P_theory : np.ndarray
        Theoretical power spectrum (no noise)
    """
    Om, h = theta

    # Fixed cosmological parameters
    As = 2.105  # 10^9 A_s
    Ob = 0.02242 / h ** 2
    ns = 0.9665
    w0 = -1.0
    wa = 0.0
    mnu = 0.0

    pk_theory = syren_new.pnl_new_emulated(
        _kcenters, As, Om, Ob, h, ns, mnu, w0, wa, a=_a
    )
    return pk_theory


def _compute_cosmic_variance_std(P_theory: np.ndarray) -> np.ndarray:
    """
    Compute cosmic variance uncertainties for power spectrum.

    Parameters
    ----------
    P_theory : np.ndarray
        Theoretical power spectrum

    Returns
    -------
    std_mode : np.ndarray
        Standard deviation per k-bin
    """
    var_single = np.abs(P_theory) ** 2
    Nk = _L ** 3 * _kcenters ** 2 * _kf / (2 * np.pi ** 2)
    var_mode = var_single * 2 / Nk
    return np.sqrt(var_mode)


def _log_prior(theta: np.ndarray) -> float:
    """Log prior probability (truncated Normal)."""
    Om, h = theta
    Om_range = _PARAM_RANGES['Om']
    h_range = _PARAM_RANGES['h']

    # Hard bounds
    if Om < Om_range[0] or Om > Om_range[1]:
        return -np.inf
    if h < h_range[0] or h > h_range[1]:
        return -np.inf

    # Normal prior (sigma = 10% of range)
    Om_mean = (Om_range[0] + Om_range[1]) / 2
    h_mean = (h_range[0] + h_range[1]) / 2
    Om_std = 0.1 * (Om_range[1] - Om_range[0])
    h_std = 0.1 * (h_range[1] - h_range[0])

    log_prior_Om = -0.5 * ((Om - Om_mean) / Om_std) ** 2
    log_prior_h = -0.5 * ((h - h_mean) / h_std) ** 2

    return log_prior_Om + log_prior_h


def _log_likelihood(theta: np.ndarray, x_obs: np.ndarray) -> float:
    """Log likelihood with heteroskedastic cosmic variance noise."""
    P_theory = _forward_model_deterministic(theta)
    std_mode = _compute_cosmic_variance_std(P_theory)

    log_like = -0.5 * np.sum(((x_obs - P_theory) / std_mode) ** 2)
    log_like -= np.sum(np.log(std_mode))
    return log_like


def _log_probability(theta: np.ndarray, x_obs: np.ndarray) -> float:
    """Log posterior probability."""
    lp = _log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(theta, x_obs)


def _run_mcmc_inference(
    x_obs: np.ndarray,
    num_samples: int = 2000,
    warmup_steps: int = 500,
    num_chains: int = 4,
    verbose: bool = False
) -> np.ndarray:
    """
    Run MCMC inference using emcee sampler.

    Parameters
    ----------
    x_obs : np.ndarray
        Observed power spectrum
    num_samples : int
        Number of MCMC samples per chain
    warmup_steps : int
        Burn-in steps
    num_chains : int
        Number of parallel walkers
    verbose : bool
        Print progress

    Returns
    -------
    samples : np.ndarray
        Shape (num_samples * num_chains, 2) for [Om, h]
    """
    Om_range = _PARAM_RANGES['Om']
    h_range = _PARAM_RANGES['h']

    # Initialize walkers randomly within prior range
    pos = np.random.rand(num_chains, 2)
    pos[:, 0] = pos[:, 0] * (Om_range[1] - Om_range[0]) + Om_range[0]
    pos[:, 1] = pos[:, 1] * (h_range[1] - h_range[0]) + h_range[0]

    sampler = emcee.EnsembleSampler(num_chains, 2, _log_probability, args=(x_obs,))

    if verbose:
        print(f"Running MCMC: {num_samples} samples, {warmup_steps} warmup, {num_chains} walkers")

    # Warmup
    state = sampler.run_mcmc(pos, warmup_steps, progress=verbose)
    sampler.reset()

    # Production
    sampler.run_mcmc(state, num_samples, progress=verbose)

    return sampler.get_chain(flat=True)


def generate_cosmology_posterior(
    n_samples: int = 8000,
    theta_true: Optional[np.ndarray] = None,
    warmup_steps: int = 500,
    num_chains: int = 4,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Generate cosmology posterior samples using MCMC with syren emulator.

    Runs emcee MCMC to sample from the posterior P(Om, h | P(k)_obs) where
    P(k) is the matter power spectrum computed using the syren emulator.

    Parameters
    ----------
    n_samples : int
        Number of MCMC samples per chain (total samples = n_samples * num_chains)
    theta_true : np.ndarray, optional
        True parameters [Om, h]. If None, samples randomly from prior center.
    warmup_steps : int
        Number of burn-in steps for MCMC
    num_chains : int
        Number of parallel walkers for emcee
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print MCMC progress

    Returns
    -------
    samples : np.ndarray
        Shape (n_samples * num_chains, 2) for [Om, h]
    info : dict
        Contains true parameters, observation, and metadata
    """
    if seed is not None:
        np.random.seed(seed)

    # Default true parameters at center of prior
    if theta_true is None:
        Om_range = _PARAM_RANGES['Om']
        h_range = _PARAM_RANGES['h']
        theta_true = np.array([
            (Om_range[0] + Om_range[1]) / 2,
            (h_range[0] + h_range[1]) / 2
        ])

    theta_true = np.asarray(theta_true)

    # Generate observation: forward model + cosmic variance noise
    P_theory = _forward_model_deterministic(theta_true)
    std_noise = _compute_cosmic_variance_std(P_theory)
    x_obs = P_theory + std_noise * np.random.randn(*P_theory.shape)


    # Handle any NaN values
    mask = np.isnan(x_obs)
    if mask.any():
        x_obs[mask] = np.nanmean(x_obs)

    # Run MCMC
    samples_per_chain = n_samples // num_chains
    samples = _run_mcmc_inference(
        x_obs,
        num_samples=samples_per_chain,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        verbose=verbose
    )

    info = {
        'theta_true': theta_true,
        'param_names': ['Om', 'h'],
        'param_ranges': _PARAM_RANGES,
        'observation': x_obs,
        'n_samples': len(samples),
    }

    return samples, info
