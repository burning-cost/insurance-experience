"""Utility functions for experience rating calculations.

These are the building blocks used across all model tiers. They are also
exposed as public API for users who want to construct credibility factors
manually or integrate with non-standard workflows.
"""

from __future__ import annotations

import numpy as np

from ._types import ClaimsHistory


def credibility_factor(posterior_premium: float, prior_premium: float) -> float:
    """Compute the multiplicative credibility factor.

    The credibility factor CF satisfies:
        posterior_premium = prior_premium * CF

    Parameters
    ----------
    posterior_premium : float
        The experience-adjusted (a posteriori) premium.
    prior_premium : float
        The a priori (GLM-based) premium.

    Returns
    -------
    float
        Credibility factor. Values > 1 indicate the policy is worse than
        the GLM base rate; values < 1 indicate better-than-average experience.

    Raises
    ------
    ValueError
        If prior_premium is zero or negative.
    """
    if prior_premium <= 0.0:
        raise ValueError(f"prior_premium must be positive, got {prior_premium}")
    return posterior_premium / prior_premium


def posterior_premium(
    prior_premium: float, cf: float, calibration_factor: float = 1.0
) -> float:
    """Compute the posterior premium from a credibility factor.

    Parameters
    ----------
    prior_premium : float
        The a priori (GLM-based) premium.
    cf : float
        Credibility factor (posterior / prior ratio).
    calibration_factor : float
        Optional portfolio-level calibration factor from balance_calibrate().
        Default 1.0 (no calibration applied).

    Returns
    -------
    float
        Posterior premium after experience adjustment and optional calibration.
    """
    return prior_premium * cf * calibration_factor


def seniority_weights(
    n_periods: int,
    p: float,
    q: float,
    exposures: list[float] | None = None,
) -> np.ndarray:
    """Compute seniority (recency) weights for claims history periods.

    In the dynamic Poisson-gamma model, the effective weight on period t's
    claims decays geometrically with age. This function computes the
    relative weight assigned to each period in a t-period history when
    fitting or predicting.

    The weight assigned to period s (0-indexed from oldest) in a history
    of length t is proportional to (p*q)^(t-s). The most recent period
    receives weight 1; the oldest receives weight (p*q)^(t-1).

    Parameters
    ----------
    n_periods : int
        Number of observation periods.
    p : float
        Transition parameter controlling how strongly the current state
        reverts to mean. Must be in (0, 1].
    q : float
        Decay parameter controlling how quickly older claims are
        downweighted. Must be in (0, 1].
    exposures : list[float] or None
        Per-period exposure weights. If None, uniform exposure is assumed.

    Returns
    -------
    np.ndarray
        Array of shape (n_periods,) with seniority weights, normalised
        to sum to 1. Most recent period is last.

    Examples
    --------
    >>> seniority_weights(3, p=0.9, q=0.8)
    array([0.182..., 0.274..., 0.543...])  # most recent gets highest weight
    """
    if not (0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")
    if not (0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")

    decay = p * q
    # Weights for periods 0, 1, ..., n-1 (oldest to most recent)
    raw = np.array([decay ** (n_periods - 1 - s) for s in range(n_periods)])

    if exposures is not None:
        raw = raw * np.asarray(exposures, dtype=float)

    total = raw.sum()
    if total == 0.0:
        return np.ones(n_periods) / n_periods
    return raw / total


def exposure_weighted_mean(
    counts: list[int],
    exposures: list[float],
) -> float:
    """Compute exposure-weighted mean claim frequency.

    Parameters
    ----------
    counts : list[int]
        Claim counts per period.
    exposures : list[float]
        Exposure (e.g., years on risk) per period.

    Returns
    -------
    float
        Total claims divided by total exposure. Returns 0.0 if total
        exposure is zero.
    """
    total_exp = sum(exposures)
    if total_exp == 0.0:
        return 0.0
    return sum(counts) / total_exp


def history_sufficient_stat(
    history: ClaimsHistory,
    theta_ref: float | None = None,
) -> float:
    """Compute the log-likelihood sufficient statistic for a claims history.

    For Poisson models with a reference intensity theta_ref, the sufficient
    statistic is the log-likelihood of the observed claim counts:

        L(Y; theta) = sum_t [ Y_t * log(theta * e_t) - theta * e_t ]

    When theta_ref is None, uses the empirical frequency as reference.

    This is the input to the surrogate model's g(.) function.

    Parameters
    ----------
    history : ClaimsHistory
        The policy's claims history.
    theta_ref : float or None
        Reference intensity (claims per exposure unit). If None, uses the
        empirical claim frequency from the history.

    Returns
    -------
    float
        Log-likelihood sufficient statistic.
    """
    assert history.exposures is not None
    if theta_ref is None:
        theta_ref = history.claim_frequency
        if theta_ref == 0.0:
            theta_ref = 1e-6  # avoid log(0)

    log_lik = 0.0
    for y_t, e_t in zip(history.claim_counts, history.exposures):
        rate = theta_ref * e_t
        log_lik += y_t * np.log(rate) - rate
    return log_lik
