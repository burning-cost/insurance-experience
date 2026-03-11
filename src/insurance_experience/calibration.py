"""Balance calibration for a posteriori rating models.

The balance property states that the sum of posterior premiums should equal
the sum of observed claims across the portfolio (weighted by exposure). This
is the self-financing constraint: experience rating redistributes the total
premium, but does not change it.

GLMs satisfy this automatically (via the score equation for the intercept).
Credibility and neural models do not — calibration enforces it post-hoc via
a multiplicative rescaling factor.

Reference:
  Wüthrich, 'Bias Regularization in Neural Network Models', EAJ 10 (2020).
  Lindholm & Wüthrich, 'The Balance Property in Insurance Pricing', SAJ 2025.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from ._types import CalibrationResult, ClaimsHistory


def balance_calibrate(
    predict_fn: Callable[[ClaimsHistory], float],
    histories: list[ClaimsHistory],
    exposure_weighted: bool = True,
) -> CalibrationResult:
    """Compute the portfolio-level balance calibration factor.

    The calibration factor delta satisfies:
        sum_i [delta * CF_i * mu_prior_i * e_i] = sum_i [Y_i * e_i]

    For multiplicative models:
        delta = sum(actual * exposure) / sum(posterior * exposure)

    This function computes delta and returns it as a CalibrationResult.
    Apply delta to all posterior premiums to restore portfolio balance.

    Parameters
    ----------
    predict_fn : callable
        Function mapping ClaimsHistory -> float, returning the credibility
        factor (not the posterior premium). Typically model.predict.
    histories : list[ClaimsHistory]
        Calibration portfolio. Typically the same data used for fitting,
        but can be a hold-out period for forward-looking calibration.
    exposure_weighted : bool
        If True, weight each policy by its total exposure when computing
        the balance check. If False, use simple sums. Default True.

    Returns
    -------
    CalibrationResult
        Contains calibration_factor, sum_actual, sum_predicted, n_policies.

    Examples
    --------
    >>> model = StaticCredibilityModel().fit(histories)
    >>> result = balance_calibrate(model.predict, histories)
    >>> print(f"Calibration factor: {result.calibration_factor:.4f}")
    >>> # Apply to predictions:
    >>> posterior = prior * model.predict(h) * result.calibration_factor
    """
    sum_actual = 0.0
    sum_predicted = 0.0

    for h in histories:
        assert h.exposures is not None
        cf = predict_fn(h)
        posterior = h.prior_premium * cf

        if exposure_weighted:
            weight = h.total_exposure
        else:
            weight = 1.0

        # Actual: total claims weighted by exposure (as a rate)
        # We sum actual * exposure / total_exposure * weight to get total weighted claims
        actual_rate = h.claim_frequency  # total_claims / total_exposure
        sum_actual += actual_rate * weight
        sum_predicted += posterior * weight

    if sum_predicted <= 0.0:
        calibration_factor = 1.0
    else:
        calibration_factor = sum_actual / sum_predicted

    return CalibrationResult(
        calibration_factor=float(calibration_factor),
        sum_actual=float(sum_actual),
        sum_predicted=float(sum_predicted),
        n_policies=len(histories),
    )


def apply_calibration(
    credibility_factor: float,
    calibration_result: CalibrationResult,
) -> float:
    """Apply a calibration factor to a single credibility factor.

    Parameters
    ----------
    credibility_factor : float
        The raw credibility factor from a model's predict().
    calibration_result : CalibrationResult
        Output of balance_calibrate().

    Returns
    -------
    float
        Calibrated credibility factor = CF * delta.
    """
    return credibility_factor * calibration_result.calibration_factor


def calibrated_predict_fn(
    predict_fn: Callable[[ClaimsHistory], float],
    calibration_result: CalibrationResult,
) -> Callable[[ClaimsHistory], float]:
    """Wrap a predict function to apply calibration automatically.

    Parameters
    ----------
    predict_fn : callable
        Original predict function (ClaimsHistory -> credibility_factor).
    calibration_result : CalibrationResult
        Output of balance_calibrate().

    Returns
    -------
    callable
        New predict function that applies calibration to each prediction.

    Examples
    --------
    >>> cal = balance_calibrate(model.predict, histories)
    >>> calibrated = calibrated_predict_fn(model.predict, cal)
    >>> cf = calibrated(new_history)  # already calibrated
    """
    delta = calibration_result.calibration_factor

    def _calibrated(history: ClaimsHistory) -> float:
        return predict_fn(history) * delta

    return _calibrated


def balance_report(
    predict_fn: Callable[[ClaimsHistory], float],
    histories: list[ClaimsHistory],
    by_n_periods: bool = False,
) -> pl.DataFrame:
    """Generate a portfolio-level balance report.

    Shows the actual vs predicted claim frequency, and the credibility
    factor distribution, optionally broken down by number of observed
    periods (to check whether newer or longer-tenured policies have
    systematic bias).

    Parameters
    ----------
    predict_fn : callable
        Function mapping ClaimsHistory -> credibility_factor.
    histories : list[ClaimsHistory]
        Portfolio to assess.
    by_n_periods : bool
        If True, break down by number of observed periods. Default False.

    Returns
    -------
    pl.DataFrame
        Summary statistics: policy_id, n_periods, actual_frequency,
        prior_premium, credibility_factor, posterior_premium,
        and residual (actual / posterior).
    """
    rows = []
    for h in histories:
        assert h.exposures is not None
        cf = predict_fn(h)
        posterior = h.prior_premium * cf
        actual_freq = h.claim_frequency
        residual = actual_freq / posterior if posterior > 0 else float("nan")
        rows.append(
            {
                "policy_id": h.policy_id,
                "n_periods": h.n_periods,
                "total_exposure": h.total_exposure,
                "actual_frequency": actual_freq,
                "prior_premium": h.prior_premium,
                "credibility_factor": cf,
                "posterior_premium": posterior,
                "residual": residual,
            }
        )
    df = pl.DataFrame(rows)

    if by_n_periods:
        return (
            df.group_by("n_periods")
            .agg(
                [
                    pl.col("actual_frequency").mean().alias("mean_actual"),
                    pl.col("posterior_premium").mean().alias("mean_posterior"),
                    pl.col("credibility_factor").mean().alias("mean_cf"),
                    pl.col("residual").mean().alias("mean_residual"),
                    pl.len().alias("n_policies"),
                ]
            )
            .sort("n_periods")
        )

    return df
