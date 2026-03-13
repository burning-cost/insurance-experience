"""Core data types for insurance experience rating.

The ClaimsHistory dataclass represents a single policy's claims sequence.
All models in this library accept lists of ClaimsHistory as training input
and a single ClaimsHistory at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClaimsHistory:
    """Claims history for a single policy.

    This is the fundamental input type for all experience rating models.
    Each instance represents one policy's observed claims sequence across
    one or more periods (typically policy years).

    Parameters
    ----------
    policy_id : str
        Unique identifier for the policy.
    periods : list[int]
        Year indices for each observation period. Typically [1, 2, 3, ...].
        Must be non-empty. Periods need not be contiguous but must be unique
        and in ascending order.
    claim_counts : list[int]
        Observed claim count in each period. Must have the same length as
        ``periods``. Non-negative integers.
    claim_amounts : list[float] or None
        Aggregate claim amount (incurred) in each period. Set to None for
        frequency-only models. When provided, must have the same length as
        ``periods``.
    exposures : list[float]
        Risk exposure in each period (e.g., years on risk, vehicle-years).
        Must have the same length as ``periods``. Defaults to 1.0 for each
        period if not supplied.
    prior_premium : float
        The a priori (GLM-based) expected annual claim cost for the next
        period. This is the base rate that experience rating will adjust.
        Must be strictly positive.

    Examples
    --------
    >>> history = ClaimsHistory(
    ...     policy_id="POL001",
    ...     periods=[1, 2, 3],
    ...     claim_counts=[0, 1, 0],
    ...     exposures=[1.0, 1.0, 0.8],
    ...     prior_premium=450.0,
    ... )
    """

    policy_id: str
    periods: list[int]
    claim_counts: list[int]
    claim_amounts: Optional[list[float]] = None
    exposures: Optional[list[float]] = None
    prior_premium: float = 1.0

    def __post_init__(self) -> None:
        self._validate()
        # Normalise exposures to a list if not provided
        if self.exposures is None:
            self.exposures = [1.0] * len(self.periods)

    def _validate(self) -> None:
        if len(self.periods) == 0:
            raise ValueError("periods must be non-empty")
        if len(self.claim_counts) != len(self.periods):
            raise ValueError(
                f"claim_counts length ({len(self.claim_counts)}) must match "
                f"periods length ({len(self.periods)})"
            )
        if self.claim_amounts is not None and len(self.claim_amounts) != len(
            self.periods
        ):
            raise ValueError(
                f"claim_amounts length ({len(self.claim_amounts)}) must match "
                f"periods length ({len(self.periods)})"
            )
        exposures = self.exposures
        if exposures is not None:
            if len(exposures) != len(self.periods):
                raise ValueError(
                    f"exposures length ({len(exposures)}) must match "
                    f"periods length ({len(self.periods)})"
                )
            if any(e <= 0.0 for e in exposures):
                raise ValueError("All exposures must be strictly positive")
        if any(c < 0 for c in self.claim_counts):
            raise ValueError("claim_counts must be non-negative")
        if self.prior_premium <= 0.0:
            raise ValueError("prior_premium must be strictly positive")
        if len(self.periods) != len(set(self.periods)):
            raise ValueError("periods must be unique")

    @property
    def n_periods(self) -> int:
        """Number of observation periods."""
        return len(self.periods)

    @property
    def total_claims(self) -> int:
        """Total claim count across all periods."""
        return sum(self.claim_counts)

    @property
    def total_exposure(self) -> float:
        """Total exposure across all periods."""
        assert self.exposures is not None
        return sum(self.exposures)

    @property
    def claim_frequency(self) -> float:
        """Observed claim frequency (claims per unit exposure).

        Returns 0.0 if total exposure is zero (guard only — exposures
        are validated to be positive).
        """
        total_exp = self.total_exposure
        if total_exp == 0.0:
            return 0.0
        return self.total_claims / total_exp

    @property
    def exposure_weighted_counts(self) -> list[float]:
        """Claim counts adjusted by exposure: Y_t / e_t per period."""
        assert self.exposures is not None
        return [c / e for c, e in zip(self.claim_counts, self.exposures)]


@dataclass
class CalibrationResult:
    """Output of the balance calibration step.

    Parameters
    ----------
    calibration_factor : float
        Multiplicative factor applied to all posterior premiums to restore
        portfolio balance. Values close to 1.0 indicate the model was
        already approximately balanced before calibration.
    sum_actual : float
        Sum of observed claim counts (weighted by exposure) over the
        calibration portfolio.
    sum_predicted : float
        Sum of posterior premiums (weighted by exposure) before calibration.
    n_policies : int
        Number of policies in the calibration portfolio.
    """

    calibration_factor: float
    sum_actual: float
    sum_predicted: float
    n_policies: int

    @property
    def relative_bias(self) -> float:
        """Relative bias before calibration: (predicted - actual) / actual."""
        if self.sum_actual == 0.0:
            return float("nan")
        return (self.sum_predicted - self.sum_actual) / self.sum_actual
