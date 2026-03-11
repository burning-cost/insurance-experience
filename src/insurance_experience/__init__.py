"""insurance-experience: Individual policy-level Bayesian posterior experience rating.

Four progressive model tiers for updating a priori GLM premiums with claims history:

1. StaticCredibilityModel  — Bühlmann-Straub at individual policy level
2. DynamicPoissonGammaModel — State-space model with seniority weighting (Ahn et al. 2023)
3. SurrogateModel          — IS-based surrogate for intractable posteriors (Calcetero et al. 2024)
4. DeepAttentionModel      — Neural attention credibility (Wüthrich 2024) [requires torch]

All models share the same interface:
    model.fit(histories)           # list[ClaimsHistory]
    cf = model.predict(history)    # returns credibility factor (float)
    posterior = prior * cf

Quick start::

    from insurance_experience import ClaimsHistory, StaticCredibilityModel
    from insurance_experience import balance_calibrate

    histories = [
        ClaimsHistory("POL001", periods=[1, 2, 3], claim_counts=[0, 1, 0],
                      exposures=[1.0, 1.0, 0.8], prior_premium=450.0),
        ClaimsHistory("POL002", periods=[1, 2, 3], claim_counts=[2, 1, 2],
                      exposures=[1.0, 1.0, 1.0], prior_premium=450.0),
    ]

    model = StaticCredibilityModel()
    model.fit(histories)

    cf = model.predict(histories[0])
    posterior = histories[0].prior_premium * cf

    # Enforce balance property
    cal = balance_calibrate(model.predict, histories)
    posterior_balanced = posterior * cal.calibration_factor
"""

from ._types import CalibrationResult, ClaimsHistory
from .calibration import (
    apply_calibration,
    balance_calibrate,
    balance_report,
    calibrated_predict_fn,
)
from .dynamic import DynamicPoissonGammaModel
from .static import StaticCredibilityModel
from .surrogate import SurrogateModel
from .utils import (
    credibility_factor,
    exposure_weighted_mean,
    history_sufficient_stat,
    posterior_premium,
    seniority_weights,
)

__version__ = "0.1.0"
__all__ = [
    # Data types
    "ClaimsHistory",
    "CalibrationResult",
    # Models
    "StaticCredibilityModel",
    "DynamicPoissonGammaModel",
    "SurrogateModel",
    "DeepAttentionModel",
    # Calibration
    "balance_calibrate",
    "apply_calibration",
    "calibrated_predict_fn",
    "balance_report",
    # Utilities
    "credibility_factor",
    "posterior_premium",
    "seniority_weights",
    "exposure_weighted_mean",
    "history_sufficient_stat",
    # Version
    "__version__",
]


def __getattr__(name: str):
    """Lazy import for optional torch-dependent classes."""
    if name == "DeepAttentionModel":
        from .attention import DeepAttentionModel

        return DeepAttentionModel
    raise AttributeError(f"module 'insurance_experience' has no attribute {name!r}")
