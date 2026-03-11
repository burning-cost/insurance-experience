"""Bühlmann-Straub credibility model at individual policy level.

This is the classical linear credibility formula applied to a single policy's
multi-period claims history, rather than the usual group-level application.
The key insight is that with enough policy-years of data, an individual policy
accumulates sufficient credibility to be rated on its own experience.

The model estimates a single structural parameter kappa = sigma^2 / tau^2
(the ratio of within-policy variance to between-policy variance) from the
portfolio, then applies the Bühlmann credibility formula:

    omega_t = t / (t + kappa)
    CF = omega_t * Y_bar / mu + (1 - omega_t)

where Y_bar is the exposure-weighted empirical frequency and mu is the
a priori rate.

Reference: Bühlmann & Gisler, 'A Course in Credibility Theory', Springer 2005.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from ._types import ClaimsHistory


class StaticCredibilityModel:
    """Bühlmann-Straub credibility at individual policy level.

    Fits the structural parameter kappa from a portfolio of policy histories
    using the method of moments estimator. Predicts a multiplicative
    credibility factor for each policy.

    The model assumes:
        Y_{t} | Theta ~ Poisson(mu * Theta * e_t)
        Theta ~ distribution with E[Theta]=1, Var[Theta]=tau^2

    This gives the credibility approximation:
        E[Theta | Y_{1:t}] ≈ omega_t * Y_bar / mu + (1 - omega_t)

    Parameters
    ----------
    kappa : float or None
        Known credibility coefficient kappa = sigma^2 / tau^2. If None,
        kappa is estimated from the portfolio during fit(). Set explicitly
        to share a kappa estimated from a larger portfolio.
    min_kappa : float
        Lower bound on the estimated kappa. Prevents omega from reaching 1
        when the portfolio is very homogeneous. Default 0.1.
    max_kappa : float
        Upper bound on the estimated kappa. Prevents degenerate credibility
        for very heterogeneous portfolios. Default 1000.0.

    Attributes
    ----------
    kappa_ : float
        Fitted credibility coefficient (after calling fit()).
    within_variance_ : float
        Estimated within-policy variance component sigma^2.
    between_variance_ : float
        Estimated between-policy variance component tau^2.
    portfolio_mean_ : float
        Exposure-weighted mean claim frequency across the portfolio.
    is_fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> histories = [ClaimsHistory("P1", [1,2,3], [0,1,0], prior_premium=400.0),
    ...              ClaimsHistory("P2", [1,2,3], [2,1,2], prior_premium=400.0)]
    >>> model = StaticCredibilityModel()
    >>> model.fit(histories)
    StaticCredibilityModel(kappa=...)
    >>> cf = model.predict(histories[0])
    >>> posterior = histories[0].prior_premium * cf
    """

    def __init__(
        self,
        kappa: Optional[float] = None,
        min_kappa: float = 0.1,
        max_kappa: float = 1000.0,
    ) -> None:
        self.kappa = kappa
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa

        # Set after fit()
        self.kappa_: Optional[float] = None
        self.within_variance_: Optional[float] = None
        self.between_variance_: Optional[float] = None
        self.portfolio_mean_: Optional[float] = None
        self.is_fitted_: bool = False

    def fit(self, histories: list[ClaimsHistory]) -> "StaticCredibilityModel":
        """Estimate structural parameters from a portfolio of policy histories.

        Uses the method of moments estimator for Bühlmann-Straub:
        - within-policy variance estimated from period-to-period fluctuations
        - between-policy variance estimated from cross-sectional spread

        Parameters
        ----------
        histories : list[ClaimsHistory]
            Portfolio of policy histories. Must contain at least 2 policies
            with at least 1 period each for parameter estimation.

        Returns
        -------
        StaticCredibilityModel
            self (fitted model).

        Raises
        ------
        ValueError
            If fewer than 2 histories are provided, or if all histories have
            only 1 period (insufficient for within-variance estimation).
        """
        if len(histories) < 2:
            raise ValueError(
                "At least 2 policy histories are required to estimate kappa"
            )

        if self.kappa is not None:
            # Use provided kappa, just estimate portfolio mean
            self.kappa_ = self.kappa
            self.within_variance_ = float("nan")
            self.between_variance_ = float("nan")
            self.portfolio_mean_ = self._portfolio_mean(histories)
            self.is_fitted_ = True
            return self

        self.kappa_, self.within_variance_, self.between_variance_ = (
            self._estimate_kappa(histories)
        )
        self.portfolio_mean_ = self._portfolio_mean(histories)
        self.is_fitted_ = True
        return self

    def predict(self, history: ClaimsHistory) -> float:
        """Compute the credibility factor for a single policy.

        Parameters
        ----------
        history : ClaimsHistory
            The policy's claims history. The prior_premium field is used as
            the a priori rate mu.

        Returns
        -------
        float
            Credibility factor CF in (0, inf). Multiply by prior_premium to
            obtain the posterior premium.
            CF = omega_t * (Y_bar / mu) + (1 - omega_t)

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        self._check_fitted()
        assert self.kappa_ is not None
        assert self.exposures_ok(history)

        assert history.exposures is not None
        t = history.total_exposure  # effective credibility weight
        omega_t = t / (t + self.kappa_)

        mu = history.prior_premium  # a priori rate (expected frequency)
        y_bar = history.claim_frequency  # empirical frequency

        # Credibility factor: omega * (empirical / prior) + (1 - omega)
        if mu <= 0.0:
            return 1.0

        cf = omega_t * (y_bar / mu) + (1.0 - omega_t)
        return max(cf, 0.0)  # guard against negative values from rounding

    def predict_batch(self, histories: list[ClaimsHistory]) -> pl.DataFrame:
        """Score a batch of policies and return a Polars DataFrame.

        Parameters
        ----------
        histories : list[ClaimsHistory]
            Policies to score.

        Returns
        -------
        pl.DataFrame
            Columns: policy_id, prior_premium, credibility_factor,
            posterior_premium.
        """
        self._check_fitted()
        rows = []
        for h in histories:
            cf = self.predict(h)
            rows.append(
                {
                    "policy_id": h.policy_id,
                    "prior_premium": h.prior_premium,
                    "credibility_factor": cf,
                    "posterior_premium": h.prior_premium * cf,
                }
            )
        return pl.DataFrame(rows)

    def credibility_weight(self, history: ClaimsHistory) -> float:
        """Return the Bühlmann credibility weight omega_t for a policy.

        Parameters
        ----------
        history : ClaimsHistory
            The policy's claims history.

        Returns
        -------
        float
            omega_t = t / (t + kappa) in [0, 1], where t is total exposure.
        """
        self._check_fitted()
        assert self.kappa_ is not None
        assert history.exposures is not None
        t = history.total_exposure
        return t / (t + self.kappa_)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_kappa(
        self, histories: list[ClaimsHistory]
    ) -> tuple[float, float, float]:
        """Bühlmann-Straub method of moments estimation.

        Returns (kappa, within_variance, between_variance).
        """
        # Compute per-policy exposure-weighted mean frequencies
        assert all(h.exposures is not None for h in histories)
        n = len(histories)

        exposures = [np.array(h.exposures, dtype=float) for h in histories]
        counts = [np.array(h.claim_counts, dtype=float) for h in histories]
        rates = [c / e for c, e in zip(counts, exposures)]  # Y_t / e_t

        total_exp = [float(e.sum()) for e in exposures]
        y_bars = [float((c.sum()) / t) for c, t in zip(counts, total_exp)]

        grand_total_exp = sum(total_exp)
        grand_mean = sum(
            c.sum() for c in counts
        ) / grand_total_exp  # E_i[Y_bar_i]

        # --- Within-policy variance estimate (sigma^2) ---
        # sigma^2 = E[Var(Y_t | Theta)] = mu (for Poisson)
        # MoM estimator: average exposure-weighted within-period variance
        within_num = 0.0
        within_denom = 0.0
        for i, (rate_i, exp_i, y_bar_i) in enumerate(
            zip(rates, exposures, y_bars)
        ):
            t_i = len(rate_i)
            if t_i < 2:
                continue
            for t in range(t_i):
                within_num += exp_i[t] * (rate_i[t] - y_bar_i) ** 2
                within_denom += 1
        if within_denom == 0 or within_num == 0:
            # Fallback: use portfolio mean as within-variance estimate
            sigma2 = grand_mean
        else:
            # Unbiased estimator: divide by (T_i - 1) per policy
            total_periods = sum(len(r) for r in rates)
            denom2 = total_periods - n
            if denom2 <= 0:
                sigma2 = grand_mean
            else:
                sigma2 = within_num / denom2

        # --- Between-policy variance estimate (tau^2) ---
        # tau^2 = Var(Theta)
        # Between-policy sum of squares
        c_const = grand_total_exp - sum(t**2 for t in total_exp) / grand_total_exp
        if c_const <= 0:
            tau2 = max(sigma2 / 10.0, 1e-6)
        else:
            between_ss = sum(
                t_i * (y_bar_i - grand_mean) ** 2
                for t_i, y_bar_i in zip(total_exp, y_bars)
            )
            tau2 = max((between_ss - (n - 1) * sigma2) / c_const, 1e-8)

        kappa = sigma2 / tau2
        kappa = float(np.clip(kappa, self.min_kappa, self.max_kappa))
        return kappa, float(sigma2), float(tau2)

    def _portfolio_mean(self, histories: list[ClaimsHistory]) -> float:
        total_claims = sum(h.total_claims for h in histories)
        total_exp = sum(h.total_exposure for h in histories)
        if total_exp == 0.0:
            return 0.0
        return total_claims / total_exp

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict()."
            )

    @staticmethod
    def exposures_ok(history: ClaimsHistory) -> bool:
        return history.exposures is not None and len(history.exposures) > 0

    def __repr__(self) -> str:
        if self.is_fitted_:
            return f"StaticCredibilityModel(kappa={self.kappa_:.4f})"
        return "StaticCredibilityModel(unfitted)"
