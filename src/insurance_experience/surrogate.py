"""Surrogate model for intractable Bayesian posterior premiums.

Implements the Calcetero/Badescu/Lin (2024) framework for approximating
Bayesian posterior premiums when the true posterior is analytically intractable.

The approach:
1. Select a representative sub-portfolio via stratified sampling.
2. For each sub-portfolio policy, compute the exact Bayesian posterior via
   importance sampling (IS). The IS weights reuse a single set of prior samples
   across all policies, making this computationally efficient at scale.
3. Fit a scalar function g(.) mapping a sufficient statistic of the claim
   history to the log credibility factor, using weighted least squares (WLS).
4. At prediction time, evaluate g(sufficient_stat) and return exp(g(.)) as
   the credibility factor.

The sufficient statistic used here is the log-likelihood of the observed
claims under a reference Poisson intensity (the empirical portfolio frequency).
For exponential family models, this is sufficient — for more complex likelihoods,
users can supply a custom sufficient_stat_fn.

Reference: Calcetero Vanegas, Badescu, Lin, Insurance: Mathematics and Economics 118 (2024).
arXiv:2211.06568.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import polars as pl

from ._types import ClaimsHistory
from .utils import history_sufficient_stat


class SurrogateModel:
    """Surrogate model for large-portfolio Bayesian experience rating.

    Fits an approximate g(.) function on a sub-portfolio using importance
    sampling, then applies it to the full portfolio analytically.

    The posterior premium for policy i is:
        mu_post_i = mu_prior_i * exp(g(L_i, n_i))
    where L_i is the log-likelihood sufficient statistic and n_i is the
    number of periods observed.

    The form of g(.) is a polynomial in L and n:
        g(L, n) = theta_0 + theta_1 * L + theta_2 * n + theta_3 * L * n

    This is flexible enough to capture Bühlmann-Straub as a special case
    (linear in L = Y_bar) while allowing non-linear responses.

    Parameters
    ----------
    prior_model : callable or None
        Function mapping ClaimsHistory -> float, returning the a priori
        premium for any policy. If None, uses history.prior_premium directly.
        This should be a GLM or similar model already fitted on the portfolio.
    n_is_samples : int
        Number of importance sampling draws from the prior. More samples
        give a better IS estimate but are slower. Default 2000.
    subsample_frac : float
        Fraction of the portfolio to use in the sub-portfolio for fitting g.
        Default 0.10 (10%). Minimum 20 policies regardless of fraction.
    poly_degree : int
        Polynomial degree for g(.). Degree 1 gives linear g; degree 2
        allows quadratic response. Default 1.
    random_state : int or None
        Random seed for IS sampling and sub-portfolio selection.
    sufficient_stat_fn : callable or None
        Custom function mapping ClaimsHistory -> float for the sufficient
        statistic. If None, uses the log-likelihood under Poisson(theta_ref).

    Attributes
    ----------
    theta_ : np.ndarray
        Fitted WLS coefficients for g(.).
    theta_ref_ : float
        Reference Poisson intensity used in the sufficient statistic.
    is_fitted_ : bool
        True after fit() has been called.
    """

    def __init__(
        self,
        prior_model: Optional[Callable[[ClaimsHistory], float]] = None,
        n_is_samples: int = 2000,
        subsample_frac: float = 0.10,
        poly_degree: int = 1,
        random_state: Optional[int] = None,
        sufficient_stat_fn: Optional[Callable[[ClaimsHistory], float]] = None,
    ) -> None:
        self.prior_model = prior_model
        self.n_is_samples = n_is_samples
        self.subsample_frac = subsample_frac
        self.poly_degree = poly_degree
        self.random_state = random_state
        self.sufficient_stat_fn = sufficient_stat_fn

        self.theta_: Optional[np.ndarray] = None
        self.theta_ref_: Optional[float] = None
        self.is_fitted_: bool = False
        self._rng: Optional[np.random.Generator] = None

    def fit(self, histories: list[ClaimsHistory]) -> "SurrogateModel":
        """Fit the surrogate g(.) function on a representative sub-portfolio.

        Steps:
        1. Draw a sub-portfolio (random stratified selection by n_periods).
        2. Estimate reference intensity theta_ref as portfolio mean frequency.
        3. Draw IS samples Theta_k ~ prior once (shared across all policies).
        4. For each sub-portfolio policy, compute IS-based Bayesian premium.
        5. Fit g(.) by WLS minimising weighted squared log-credibility error.

        Parameters
        ----------
        histories : list[ClaimsHistory]
            Full training portfolio.

        Returns
        -------
        SurrogateModel
            self (fitted model).
        """
        self._rng = np.random.default_rng(self.random_state)

        # Estimate reference intensity
        total_claims = sum(h.total_claims for h in histories)
        total_exp = sum(h.total_exposure for h in histories)
        self.theta_ref_ = total_claims / total_exp if total_exp > 0 else 1.0

        # Select sub-portfolio
        n_sub = max(20, int(len(histories) * self.subsample_frac))
        n_sub = min(n_sub, len(histories))
        idx = self._rng.choice(len(histories), size=n_sub, replace=False)
        sub_histories = [histories[i] for i in idx]

        # Draw IS samples from prior: Theta_k ~ Gamma(alpha_prior, beta_prior)
        # We use a weakly informative prior: Theta ~ Gamma(1, 1) -> E[Theta] = 1
        prior_alpha = 1.0
        prior_beta = 1.0
        theta_samples = self._rng.gamma(
            prior_alpha, 1.0 / prior_beta, size=self.n_is_samples
        )

        # Compute IS-based posterior premiums for sub-portfolio
        sub_L = []
        sub_n = []
        sub_log_cf = []
        sub_weights = []

        for hist in sub_histories:
            mu_prior = self._get_prior(hist)
            L_i = self._sufficient_stat(hist)
            n_i = float(hist.n_periods)

            mu_post_is, se_i = self._is_posterior(hist, theta_samples, mu_prior)

            if mu_prior <= 0.0 or mu_post_is <= 0.0:
                continue
            log_cf_i = np.log(mu_post_is / mu_prior)
            w_i = 1.0 / max(se_i**2, 1e-8)

            sub_L.append(L_i)
            sub_n.append(n_i)
            sub_log_cf.append(log_cf_i)
            sub_weights.append(w_i)

        if len(sub_L) < 4:
            # Degenerate case: not enough data, return identity (CF = 1)
            self.theta_ = np.zeros(self._n_features())
            self.is_fitted_ = True
            return self

        # Design matrix for g(L, n)
        X = self._design_matrix(np.array(sub_L), np.array(sub_n))
        y = np.array(sub_log_cf)
        w = np.array(sub_weights)
        w = w / w.sum()

        # WLS: solve (X^T W X) theta = X^T W y
        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ (w[:, None] * y[:, None]).ravel()

        try:
            self.theta_ = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            self.theta_ = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

        self.is_fitted_ = True
        return self

    def predict(self, history: ClaimsHistory) -> float:
        """Compute the credibility factor for a single policy.

        Parameters
        ----------
        history : ClaimsHistory
            The policy's claims history.

        Returns
        -------
        float
            Credibility factor CF = exp(g(L, n)). Multiply by prior_premium
            to obtain the posterior premium.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        self._check_fitted()
        assert self.theta_ is not None

        L = self._sufficient_stat(history)
        n = float(history.n_periods)
        x = self._design_matrix(np.array([L]), np.array([n]))[0]
        log_cf = float(x @ self.theta_)
        return float(np.exp(log_cf))

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
            posterior_premium, sufficient_stat.
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
                    "sufficient_stat": self._sufficient_stat(h),
                }
            )
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_posterior(
        self,
        history: ClaimsHistory,
        theta_samples: np.ndarray,
        mu_prior: float,
    ) -> tuple[float, float]:
        """Estimate the Bayesian posterior premium via importance sampling.

        Computes E[mu(Theta) | Y_{1:n}] using self-normalised IS:
            E_hat = sum_k w_k * mu(Theta_k) / sum_k w_k
        where w_k = exp(log-likelihood of Y_{1:n} | Theta_k) and
        Theta_k ~ prior (Gamma(1,1) in this implementation).

        Returns (posterior_mean, standard_error).
        """
        assert history.exposures is not None
        log_weights = np.zeros(len(theta_samples))
        for y_t, e_t in zip(history.claim_counts, history.exposures):
            # Y_t | Theta ~ Poi(mu_prior * e_t * Theta)
            rate = mu_prior * e_t * theta_samples
            # Poisson log PMF: y*log(rate) - rate - log(y!)
            log_weights += y_t * np.log(rate + 1e-300) - rate

        # Stabilise: subtract max log weight
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()

        # Posterior premium: E[mu_prior * Theta | data] = mu_prior * E[Theta | data]
        post_mean = float(mu_prior * np.sum(weights * theta_samples))

        # Effective sample size for SE calculation
        ess = 1.0 / np.sum(weights**2)
        if ess < 2:
            se = abs(post_mean) * 0.5  # pessimistic SE
        else:
            # Monte Carlo SE
            mean_theta = np.sum(weights * theta_samples)
            var_theta = np.sum(weights * (theta_samples - mean_theta) ** 2)
            se = float(mu_prior * np.sqrt(var_theta / max(ess, 1)))

        return post_mean, se

    def _sufficient_stat(self, history: ClaimsHistory) -> float:
        if self.sufficient_stat_fn is not None:
            return self.sufficient_stat_fn(history)
        return history_sufficient_stat(history, theta_ref=self.theta_ref_)

    def _get_prior(self, history: ClaimsHistory) -> float:
        if self.prior_model is not None:
            return self.prior_model(history)
        return history.prior_premium

    def _design_matrix(self, L: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Build polynomial design matrix for g(L, n)."""
        cols = [np.ones(len(L))]
        if self.poly_degree >= 1:
            cols.extend([L, n])
        if self.poly_degree >= 2:
            cols.extend([L**2, n**2, L * n])
        return np.column_stack(cols)

    def _n_features(self) -> int:
        if self.poly_degree >= 2:
            return 6
        if self.poly_degree >= 1:
            return 3
        return 1

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict()."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"SurrogateModel(poly_degree={self.poly_degree}, "
                f"n_is_samples={self.n_is_samples})"
            )
        return f"SurrogateModel(poly_degree={self.poly_degree}, unfitted)"
