"""Dynamic Poisson-gamma state-space experience rating model.

This implements the Ahn/Jeong/Lu/Wüthrich (2023) model for individual-level
a posteriori premium calculation with seniority weighting.

The key improvement over static credibility is that older claims are
geometrically downweighted via the decay parameters (p, q). This is
actuarially sensible: a fleet's accident record from 5 years ago is less
predictive than last year's record, especially if risk characteristics
(driver pool, vehicle fleet, safety measures) have changed.

Model specification:
    Y_t | Theta_t ~ Poisson(mu * e_t * Theta_t)
    Theta_t | Theta_{t-1}, Y_{t-1} has Gamma distribution with:
        alpha_{t+1} = p*q*(alpha_t + Y_t) + (1-p)*beta_{t+1}
        beta_{t+1}  = q*(beta_t + mu*e_t)
    Initial state: Theta_1 ~ Gamma(alpha_0, beta_0) where alpha_0/beta_0 = 1

The marginal distribution of Y_t is Negative Binomial, making the
log-likelihood tractable for empirical Bayes fitting.

Reference: Ahn, Jeong, Lu, Wüthrich, 'Dynamic Bayesian Credibility', arXiv:2308.16058.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
from scipy import optimize
from scipy.special import gammaln

from ._types import ClaimsHistory


def _negbin_logpmf(k: int, r: float, mu_nb: float) -> float:
    """Log PMF of Negative Binomial parameterised by mean and shape.

    Y ~ NegBin(r, p) where E[Y] = mu_nb = r*(1-p)/p.
    Equivalent to NB2 parameterisation: Var[Y] = mu + mu^2/r.

    Parameters
    ----------
    k : int
        Observed count.
    r : float
        Shape parameter (> 0).
    mu_nb : float
        Mean parameter (> 0).
    """
    if r <= 0 or mu_nb <= 0:
        return -1e10
    p_nb = r / (r + mu_nb)
    log_pmf = (
        gammaln(r + k)
        - gammaln(r)
        - gammaln(k + 1)
        + r * np.log(p_nb)
        + k * np.log(1.0 - p_nb + 1e-300)
    )
    return float(log_pmf)


class DynamicPoissonGammaModel:
    """Poisson-gamma conjugate state-space model with seniority weighting.

    The model fits two decay parameters (p, q) from the portfolio using
    empirical Bayes maximum likelihood. At prediction time, it runs the
    forward recursion to compute the posterior state (alpha_t, beta_t)
    and returns the corresponding posterior mean as a credibility factor.

    Parameters
    ----------
    p0 : float
        Initial guess for the state-reversion parameter p. Must be in (0, 1].
    q0 : float
        Initial guess for the decay parameter q. Must be in (0, 1].
    alpha0 : float
        Initial shape parameter of the prior Gamma distribution for Theta.
        Must be > 0. Default 1.0 (prior mode at 1 with matching scale).
    beta0_multiplier : float
        The initial rate parameter beta_0 is set to alpha0 * beta0_multiplier
        per unit of a priori rate. This keeps the prior mean at 1.0.
        Default 1.0.
    bounds : tuple
        (lower, upper) bounds for p and q during optimisation.
        Default ((0.01, 0.99), (0.01, 0.99)).

    Attributes
    ----------
    p_ : float
        Fitted state-reversion parameter (after fit()).
    q_ : float
        Fitted decay parameter (after fit()).
    loglik_ : float
        Log-likelihood at the fitted parameters.
    is_fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> histories = [
    ...     ClaimsHistory(f"P{i}", [1,2,3],
    ...                   rng.poisson(1.2, size=3).tolist(),
    ...                   prior_premium=400.0)
    ...     for i in range(100)
    ... ]
    >>> model = DynamicPoissonGammaModel()
    >>> model.fit(histories)
    DynamicPoissonGammaModel(p=..., q=...)
    """

    def __init__(
        self,
        p0: float = 0.5,
        q0: float = 0.8,
        alpha0: float = 1.0,
        beta0_multiplier: float = 1.0,
        bounds: tuple = ((0.01, 0.99), (0.01, 0.99)),
    ) -> None:
        self.p0 = p0
        self.q0 = q0
        self.alpha0 = alpha0
        self.beta0_multiplier = beta0_multiplier
        self.bounds = bounds

        self.p_: Optional[float] = None
        self.q_: Optional[float] = None
        self.loglik_: Optional[float] = None
        self.is_fitted_: bool = False

    def fit(
        self, histories: list[ClaimsHistory], verbose: bool = False
    ) -> "DynamicPoissonGammaModel":
        """Fit p and q by empirical Bayes MLE on the portfolio log-likelihood.

        The log-likelihood sums over all policies and all periods. Within each
        policy, the marginal Y_t is Negative Binomial with state-dependent
        shape and mean parameters derived from the forward recursion.

        Parameters
        ----------
        histories : list[ClaimsHistory]
            Training portfolio. At least 10 policies recommended for stable
            parameter estimation.
        verbose : bool
            If True, print optimisation progress.

        Returns
        -------
        DynamicPoissonGammaModel
            self (fitted model).
        """
        if len(histories) < 2:
            raise ValueError(
                "At least 2 policy histories are required for fitting"
            )

        def neg_loglik(params: np.ndarray) -> float:
            p, q = float(params[0]), float(params[1])
            total_ll = 0.0
            for history in histories:
                total_ll += self._policy_loglik(history, p, q)
            return -total_ll

        x0 = np.array([self.p0, self.q0])
        bounds_list = [self.bounds[0], self.bounds[1]]

        result = optimize.minimize(
            neg_loglik,
            x0,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6},
        )

        if verbose and not result.success:
            print(f"Optimisation warning: {result.message}")

        self.p_ = float(result.x[0])
        self.q_ = float(result.x[1])
        self.loglik_ = float(-result.fun)
        self.is_fitted_ = True
        return self

    def predict(self, history: ClaimsHistory) -> float:
        """Compute the credibility factor for a single policy.

        Runs the forward recursion using the fitted (p, q) parameters to
        propagate the Gamma posterior through the observed claim sequence.
        The credibility factor is the ratio of the posterior mean to the
        a priori rate.

        Parameters
        ----------
        history : ClaimsHistory
            The policy's claims history.

        Returns
        -------
        float
            Credibility factor CF = mu_post / mu_prior. Multiply by
            prior_premium to obtain the posterior premium.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        self._check_fitted()
        assert self.p_ is not None
        assert self.q_ is not None

        alpha_t, beta_t = self._forward_recursion(
            history, self.p_, self.q_
        )
        mu = history.prior_premium
        mu_post = (alpha_t / beta_t) * mu
        cf = mu_post / mu
        return max(cf, 0.0)

    def predict_posterior_params(
        self, history: ClaimsHistory
    ) -> tuple[float, float]:
        """Return the posterior Gamma parameters (alpha, beta) after recursion.

        The posterior distribution of Theta_{t+1} | Y_{1:t} is:
            Gamma(alpha_{t+1}, beta_{t+1})
        with mean alpha_{t+1} / beta_{t+1}.

        This is useful for uncertainty quantification: the posterior variance
        is alpha / beta^2.

        Parameters
        ----------
        history : ClaimsHistory
            The policy's claims history.

        Returns
        -------
        tuple[float, float]
            (alpha_{t+1}, beta_{t+1}) — shape and rate of posterior Gamma.
        """
        self._check_fitted()
        assert self.p_ is not None
        assert self.q_ is not None
        return self._forward_recursion(history, self.p_, self.q_)

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
            posterior_premium, posterior_alpha, posterior_beta,
            posterior_variance.
        """
        self._check_fitted()
        rows = []
        for h in histories:
            cf = self.predict(h)
            alpha_t, beta_t = self.predict_posterior_params(h)
            rows.append(
                {
                    "policy_id": h.policy_id,
                    "prior_premium": h.prior_premium,
                    "credibility_factor": cf,
                    "posterior_premium": h.prior_premium * cf,
                    "posterior_alpha": alpha_t,
                    "posterior_beta": beta_t,
                    "posterior_variance": alpha_t / (beta_t**2),
                }
            )
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Core recursion (also used in log-likelihood)
    # ------------------------------------------------------------------

    def _forward_recursion(
        self,
        history: ClaimsHistory,
        p: float,
        q: float,
    ) -> tuple[float, float]:
        """Run the Gamma state recursion through the claims history.

        Returns the posterior parameters (alpha, beta) after observing
        all periods in ``history``. The initial prior is set so that the
        prior mean of Theta is 1 (i.e., alpha_0 / beta_0 = 1).

        Parameters
        ----------
        history : ClaimsHistory
            Policy's claims history.
        p : float
            State-reversion parameter.
        q : float
            Decay parameter.

        Returns
        -------
        tuple[float, float]
            (alpha_{T+1}, beta_{T+1}) — the one-step-ahead posterior state.
        """
        assert history.exposures is not None
        mu = history.prior_premium

        # Initialise prior: Theta ~ Gamma(alpha0, beta0) with E[Theta]=1
        alpha = self.alpha0
        beta = self.alpha0 * self.beta0_multiplier  # alpha0 / beta0 = 1

        for y_t, e_t in zip(history.claim_counts, history.exposures):
            # Bayesian update: observe Y_t ~ Poi(mu * e_t * Theta)
            # Posterior: Theta | Y_t ~ Gamma(alpha + Y_t, beta + mu*e_t)
            alpha_post = alpha + float(y_t)
            beta_post = beta + mu * e_t

            # State transition to next period
            beta_next = q * (beta_post + mu * e_t)
            alpha_next = p * q * alpha_post + (1.0 - p) * beta_next

            # Guard against numerical underflow
            alpha = max(alpha_next, 1e-10)
            beta = max(beta_next, 1e-10)

        return float(alpha), float(beta)

    def _policy_loglik(
        self,
        history: ClaimsHistory,
        p: float,
        q: float,
    ) -> float:
        """Compute the marginal log-likelihood for a single policy.

        The marginal distribution of Y_t given Y_{1:t-1} is Negative Binomial.
        The shape parameter r_t = alpha_t and the mean is mu * e_t * alpha_t / beta_t.

        Summing these marginal log-likelihoods over all periods gives the
        complete-data log-likelihood tractable via the forward recursion.
        """
        assert history.exposures is not None
        mu = history.prior_premium

        alpha = self.alpha0
        beta = self.alpha0 * self.beta0_multiplier

        total_ll = 0.0
        for y_t, e_t in zip(history.claim_counts, history.exposures):
            # Marginal: Y_t | Y_{1:t-1} ~ NegBin(r=alpha, mu=alpha/beta * mu*e_t)
            r_t = alpha
            mu_t = (alpha / beta) * mu * e_t
            total_ll += _negbin_logpmf(int(y_t), r_t, mu_t)

            # Update state
            alpha_post = alpha + float(y_t)
            beta_post = beta + mu * e_t
            beta_next = q * (beta_post + mu * e_t)
            alpha_next = p * q * alpha_post + (1.0 - p) * beta_next

            alpha = max(alpha_next, 1e-10)
            beta = max(beta_next, 1e-10)

        return total_ll

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict()."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"DynamicPoissonGammaModel(p={self.p_:.4f}, q={self.q_:.4f})"
            )
        return "DynamicPoissonGammaModel(unfitted)"
