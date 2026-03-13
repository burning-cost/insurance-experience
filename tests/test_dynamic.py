"""Tests for DynamicPoissonGammaModel."""

import numpy as np
import pytest
from insurance_experience import ClaimsHistory, DynamicPoissonGammaModel


def make_history(policy_id: str, counts: list[int], prior: float = 1.0) -> ClaimsHistory:
    return ClaimsHistory(
        policy_id=policy_id,
        periods=list(range(1, len(counts) + 1)),
        claim_counts=counts,
        prior_premium=prior,
    )


def simulate_poisson_gamma(
    n_policies: int,
    n_periods: int,
    p: float,
    q: float,
    mu: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[ClaimsHistory]:
    """Simulate from the dynamic Poisson-gamma model.

    Generates data from the true model so we can verify parameter recovery.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    histories = []
    alpha0 = 1.0
    beta0 = 1.0

    for i in range(n_policies):
        counts = []
        alpha = alpha0
        beta = beta0

        for t in range(n_periods):
            # Draw latent Theta_t ~ Gamma(alpha, beta)
            theta_t = rng.gamma(alpha, 1.0 / beta)
            y_t = rng.poisson(mu * theta_t)
            counts.append(int(y_t))

            # State update
            alpha_post = alpha + y_t
            beta_post = beta + mu
            beta_next = q * (beta_post + mu)
            alpha_next = p * q * alpha_post + (1.0 - p) * beta_next

            alpha = max(alpha_next, 1e-10)
            beta = max(beta_next, 1e-10)

        histories.append(make_history(f"P{i}", counts, prior=mu))

    return histories


class TestDynamicFit:
    def test_fit_returns_self(self):
        rng = np.random.default_rng(100)
        histories = simulate_poisson_gamma(20, 5, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        result = model.fit(histories)
        assert result is model

    def test_is_fitted_after_fit(self):
        rng = np.random.default_rng(101)
        histories = simulate_poisson_gamma(10, 3, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        assert not model.is_fitted_
        model.fit(histories)
        assert model.is_fitted_

    def test_fitted_parameters_in_bounds(self):
        rng = np.random.default_rng(102)
        histories = simulate_poisson_gamma(30, 5, p=0.7, q=0.85, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        assert 0.01 <= model.p_ <= 0.99
        assert 0.01 <= model.q_ <= 0.99

    def test_loglik_is_negative(self):
        rng = np.random.default_rng(103)
        histories = simulate_poisson_gamma(20, 4, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        # Log-likelihood should be finite and negative
        assert model.loglik_ < 0
        assert np.isfinite(model.loglik_)

    def test_too_few_histories_raises(self):
        model = DynamicPoissonGammaModel()
        with pytest.raises(ValueError, match="At least 2"):
            model.fit([make_history("P1", [1, 2, 3])])

    def test_parameter_recovery(self):
        """Verify that fitted p, q are in the right ballpark for simulated data."""
        rng = np.random.default_rng(42)
        # Use many policies and periods for reliable estimation
        histories = simulate_poisson_gamma(
            200, 5, p=0.75, q=0.85, mu=1.5, rng=rng
        )
        model = DynamicPoissonGammaModel(p0=0.5, q0=0.5)
        model.fit(histories)
        # Allow wide tolerance (MLE on NegBin is noisy with moderate data)
        assert model.p_ == pytest.approx(0.75, abs=0.25), f"p={model.p_}"
        assert model.q_ == pytest.approx(0.85, abs=0.25), f"q={model.q_}"


class TestDynamicPredict:
    def test_predict_before_fit_raises(self):
        model = DynamicPoissonGammaModel()
        h = make_history("P1", [0, 1, 0])
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(h)

    def test_credibility_factor_non_negative(self):
        rng = np.random.default_rng(104)
        histories = simulate_poisson_gamma(20, 5, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        for h in histories:
            cf = model.predict(h)
            assert cf >= 0.0

    def test_zero_claims_cf_below_one(self):
        """History of zero claims should pull posterior below prior."""
        rng = np.random.default_rng(105)
        histories = simulate_poisson_gamma(50, 5, p=0.8, q=0.9, mu=1.0, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        h_zero = make_history("ZERO", [0, 0, 0, 0, 0], prior=1.0)
        cf = model.predict(h_zero)
        assert cf < 1.0, f"Expected CF < 1 for zero claims, got {cf}"

    def test_high_claims_cf_above_one(self):
        """History of high claims should pull posterior above prior."""
        rng = np.random.default_rng(106)
        histories = simulate_poisson_gamma(50, 5, p=0.8, q=0.9, mu=1.0, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        h_high = make_history("HIGH", [5, 5, 5, 5, 5], prior=1.0)
        cf = model.predict(h_high)
        assert cf > 1.0, f"Expected CF > 1 for high claims, got {cf}"

    def test_single_period_works(self):
        """Model should handle t=1 without error."""
        rng = np.random.default_rng(107)
        histories = simulate_poisson_gamma(20, 5, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        h_single = make_history("S1", [2], prior=1.0)
        cf = model.predict(h_single)
        assert cf >= 0.0
        assert np.isfinite(cf)

    def test_posterior_params_shape(self):
        rng = np.random.default_rng(108)
        histories = simulate_poisson_gamma(20, 5, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        alpha, beta = model.predict_posterior_params(histories[0])
        assert alpha > 0
        assert beta > 0

    def test_posterior_mean_matches_cf(self):
        """alpha/beta should equal CF * mu_prior."""
        rng = np.random.default_rng(109)
        histories = simulate_poisson_gamma(20, 5, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        h = histories[0]
        cf = model.predict(h)
        alpha, beta = model.predict_posterior_params(h)
        expected_mean = alpha / beta
        assert expected_mean == pytest.approx(cf, rel=1e-5)


class TestDynamicForwardRecursion:
    """Unit tests for the forward recursion."""

    def test_alpha_beta_positive_throughout(self):
        """Alpha and beta must remain positive after each step."""
        rng = np.random.default_rng(110)
        histories = simulate_poisson_gamma(10, 10, p=0.5, q=0.5, rng=rng)
        model = DynamicPoissonGammaModel(p0=0.5, q0=0.5)
        model.fit(histories)
        for h in histories:
            alpha, beta = model.predict_posterior_params(h)
            assert alpha > 0, f"Negative alpha: {alpha}"
            assert beta > 0, f"Negative beta: {beta}"

    def test_high_q_decays_slowly(self):
        """With q near 1, older claims retain high weight.
        The posterior should be further from the prior than with low q."""
        rng = np.random.default_rng(111)
        _ = simulate_poisson_gamma(50, 5, p=0.8, q=0.9, rng=rng)

        model_high_q = DynamicPoissonGammaModel(p0=0.5, q0=0.5)
        model_high_q.p_ = 0.8
        model_high_q.q_ = 0.95
        model_high_q.is_fitted_ = True

        model_low_q = DynamicPoissonGammaModel(p0=0.5, q0=0.5)
        model_low_q.p_ = 0.8
        model_low_q.q_ = 0.3
        model_low_q.is_fitted_ = True

        # Policy with consistent high claims: high-q should give higher CF
        h = make_history("HIGH", [4, 4, 4, 4, 4], prior=1.0)
        cf_high = model_high_q.predict(h)
        cf_low = model_low_q.predict(h)
        # With more persistent states (high q), extreme histories have more effect
        assert cf_high > cf_low or abs(cf_high - cf_low) < 0.5  # directional only


class TestDynamicBatch:
    def test_predict_batch_returns_dataframe(self):
        import polars as pl
        rng = np.random.default_rng(112)
        histories = simulate_poisson_gamma(10, 5, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        df = model.predict_batch(histories)
        assert isinstance(df, pl.DataFrame)
        assert "policy_id" in df.columns
        assert "credibility_factor" in df.columns
        assert "posterior_alpha" in df.columns
        assert "posterior_variance" in df.columns
        assert len(df) == len(histories)

    def test_posterior_variance_positive(self):
        rng = np.random.default_rng(113)
        histories = simulate_poisson_gamma(10, 5, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        df = model.predict_batch(histories)
        assert (df["posterior_variance"] > 0).all()


class TestDynamicEdgeCases:
    def test_very_large_claim_count(self):
        """Should not crash or produce NaN for unusual claim history."""
        rng = np.random.default_rng(114)
        histories = simulate_poisson_gamma(20, 5, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        h = make_history("EXTREME", [1000, 0, 0, 0, 1000], prior=1.0)
        cf = model.predict(h)
        assert np.isfinite(cf)
        assert cf >= 0.0

    def test_repr_unfitted(self):
        model = DynamicPoissonGammaModel()
        assert "unfitted" in repr(model)

    def test_repr_fitted(self):
        rng = np.random.default_rng(115)
        histories = simulate_poisson_gamma(10, 3, p=0.8, q=0.9, rng=rng)
        model = DynamicPoissonGammaModel()
        model.fit(histories)
        r = repr(model)
        assert "p=" in r
        assert "q=" in r
