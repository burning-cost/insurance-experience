"""Tests for StaticCredibilityModel (Bühlmann-Straub at policy level)."""

import numpy as np
import pytest
from insurance_experience import ClaimsHistory, StaticCredibilityModel


def make_history(policy_id: str, counts: list[int], prior: float = 400.0) -> ClaimsHistory:
    return ClaimsHistory(
        policy_id=policy_id,
        periods=list(range(1, len(counts) + 1)),
        claim_counts=counts,
        prior_premium=prior,
    )


def make_portfolio(n: int, rng: np.random.Generator, mean: float = 1.0) -> list[ClaimsHistory]:
    """Generate a simple homogeneous portfolio."""
    histories = []
    for i in range(n):
        counts = rng.poisson(mean, size=3).tolist()
        histories.append(make_history(f"P{i}", counts, prior=mean))
    return histories


class TestStaticFit:
    def test_fit_returns_self(self):
        rng = np.random.default_rng(1)
        histories = make_portfolio(20, rng)
        model = StaticCredibilityModel()
        result = model.fit(histories)
        assert result is model

    def test_is_fitted_after_fit(self):
        rng = np.random.default_rng(2)
        histories = make_portfolio(10, rng)
        model = StaticCredibilityModel()
        assert not model.is_fitted_
        model.fit(histories)
        assert model.is_fitted_

    def test_kappa_is_positive(self):
        rng = np.random.default_rng(3)
        histories = make_portfolio(30, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        assert model.kappa_ > 0

    def test_too_few_histories_raises(self):
        model = StaticCredibilityModel()
        with pytest.raises(ValueError, match="At least 2"):
            model.fit([make_history("P1", [1, 2, 3])])

    def test_provided_kappa_not_reestimated(self):
        rng = np.random.default_rng(4)
        histories = make_portfolio(20, rng)
        model = StaticCredibilityModel(kappa=5.0)
        model.fit(histories)
        assert model.kappa_ == pytest.approx(5.0)

    def test_kappa_bounded(self):
        rng = np.random.default_rng(5)
        histories = make_portfolio(5, rng)  # small portfolio, potentially degenerate
        model = StaticCredibilityModel(min_kappa=0.1, max_kappa=100.0)
        model.fit(histories)
        assert model.kappa_ >= 0.1
        assert model.kappa_ <= 100.0


class TestStaticPredict:
    def test_predict_before_fit_raises(self):
        model = StaticCredibilityModel()
        h = make_history("P1", [0, 1, 0])
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(h)

    def test_credibility_factor_range(self):
        """CF should be non-negative for valid inputs."""
        rng = np.random.default_rng(6)
        histories = make_portfolio(30, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        for h in histories:
            cf = model.predict(h)
            assert cf >= 0.0, f"Negative CF: {cf}"

    def test_zero_claims_cf_less_than_one(self):
        """Policy with zero claims should get CF < 1 (better than prior)."""
        rng = np.random.default_rng(7)
        histories = make_portfolio(30, rng, mean=1.5)
        model = StaticCredibilityModel()
        model.fit(histories)
        zero_claim_history = make_history("NEW", [0, 0, 0], prior=1.5)
        cf = model.predict(zero_claim_history)
        assert cf < 1.0, f"Expected CF < 1 for zero claims, got {cf}"

    def test_high_claims_cf_greater_than_one(self):
        """Policy with many claims should get CF > 1 (worse than prior)."""
        rng = np.random.default_rng(8)
        histories = make_portfolio(30, rng, mean=1.0)
        model = StaticCredibilityModel()
        model.fit(histories)
        high_claim_history = make_history("BAD", [5, 5, 5], prior=1.0)
        cf = model.predict(high_claim_history)
        assert cf > 1.0, f"Expected CF > 1 for high claims, got {cf}"

    def test_single_period_history(self):
        """Should work with single-period policy."""
        rng = np.random.default_rng(9)
        histories = make_portfolio(20, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        single = make_history("SINGLE", [2])
        cf = model.predict(single)
        assert cf >= 0.0

    def test_credibility_weight_range(self):
        """omega_t should be in [0, 1]."""
        rng = np.random.default_rng(10)
        histories = make_portfolio(30, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        for h in histories:
            omega = model.credibility_weight(h)
            assert 0.0 <= omega <= 1.0


class TestStaticConsistency:
    def test_more_periods_higher_credibility(self):
        """Longer tenure should give higher credibility weight."""
        rng = np.random.default_rng(11)
        histories = make_portfolio(30, rng)
        model = StaticCredibilityModel()
        model.fit(histories)

        h_short = ClaimsHistory("SHORT", [1, 2], [1, 2, 3][:2], prior_premium=1.0)
        h_long = ClaimsHistory("LONG", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], prior_premium=1.0)
        omega_short = model.credibility_weight(h_short)
        omega_long = model.credibility_weight(h_long)
        assert omega_long > omega_short

    def test_prior_only_with_kappa_infinity(self):
        """With kappa -> infinity, CF should approach 1 (pure prior)."""
        rng = np.random.default_rng(12)
        histories = make_portfolio(20, rng)
        model = StaticCredibilityModel(kappa=1e6)
        model.fit(histories)
        h = make_history("P0", [5, 5, 5], prior=1.0)  # high claims
        cf = model.predict(h)
        assert cf == pytest.approx(1.0, abs=0.01), f"Expected CF ≈ 1, got {cf}"

    def test_kappa_recovery_heterogeneous_portfolio(self):
        """With clear heterogeneity, kappa should be low (high credibility)."""
        rng = np.random.default_rng(42)
        # Two risk classes: low (lambda=0.5) and high (lambda=2.0)
        histories = []
        for i in range(25):
            counts = rng.poisson(0.5, size=5).tolist()
            histories.append(make_history(f"L{i}", counts, prior=1.25))
        for i in range(25):
            counts = rng.poisson(2.0, size=5).tolist()
            histories.append(make_history(f"H{i}", counts, prior=1.25))
        model = StaticCredibilityModel()
        model.fit(histories)
        # Kappa should be reasonably estimated (not degenerate)
        assert model.kappa_ > 0
        assert model.kappa_ < 1000


class TestStaticBatch:
    def test_predict_batch_returns_dataframe(self):
        import polars as pl
        rng = np.random.default_rng(13)
        histories = make_portfolio(10, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        df = model.predict_batch(histories)
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) >= {"policy_id", "prior_premium", "credibility_factor", "posterior_premium"}
        assert len(df) == len(histories)

    def test_batch_posterior_equals_prior_times_cf(self):
        rng = np.random.default_rng(14)
        histories = make_portfolio(10, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        df = model.predict_batch(histories)
        for row in df.iter_rows(named=True):
            expected = row["prior_premium"] * row["credibility_factor"]
            assert row["posterior_premium"] == pytest.approx(expected, rel=1e-6)
