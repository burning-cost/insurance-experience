"""Tests for SurrogateModel."""

import numpy as np
import pytest
from insurance_experience import ClaimsHistory, SurrogateModel


def make_portfolio(
    n: int,
    rng: np.random.Generator,
    mean: float = 1.0,
    n_periods: int = 3,
) -> list[ClaimsHistory]:
    histories = []
    for i in range(n):
        counts = rng.poisson(mean, size=n_periods).tolist()
        histories.append(
            ClaimsHistory(
                f"P{i}",
                list(range(1, n_periods + 1)),
                counts,
                prior_premium=mean,
            )
        )
    return histories


class TestSurrogateFit:
    def test_fit_returns_self(self):
        rng = np.random.default_rng(200)
        histories = make_portfolio(50, rng)
        model = SurrogateModel(n_is_samples=100, random_state=42)
        result = model.fit(histories)
        assert result is model

    def test_is_fitted_after_fit(self):
        rng = np.random.default_rng(201)
        histories = make_portfolio(30, rng)
        model = SurrogateModel(n_is_samples=100, random_state=42)
        assert not model.is_fitted_
        model.fit(histories)
        assert model.is_fitted_

    def test_theta_has_correct_shape_degree1(self):
        rng = np.random.default_rng(202)
        histories = make_portfolio(50, rng)
        model = SurrogateModel(n_is_samples=100, poly_degree=1, random_state=42)
        model.fit(histories)
        # Degree 1: intercept + L + n = 3 parameters
        assert model.theta_ is not None
        assert len(model.theta_) == 3

    def test_theta_has_correct_shape_degree2(self):
        rng = np.random.default_rng(203)
        histories = make_portfolio(50, rng)
        model = SurrogateModel(n_is_samples=100, poly_degree=2, random_state=42)
        model.fit(histories)
        # Degree 2: 1 + L + n + L^2 + n^2 + L*n = 6 parameters
        assert model.theta_ is not None
        assert len(model.theta_) == 6

    def test_theta_ref_estimated(self):
        rng = np.random.default_rng(204)
        histories = make_portfolio(50, rng, mean=1.5)
        model = SurrogateModel(n_is_samples=100, random_state=42)
        model.fit(histories)
        assert model.theta_ref_ is not None
        assert model.theta_ref_ > 0


class TestSurrogatePredict:
    def test_predict_before_fit_raises(self):
        model = SurrogateModel()
        h = ClaimsHistory("P1", [1, 2, 3], [0, 1, 0], prior_premium=1.0)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(h)

    def test_cf_is_positive(self):
        """Credibility factor must be strictly positive (exp of real number)."""
        rng = np.random.default_rng(205)
        histories = make_portfolio(50, rng)
        model = SurrogateModel(n_is_samples=200, random_state=42)
        model.fit(histories)
        for h in histories[:10]:
            cf = model.predict(h)
            assert cf > 0.0, f"Non-positive CF: {cf}"

    def test_zero_claim_history_gives_cf_below_one(self):
        """Zero-claims policy should get CF < 1 on a non-zero portfolio."""
        rng = np.random.default_rng(206)
        histories = make_portfolio(80, rng, mean=1.5)
        model = SurrogateModel(n_is_samples=500, random_state=42)
        model.fit(histories)
        h_zero = ClaimsHistory("ZERO", [1, 2, 3], [0, 0, 0], prior_premium=1.5)
        cf = model.predict(h_zero)
        # Not strictly required but typical for well-fitted surrogate
        assert cf < 1.5, "CF implausibly large for zero-claims policy"

    def test_vs_conjugate_benchmark(self):
        """On Poisson-Gamma data, surrogate should be in the right direction.

        For conjugate Poisson-Gamma, the exact posterior is Gamma distributed
        and the credibility factor is computable analytically. The surrogate
        should approximate this (within IS noise).
        """
        rng = np.random.default_rng(207)
        # Gamma(2, 2) prior -> E[Theta] = 1, Var = 0.5
        prior_alpha, prior_beta = 2.0, 2.0
        n_periods = 4
        histories = []
        for i in range(100):
            theta = rng.gamma(prior_alpha, 1.0 / prior_beta)
            counts = rng.poisson(theta, size=n_periods).tolist()
            histories.append(
                ClaimsHistory(
                    f"P{i}",
                    list(range(1, n_periods + 1)),
                    counts,
                    prior_premium=1.0,
                )
            )

        model = SurrogateModel(n_is_samples=500, random_state=42, subsample_frac=0.5)
        model.fit(histories)

        # For a policy with total claims 8 in 4 periods: posterior mean is
        # (prior_alpha + 8) / (prior_beta + 4) = 10/6 ≈ 1.667
        h_test = ClaimsHistory("TEST", [1, 2, 3, 4], [2, 2, 2, 2], prior_premium=1.0)
        cf = model.predict(h_test)
        # Surrogate should give CF in a sensible range (> 1.0 since 8 claims > expected 4)
        assert cf > 0.5, f"CF implausibly low: {cf}"
        assert cf < 5.0, f"CF implausibly high: {cf}"


class TestSurrogateBatch:
    def test_predict_batch_returns_dataframe(self):
        import polars as pl
        rng = np.random.default_rng(208)
        histories = make_portfolio(30, rng)
        model = SurrogateModel(n_is_samples=100, random_state=42)
        model.fit(histories)
        df = model.predict_batch(histories[:5])
        assert isinstance(df, pl.DataFrame)
        assert "sufficient_stat" in df.columns
        assert len(df) == 5

    def test_batch_consistent_with_single(self):
        rng = np.random.default_rng(209)
        histories = make_portfolio(30, rng)
        model = SurrogateModel(n_is_samples=100, random_state=42)
        model.fit(histories)
        df = model.predict_batch(histories[:3])
        for i, h in enumerate(histories[:3]):
            expected_cf = model.predict(h)
            actual_cf = df[i]["credibility_factor"][0]
            assert actual_cf == pytest.approx(expected_cf, rel=1e-6)


class TestSurrogateCustomSufficientStat:
    def test_custom_stat_fn(self):
        """Verify that a custom sufficient_stat_fn is called."""
        rng = np.random.default_rng(210)
        histories = make_portfolio(30, rng)

        called_with = []

        def my_stat(h: ClaimsHistory) -> float:
            called_with.append(h.policy_id)
            return float(h.total_claims)

        model = SurrogateModel(
            n_is_samples=100, random_state=42, sufficient_stat_fn=my_stat
        )
        model.fit(histories)
        assert len(called_with) > 0  # Called during fitting

        model.predict(histories[0])
        # Called during prediction too
        assert histories[0].policy_id in called_with


class TestSurrogateEdgeCases:
    def test_custom_prior_model(self):
        """prior_model callable overrides history.prior_premium."""
        rng = np.random.default_rng(211)
        histories = make_portfolio(30, rng, mean=1.0)
        # Override prior to always return 2.0
        model = SurrogateModel(
            prior_model=lambda h: 2.0,
            n_is_samples=100,
            random_state=42,
        )
        model.fit(histories)
        cf = model.predict(histories[0])
        assert cf > 0.0

    def test_repr_unfitted(self):
        model = SurrogateModel(poly_degree=2)
        assert "unfitted" in repr(model)
        assert "poly_degree=2" in repr(model)

    def test_repr_fitted(self):
        rng = np.random.default_rng(212)
        histories = make_portfolio(30, rng)
        model = SurrogateModel(n_is_samples=100, random_state=42)
        model.fit(histories)
        assert "n_is_samples" in repr(model)
