"""Tests for balance calibration functions."""

import math
import numpy as np
import pytest
import polars as pl
from insurance_experience import (
    ClaimsHistory,
    StaticCredibilityModel,
    CalibrationResult,
    balance_calibrate,
    apply_calibration,
    calibrated_predict_fn,
    balance_report,
)


def make_portfolio(n: int, rng: np.random.Generator, mean: float = 1.0) -> list[ClaimsHistory]:
    histories = []
    for i in range(n):
        counts = rng.poisson(mean, size=4).tolist()
        histories.append(
            ClaimsHistory(
                f"P{i}",
                [1, 2, 3, 4],
                counts,
                prior_premium=mean,
            )
        )
    return histories


class TestBalanceCalibrate:
    def test_returns_calibration_result(self):
        rng = np.random.default_rng(300)
        histories = make_portfolio(30, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        result = balance_calibrate(model.predict, histories)
        assert isinstance(result, CalibrationResult)

    def test_calibration_factor_finite_and_positive(self):
        rng = np.random.default_rng(301)
        histories = make_portfolio(30, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        result = balance_calibrate(model.predict, histories)
        assert result.calibration_factor > 0
        assert math.isfinite(result.calibration_factor)

    def test_n_policies_correct(self):
        rng = np.random.default_rng(302)
        histories = make_portfolio(25, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        result = balance_calibrate(model.predict, histories)
        assert result.n_policies == 25

    def test_identity_model_factor_near_one(self):
        """If predict returns CF=1 for all policies, calibration = actual/prior."""
        histories = [
            ClaimsHistory("P1", [1, 2], [1, 1], prior_premium=1.0),
            ClaimsHistory("P2", [1, 2], [1, 1], prior_premium=1.0),
        ]
        # Identity predict: CF = 1
        result = balance_calibrate(lambda h: 1.0, histories)
        # actual frequency = 1.0 (2 claims / 2 exposure), prior = 1.0
        # so calibration factor should be ~ 1.0
        assert result.calibration_factor == pytest.approx(1.0, abs=0.01)

    def test_sum_actual_matches_observed(self):
        rng = np.random.default_rng(303)
        histories = make_portfolio(20, rng, mean=1.5)
        model = StaticCredibilityModel()
        model.fit(histories)
        result = balance_calibrate(model.predict, histories)
        # sum_actual should be sum of (claim_freq * exposure) for each policy
        expected_actual = sum(h.claim_frequency * h.total_exposure for h in histories)
        assert result.sum_actual == pytest.approx(expected_actual, rel=1e-5)


class TestApplyCalibration:
    def test_apply_scales_cf(self):
        result = CalibrationResult(
            calibration_factor=1.2,
            sum_actual=120.0,
            sum_predicted=100.0,
            n_policies=10,
        )
        cf = 0.9
        calibrated = apply_calibration(cf, result)
        assert calibrated == pytest.approx(0.9 * 1.2, rel=1e-6)

    def test_apply_with_factor_one_unchanged(self):
        result = CalibrationResult(1.0, 100.0, 100.0, 10)
        cf = 1.5
        assert apply_calibration(cf, result) == pytest.approx(1.5)


class TestCalibratedPredictFn:
    def test_wraps_predict_correctly(self):
        rng = np.random.default_rng(304)
        histories = make_portfolio(30, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        result = balance_calibrate(model.predict, histories)
        calibrated = calibrated_predict_fn(model.predict, result)

        h = histories[0]
        raw_cf = model.predict(h)
        cal_cf = calibrated(h)
        assert cal_cf == pytest.approx(raw_cf * result.calibration_factor, rel=1e-6)

    def test_calibrated_fn_applies_to_all(self):
        rng = np.random.default_rng(305)
        histories = make_portfolio(20, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        result = balance_calibrate(model.predict, histories)
        calibrated = calibrated_predict_fn(model.predict, result)

        for h in histories:
            cf_cal = calibrated(h)
            assert cf_cal >= 0.0


class TestBalanceProperty:
    def test_after_calibration_sum_matches_actual(self):
        """After applying calibration factor, portfolio sum should balance."""
        rng = np.random.default_rng(306)
        histories = make_portfolio(50, rng, mean=1.0)
        model = StaticCredibilityModel()
        model.fit(histories)

        result = balance_calibrate(model.predict, histories)
        calibrated = calibrated_predict_fn(model.predict, result)

        # Recompute balance with calibrated predictions
        sum_actual = 0.0
        sum_predicted = 0.0
        for h in histories:
            cf = calibrated(h)
            posterior = h.prior_premium * cf
            weight = h.total_exposure
            sum_actual += h.claim_frequency * weight
            sum_predicted += posterior * weight

        assert sum_predicted == pytest.approx(sum_actual, rel=1e-5)


class TestBalanceReport:
    def test_returns_dataframe(self):
        rng = np.random.default_rng(307)
        histories = make_portfolio(20, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        df = balance_report(model.predict, histories)
        assert isinstance(df, pl.DataFrame)
        assert "policy_id" in df.columns
        assert "credibility_factor" in df.columns
        assert "residual" in df.columns

    def test_by_n_periods_groupby(self):
        rng = np.random.default_rng(308)
        histories = make_portfolio(30, rng)
        model = StaticCredibilityModel()
        model.fit(histories)
        df = balance_report(model.predict, histories, by_n_periods=True)
        assert isinstance(df, pl.DataFrame)
        assert "n_periods" in df.columns
        assert "mean_cf" in df.columns
        assert "n_policies" in df.columns

    def test_residual_column_finite(self):
        rng = np.random.default_rng(309)
        histories = make_portfolio(15, rng, mean=1.5)
        model = StaticCredibilityModel()
        model.fit(histories)
        df = balance_report(model.predict, histories)
        residuals = df["residual"].to_list()
        assert all(math.isfinite(r) for r in residuals if not math.isnan(r))
