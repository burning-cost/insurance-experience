"""Tests for ClaimsHistory and CalibrationResult data types."""

import pytest
from insurance_experience import ClaimsHistory, CalibrationResult


class TestClaimsHistoryConstruction:
    def test_basic_construction(self):
        h = ClaimsHistory(
            policy_id="POL001",
            periods=[1, 2, 3],
            claim_counts=[0, 1, 0],
            prior_premium=400.0,
        )
        assert h.policy_id == "POL001"
        assert h.n_periods == 3
        assert h.total_claims == 1

    def test_default_exposures(self):
        h = ClaimsHistory("P1", [1, 2], [0, 1], prior_premium=100.0)
        assert h.exposures == [1.0, 1.0]
        assert h.total_exposure == 2.0

    def test_custom_exposures(self):
        h = ClaimsHistory(
            "P1", [1, 2], [0, 1], exposures=[0.5, 0.75], prior_premium=100.0
        )
        assert h.total_exposure == pytest.approx(1.25)

    def test_claim_amounts_optional(self):
        h = ClaimsHistory("P1", [1], [2], prior_premium=100.0)
        assert h.claim_amounts is None

    def test_claim_amounts_provided(self):
        h = ClaimsHistory("P1", [1], [2], claim_amounts=[500.0], prior_premium=100.0)
        assert h.claim_amounts == [500.0]

    def test_single_period(self):
        h = ClaimsHistory("P1", [1], [3], prior_premium=100.0)
        assert h.n_periods == 1
        assert h.total_claims == 3

    def test_zero_claims(self):
        h = ClaimsHistory("P1", [1, 2, 3], [0, 0, 0], prior_premium=100.0)
        assert h.total_claims == 0
        assert h.claim_frequency == pytest.approx(0.0)


class TestClaimsHistoryValidation:
    def test_empty_periods_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ClaimsHistory("P1", [], [], prior_premium=100.0)

    def test_mismatched_counts_raises(self):
        with pytest.raises(ValueError, match="claim_counts length"):
            ClaimsHistory("P1", [1, 2], [0], prior_premium=100.0)

    def test_mismatched_amounts_raises(self):
        with pytest.raises(ValueError, match="claim_amounts length"):
            ClaimsHistory("P1", [1, 2], [0, 0], claim_amounts=[100.0], prior_premium=100.0)

    def test_mismatched_exposures_raises(self):
        with pytest.raises(ValueError, match="exposures length"):
            ClaimsHistory("P1", [1, 2], [0, 0], exposures=[1.0], prior_premium=100.0)

    def test_negative_exposure_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ClaimsHistory("P1", [1], [0], exposures=[-1.0], prior_premium=100.0)

    def test_zero_exposure_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ClaimsHistory("P1", [1], [0], exposures=[0.0], prior_premium=100.0)

    def test_negative_claims_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ClaimsHistory("P1", [1], [-1], prior_premium=100.0)

    def test_non_positive_prior_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ClaimsHistory("P1", [1], [0], prior_premium=0.0)

    def test_duplicate_periods_raises(self):
        with pytest.raises(ValueError, match="unique"):
            ClaimsHistory("P1", [1, 1], [0, 0], prior_premium=100.0)

    def test_negative_prior_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ClaimsHistory("P1", [1], [0], prior_premium=-1.0)


class TestClaimsHistoryProperties:
    def test_claim_frequency(self):
        h = ClaimsHistory("P1", [1, 2], [4, 2], exposures=[2.0, 1.0], prior_premium=100.0)
        assert h.claim_frequency == pytest.approx(6.0 / 3.0)

    def test_exposure_weighted_counts(self):
        h = ClaimsHistory("P1", [1, 2], [4, 2], exposures=[2.0, 1.0], prior_premium=100.0)
        rates = h.exposure_weighted_counts
        assert rates[0] == pytest.approx(2.0)
        assert rates[1] == pytest.approx(2.0)

    def test_total_claims_property(self):
        h = ClaimsHistory("P1", [1, 2, 3], [1, 2, 3], prior_premium=100.0)
        assert h.total_claims == 6


class TestCalibrationResult:
    def test_relative_bias_zero(self):
        r = CalibrationResult(1.0, 100.0, 100.0, 50)
        assert r.relative_bias == pytest.approx(0.0)

    def test_relative_bias_positive(self):
        r = CalibrationResult(0.9, 100.0, 110.0, 50)
        assert r.relative_bias == pytest.approx(0.1)

    def test_relative_bias_negative(self):
        r = CalibrationResult(1.1, 100.0, 90.0, 50)
        assert r.relative_bias == pytest.approx(-0.1)

    def test_zero_actual_gives_nan(self):
        import math
        r = CalibrationResult(1.0, 0.0, 10.0, 10)
        assert math.isnan(r.relative_bias)
