"""Tests for the experiment-design maths.

These are checked against values a reader can verify independently, because the whole
recommendation in this study turns on one number -- how long a test would take -- and a
silent arithmetic error there would invert the conclusion.
"""

from __future__ import annotations

import math

import pytest

from activation.experiment import (
    detectable_effect,
    peeking_inflation,
    sample_size_for_proportions,
    srm_check,
    two_proportion_test,
)


class TestSampleSize:
    def test_matches_hand_calculation(self):
        """Baseline 20%, detect +2pp, alpha 0.05, power 80%.

        n = (1.96 + 0.8416)^2 * [0.2(0.8) + 0.22(0.78)] / 0.02^2 = 6,507 per arm.
        Any implementation that disagrees with this by more than rounding is wrong.
        """
        plan = sample_size_for_proportions(0.20, 0.02)
        assert plan.per_arm == pytest.approx(6507, rel=0.01)
        assert plan.total == 2 * plan.per_arm
        assert plan.relative_mde == pytest.approx(0.10)

    def test_smaller_effects_cost_quadratically(self):
        """Halving the detectable effect roughly quadruples the sample. This is the
        single most important intuition in experiment design and the reason small
        products cannot test small effects."""
        big = sample_size_for_proportions(0.20, 0.04).per_arm
        small = sample_size_for_proportions(0.20, 0.02).per_arm
        assert small / big == pytest.approx(4.0, rel=0.05)

    def test_more_arms_cost_more_per_arm(self):
        """Bonferroni tightens alpha, so each arm needs more users. A four-arm test is
        not 'the same test with more options'."""
        two = sample_size_for_proportions(0.20, 0.02, arms=2).per_arm
        four = sample_size_for_proportions(0.20, 0.02, arms=4).per_arm
        assert four > two
        assert sample_size_for_proportions(0.20, 0.02, arms=4).alpha < 0.05

    def test_higher_power_costs_more(self):
        low = sample_size_for_proportions(0.20, 0.02, power=0.80).per_arm
        high = sample_size_for_proportions(0.20, 0.02, power=0.95).per_arm
        assert high > low

    def test_days_at_traffic(self):
        plan = sample_size_for_proportions(0.20, 0.02)
        assert plan.days_at(1000) == pytest.approx(plan.total / 1000)
        assert math.isinf(plan.days_at(0))

    @pytest.mark.parametrize("baseline,mde", [(0.0, 0.02), (1.0, 0.02), (0.2, 0.0),
                                              (0.2, -0.01), (0.99, 0.05)])
    def test_rejects_impossible_inputs(self, baseline, mde):
        with pytest.raises(ValueError):
            sample_size_for_proportions(baseline, mde)


class TestDetectableEffect:
    def test_round_trips_against_sample_size(self):
        """The inverse of the sizing calculation should land back near where it started."""
        plan = sample_size_for_proportions(0.20, 0.02)
        recovered = detectable_effect(plan.per_arm, 0.20)
        assert recovered == pytest.approx(0.02, rel=0.05)

    def test_more_traffic_detects_smaller_effects(self):
        assert detectable_effect(100_000, 0.20) < detectable_effect(1_000, 0.20)


class TestSampleRatioMismatch:
    def test_clean_split_passes(self):
        result = srm_check({"control": 50_012, "treatment": 49_988})
        assert result["passed"]
        assert result["p_value"] > 0.05

    def test_detects_a_real_mismatch(self):
        """49.3/50.7 on 100k users is not luck -- it means users were dropped
        non-randomly, and the readout is uninterpretable rather than merely noisy."""
        result = srm_check({"control": 49_300, "treatment": 50_700})
        assert not result["passed"]
        assert result["p_value"] < 0.001

    def test_supports_uneven_intended_splits(self):
        """A deliberate 90/10 ramp must not be flagged as a mismatch."""
        result = srm_check({"control": 90_000, "treatment": 10_000},
                           expected={"control": 0.9, "treatment": 0.1})
        assert result["passed"]


class TestReadout:
    def test_reports_effect_and_interval(self):
        result = two_proportion_test(2_000, 10_000, 2_200, 10_000)
        assert result["rate_control"] == pytest.approx(0.20)
        assert result["rate_treatment"] == pytest.approx(0.22)
        assert result["absolute_difference"] == pytest.approx(0.02)
        assert result["relative_difference"] == pytest.approx(0.10)
        assert result["ci_low"] < result["absolute_difference"] < result["ci_high"]
        assert result["significant"]

    def test_underpowered_test_is_not_significant(self):
        """The same 2pp effect on 500 users per arm is invisible. This is what running
        an undersized test buys you: not a wrong answer, but no answer."""
        result = two_proportion_test(100, 500, 110, 500)
        assert not result["significant"]
        assert result["ci_low"] < 0 < result["ci_high"]

    def test_no_difference_is_not_significant(self):
        result = two_proportion_test(2_000, 10_000, 2_000, 10_000)
        assert result["absolute_difference"] == pytest.approx(0.0)
        assert not result["significant"]


class TestPeeking:
    def test_single_look_is_nominal(self):
        assert peeking_inflation(1) == pytest.approx(0.05)

    def test_daily_peeking_inflates_false_positives(self):
        """A fortnight of daily dashboard checks turns a 1-in-20 false-positive rate
        into roughly 1-in-5."""
        assert peeking_inflation(14) > 0.15
        assert peeking_inflation(50) > peeking_inflation(14)

    def test_monotone(self):
        rates = [peeking_inflation(n) for n in (1, 2, 5, 10, 20, 50)]
        assert rates == sorted(rates)
