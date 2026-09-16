"""Guard the committed findings against silent drift.

`results/findings.json` is committed, and the README and memo quote it. That only means
anything if a change to the analysis that moves the headline is visible in review, so
these tests assert the claims the write-up actually makes -- not every number, which
would turn a refactor into a chore, but the ones a reader would act on.

They read the committed JSON rather than recomputing it, so they run in CI without the
89MB dataset. Regenerate with `make study` after any deliberate change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FINDINGS = Path(__file__).resolve().parents[1] / "results" / "findings.json"


@pytest.fixture(scope="module")
def findings() -> dict:
    return json.loads(FINDINGS.read_text())


def test_population_matches_the_eligibility_rule(findings):
    data = findings["dataset"]
    assert data["users_eligible"] < data["users_total"]
    assert data["eligibility_rule"] == "at least 20 ratings on day 0"


def test_headline_gap_is_what_the_writeup_claims(findings):
    opportunity = findings["opportunity"]
    assert opportunity["return_rate_single"] == pytest.approx(0.198, abs=0.005)
    assert opportunity["return_rate_multi"] == pytest.approx(0.524, abs=0.005)
    assert opportunity["single_session_share"] == pytest.approx(0.785, abs=0.005)


def test_gap_survives_inside_every_volume_quartile(findings):
    """The finding the whole study rests on. If any quartile inverts, the headline is
    volume in disguise and the recommendation changes."""
    quartiles = findings["sessions_controlling_for_volume"]
    assert len(quartiles) == 4
    for row in quartiles:
        assert row["return_multi"] > row["return_single"], row["volume_quartile"]
        assert row["lift_x"] > 1.5, row["volume_quartile"]


def test_lift_attenuates_as_volume_rises(findings):
    """Part of the gap *is* volume: the lift shrinks from Q1 to Q4. Saying so is what
    separates an honest read from an oversold one."""
    lifts = [row["lift_x"] for row in findings["sessions_controlling_for_volume"]]
    assert lifts == sorted(lifts, reverse=True)


def test_adjusted_effect_survives_the_model(findings):
    term = findings["logistic_model"]["terms"]["multi_session"]
    assert term["odds_ratio"] > 2.5
    assert term["ci_low_or"] > 1.0
    assert term["p_value"] < 1e-6


def test_model_explains_little_and_says_so(findings):
    """A pseudo-R2 near 0.1 is the honest headline, and the write-up quotes it. If a
    change pushes it far higher, something has leaked from the outcome window."""
    assert findings["logistic_model"]["pseudo_r2"] < 0.2


def test_effect_holds_in_every_cohort(findings):
    years = findings["stability_by_year"]
    assert len(years) >= 10
    assert all(row["lift_x"] > 1.5 for row in years)


def test_conclusion_does_not_depend_on_the_session_definition(findings):
    rows = findings["session_gap_sensitivity"]
    assert {row["gap_minutes"] for row in rows} == {10, 30, 60, 120}
    assert all(row["lift_x"] > 2.0 for row in rows)


def test_the_experiment_is_infeasible_at_this_traffic(findings):
    """The recommendation is 'do not run the obvious test'. It only stands while the
    2pp design still costs years."""
    assert findings["experiment_designs"]["mde_2pp"]["years_at_current_traffic"] > 2
    quarter = findings["experiment_designs"]["what_one_quarter_buys"]
    assert quarter["detectable_absolute_pp"] > 5


def test_every_opportunity_scenario_is_below_the_naive_bound(findings):
    opportunity = findings["opportunity"]
    ceiling = opportunity["naive_upper_bound_pp"]
    assert all(s["absolute_lift_pp"] < ceiling
               for s in opportunity["scenarios"].values())
