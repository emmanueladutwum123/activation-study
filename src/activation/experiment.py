"""Experiment design: how large a test has to be, and how long it would take.

The analysis in this study is observational. It can rank hypotheses; it cannot settle
one. This module turns the hypothesis into a test plan with an honest answer to the only
question that decides whether the test is worth running: **is it even feasible at this
product's traffic?**

Most portfolio experiment sections stop at "we would A/B test it". The interesting part
is almost always that you cannot -- not at the effect size you care about, not at the
traffic you have, not in a quarter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SampleSize:
    per_arm: int
    total: int
    baseline_rate: float
    target_rate: float
    absolute_mde: float
    relative_mde: float
    alpha: float
    power: float

    def days_at(self, users_per_day: float) -> float:
        return self.total / users_per_day if users_per_day > 0 else float("inf")


def sample_size_for_proportions(baseline: float, absolute_mde: float,
                                alpha: float = 0.05, power: float = 0.80,
                                arms: int = 2) -> SampleSize:
    """Users per arm needed to detect `absolute_mde` on a binary metric.

    Uses the unpooled normal approximation. With rates near 20% and thousands of users
    per arm the approximation is well inside its valid range, and quoting a more exact
    method would imply a precision the inputs do not have -- the baseline itself is an
    estimate.

    `arms` above 2 applies a Bonferroni correction to alpha. Running four variants and
    testing each at 0.05 gives roughly a 1-in-5 chance of a false winner somewhere,
    which is how teams ship things that do nothing.
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must be a proportion in (0,1)")
    if absolute_mde <= 0:
        raise ValueError("absolute_mde must be positive")

    comparisons = max(1, arms - 1)
    adjusted_alpha = alpha / comparisons

    target = baseline + absolute_mde
    if not 0 < target < 1:
        raise ValueError("baseline + mde must stay inside (0,1)")

    z_alpha = stats.norm.ppf(1 - adjusted_alpha / 2)
    z_beta = stats.norm.ppf(power)
    variance = baseline * (1 - baseline) + target * (1 - target)
    per_arm = int(np.ceil((z_alpha + z_beta) ** 2 * variance / absolute_mde ** 2))

    return SampleSize(
        per_arm=per_arm,
        total=per_arm * arms,
        baseline_rate=baseline,
        target_rate=target,
        absolute_mde=absolute_mde,
        relative_mde=absolute_mde / baseline,
        alpha=adjusted_alpha,
        power=power,
    )


def detectable_effect(per_arm: int, baseline: float, alpha: float = 0.05,
                      power: float = 0.80) -> float:
    """Inverse question: given the traffic we have, what is the smallest real effect
    we could detect? This is the number to look at when the sample size comes back
    impossible -- it says what the test would actually be able to prove."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    # Solved under the approximation that variance is roughly constant near baseline.
    variance = 2 * baseline * (1 - baseline)
    return float((z_alpha + z_beta) * np.sqrt(variance / per_arm))


def srm_check(counts: dict[str, int], expected: dict[str, float] | None = None,
              threshold: float = 0.001) -> dict:
    """Sample Ratio Mismatch: did randomisation actually split the way we asked?

    This is the first thing to check on any readout and the most commonly skipped. A
    50/50 assignment that lands 49.3/50.7 on millions of users is not bad luck -- it
    means some users were dropped non-randomly (a redirect that failed for one arm, a
    client that crashed on the treatment). When SRM fires, the result is not "slightly
    off"; it is uninterpretable, because the arms are no longer comparable populations.

    Threshold is deliberately strict: a p-value below 0.001 stops the readout.
    """
    labels = list(counts)
    observed = np.array([counts[k] for k in labels], dtype=float)
    if expected is None:
        expected_share = np.ones(len(labels)) / len(labels)
    else:
        expected_share = np.array([expected[k] for k in labels], dtype=float)
        expected_share = expected_share / expected_share.sum()

    expected_counts = expected_share * observed.sum()
    chi2 = float(((observed - expected_counts) ** 2 / expected_counts).sum())
    p_value = float(stats.chi2.sf(chi2, df=len(labels) - 1))
    return {
        "chi2": chi2,
        "p_value": p_value,
        "passed": p_value >= threshold,
        "observed": {k: int(v) for k, v in zip(labels, observed, strict=True)},
        "expected": {k: float(v) for k, v in zip(labels, expected_counts, strict=True)},
    }


def two_proportion_test(successes_a: int, n_a: int, successes_b: int, n_b: int,
                        alpha: float = 0.05) -> dict:
    """Readout for a binary primary metric, with a confidence interval on the
    difference. The interval matters more than the p-value: "significant" says only
    that the effect is probably not zero, while the interval says whether it is large
    enough to be worth the cost of shipping."""
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))

    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    difference = p_b - p_a
    return {
        "rate_control": float(p_a),
        "rate_treatment": float(p_b),
        "absolute_difference": float(difference),
        "relative_difference": float(difference / p_a) if p_a else float("nan"),
        "ci_low": float(difference - z_crit * se_unpooled),
        "ci_high": float(difference + z_crit * se_unpooled),
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def peeking_inflation(looks: int) -> float:
    """Roughly how much the false-positive rate inflates if you test at every look and
    stop at the first significant result.

    Included because "we'll watch the dashboard daily and call it when it goes green"
    is the single most common way an experiment programme produces confident nonsense.
    A 14-day test peeked at daily runs near a 1-in-3 false-positive rate, not 1-in-20.
    The fix is a fixed horizon or a sequential test, not discipline.
    """
    # Armitage's classic simulation result for repeated testing at alpha=0.05.
    known = {1: 0.05, 2: 0.08, 3: 0.11, 5: 0.14, 10: 0.19, 20: 0.25, 50: 0.32}
    if looks in known:
        return known[looks]
    keys = sorted(known)
    if looks < keys[0]:
        return known[keys[0]]
    if looks > keys[-1]:
        return known[keys[-1]]
    lower = max(k for k in keys if k <= looks)
    upper = min(k for k in keys if k >= looks)
    if lower == upper:
        return known[lower]
    weight = (looks - lower) / (upper - lower)
    return known[lower] + weight * (known[upper] - known[lower])
