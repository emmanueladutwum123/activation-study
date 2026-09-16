"""Run the whole study and write `results/findings.json` plus the figures.

Every number quoted in `docs/` and `README.md` comes out of this file. The JSON is
committed so a reader can diff findings across changes to the analysis, and so the
tests can assert the headline results have not silently moved.

Run: ``make study``  (or ``PYTHONPATH=src python analysis/run_study.py``)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from activation import analysis, experiment, features

RESULTS = Path("results")
FIGURES = Path("figures")


def signup_rate(users: pd.DataFrame) -> dict:
    """Actual new-user rate, needed to turn a sample size into a calendar date.

    Measured over the last two full years rather than the whole history: a product's
    current traffic is what an experiment runs on, and averaging over fourteen years
    would flatter a service whose early growth is long past.
    """
    first = pd.to_datetime(users["first_date"])
    last_year = int(first.dt.year.max())
    window = users[first.dt.year.between(last_year - 2, last_year - 1)]
    days = 365.25 * 2
    return {
        "window_years": f"{last_year - 2}-{last_year - 1}",
        "users_in_window": int(len(window)),
        "users_per_day": float(len(window) / days),
    }


def make_figures(retention, by_sessions, volume_control, stability) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })
    FIGURES.mkdir(exist_ok=True)

    # 1. The shape of the problem.
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(retention["day"], retention["share"] * 100, marker="o", color="#1f4e79")
    ax.set_xscale("log")
    ax.set_xticks(retention["day"])
    ax.set_xticklabels(retention["day"])
    ax.set_xlabel("days since first activity")
    ax.set_ylabel("still active (%)")
    ax.set_title("Retention decays fast and then flattens", loc="left", fontsize=10)
    fig.savefig(FIGURES / "01_retention_curve.png")
    plt.close(fig)

    # 2. The signal, and that it lasts.
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = {1: "#b03030", 2: "#d18b2c", 3: "#1f7a3f"}
    for sessions, chunk in by_sessions.groupby("d0_sessions"):
        ax.plot(chunk["day"], chunk["share"] * 100, marker="o",
                color=colors.get(sessions, "#555"),
                label=f"{sessions}{'+' if sessions == 3 else ''} day-0 session"
                      f"{'s' if sessions > 1 else ''}")
    ax.set_xscale("log")
    ax.set_xticks(sorted(by_sessions["day"].unique()))
    ax.set_xticklabels(sorted(by_sessions["day"].unique()))
    ax.set_xlabel("days since first activity")
    ax.set_ylabel("still active (%)")
    ax.set_title("A second session on day one separates users for a year",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "02_retention_by_sessions.png")
    plt.close(fig)

    # 3. The confounding check -- the most important chart in the study.
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(volume_control))
    ax.bar(x - 0.2, volume_control["return_single"] * 100, 0.4,
           label="1 session", color="#b03030")
    ax.bar(x + 0.2, volume_control["return_multi"] * 100, 0.4,
           label="2+ sessions", color="#1f7a3f")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n(med. {int(m)} ratings)" for q, m in
                        zip(volume_control["volume_quartile"],
                            volume_control["median_d0_ratings"],
                            strict=True)], fontsize=8)
    ax.set_xlabel("day-0 rating volume quartile")
    ax.set_ylabel("returned by day 7 (%)")
    ax.set_title("The gap survives inside every volume quartile",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "03_sessions_vs_volume.png")
    plt.close(fig)

    # 4. Does it hold across eras.
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(stability["year"], stability["lift_x"], marker="o", color="#1f4e79")
    ax.axhline(1.0, color="#999", linestyle="--", linewidth=1)
    ax.set_ylim(0, max(3.6, stability["lift_x"].max() * 1.15))
    ax.set_xlabel("signup year")
    ax.set_ylabel("return rate lift (x)")
    ax.set_title("Holds in every cohort from 1996 to 2008", loc="left", fontsize=10)
    fig.savefig(FIGURES / "04_stability_by_year.png")
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    ratings, users = features.load()

    table = features.build(ratings, users)
    population = features.eligible(table)

    print(f"users total        : {len(table):,}")
    print(f"eligible population: {len(population):,} "
          f"({len(population)/len(table):.1%})")

    retention = analysis.retention_curve(ratings, population)
    by_sessions = analysis.retention_by_sessions(ratings, population)
    volume_control = analysis.sessions_controlling_for_volume(population)
    model = analysis.logistic_model(population)
    stability = analysis.stability_by_year(population)
    sizing = analysis.opportunity(population)
    sensitivity = features.session_gap_sensitivity(ratings, users)
    traffic = signup_rate(users)

    # Experiment design, sized against the population the intervention targets:
    # single-session users, whose return rate is the baseline we would move.
    baseline = sizing["return_rate_single"]

    # The intervention is *triggered*: it only fires for a user who finishes a first
    # session and does not come back on their own. So enrolment is all new users, but
    # only the triggered share is analysable -- and the trigger has to be evaluated
    # identically in both arms. Defining the analysis group by anything that happens
    # after treatment (say, "users who opened the email") would reintroduce exactly the
    # selection this experiment exists to eliminate.
    trigger_share = sizing["single_session_share"]
    enrolled_per_day = traffic["users_per_day"] * trigger_share

    designs = {}
    for mde_pp in (1.0, 2.0, 3.0, 5.0):
        plan = experiment.sample_size_for_proportions(baseline, mde_pp / 100)
        days = plan.days_at(enrolled_per_day)
        designs[f"mde_{mde_pp:g}pp"] = {
            "absolute_mde_pp": mde_pp,
            "relative_mde": plan.relative_mde,
            "per_arm_triggered": plan.per_arm,
            "total_triggered": plan.total,
            "total_enrolled": int(round(plan.total / trigger_share)),
            "days_at_current_traffic": days,
            "years_at_current_traffic": days / 365.25,
        }

    # The inverse question, which is the one that actually matters here: given a
    # quarter, what could this product prove?
    quarter_triggered = int(enrolled_per_day * 90)
    designs["what_one_quarter_buys"] = {
        "triggered_users_in_90_days": quarter_triggered,
        "per_arm": quarter_triggered // 2,
        "detectable_absolute_pp": float(
            experiment.detectable_effect(max(quarter_triggered // 2, 1), baseline) * 100),
    }

    findings = {
        "dataset": {
            "name": "MovieLens 10M",
            "ratings": int(len(ratings)),
            "users_total": int(len(table)),
            "users_eligible": int(len(population)),
            "eligibility_rule": f"at least {features.MIN_DAY0_RATINGS} ratings on day 0",
            "cohort_span": [str(users["first_date"].min().date()),
                            str(users["first_date"].max().date())],
        },
        "retention_curve": retention.to_dict(orient="records"),
        "retention_by_sessions": by_sessions.to_dict(orient="records"),
        "sessions_controlling_for_volume": volume_control.to_dict(orient="records"),
        "logistic_model": model,
        "stability_by_year": stability.to_dict(orient="records"),
        "session_gap_sensitivity": sensitivity.to_dict(orient="records"),
        "opportunity": sizing,
        "traffic": traffic,
        "experiment_designs": designs,
        "peeking_false_positive_rate": {
            "one_look": experiment.peeking_inflation(1),
            "daily_for_14_days": experiment.peeking_inflation(14),
        },
    }

    (RESULTS / "findings.json").write_text(json.dumps(findings, indent=2))
    make_figures(retention, by_sessions, volume_control, stability)

    print("\n--- headline ---")
    print(f"single-session share : {sizing['single_session_share']:.1%}")
    print(f"return rate, 1 sess  : {sizing['return_rate_single']:.1%}")
    print(f"return rate, 2+ sess : {sizing['return_rate_multi']:.1%}")
    print(f"odds ratio (adjusted): {model['terms']['multi_session']['odds_ratio']:.2f}")
    plan_2pp = designs["mde_2pp"]
    quarter = designs["what_one_quarter_buys"]
    print(f"test for +2pp        : {plan_2pp['per_arm_triggered']:,}/arm, "
          f"{plan_2pp['years_at_current_traffic']:.1f} years at current traffic")
    print(f"one quarter buys     : MDE of "
          f"{quarter['detectable_absolute_pp']:.1f}pp "
          f"({quarter['detectable_absolute_pp']/(baseline*100):.0%} relative)")
    print(f"\nwrote {RESULTS}/findings.json and {FIGURES}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
