"""The analysis itself: retention curves, the activation signal, and the confounding
checks that decide whether the signal is worth acting on.

Everything returns plain dataframes and dicts so the runner can serialise them. The
numbers quoted in `docs/` are generated from here, not typed by hand -- a document that
quotes a stale figure is worse than one that quotes none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def retention_curve(ratings: pd.DataFrame, population: pd.DataFrame,
                    horizons=(1, 7, 14, 30, 60, 90, 180, 365)) -> pd.DataFrame:
    """Share of the population still active on or after each horizon."""
    rows = []
    total = len(population)
    ids = set(population["user_id"])
    for day in horizons:
        active = ratings.loc[ratings["days_since_first"] >= day, "user_id"].unique()
        alive = len(ids.intersection(active))
        rows.append({"day": day, "users_active": alive, "share": alive / total})
    return pd.DataFrame(rows)


def retention_by_sessions(ratings: pd.DataFrame, population: pd.DataFrame,
                          horizons=(7, 30, 90, 180, 365),
                          max_sessions: int = 3) -> pd.DataFrame:
    """Retention curve split by how many sessions the user had on day 0.

    The question this answers is durability: a day-0 signal that only predicts day-7
    behaviour is probably just measuring the same burst of activity twice. One that
    still separates users a year later is describing something real about them.
    """
    grouped = population.assign(
        bucket=population["d0_sessions"].clip(upper=max_sessions)
    )
    rows = []
    for day in horizons:
        active = set(ratings.loc[ratings["days_since_first"] >= day, "user_id"].unique())
        for bucket, chunk in grouped.groupby("bucket"):
            share = chunk["user_id"].isin(active).mean()
            rows.append({"day": day, "d0_sessions": int(bucket),
                         "users": len(chunk), "share": float(share)})
    return pd.DataFrame(rows)


def lift_by_bucket(population: pd.DataFrame, column: str, bins) -> pd.DataFrame:
    """Return rate by bucket of some day-0 feature."""
    working = population.copy()
    working["bucket"] = pd.cut(working[column], bins, right=False)
    table = working.groupby("bucket", observed=True).agg(
        users=("user_id", "size"),
        return_rate=("returned", "mean"),
    ).reset_index()
    table["bucket"] = table["bucket"].astype(str)
    return table


def sessions_controlling_for_volume(population: pd.DataFrame,
                                    quantiles: int = 4) -> pd.DataFrame:
    """The check that decides the whole study.

    Users with two day-0 sessions also rate more, so the raw session effect could be
    volume wearing a disguise. Slicing by volume quartile and comparing within each
    slice separates them: if the gap survives inside every quartile, session count is
    carrying information that volume is not.
    """
    working = population.copy()
    working["volume_quartile"] = pd.qcut(
        working["d0_ratings"], quantiles,
        labels=[f"Q{i+1}" for i in range(quantiles)])
    rows = []
    for quartile, chunk in working.groupby("volume_quartile", observed=True):
        single = chunk[chunk["multi_session"] == 0]
        multi = chunk[chunk["multi_session"] == 1]
        rows.append({
            "volume_quartile": str(quartile),
            "median_d0_ratings": float(chunk["d0_ratings"].median()),
            "n_single": len(single),
            "n_multi": len(multi),
            "return_single": float(single["returned"].mean()),
            "return_multi": float(multi["returned"].mean()),
            "lift_x": float(multi["returned"].mean() / single["returned"].mean()),
        })
    return pd.DataFrame(rows)


def logistic_model(population: pd.DataFrame) -> dict:
    """Day-0 behaviour -> P(return), with every predictor competing at once.

    Reported as odds ratios because a logit coefficient means nothing to a reader, and
    with the pseudo-R2 stated plainly: this model explains a *small* fraction of the
    variance. That is the honest headline. Day-0 behaviour shifts the odds of returning
    substantially, and still leaves most of the outcome unexplained -- which is what you
    would expect, and a useful corrective to any claim that retention has been "solved".
    """
    working = population.dropna(subset=["d0_ratings", "d0_median_popularity"]).copy()
    working["log_volume"] = np.log(working["d0_ratings"])
    working["log_popularity"] = np.log(working["d0_median_popularity"].clip(lower=1))

    predictors = ["multi_session", "log_volume", "log_popularity", "d0_rating_std"]
    design = sm.add_constant(working[predictors].astype(float))
    fit = sm.Logit(working["returned"].astype(int), design).fit(disp=0)

    return {
        "n": int(fit.nobs),
        "pseudo_r2": float(fit.prsquared),
        "terms": {
            name: {
                "coefficient": float(fit.params[name]),
                "odds_ratio": float(np.exp(fit.params[name])),
                "p_value": float(fit.pvalues[name]),
                "ci_low_or": float(np.exp(fit.conf_int().loc[name, 0])),
                "ci_high_or": float(np.exp(fit.conf_int().loc[name, 1])),
            }
            for name in design.columns
        },
    }


def stability_by_year(population: pd.DataFrame, min_users: int = 800) -> pd.DataFrame:
    """Does the effect hold in every signup cohort, or is it an artifact of one era?

    Fourteen years is long enough for the product, the internet and the user base to
    have changed completely. An effect that appears only in some years is a story about
    those years.
    """
    rows = []
    for year, chunk in population.groupby("signup_year"):
        if len(chunk) < min_users:
            continue
        single = chunk.loc[chunk["multi_session"] == 0, "returned"].mean()
        multi = chunk.loc[chunk["multi_session"] == 1, "returned"].mean()
        rows.append({
            "year": int(year),
            "users": len(chunk),
            "return_single": float(single),
            "return_multi": float(multi),
            "lift_x": float(multi / single) if single else float("nan"),
        })
    return pd.DataFrame(rows)


def opportunity(population: pd.DataFrame) -> dict:
    """Size the prize -- and then immediately say why the headline number is wrong.

    The naive calculation (move every single-session user to the multi-session return
    rate) is the number a portfolio project would print in bold. It is not a forecast,
    it is an upper bound that assumes the entire observed gap is causal. Users who come
    back within a day do so *because they were more interested*, and no notification
    manufactures that interest. The attenuated scenarios below exist so the reader sees
    the range, not the ceiling.
    """
    single = population[population["multi_session"] == 0]
    multi = population[population["multi_session"] == 1]

    rate_single = float(single["returned"].mean())
    rate_multi = float(multi["returned"].mean())
    gap = rate_multi - rate_single

    scenarios = {}
    # Two things have to go right: some share of single-session users can be induced
    # back for a second session at all, and the induced session has to carry some share
    # of the effect the spontaneous one does.
    for conversion in (0.05, 0.10, 0.20):
        for attribution in (0.10, 0.25, 0.50):
            lifted = conversion * attribution * gap
            scenarios[f"convert{int(conversion*100)}_attrib{int(attribution*100)}"] = {
                "converted_share": conversion,
                "causal_share_of_gap": attribution,
                "absolute_lift_pp": float(lifted * 100),
                "new_return_rate": float(rate_single + lifted),
                "users_retained_extra": int(round(len(single) * lifted)),
            }

    return {
        "single_session_users": int(len(single)),
        "single_session_share": float(len(single) / len(population)),
        "return_rate_single": rate_single,
        "return_rate_multi": rate_multi,
        "observed_gap_pp": float(gap * 100),
        "naive_upper_bound_pp": float(gap * 100),
        "scenarios": scenarios,
    }
