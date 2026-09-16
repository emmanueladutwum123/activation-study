"""Build the per-user feature table the whole study rests on.

Two rules govern everything here, and both exist to stop the analysis lying to itself.

**Predictors come only from day 0; the outcome comes only from day 7 onwards.** There
is a six-day gap between them. That gap is not conservatism, it is what makes the
question answerable: a feature computed over a window that overlaps the outcome window
would predict the outcome trivially and mean nothing.

**Users whose retention is forced by the dataset's construction are excluded.**
MovieLens only publishes users with at least 20 ratings in total. A user who rated 12
movies on day 0 is therefore *guaranteed* to rate more later -- their "retention" is an
artifact of the filter, not behaviour. Left in, they invert the headline result: the
raw data appears to show that rating *less* on day one predicts coming back, which is
exactly backwards and exactly the kind of finding that gets shipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")

# A gap longer than this starts a new session. 30 minutes is the common analytics
# default; `session_gap_sensitivity` below checks the conclusion does not depend on it.
SESSION_GAP_SECONDS = 1_800

# Day 0 is the first 24 hours. The outcome window opens at day 7.
DAY0_END_DAYS = 1.0
RETURN_HORIZON_DAYS = 7.0

# MovieLens' own inclusion threshold. A user must clear it *within day 0* to be
# eligible, so that later activity is behaviour rather than bookkeeping.
MIN_DAY0_RATINGS = 20


def label_sessions(day0: pd.DataFrame, gap_seconds: int = SESSION_GAP_SECONDS) -> pd.DataFrame:
    """Assign a session index to each day-0 rating, per user."""
    day0 = day0.sort_values(["user_id", "timestamp"])
    gap = day0.groupby("user_id")["timestamp"].diff()
    starts = gap.isna() | (gap > gap_seconds)
    day0 = day0.assign(session_idx=starts.groupby(day0["user_id"]).cumsum())
    return day0


def build(ratings: pd.DataFrame, users: pd.DataFrame,
          gap_seconds: int = SESSION_GAP_SECONDS) -> pd.DataFrame:
    """Return one row per user with day-0 behaviour and the day-7 outcome."""
    returned = ratings.loc[
        ratings["days_since_first"] >= RETURN_HORIZON_DAYS, "user_id"
    ].unique()
    users = users.copy()
    users["returned"] = users["user_id"].isin(returned)

    day0 = ratings[ratings["days_since_first"] < DAY0_END_DAYS].copy()
    day0 = label_sessions(day0, gap_seconds)

    # Popularity is measured over the whole corpus, not day 0: it is a property of the
    # title, not of this user's session.
    popularity = ratings.groupby("movie_id")["user_id"].size()
    day0["movie_pop"] = day0["movie_id"].map(popularity)

    grouped = day0.groupby("user_id")
    features = pd.DataFrame({
        "d0_ratings": grouped.size(),
        "d0_sessions": grouped["session_idx"].max(),
        "d0_span_hours": grouped["timestamp"].apply(
            lambda s: (s.max() - s.min()) / 3600.0),
        "d0_median_popularity": grouped["movie_pop"].median(),
        "d0_rating_std": grouped["rating"].std().fillna(0.0),
        "d0_mean_rating": grouped["rating"].mean(),
    }).reset_index()

    table = users.merge(features, on="user_id", how="left")
    table["multi_session"] = (table["d0_sessions"] > 1).astype(int)
    table["eligible"] = table["d0_ratings"] >= MIN_DAY0_RATINGS
    table["signup_year"] = pd.to_datetime(table["first_date"]).dt.year
    return table


def eligible(table: pd.DataFrame) -> pd.DataFrame:
    """The analysis population: users whose later activity was not forced."""
    return table[table["eligible"]].copy()


def session_gap_sensitivity(ratings: pd.DataFrame, users: pd.DataFrame,
                            gaps=(600, 1_800, 3_600, 7_200)) -> pd.DataFrame:
    """Does the headline hold if "a session" is defined differently?

    The 30-minute gap is a convention, and a result that only exists at 30 minutes is a
    result about the convention. This re-derives the effect at four thresholds so the
    reader can see whether the conclusion moves.
    """
    rows = []
    for gap in gaps:
        table = eligible(build(ratings, users, gap_seconds=gap))
        single = table.loc[table["multi_session"] == 0, "returned"].mean()
        multi = table.loc[table["multi_session"] == 1, "returned"].mean()
        rows.append({
            "gap_minutes": gap // 60,
            "share_multi_session": float((table["multi_session"] == 1).mean()),
            "return_single": float(single),
            "return_multi": float(multi),
            "lift_x": float(multi / single),
        })
    return pd.DataFrame(rows)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")
    users = pd.read_parquet(PROCESSED / "users.parquet")
    return ratings, users
