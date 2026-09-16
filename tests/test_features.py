"""Tests for the feature build.

The eligibility rule and the session boundary are the two places where this study could
quietly produce a wrong answer, so both are pinned here against small hand-built
fixtures where the right answer is obvious by inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from activation import features

HOUR = 3_600
DAY = 86_400


def make_ratings(events):
    """events: list of (user_id, offset_seconds_from_that_user's_first, rating)."""
    rows = []
    for user_id, offset, rating in events:
        rows.append({"user_id": user_id, "movie_id": 1 + (offset % 7),
                     "rating": rating, "timestamp": 1_000_000 + offset})
    frame = pd.DataFrame(rows)
    first = frame.groupby("user_id")["timestamp"].transform("min")
    frame["days_since_first"] = (frame["timestamp"] - first) / DAY
    frame["week_since_first"] = np.floor(frame["days_since_first"] / 7).astype(int)
    return frame


def make_users(ratings):
    grouped = ratings.groupby("user_id")["timestamp"]
    users = pd.DataFrame({"first_ts": grouped.min(), "last_ts": grouped.max(),
                          "n_ratings": grouped.size()}).reset_index()
    users["first_date"] = pd.to_datetime(users["first_ts"], unit="s")
    users["last_date"] = pd.to_datetime(users["last_ts"], unit="s")
    users["cohort_month"] = users["first_date"].dt.to_period("M").astype(str)
    users["lifespan_days"] = (users["last_ts"] - users["first_ts"]) / DAY
    return users


class TestSessions:
    def test_one_continuous_burst_is_one_session(self):
        # 25 ratings five minutes apart: never a gap over 30 minutes.
        events = [(1, i * 300, 4.0) for i in range(25)]
        ratings = make_ratings(events)
        table = features.build(ratings, make_users(ratings))
        assert table.loc[0, "d0_sessions"] == 1
        assert table.loc[0, "multi_session"] == 0

    def test_a_gap_starts_a_new_session(self):
        events = [(1, i * 300, 4.0) for i in range(20)]
        events += [(1, 6 * HOUR + i * 300, 4.0) for i in range(5)]
        ratings = make_ratings(events)
        table = features.build(ratings, make_users(ratings))
        assert table.loc[0, "d0_sessions"] == 2
        assert table.loc[0, "multi_session"] == 1

    def test_gap_threshold_is_respected(self):
        """29 minutes is the same session; 31 minutes is not. The boundary is the
        definition, so it is worth pinning exactly."""
        for gap_minutes, expected in ((29, 1), (31, 2)):
            events = [(1, i * 60, 4.0) for i in range(20)]
            events += [(1, 19 * 60 + gap_minutes * 60 + i * 60, 4.0) for i in range(3)]
            ratings = make_ratings(events)
            table = features.build(ratings, make_users(ratings))
            assert table.loc[0, "d0_sessions"] == expected, gap_minutes

    def test_activity_after_24h_is_not_day_zero(self):
        events = [(1, i * 300, 4.0) for i in range(20)]
        events += [(1, 30 * HOUR, 4.0)]  # next day
        ratings = make_ratings(events)
        table = features.build(ratings, make_users(ratings))
        assert table.loc[0, "d0_ratings"] == 20
        assert table.loc[0, "d0_sessions"] == 1


class TestOutcome:
    def test_return_requires_activity_at_day_seven(self):
        """Days 1-6 are deliberately neither predictor nor outcome. A user active only
        on day 3 has not 'returned' under this definition -- the gap is what keeps the
        two windows from overlapping."""
        events = [(1, i * 300, 4.0) for i in range(20)] + [(1, 3 * DAY, 4.0)]
        ratings = make_ratings(events)
        table = features.build(ratings, make_users(ratings))
        assert not table.loc[0, "returned"]

        events = [(2, i * 300, 4.0) for i in range(20)] + [(2, 9 * DAY, 4.0)]
        ratings = make_ratings(events)
        table = features.build(ratings, make_users(ratings))
        assert table.loc[0, "returned"]


class TestEligibility:
    def test_excludes_users_whose_retention_is_forced(self):
        """The rule that decides the headline.

        MovieLens only ships users with 20+ ratings total. A user with 5 on day 0 is
        guaranteed more later, so counting them as 'retained' measures the dataset's
        filter rather than the user. Left in, they reverse the result.
        """
        thin = [(1, i * 300, 4.0) for i in range(5)] + [(1, 10 * DAY + i, 4.0)
                                                        for i in range(15)]
        thick = [(2, i * 300, 4.0) for i in range(25)]
        ratings = make_ratings(thin + thick)
        table = features.build(ratings, make_users(ratings))

        assert not table.set_index("user_id").loc[1, "eligible"]
        assert table.set_index("user_id").loc[2, "eligible"]
        assert set(features.eligible(table)["user_id"]) == {2}

    def test_threshold_is_the_documented_one(self):
        assert features.MIN_DAY0_RATINGS == 20


class TestNoLeakage:
    def test_day_zero_features_ignore_later_activity(self):
        """Two users with identical day-0 behaviour must get identical features, no
        matter how differently their lives go afterwards. If later activity leaked into
        a predictor, the model would be reading the answer."""
        base = [(i * 300, 4.0) for i in range(30)]
        events = [(1, off, r) for off, r in base]
        events += [(2, off, r) for off, r in base]
        events += [(2, 40 * DAY + i * 300, 5.0) for i in range(200)]
        ratings = make_ratings(events)
        table = features.build(ratings, make_users(ratings)).set_index("user_id")

        for column in ("d0_ratings", "d0_sessions", "d0_span_hours", "d0_rating_std"):
            assert table.loc[1, column] == pytest.approx(table.loc[2, column]), column
        assert not table.loc[1, "returned"]
        assert table.loc[2, "returned"]
