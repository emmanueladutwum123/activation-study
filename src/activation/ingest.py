"""Turn the raw MovieLens dump into the two tables the analysis actually needs.

The raw file is 253MB of ``::``-delimited text. Everything downstream reads parquet
instead, for three reasons that matter more than speed:

* **Types survive.** Re-parsing text means re-deciding, every time, whether a rating is
  a float and a timestamp is an integer. One script decides once.
* **The derived user table is the unit of analysis.** Almost every question in this
  study is asked per user, not per rating. Materialising a user table once means the
  rest of the code reads 70k rows instead of 10M.
* **The raw file can be deleted.** Parquet is ~5x smaller, which matters on a laptop.

Run: ``python -m activation.ingest``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/ml-10M100K")
OUT_DIR = Path("data/processed")

SECONDS_PER_DAY = 86_400


def load_ratings(path: Path) -> pd.DataFrame:
    """Read ``user::movie::rating::timestamp``.

    Pandas cannot use its fast C engine on a multi-character separator, so the ``::``
    is split manually on a single-character read. On 10M rows that is the difference
    between seconds and minutes.
    """
    frame = pd.read_csv(
        path,
        sep=":",
        header=None,
        engine="c",
        usecols=[0, 2, 4, 6],
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "float32",
               "timestamp": "int64"},
    )
    return frame


def load_movies(path: Path) -> pd.DataFrame:
    rows = []
    # movies.dat is Latin-1, not UTF-8, and a handful of titles carry accents that make
    # a UTF-8 read fail outright rather than degrade.
    with path.open("r", encoding="latin-1") as handle:
        for line in handle:
            movie_id, title, genres = line.rstrip("\n").split("::")
            rows.append((int(movie_id), title, genres))
    return pd.DataFrame(rows, columns=["movie_id", "title", "genres"])


def build_user_table(ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per user: when they arrived, what they did, and how long they stayed.

    "Signup" is the first rating we observe. MovieLens has no account-creation event, so
    first activity is the only arrival signal available -- which means every user in the
    dataset is, by construction, already activated to the extent of one rating. That
    biases every retention number upward relative to a real product, where a large share
    of signups never act at all. Stated here because it changes how the numbers should
    be read, not because it invalidates them.
    """
    grouped = ratings.groupby("user_id", sort=False)["timestamp"]
    users = pd.DataFrame({
        "first_ts": grouped.min(),
        "last_ts": grouped.max(),
        "n_ratings": grouped.size(),
    }).reset_index()

    users["first_date"] = pd.to_datetime(users["first_ts"], unit="s")
    users["last_date"] = pd.to_datetime(users["last_ts"], unit="s")
    users["cohort_month"] = users["first_date"].dt.to_period("M").astype(str)
    users["lifespan_days"] = (users["last_ts"] - users["first_ts"]) / SECONDS_PER_DAY
    return users


def add_relative_time(ratings: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Attach each rating's age relative to its own user's first activity.

    Calendar time is the wrong clock for a retention question: a user who joined in 1998
    and one who joined in 2007 are at completely different points in the product's life.
    Days-since-signup puts every user on the same axis.
    """
    first = users.set_index("user_id")["first_ts"]
    ratings = ratings.copy()
    ratings["days_since_first"] = (
        (ratings["timestamp"] - ratings["user_id"].map(first)) / SECONDS_PER_DAY
    ).astype("float32")
    ratings["week_since_first"] = np.floor(
        ratings["days_since_first"] / 7.0
    ).astype("int32")
    return ratings


def main() -> int:
    if not RAW_DIR.exists():
        print(f"missing {RAW_DIR}; run scripts/fetch_data.sh first", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("reading ratings...", flush=True)
    ratings = load_ratings(RAW_DIR / "ratings.dat")
    print(f"  {len(ratings):,} ratings, {ratings['user_id'].nunique():,} users")

    movies = load_movies(RAW_DIR / "movies.dat")
    print(f"  {len(movies):,} movies")

    users = build_user_table(ratings)
    ratings = add_relative_time(ratings, users)

    ratings.to_parquet(OUT_DIR / "ratings.parquet", index=False,
                       compression="zstd")
    users.to_parquet(OUT_DIR / "users.parquet", index=False, compression="zstd")
    movies.to_parquet(OUT_DIR / "movies.parquet", index=False, compression="zstd")

    span = f"{users['first_date'].min():%Y-%m} to {users['first_date'].max():%Y-%m}"
    print(f"\nwrote {OUT_DIR}/  --  cohorts span {span}")
    for name in ("ratings", "users", "movies"):
        size_mb = (OUT_DIR / f"{name}.parquet").stat().st_size / 1e6
        print(f"  {name}.parquet  {size_mb:6.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
