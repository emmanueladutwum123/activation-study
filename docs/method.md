# Method, and what could still be wrong with it

## Definitions

| Term | Definition | Why this one |
|---|---|---|
| **Signup** | The user's first rating | MovieLens has no account-creation event; first activity is the only arrival signal in the data |
| **Day 0** | First 24 hours after signup | A calendar day would cut a user who joined at 23:00 mid-session |
| **Session** | Ratings separated by < 30 minutes | The common analytics default; the conclusion is re-derived at 10/60/120 minutes and does not depend on it |
| **Returned** | Any activity on day 7 or later | Six clear days between the predictor window and the outcome window |
| **Eligible** | ≥ 20 ratings *within day 0* | See below — this is the single most consequential choice in the study |

Every user is placed on a days-since-signup clock rather than a calendar one. A user who
joined in 1998 and one who joined in 2007 are at completely different points in the
product's life, and calendar time would compare them as if they were not.

## The eligibility rule, and why it is not optional

MovieLens only publishes users with **20 or more ratings in total**. A user who rated 12
movies on day 0 has therefore already been guaranteed, by the dataset's construction, to
rate at least 8 more later. Their "retention" is bookkeeping, not behaviour.

Requiring 20 ratings *within day 0* makes later activity unforced. It keeps 65,947 of
69,878 users (94.4%).

Left in, the excluded users invert the headline: the raw data appears to show that rating
*less* on day one predicts coming back. That finding is an artifact of the publication
filter, and it is exactly the kind of result that gets shipped because it is
counterintuitive and therefore feels like insight.

## Separating the signal from its confounder

Users with two day-0 sessions also rate more, so the session effect could be volume
wearing a disguise. Two checks, both reported:

1. **Stratification.** Compare single- and multi-session users *inside* each quartile of
   day-0 rating volume. The gap survives in all four (lift 2.73×, 2.58×, 2.19×, 1.87×).
2. **Regression.** A logistic model with `multi_session`, `log(volume)`,
   `log(median catalogue popularity)` and `rating variance` competing at once. The session
   term holds at an odds ratio of 3.26 (95% CI 3.13–3.40).

The lift **attenuates monotonically** as volume rises, from 2.73× in Q1 to 1.87× in Q4.
Part of the raw gap really is volume. Both facts are the finding; reporting only the first
would oversell it.

The model's pseudo-R² is **0.099**. Day-0 behaviour shifts the odds of returning
substantially and leaves most of the outcome unexplained — which is what should be
expected, and a corrective to any claim that retention has been "solved" by a feature
table.

## Robustness

- **Session definition.** Re-derived at 10, 30, 60 and 120-minute gaps. The multi-session
  share moves a lot (32.5% → 15.6%) and the lift barely does (2.46× → 2.73×). The result
  is not an artifact of the 30-minute convention.
- **Era.** Re-derived per signup year, 1996–2008. Lift ranges 1.84×–3.16× with no year
  below 1.8×. Fourteen years is long enough for the product, the internet and the user
  base to have changed completely; an effect that appeared in only some years would be a
  story about those years.
- **Horizon.** The split is checked at day 7, 30, 90, 180 and 365. A day-0 signal that
  only predicts day-7 behaviour is probably measuring the same burst of activity twice.
  This one is still 3.4× apart at a year.

## Threats to validity

**This is observational.** Nothing here identifies a causal effect. Users who return
within a day were more interested to begin with, and the study cannot separate "the second
session caused retention" from "whatever caused the second session also caused retention."
The opportunity sizing is built around that fact rather than around it: the naive number
is labelled a ceiling, and the scenario grid attenuates it by both an inducement rate and
a causal share.

**MovieLens users are already activated.** Every user in the data rated at least one
movie. In a real product a large share of signups never act at all, and those users are
missing here entirely. Every retention rate in this study is therefore biased upward
relative to what a real funnel would show — the *shape* of the finding travels, the levels
do not.

**Ratings are not sessions.** The only observable is a rating event. A user browsing for
an hour without rating anything is invisible, which means "sessions" here are rating
sessions, and any nudge built on this would be triggered by a proxy for engagement rather
than engagement itself.

**The traffic figure is a floor.** 10.5 new users/day is measured over 2007–2008, the last
two full years, because a product's current traffic is what an experiment runs on. A real
product deciding this question should substitute its own number — the infeasibility
conclusion is arithmetic, and it flips at roughly 100 users/day.

**Popularity is measured over the whole corpus**, including the future relative to any
given user. It is used as a property of the title rather than of the session, so this is
deliberate, but it is a lookahead and worth naming.

## What would settle it

A randomised nudge, with assignment at signup, triggered identically in both arms, read
out on second-session rate rather than day-7 return. The trigger must be evaluated the
same way for control users who never see the nudge — defining the analysis group by
anything that happens *after* treatment ("users who opened the email") reintroduces
exactly the selection the experiment exists to remove.

At this product's traffic that test needs 4.3 years to detect +2pp on retention, which is
the finding in [`recommendation.md`](recommendation.md). On second-session rate, where the
effect the mechanism targets is much larger, it is a matter of weeks.

## Reproducing

```bash
make setup     # venv + pinned deps
make data      # download (63MB) -> parquet; raw .dat files deleted afterwards
make all       # lint, tests, study
```

`make study` rewrites `results/findings.json` and `figures/`. The run is deterministic:
re-running on an unchanged dataset reproduces the committed JSON byte-for-byte.
`tests/test_findings.py` asserts the claims the write-ups make against that file, so an
analysis change that moves a headline fails CI rather than quietly ageing the prose.
