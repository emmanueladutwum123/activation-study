# Recommendation: ship the second-session nudge, don't A/B test it

*One page. The evidence is in [`../README.md`](../README.md); the method and its holes
are in [`method.md`](method.md). Every number is read from
[`../results/findings.json`](../results/findings.json).*

## The decision

**Ship a day-0 re-engagement nudge to every new user who finishes a first session and
does not come back the same day. Do not gate it behind an A/B test.**

That is an unusual recommendation, so the reasoning matters more than the conclusion.

## Why act at all

78.5% of new users have exactly one session on day 0, and 19.8% of them are still around
on day 7. The users who came back for a second session that day return at 52.4%. The gap
holds inside every quartile of day-0 rating volume, in every signup cohort from 1996 to
2008, and at every definition of "session" between a 10-minute and a 2-hour gap. Adjusted
for volume, catalogue popularity and rating variance, the odds ratio is 3.26.

This is the strongest day-0 signal in the dataset, it is measurable in real time, and the
population it identifies is four-fifths of all new users.

## Why not to believe the size of it

The gap is **selection, mostly**. Users who return within a day do so because they were
more interested, and a notification does not manufacture interest. Two things must both
go right for any of the gap to be recoverable: some share of single-session users must be
inducible back at all, and the induced session must carry some share of the effect the
spontaneous one does.

Under plausible values for both — 5–20% converted, 10–50% of the effect carried — the
realistic gain is **+0.2pp to +3.3pp** on day-7 return, against a naive ceiling of
+32.6pp. Plan against the low end.

## Why not to test it

The nudge only fires for single-session users, so at 10.5 new users per day a clean
readout on day-7 return needs **6,468 per arm to detect +2pp — 4.3 years**. A quarter of
traffic buys a minimum detectable effect of **8.2pp**, a 41% relative improvement that no
re-engagement email has ever produced.

A quarter-long test would therefore return "no significant difference" whether or not the
feature works, and that result would be read as "the idea doesn't work." Running it is
worse than not running it: it converts an unknown into a false negative with a number
attached. And peeking daily for two weeks to get an answer sooner pushes the false-positive
rate from 5% to 21.4%, which trades the false negative for a false positive.

## What to measure instead

Judge the nudge on the intermediate metric its mechanism actually moves:

1. **Primary: second-session rate among nudged users.** This is the step the feature is
   trying to cause, the effect there is large enough to see in weeks rather than years,
   and it is the link in the chain the data says is doing the work.
2. **Secondary: day-7 return, monitored, not gated.** Directionally useful over quarters;
   never the accept/reject criterion at this traffic.
3. **Guardrail: unsubscribe and notification-disable rate.** The cost of being wrong here
   is not a flat result, it is a permanently unreachable user.

If second-session rate does not move, the feature does not work and no retention readout
is needed. If it moves and day-7 return does not follow over several quarters, the causal
story is wrong — the second session was a symptom, not a cause — and that is worth knowing.

## What would change this recommendation

- **Traffic rising by an order of magnitude.** The infeasibility is arithmetic, not
  principle; at 100+ new users a day, the +2pp test fits in a quarter and should be run.
- **A cheaper randomisation unit.** If the nudge can be varied *within* user (timing,
  content) rather than on/off, the comparison stops needing two arms of naive users.
- **Second-session rate moving with no retention change over a year.** That breaks the
  mechanism and the nudge should be withdrawn, not tuned.

## What this study cannot tell you

Whether the nudge works. Nothing observational can. What it can tell you is that the
target is the right one, the effect worth expecting is small, and the experiment everyone
will ask for cannot be run here — which is enough to decide what to build and what to
watch.
