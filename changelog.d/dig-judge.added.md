- Add a second-stage dig Judge: every item Jev keeps is re-scored 0-100 in
  batched calls by a small chat model (`DISCOVERY_JUDGE_*`, defaulting to the
  planner gateway key and gpt-4.1-mini); scores under 50 are dropped as thin,
  spammy, or redundant and kept items carry `judge_score` in metadata.
  `LAST30DAYS_X_DIG_JUDGE=0` disables; provider failures fail open.
