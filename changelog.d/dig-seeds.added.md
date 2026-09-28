- Add deterministic dig seeding: every round fires broad queries mined from
  the shared cross-lane corpus (recurring capitalized names in retrieved
  items plus the topic's own proper names) before the planner adds its own —
  coverage no longer depends on planner phrasing luck, an entity surfacing on
  one lane seeds queries on the others, and recurring voices on X get
  account-scoped `from:` seeds. `LAST30DAYS_X_DIG_SEEDS` tunes the per-round
  count (default 3, `0` disables).
