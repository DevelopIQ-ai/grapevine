- Add deterministic dig seeding: every round fires broad queries mined from
  the corpus so far (recurring capitalized names in retrieved titles plus
  the topic's own proper names) before the planner adds its own — coverage no
  longer depends on planner phrasing luck. `LAST30DAYS_X_DIG_SEEDS` tunes the
  per-round count (default 3, `0` disables).
