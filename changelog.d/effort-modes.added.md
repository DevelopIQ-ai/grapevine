- Add `--effort low|normal|high|ultra` depth modes: low = `--quick`, normal =
  default, high = `--deep`, and `ultra` = deep plus maximum dig fan-out (5 dig
  rounds, 5 follow-up queries per round, 40-page GetXAPI lanes, 30/120
  per-stream/ranked pool caps) — explicit env/flag pins still override.
- Add `--max-calls N` / `LAST30DAYS_MAX_X_CALLS`: a per-run hard stop on
  GetXAPI calls (~$0.001 each) shared across every gate instance; hitting the
  cap short-circuits all X lanes and dig rounds while other sources continue.
