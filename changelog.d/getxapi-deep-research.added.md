Ported the pixie-app GetXAPI researcher into the X lane: `--x-dig N` runs an
LLM-steered iterative dig (the discovery planner reviews interim hits and
issues up to 3 follow-up GetXAPI queries per round; `--deep` defaults to 2
rounds), a cross-run ledger at `~/.config/last30days/x-research-ledger.json`
flags re-surfaced posts as `previously_seen` (`LAST30DAYS_X_LEDGER=0`
disables), a daily call budget plus a five-minute 429 latch bounds GetXAPI
spend (`LAST30DAYS_GETXAPI_DAILY_BUDGET`, default 800), and author name, bio,
followers, and location now flow into X item metadata.
