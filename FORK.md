# Last 30 Days + GetXAPI + Jev

This fork of [Matt Van Horn’s Last 30 Days](https://github.com/mvanhorn/last30days-skill)
adds GetXAPI as an X search backend and an LLM-steered dig loop with Jev
relevance classification inside the ordinary research pass.
It keeps the original one-pass skill and other sources.
No Pixie application, database, or job infrastructure is required.

## Install

```sh
npx skills add DevelopIQ-ai/last30days-skill -g
```

Provide `GETXAPI_KEY` through your environment or private
`~/.config/last30days/.env`, then set `LAST30DAYS_X_BACKEND=getxapi`.
Do not commit that file. No X browser cookies or X account login are required.

Then use the skill normally:

```text
/last30days AI coding agents
/last30days Peter Steinberger
```

GetXAPI supports topic searches, posts by an author, and mentions by other authors.
Results include source links, timestamps, and engagement. Searches paginate within
a bounded budget, deduplicate posts, and preserve partial results on failures.
Requests use GetXAPI credits. Without an explicit backend pin, GetXAPI is the last
fallback in the ordinary X chain. Existing host-specific policies remain intact.

## Dig mode with Jev

For "dig deep / find everything" asks the engine adds `--x-dig N` (2 rounds on
`--deep`, `--effort ultra` maxes it). After initial retrieval a planner reviews
the interim corpus and fires follow-up queries on each diggable lane — X via
GetXAPI (`from:`/`@` chases, broad entity queries), Hacker News via the keyless
Algolia index, Google News via the keyless RSS search, Reddit via its keyless
search RSS, GitHub via its keyless anon search tier, and the web via the
configured grounding backend. Bluesky joins when `BSKY_*` creds are set, and
YouTube/arXiv/Techmeme join when their binaries (`yt-dlp`, `arxiv-pp-cli`,
`techmeme-pp-cli`) are on PATH. Deterministic
seed queries mined from the corpus (recurring names, `from:` author seeds) run
every round so coverage doesn't depend on planner phrasing. Every retrieved
item is relevance-classified by Jev before merging; a second-stage Judge then
scores survivors and drops thin or spammy ones. A cross-run ledger remembers
every surfaced id so repeat runs accumulate instead of repeating.

Configure native Jev with `JEV_API_KEY` or `TYPESAFE_API_KEY`, or use
`JEV_PROVIDER=vercel` with `AI_GATEWAY_API_KEY`. A separate general model plans
queries using `DISCOVERY_PLANNER_API_KEY`, `AI_GATEWAY_API_KEY`, or
`OPENAI_API_KEY`. Provider calls retry transient failures with exponential
backoff (`DISCOVERY_PROVIDER_RETRIES`/`DISCOVERY_PROVIDER_BACKOFF`), and a lane
that still fails marks `provider_failed` and stops rather than degrading
silently. GetXAPI spend is bounded by `LAST30DAYS_GETXAPI_DAILY_BUDGET` (daily)
and `--max-calls`/`LAST30DAYS_MAX_X_CALLS` (per run).

See [the dig configuration](CONFIGURATION.md#api-keys-env) for every knob.
Reinstall the skill after updating this fork: installed copies do not
automatically track checkout edits.

## Verification

```sh
uv run pytest tests/test_getxapi.py tests/test_config_x_backends.py tests/test_backend_descriptors.py tests/test_x_policy.py
```

Live checks returned ten recent author posts through the adapter and a complete
X-only engine report for “AI coding agents”: six ranked results with engagement,
six clusters, `source_status.x=ok`, and exit code 0. The focused suite passes
290 tests, including an inclusive end-date regression. Credentials were held in memory, not committed or
saved into this repository. Fresh installs still need their own credential setup.

See [CONFIGURATION.md](CONFIGURATION.md) for configuration details. Original MIT
license and attribution are preserved.
