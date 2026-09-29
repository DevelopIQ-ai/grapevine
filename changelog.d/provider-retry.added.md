Dig provider calls retry transient failures with exponential backoff, then fail loudly.

- `discovery_providers`: `ProviderError` gained a `retryable` flag (429/5xx,
  timeouts, transport errors are retryable; deterministic 4xx/validation
  failures are not). Planner, Jev, and Judge calls now go through
  `_post_retried`, which retries retryable errors with exponential backoff
  and jitter — `DISCOVERY_PROVIDER_RETRIES` (default 3 extra attempts) and
  `DISCOVERY_PROVIDER_BACKOFF` (default 2s base).
- `x_research`: a provider call that still fails after its retries marks the
  dig `provider_failed` in stats, warns "failed after retries", and stops
  that lane instead of silently degrading; the flag surfaces in the
  `x_dig`/`dig` artifacts. Items already fetched are kept flagged
  `jev_unclassified` so a judge outage can never silently empty results.
