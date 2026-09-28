"""Durable GetXAPI research machinery: daily capacity gate, cross-run ledger,
and an LLM-steered dig loop.

Ported from the pixie-app GetXAPI tool-session researcher (Supabase ledger +
Codex-driven runPagedQuery + daily usage gate). Here the Supabase ledger is a
JSON file under the config dir, the daily gate is a small state file, and the
steering model is the existing discovery Planner (AI_GATEWAY_API_KEY /
OPENAI_API_KEY) instead of a Codex MCP session. Everything is best-effort:
missing config dir, unreadable files, and provider errors all degrade to the
unledgered / ungated / undug path rather than failing the run.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import env

# ---------------------------------------------------------------------------
# Daily capacity gate (getx-daily-usage-gate.ts port)
# ---------------------------------------------------------------------------

# Ceiling on GetXAPI HTTP calls per UTC day. High enough that normal runs never
# reach it; it exists so a runaway loop or a hot recurring job cannot burn
# through the plan. LAST30DAYS_GETXAPI_DAILY_BUDGET overrides; "0" disables the
# budget (the 429/5xx retry latch below still applies).
DEFAULT_DAILY_BUDGET = 800
# How long a 429/5xx pins the provider as exhausted. Mirrors pixie's
# GETX_AUTHORITY_RETRY_SECONDS latch.
RETRY_LATCH_SECONDS = 300

GATE_FILE = "getxapi-usage.json"
LEDGER_FILE = "x-research-ledger.json"

# Ledger bounds: keep the file small and the tail recent. Both are env-tunable;
# 0 (or a negative value) lifts the cap entirely.
LEDGER_MAX_QUERIES = 400
LEDGER_MAX_IDS_PER_QUERY = 2000

# Follow-up queries the dig planner may issue per round.
DIG_QUERIES_PER_ROUND = 3


def _now() -> float:
    return time.time()


def _env_int(name: str, default: int) -> int:
    return _map_int(os.environ, name, default)


def _map_int(env_map, name: str, default: int) -> int:
    try:
        return int(env_map.get(name) or "")
    except (TypeError, ValueError):
        return default


def _state_dir() -> Path | None:
    """Ledger/gate home. Respects LAST30DAYS_CONFIG_DIR="" (no-config runs)."""
    if env.CONFIG_DIR is None:
        return None
    try:
        env.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return env.CONFIG_DIR


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    try:
        handle, tmp = tempfile.mkstemp(
            prefix=path.name + ".", dir=str(path.parent)
        )
        with os.fdopen(handle, "w") as fh:
            fh.write(json.dumps(data))
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except (OSError, UnboundLocalError):
            pass


class DailyGate:
    """Bounds GetXAPI spend per UTC day and latches provider exhaustion.

    A 429 from GetXAPI records ``retry_until``; every later call in the
    latch window short-circuits instead of spending against a saturated key.
    """

    def __init__(self, environ=None):
        env_map = os.environ if environ is None else environ
        state_dir = _state_dir()
        self._path = state_dir / GATE_FILE if state_dir else None
        self._budget = _env_int("LAST30DAYS_GETXAPI_DAILY_BUDGET", DEFAULT_DAILY_BUDGET)
        self._state: dict[str, Any] = {}
        if self._path:
            self._state = _read_json(self._path)

    def _day(self) -> str:
        return datetime.now(UTC).date().isoformat()

    def _calls_today(self) -> int:
        if self._state.get("day") != self._day():
            return 0
        try:
            return int(self._state.get("calls") or 0)
        except (TypeError, ValueError):
            return 0

    def check(self) -> str | None:
        """Return a block reason, or None when a call may proceed."""
        if self._path is None:
            return None
        retry_until = self._state.get("retry_until")
        try:
            retry_until = float(retry_until) if retry_until else 0.0
        except (TypeError, ValueError):
            retry_until = 0.0
        if retry_until > _now():
            return "recently rate limited (retry latch active)"
        if self._budget > 0 and self._calls_today() >= self._budget:
            return f"daily budget reached ({self._budget} calls/day)"
        return None

    def charge(self) -> None:
        """Record one GetXAPI HTTP call."""
        if self._path is None:
            return
        day = self._day()
        state = dict(self._state)
        state["day"] = day
        state["calls"] = (self._calls_today() + 1)
        self._state = state
        _write_json(self._path, state)

    def latch(self) -> None:
        """Record a provider 429/5xx and pin the gate for RETRY_LATCH_SECONDS."""
        if self._path is None:
            return
        state = dict(self._state)
        state["day"] = self._day()
        state["retry_until"] = _now() + RETRY_LATCH_SECONDS
        self._state = state
        _write_json(self._path, state)


# ---------------------------------------------------------------------------
# Cross-run research ledger (x-research-ledger.ts port)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Dedupe key: case- and whitespace-insensitive query text."""
    return _WS_RE.sub(" ", str(query or "").strip().lower())


class Ledger:
    """Per-query dig history shared across runs.

    ``queries[key]`` keeps the bounded set of post ids already surfaced, the
    last pagination cursor and exhaustion flag per product lane, and usage
    counters. ``record_page`` is called per page so a crash mid-run still
    leaves the ledger consistent.
    """

    def __init__(self, environ=None):
        env_map = os.environ if environ is None else environ
        enabled = str(env_map.get("LAST30DAYS_X_LEDGER") or "1") != "0"
        state_dir = _state_dir() if enabled else None
        self._path = state_dir / LEDGER_FILE if state_dir else None
        self._max_queries = _map_int(
            env_map, "LAST30DAYS_X_LEDGER_MAX_QUERIES", LEDGER_MAX_QUERIES
        )
        self._max_ids = _map_int(
            env_map, "LAST30DAYS_X_LEDGER_MAX_IDS", LEDGER_MAX_IDS_PER_QUERY
        )
        self._data: dict[str, Any] = {"schema_version": 1, "queries": {}}
        if self._path:
            loaded = _read_json(self._path)
            if loaded.get("schema_version") == 1 and isinstance(loaded.get("queries"), dict):
                self._data = loaded

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def _entry(self, key: str, create: bool = False) -> dict | None:
        queries = self._data.setdefault("queries", {})
        entry = queries.get(key)
        if entry is None and create:
            entry = {
                "first_seen": datetime.now(UTC).isoformat(),
                "runs": 0,
                "ids": [],
                "products": {},
            }
            queries[key] = entry
        return entry

    def seen_ids(self, query: str) -> set[str]:
        """Post ids already surfaced for this normalized query."""
        if not self.enabled:
            return set()
        entry = self._entry(normalize_query(query))
        return set(entry.get("ids") or []) if entry else set()

    def record_query_run(self, query: str) -> None:
        """Mark that a run touched this query (dedupe + LRU bookkeeping)."""
        if not self.enabled:
            return
        entry = self._entry(normalize_query(query), create=True)
        entry["runs"] = int(entry.get("runs") or 0) + 1
        entry["updated_at"] = datetime.now(UTC).isoformat()
        self._evict()
        if self._path:
            _write_json(self._path, self._data)

    def record_page(
        self,
        query: str,
        product: str,
        post_ids: list[str],
        next_cursor: str | None,
        has_more: bool,
    ) -> None:
        if not self.enabled:
            return
        entry = self._entry(normalize_query(query), create=True)
        ids = entry.setdefault("ids", [])
        known = set(ids)
        for pid in post_ids:
            if pid not in known:
                known.add(pid)
                ids.append(pid)
        if 0 < self._max_ids < len(ids):
            del ids[: len(ids) - self._max_ids]
        entry.setdefault("products", {})[product] = {
            "next_cursor": next_cursor,
            "has_more": bool(has_more),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        entry["updated_at"] = datetime.now(UTC).isoformat()
        self._evict()
        if self._path:
            _write_json(self._path, self._data)

    def _evict(self) -> None:
        queries = self._data.get("queries")
        if self._max_queries <= 0:
            return
        if not isinstance(queries, dict) or len(queries) <= self._max_queries:
            return
        ordered = sorted(
            queries.items(),
            key=lambda kv: str(kv[1].get("updated_at") or ""),
        )
        for key, _ in ordered[: len(queries) - self._max_queries]:
            del queries[key]


# ---------------------------------------------------------------------------
# LLM-steered dig loop (Codex-driven runPagedQuery port)
# ---------------------------------------------------------------------------

MAX_DIGEST_ITEMS = 20
MAX_DIGEST_TEXT = 240


def _digest_items(items: list[dict]) -> list[dict]:
    """Compact evidence digest for the planner: text, author, engagement."""
    digest = []
    for item in items[-MAX_DIGEST_ITEMS:]:
        if not isinstance(item, dict):
            continue
        digest.append(
            {
                "text": str(item.get("text") or item.get("body") or "")[:MAX_DIGEST_TEXT],
                "author": str(item.get("author_handle") or item.get("author") or ""),
                "url": str(item.get("url") or ""),
                "likes": (item.get("engagement") or {}).get("likes"),
                "previously_seen": bool(item.get("previously_seen")),
            }
        )
    return digest


def _dig_feedback(
    topic: str,
    tried: list[str],
    interim: list[dict],
    round_no: int,
    previous_assessment: str | None,
    queries_requested: int,
) -> dict:
    return {
        "queries": tried[-100:],
        "total_queries": len(tried),
        "accepted_count": len(interim),
        "examples": _digest_items(interim),
        "queries_requested": queries_requested,
        "previous_assessment": previous_assessment,
        "rounds": [{"number": i + 1} for i in range(round_no)],
        "context_note": (
            "Iterative X dig: queries run through GetXAPI advanced search "
            "(Latest and Top lanes). Every retrieved post is scored for "
            "relevance by a downstream classifier, so favor broad retrieval "
            "— single terms, product/vendor names, category words — over "
            "multi-keyword constructions; search engines AND terms, and a "
            "missing synonym means a total miss. Good pivots: bare entity or "
            "product names, alternate phrasings, sub-events, or "
            "from:handle / @handle lanes for recurring voices. X search "
            "supports from:, @handle, \"exact phrase\", and plain keywords; "
            "since:/until: dates are applied by the engine."
        ),
    }


def dig(
    topic: str,
    interim_items: list[dict],
    tried_queries: list[str],
    *,
    from_date: str,
    to_date: str,
    depth: str,
    token: str,
    rounds: int,
    environ=None,
    gate: "DailyGate | None" = None,
    ledger: "Ledger | None" = None,
    timeout: int = 60,
) -> tuple[list[dict], list[str], dict]:
    """LLM-steered iterative dig over GetXAPI.

    Each round the discovery Planner reviews the interim corpus and emits 1-3
    follow-up queries (or declares coverage complete). Follow-ups run through
    ``getxapi.search_exact`` under the daily gate and ledger. Returns
    ``(new_items, warnings, stats)``; new items exclude post ids already in
    ``interim_items``.
    """
    from . import discovery_providers, getxapi

    warnings: list[str] = []
    stats = {"rounds_run": 0, "queries_run": 0, "new_items": 0,
             "jev_rejected": 0}
    if rounds <= 0 or not token:
        return [], warnings, stats
    env_map = os.environ if environ is None else environ
    try:
        planner = discovery_providers.Planner(environ)
    except discovery_providers.ProviderError as exc:
        warnings.append(f"X dig skipped: {exc}")
        return [], warnings, stats

    jev = None
    if str(env_map.get("LAST30DAYS_X_DIG_JEV") or "1") != "0":
        try:
            jev = discovery_providers.Jev(environ)
        except discovery_providers.ProviderError:
            jev = None

    objective = (
        f"X/Twitter posts relevant to: {topic}. Surface posts that the queries "
        "already tried did not reach — broad single-term and entity-name "
        "queries are safe because every candidate is relevance-classified "
        "downstream; keyword stuffing hides posts whose wording differs."
    )
    filters = {"source": "x", "from": from_date, "to": to_date}
    seen_post_ids = {
        str(i.get("post_id") or "")
        for i in interim_items
        if isinstance(i, dict) and i.get("post_id")
    }
    seen_urls = {
        str(i.get("url") or "")
        for i in interim_items
        if isinstance(i, dict) and i.get("url")
    }
    tried = list(tried_queries)
    tried_norm = {normalize_query(q) for q in tried}
    queries_per_round = max(
        1, _map_int(env_map, "LAST30DAYS_X_DIG_QUERIES", DIG_QUERIES_PER_ROUND)
    )
    new_items: list[dict] = []
    previous_assessment = None

    for round_no in range(rounds):
        blocked = gate.check() if gate else None
        if blocked:
            warnings.append(f"X dig stopped: GetXAPI {blocked}")
            break
        try:
            decision = planner.plan(
                objective, filters,
                _dig_feedback(topic, tried, interim_items + new_items,
                              round_no, previous_assessment,
                              queries_per_round),
                timeout,
            )
        except discovery_providers.ProviderError as exc:
            warnings.append(f"X dig planner failed: {exc}")
            break
        previous_assessment = decision.get("coverage_summary")
        stats["rounds_run"] = round_no + 1
        if decision.get("action") != "search":
            break
        produced = False
        round_items: list[dict] = []
        for query in (decision.get("queries") or [])[:queries_per_round]:
            if normalize_query(query) in tried_norm:
                continue
            tried_norm.add(normalize_query(query))
            tried.append(query)
            stats["queries_run"] += 1
            result = getxapi.search_exact(
                query, from_date, to_date, depth=depth, token=token,
                gate=gate, ledger=ledger,
            )
            if result.get("error") and not result.get("items"):
                warnings.append(f"X dig query {query!r}: {result['error']}")
                continue
            produced = True
            for item in result.get("items") or []:
                pid = str(item.get("post_id") or "")
                url = str(item.get("url") or "")
                if (pid and pid in seen_post_ids) or (url and url in seen_urls):
                    continue
                if pid:
                    seen_post_ids.add(pid)
                if url:
                    seen_urls.add(url)
                item["dig_round"] = round_no + 1
                round_items.append(item)
        if not produced:
            # Planner asked for queries but all were repeats — treat as done.
            break
        new_items.extend(_classify_round(round_items, jev, objective,
                                       stats, warnings, timeout))
    stats["new_items"] = len(new_items)
    return new_items, warnings, stats


# Jev reject/evidence bands, mirroring discovery.py's defaults.
JEV_REJECT = 0.2
JEV_EVIDENCE = 0.8


def _classify_round(items, jev, objective, stats, warnings, timeout):
    """Relevance-classify one dig round; rejected candidates are dropped.

    Broad queries only stay safe because this filter exists. Fail-open on
    provider errors: an unclassifiable batch is kept rather than silently
    emptied.
    """
    if not jev or not items:
        return items
    from . import discovery_providers
    kept = []
    for idx, item in enumerate(items):
        try:
            judgement = jev.classify(objective, [objective], {
                "text": str(item.get("text") or item.get("body") or "")[:2000],
                "author": str(item.get("author_handle") or item.get("author") or ""),
                "url": str(item.get("url") or ""),
            }, timeout)
        except discovery_providers.ProviderError as exc:
            warnings.append(f"X dig classifier degraded: {exc}"
                            " (keeping remaining items unclassified)")
            kept.extend(items[idx:])
            break
        score = judgement["probabilities"].get("c0", 0.0)
        sufficient = judgement.get("evidence_sufficient", 0.0)
        item["jev_score"] = round(score, 3)
        if sufficient >= JEV_EVIDENCE and score <= JEV_REJECT:
            stats["jev_rejected"] += 1
            continue
        kept.append(item)
    return kept
