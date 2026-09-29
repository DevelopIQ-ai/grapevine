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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import entity_extract, env

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

    # Per-process call counter shared by every gate instance in the run —
    # getxapi constructs a fresh gate per call site, so the run cap cannot
    # live on any one instance or in the day-scoped state file.
    _run_calls: int = 0

    def __init__(self, environ=None):
        env_map = os.environ if environ is None else environ
        state_dir = _state_dir()
        self._path = state_dir / GATE_FILE if state_dir else None
        self._budget = _env_int("LAST30DAYS_GETXAPI_DAILY_BUDGET", DEFAULT_DAILY_BUDGET)
        self._max_run = _env_int("LAST30DAYS_MAX_X_CALLS", 0)
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
        if self._max_run > 0 and type(self)._run_calls >= self._max_run:
            return f"run call cap reached ({self._max_run} calls this run)"
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
        type(self)._run_calls += 1
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
        text = (
            item.get("text") or item.get("body")
            or item.get("title") or item.get("snippet") or ""
        )
        digest.append(
            {
                "text": str(text)[:MAX_DIGEST_TEXT],
                "author": str(item.get("author_handle") or item.get("author") or ""),
                "url": str(item.get("url") or item.get("hn_url") or ""),
                "likes": (item.get("engagement") or {}).get("likes"),
                "previously_seen": bool(item.get("previously_seen")),
            }
        )
    return digest


_X_CONTEXT_NOTE = (
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
)

_LANE_LABELS = {
    "x": "X/Twitter posts",
    "hackernews": "Hacker News stories",
    "grounding": "web pages",
}

_LANE_CONTEXT_NOTES = {
    "hackernews": (
        "Iterative dig: queries run through the Hacker News Algolia stories "
        "index. Every retrieved story is scored for relevance by a "
        "downstream classifier, so favor broad retrieval — single terms, "
        "product/vendor names, category words — over multi-keyword "
        "constructions; Algolia ANDs leading terms, and a missing synonym "
        "means a total miss. Good pivots: bare product or project names, "
        "'Show HN'-style launch phrasings, alternate terms for the same "
        "thing. No operators — plain keywords only."
    ),
    "grounding": (
        "Iterative dig: queries run through a web search backend. Every "
        "retrieved page is scored for relevance by a downstream classifier, "
        "so favor broad retrieval — single terms, product/vendor names, "
        "category words — over long keyword strings. Good pivots: bare "
        "product or company names, alternate phrasings, official domains, "
        "launch/announcement phrasings."
    ),
}


def _dig_feedback(
    topic: str,
    tried: list[str],
    interim: list[dict],
    round_no: int,
    previous_assessment: str | None,
    queries_requested: int,
    context_note: str | None = None,
) -> dict:
    return {
        "queries": tried[-100:],
        "total_queries": len(tried),
        "accepted_count": len(interim),
        "examples": _digest_items(interim),
        "queries_requested": queries_requested,
        "previous_assessment": previous_assessment,
        "rounds": [{"number": i + 1} for i in range(round_no)],
        "context_note": context_note or _X_CONTEXT_NOTE,
    }


def _item_dedupe_keys(item: dict) -> tuple[str, str]:
    """(id, url) dedupe keys; works for X posts and HN/web items alike."""
    return (
        str(item.get("post_id") or item.get("id") or ""),
        str(item.get("url") or item.get("hn_url") or ""),
    )


# Capitalized runs in titles/text: "Prime Sandboxes", "Docker", "Edera".
_PROPER_RUN = re.compile(
    r"[A-Z][A-Za-z0-9.&'+-]*(?:[ \t]+[A-Z][A-Za-z0-9.&'+-]*){0,3}"
)

# Title-cased junk that survives capital-run mining but names nothing.
_SEED_STOP = frozenset({
    "show hn", "ask hn", "tell hn", "hn", "show", "ask", "the", "part",
    "new", "vs", "why", "how", "what", "when", "launch", "launched",
    "launching", "announcing", "introducing", "meet", "open source",
})

# How many mined seed queries each dig round fires before the planner adds
# its own. LAST30DAYS_X_DIG_SEEDS overrides; 0 disables seeding.
DIG_SEEDS_PER_ROUND = 3


def _seed_queries(
    topic: str,
    items: list[dict],
    tried_norm: set,
    *,
    max_seeds: int,
    author_prefix: str | None = None,
) -> list[str]:
    """Deterministic broad queries mined from the corpus so far.

    The planner's phrasing is a luck surface — a run can miss a whole vendor
    because nobody typed its name. Seeds remove the luck: every round, the
    most frequent capitalized phrases in retrieved titles/text plus the
    topic's own proper names become bare queries, deduped against everything
    already tried. Jev+judge keep the noise cheap. When ``author_prefix``
    is set (``from:`` on X), recurring authors get account-scoped seeds — a
    from:lane catches a launch announcement regardless of wording.
    """
    if max_seeds <= 0:
        return []
    counts: Counter[str] = Counter()
    authors: Counter[str] = Counter()
    for it in items[-400:]:
        if not isinstance(it, dict):
            continue
        text = " ".join(
            str(it.get(field) or "")
            for field in ("title", "text", "body", "snippet")
        )[:400]
        for match in _PROPER_RUN.finditer(text):
            phrase = " ".join(match.group(0).split()).strip(".,:;!?-")
            if (
                len(phrase) < 4
                or phrase.lower() in _SEED_STOP
                or all(
                    w.lower() in entity_extract.ENTITY_STOPWORDS
                    for w in phrase.split()
                )
            ):
                continue
            counts[phrase] += 1
        if author_prefix:
            handle = str(
                it.get("author_handle") or it.get("author") or ""
            ).strip().lstrip("@")
            if 2 <= len(handle) <= 30 and " " not in handle:
                authors[handle] += 1
    seeds: list[str] = []

    def _take(phrase: str) -> bool:
        if normalize_query(phrase) not in tried_norm and phrase not in seeds:
            seeds.append(phrase)
            return True
        return False

    # Recurring voices get a reserved budget — an account-scoped query catches
    # an announcement no matter the wording, and phrase seeds shouldn't crowd
    # them out.
    if author_prefix:
        author_budget = min(2, max_seeds)
        for handle, _ in authors.most_common(10):
            if author_budget <= 0:
                break
            if _take(f"{author_prefix}{handle}"):
                author_budget -= 1
    # Recurring names in hits — strongest novelty signal.
    for phrase, _ in counts.most_common(20):
        if len(seeds) >= max_seeds:
            break
        _take(phrase)
    # Then the topic's own proper names as a guaranteed floor.
    for match in _PROPER_RUN.finditer(topic):
        if len(seeds) >= max_seeds:
            break
        phrase = " ".join(match.group(0).split()).strip(".,:;!?-")
        if len(phrase) >= 4 and phrase.lower() not in _SEED_STOP:
            _take(phrase)
    return seeds[:max_seeds]


def dig_source(
    lane: str,
    topic: str,
    interim_items: list[dict],
    tried_queries: list[str],
    *,
    search_fn,
    rounds: int,
    environ=None,
    gate: "DailyGate | None" = None,
    ledger: "Ledger | None" = None,
    timeout: int = 60,
    context_note: str | None = None,
    seed_items: list[dict] | None = None,
) -> tuple[list[dict], list[str], dict]:
    """LLM-steered iterative dig over any search lane.

    Each round the discovery Planner reviews the interim corpus and emits
    follow-up queries (or declares coverage complete). ``search_fn(query)``
    executes one query against the lane's backend and returns a dict with
    ``items`` and optional ``error``. Every batch is relevance-classified by
    Jev when configured (broad queries stay safe because weak hits are
    dropped, not kept). Returns ``(new_items, warnings, stats)``; new items
    exclude ids/urls already in ``interim_items``.
    """
    from . import discovery_providers

    warnings: list[str] = []
    stats = {"rounds_run": 0, "queries_run": 0, "new_items": 0,
             "jev_rejected": 0, "judge_rejected": 0}
    if rounds <= 0 or search_fn is None:
        return [], warnings, stats
    env_map = os.environ if environ is None else environ
    label = _LANE_LABELS.get(lane, lane)
    try:
        planner = discovery_providers.Planner(environ)
    except discovery_providers.ProviderError as exc:
        warnings.append(f"{label} dig skipped: {exc}")
        return [], warnings, stats

    jev = None
    if str(env_map.get("LAST30DAYS_X_DIG_JEV") or "1") != "0":
        try:
            jev = discovery_providers.Jev(environ)
        except discovery_providers.ProviderError:
            jev = None

    # Second-stage verdict: whatever Jev kept gets a finer keep/drop + score
    # from a small chat model. Disabled with LAST30DAYS_X_DIG_JUDGE=0.
    judge = None
    if str(env_map.get("LAST30DAYS_X_DIG_JUDGE") or "1") != "0":
        try:
            judge = discovery_providers.Judge(environ)
        except discovery_providers.ProviderError:
            judge = None

    objective = (
        f"{label} relevant to: {topic}. Surface posts that the queries "
        "already tried did not reach — broad single-term and entity-name "
        "queries are safe because every candidate is relevance-classified "
        "downstream; keyword stuffing hides posts whose wording differs."
    )
    filters = {"source": lane}
    seen_ids = set()
    seen_urls = set()
    for i in interim_items:
        if isinstance(i, dict):
            pid, url = _item_dedupe_keys(i)
            if pid:
                seen_ids.add(pid)
            if url:
                seen_urls.add(url)
    tried = list(tried_queries)
    tried_norm = {normalize_query(q) for q in tried}
    queries_per_round = max(
        1, _map_int(env_map, "LAST30DAYS_X_DIG_QUERIES", DIG_QUERIES_PER_ROUND)
    )
    seeds_per_round = max(
        0, _map_int(env_map, "LAST30DAYS_X_DIG_SEEDS", DIG_SEEDS_PER_ROUND)
    )
    new_items: list[dict] = []
    previous_assessment = None

    for round_no in range(rounds):
        blocked = gate.check() if gate else None
        if blocked:
            warnings.append(f"{label} dig stopped: {blocked}")
            break
        produced = False
        round_items: list[dict] = []

        def _run_query(query: str) -> None:
            nonlocal produced
            if normalize_query(query) in tried_norm:
                return
            tried_norm.add(normalize_query(query))
            tried.append(query)
            stats["queries_run"] += 1
            try:
                result = search_fn(query)
            except Exception as exc:
                warnings.append(f"{label} dig query {query!r}: {exc}")
                return
            if result.get("error") and not result.get("items"):
                warnings.append(f"{label} dig query {query!r}: {result['error']}")
                return
            produced = True
            ledger_key = f"{lane}:{query}"
            known = ledger.seen_ids(ledger_key) if ledger else set()
            if ledger:
                ledger.record_query_run(ledger_key)
            found_ids: list[str] = []
            for item in result.get("items") or []:
                pid, url = _item_dedupe_keys(item)
                if (pid and pid in seen_ids) or (url and url in seen_urls):
                    continue
                if pid:
                    seen_ids.add(pid)
                    found_ids.append(pid)
                if url:
                    seen_urls.add(url)
                if pid and pid in known:
                    item["previously_seen"] = True
                item["dig_round"] = round_no + 1
                round_items.append(item)
            if ledger and found_ids:
                ledger.record_page(ledger_key, lane, found_ids, None, False)

        # Deterministic coverage floor: entity phrases mined from this
        # lane's hits, the shared cross-lane corpus, and fresh dig finds —
        # an entity surfacing on HN seeds an X query too.
        for query in _seed_queries(
            topic,
            interim_items + list(seed_items or []) + new_items,
            tried_norm,
            max_seeds=seeds_per_round,
            author_prefix="from:" if lane == "x" else None,
        ):
            _run_query(query)

        try:
            decision = planner.plan(
                objective, filters,
                _dig_feedback(topic, tried, interim_items + new_items,
                              round_no, previous_assessment,
                              queries_per_round,
                              context_note=context_note
                              or _LANE_CONTEXT_NOTES.get(lane)),
                timeout,
            )
        except discovery_providers.ProviderError as exc:
            stats["provider_failed"] = True
            warnings.append(f"{label} dig planner failed after retries: {exc}")
            break
        previous_assessment = decision.get("coverage_summary")
        stats["rounds_run"] = round_no + 1
        if decision.get("action") == "search":
            for query in (decision.get("queries") or [])[:queries_per_round]:
                _run_query(query)
        if not produced:
            # Nothing new this round — planner stop, all-repeats, or all
            # queries failed. Either way the trail is cold.
            break
        new_items.extend(_judge_round(
            _classify_round(round_items, jev, objective, stats, warnings,
                            timeout),
            judge, objective, stats, warnings, timeout,
        ))
        if stats.get("provider_failed"):
            warnings.append(f"{label} dig stopped: provider failure")
            break
    stats["new_items"] = len(new_items)
    return new_items, warnings, stats


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
    seed_items: list[dict] | None = None,
) -> tuple[list[dict], list[str], dict]:
    """LLM-steered iterative dig over GetXAPI.

    Each round the discovery Planner reviews the interim corpus and emits 1-3
    follow-up queries (or declares coverage complete). Follow-ups run through
    ``getxapi.search_exact`` under the daily gate and ledger. Returns
    ``(new_items, warnings, stats)``; new items exclude post ids already in
    ``interim_items``.
    """
    from . import getxapi

    if not token:
        return [], [], {"rounds_run": 0, "queries_run": 0, "new_items": 0,
                        "jev_rejected": 0, "judge_rejected": 0}

    def _search(query: str) -> dict:
        return getxapi.search_exact(
            query, from_date, to_date, depth=depth, token=token,
            gate=gate, ledger=ledger,
        )

    return dig_source(
        "x", topic, interim_items, tried_queries,
        search_fn=_search, rounds=rounds, environ=environ, gate=gate,
        timeout=timeout, seed_items=seed_items,
    )


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
        text = "\n".join(
            part for part in [
                str(item.get("title") or "").strip(),
                str(
                    item.get("text") or item.get("body")
                    or item.get("snippet") or ""
                ).strip(),
            ] if part
        )
        try:
            judgement = jev.classify(objective, [objective], {
                "text": text[:2000],
                "author": str(item.get("author_handle") or item.get("author") or ""),
                "url": str(item.get("url") or item.get("hn_url") or ""),
            }, timeout)
        except discovery_providers.ProviderError as exc:
            stats["provider_failed"] = True
            warnings.append(
                f"Dig classifier failed after retries: {exc}"
                " (keeping remaining items unclassified)"
            )
            for remaining in items[idx:]:
                remaining["jev_unclassified"] = True
                kept.append(remaining)
            break
        score = judgement["probabilities"].get("c0", 0.0)
        sufficient = judgement.get("evidence_sufficient", 0.0)
        item["jev_score"] = round(score, 3)
        if sufficient >= JEV_EVIDENCE and score <= JEV_REJECT:
            stats["jev_rejected"] += 1
            continue
        kept.append(item)
    return kept


# Judge verdicts below this band drop the item — the coarse filter already
# passed, so a low score means thin/duplicate/spam signal, not just off-topic.
JUDGE_DROP_BELOW = 50


def _judge_round(items, judge, objective, stats, warnings, timeout):
    """Second-stage verdict on one round's Jev survivors; <50 drops.

    Fail-open like the classifier: a judge outage keeps the batch rather
    than silently emptying it.
    """
    if not judge or not items:
        return items
    from . import discovery_providers
    kept = []
    for start in range(0, len(items), judge.MAX_CANDIDATES):
        chunk = items[start:start + judge.MAX_CANDIDATES]
        candidates = [
            {
                "i": offset,
                "title": str(item.get("title") or ""),
                "text": str(
                    item.get("text") or item.get("body")
                    or item.get("snippet") or ""
                )[:1200],
            }
            for offset, item in enumerate(chunk)
        ]
        try:
            scores = judge.judge(objective, candidates, timeout)
        except discovery_providers.ProviderError as exc:
            stats["provider_failed"] = True
            warnings.append(
                f"Dig judge failed after retries: {exc}"
                " (keeping remaining items unjudged)"
            )
            kept.extend(items[start:])
            break
        for offset, item in enumerate(chunk):
            score = scores.get(offset)
            if score is None:
                kept.append(item)
                continue
            item["judge_score"] = score
            if score < JUDGE_DROP_BELOW:
                stats["judge_rejected"] += 1
                continue
            kept.append(item)
    return kept
