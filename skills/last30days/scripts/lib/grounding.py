"""Web search retrieval via Brave Search, Exa, Serper, Parallel, or a keyless floor."""

from __future__ import annotations

import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from . import dates, env, http, parallel_mcp, schema, web_search_keyless


@dataclass(frozen=True)
class GroundedClaimText:
    """Candidate text with its exact primary evidence item."""

    candidate_id: str
    title: str
    summary: str
    item: schema.SourceItem


def claim_source_map(report: schema.Report) -> dict[str, GroundedClaimText]:
    """Expose only candidate claims that have a clean primary-item trace.

    Freshness verification deliberately starts here instead of scanning all
    report prose. A candidate without a primary ``SourceItem`` cannot produce
    an auditable per-claim verdict.
    """
    grounded: dict[str, GroundedClaimText] = {}
    for candidate in report.ranked_candidates:
        item = schema.candidate_primary_item(candidate)
        if item is None:
            continue
        grounded[candidate.candidate_id] = GroundedClaimText(
            candidate_id=candidate.candidate_id,
            title=candidate.title,
            summary=candidate.snippet or item.snippet or item.body,
            item=item,
        )
    return grounded


# ---------------------------------------------------------------------------
# Brave Search API
# ---------------------------------------------------------------------------

def brave_search(
    query: str, date_range: tuple[str, str], api_key: str, count: int = 5,
) -> tuple[list[dict], dict]:
    url = (
        "https://api.search.brave.com/res/v1/web/search?"
        + urllib.parse.urlencode(
            {
                "q": query,
                "count": count,
                "freshness": f"{date_range[0]}to{date_range[1]}",
            }
        )
    )
    data = http.request("GET", url, headers={"X-Subscription-Token": api_key}, timeout=15)
    items = []
    for i, r in enumerate((data.get("web", {}).get("results", []))[:count]):
        raw_date = r.get("page_age") or ""
        pub_date = _normalize_date(raw_date[:10]) if raw_date else None
        if not _in_date_range(pub_date, date_range):
            continue
        items.append({
            "id": f"WB{i + 1}",
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source_domain": _domain(r.get("url", "")),
            "snippet": r.get("description", ""),
            "date": pub_date,
            "relevance": 0.8,
            "why_relevant": "Brave web search",
        })
    artifact = {"label": "brave", "webSearchQueries": [query], "resultCount": len(items)}
    return items, artifact


# ---------------------------------------------------------------------------
# Exa AI Search
# ---------------------------------------------------------------------------

def exa_search(
    query: str, date_range: tuple[str, str], api_key: str, count: int = 5,
) -> tuple[list[dict], dict]:
    data = http.request(
        "POST", "https://api.exa.ai/search",
        headers={"x-api-key": api_key},
        json_data={
            "query": query,
            "type": "auto",
            "numResults": count,
            "startPublishedDate": f"{date_range[0]}T00:00:00.000Z",
            "endPublishedDate": f"{date_range[1]}T23:59:59.999Z",
            "contents": {"text": {"maxCharacters": 2000}},
        },
        timeout=15,
    )
    items = []
    for i, r in enumerate((data.get("results", []))[:count]):
        if not isinstance(r, dict):
            continue
        url = r.get("url", "")
        if not url:
            continue
        raw_date = r.get("publishedDate") or ""
        pub_date = _normalize_date(raw_date.split("T")[0] if "T" in raw_date else raw_date[:10]) if raw_date else None
        if not _in_date_range(pub_date, date_range):
            continue
        items.append({
            "id": f"WE{i + 1}",
            "title": r.get("title", ""),
            "url": url,
            "source_domain": _domain(url),
            "snippet": (r.get("text") or "")[:500],
            "date": pub_date,
            "relevance": 0.8,
            "why_relevant": "Exa web search",
        })
    artifact = {"label": "exa", "webSearchQueries": [query], "resultCount": len(items)}
    return items, artifact


# ---------------------------------------------------------------------------
# Serper (Google Search wrapper)
# ---------------------------------------------------------------------------

def serper_search(
    query: str, date_range: tuple[str, str], api_key: str, count: int = 5,
) -> tuple[list[dict], dict]:
    data = http.request(
        "POST", "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key},
        json_data={
            "q": query,
            "num": count,
            "tbs": f"cdr:1,cd_min:{_serper_date_param(date_range[0])},cd_max:{_serper_date_param(date_range[1])}",
        },
        timeout=15,
    )
    items = []
    for i, r in enumerate((data.get("organic", []))[:count]):
        raw_date = r.get("date") or ""
        pub_date = _parse_serper_date(raw_date)
        if not _in_date_range(pub_date, date_range):
            continue
        items.append({
            "id": f"WS{i + 1}",
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "source_domain": _domain(r.get("link", "")),
            "snippet": r.get("snippet", ""),
            "date": pub_date,
            "relevance": 0.8,
            "why_relevant": "Serper web search",
        })
    artifact = {"label": "serper", "webSearchQueries": [query], "resultCount": len(items)}
    return items, artifact


# ---------------------------------------------------------------------------
# Parallel AI Search
# ---------------------------------------------------------------------------

def parallel_search(
    query: str, date_range: tuple[str, str], api_key: str, count: int = 5,
) -> tuple[list[dict], dict]:
    data = http.request(
        "POST", "https://api.parallel.ai/v1/search",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_data={
            "search_queries": [query],
            "advanced_settings": {"max_results": count},
        },
        timeout=15,
    )
    items = []
    for i, r in enumerate((data.get("results", []))[:count]):
        if not isinstance(r, dict):
            continue
        url = r.get("url", "")
        if not url:
            continue
        raw_date = r.get("publish_date") or ""
        pub_date = _normalize_date(raw_date[:10]) if raw_date else None
        if not _in_date_range(pub_date, date_range):
            continue
        items.append({
            "id": f"WP{i + 1}",
            "title": r.get("title", ""),
            "url": url,
            "source_domain": _domain(url),
            "snippet": ((r.get("excerpts") or [""])[0] or "")[:500],
            "date": pub_date,
            "relevance": 0.8,
            "why_relevant": "Parallel AI web search",
        })
    artifact = {"label": "parallel", "webSearchQueries": [query], "resultCount": len(items)}
    return items, artifact


# ---------------------------------------------------------------------------
# TinyFish Search API
# ---------------------------------------------------------------------------

def tinyfish_search(
    query: str, date_range: tuple[str, str], api_key: str, count: int = 5,
) -> tuple[list[dict], dict]:
    url = (
        "https://api.search.tinyfish.ai?"
        + urllib.parse.urlencode(
            {
                "query": query,
                "from_date": date_range[0],
                "to_date": date_range[1],
                "purpose": "Find recent discussion, news, and community coverage of the query",
            }
        )
    )
    data = http.request("GET", url, headers={"X-API-Key": api_key}, timeout=20)
    # TinyFish date fields are sparse/relative and the API's own from/to params
    # are advisory, so filter first, then slice — an undated early result must
    # not displace a dated later one.
    items = []
    for i, r in enumerate(data.get("results") or []):
        if not isinstance(r, dict):
            continue
        url_i = r.get("url", "")
        if not url_i:
            continue
        pub_date = _parse_tinyfish_date(str(r.get("date") or ""))
        if not _in_date_range(pub_date, date_range):
            continue
        items.append({
            "id": f"WT{i + 1}",
            "title": r.get("title", ""),
            "url": url_i,
            "source_domain": _domain(url_i),
            "snippet": r.get("snippet", ""),
            "date": pub_date,
            "relevance": 0.8,
            "why_relevant": "TinyFish web search",
        })
    items = items[:count]
    artifact = {"label": "tinyfish", "webSearchQueries": [query], "resultCount": len(items)}
    return items, artifact


# ---------------------------------------------------------------------------
# Google News RSS (keyless — no API key, free)
# ---------------------------------------------------------------------------

def googlenews_search(
    query: str, date_range: tuple[str, str], count: int = 25,
) -> tuple[list[dict], dict]:
    url = (
        "https://news.google.com/rss/search?"
        + urllib.parse.urlencode(
            {
                "q": query,
                "hl": "en-US",
                "gl": "US",
                "ceid": "US:en",
            }
        )
    )
    text = http.get_text(url, accept="application/rss+xml")
    root = None
    if text:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            root = None
    items = []
    if root is not None:
        for i, node in enumerate(root.findall(".//item")):
            link = (node.findtext("link") or "").strip()
            if not link:
                continue
            pub_date = _rfc822_date(node.findtext("pubDate") or "")
            if not _in_date_range(pub_date, date_range):
                continue
            publisher = (node.findtext("source") or "").strip()
            items.append({
                "id": f"GN{i + 1}",
                "title": (node.findtext("title") or "").strip(),
                "url": link,
                "source_domain": publisher or _domain(link),
                "snippet": "",
                "date": pub_date,
                "relevance": 0.8,
                "why_relevant": "Google News",
            })
    items = items[:count]
    artifact = {"label": "googlenews", "webSearchQueries": [query],
                "resultCount": len(items)}
    return items, artifact


def _rfc822_date(raw: str) -> str | None:
    try:
        return parsedate_to_datetime(raw.strip()).date().isoformat()
    except (TypeError, ValueError):
        return None


_TINYFISH_REL_RE = re.compile(
    r"^(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago$", re.IGNORECASE
)
_TINYFISH_REL_DAYS = {
    "second": 0, "minute": 0, "hour": 0,
    "day": 1, "week": 7, "month": 30, "year": 365,
}


def _parse_tinyfish_date(raw: str) -> str | None:
    """Parse TinyFish result dates: 'Aug 30, 2026' or relative '3 days ago'."""
    if not raw:
        return None
    raw = raw.strip()
    normalized = _normalize_date(raw)
    if normalized:
        return normalized
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    m = _TINYFISH_REL_RE.match(raw)
    if m:
        days = int(m.group(1)) * _TINYFISH_REL_DAYS[m.group(2).lower()]
        return (datetime.now().date() - timedelta(days=days)).isoformat()
    return None


def _parse_serper_date(raw: str) -> str | None:
    if not raw:
        return None
    normalized = _normalize_date(raw)
    if normalized:
        return normalized
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None




# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def web_search(
    query: str,
    date_range: tuple[str, str],
    config: dict,
    backend: str = "auto",
) -> tuple[list[dict], dict]:
    """Run web search with the specified or auto-detected backend."""
    if backend == "auto":
        if config.get("BRAVE_API_KEY"):
            backend = "brave"
        elif config.get("EXA_API_KEY"):
            backend = "exa"
        elif config.get("SERPER_API_KEY"):
            backend = "serper"
        elif config.get("PARALLEL_API_KEY"):
            backend = "parallel"
        elif config.get("TINYFISH_API_KEY"):
            backend = "tinyfish"
        elif env.keyless_web_allowed(config):
            # No paid key and the host has no native search -> use the keyless
            # floor. On a native-search host this branch is skipped (the model
            # supplies web results itself), so the engine returns nothing here.
            backend = "keyless"
        else:
            return [], {}
    items: list[dict] = []
    artifact: dict = {}
    if backend == "brave":
        key = config.get("BRAVE_API_KEY")
        if not key:
            raise RuntimeError("BRAVE_API_KEY is required when web_backend='brave'")
        items, artifact = brave_search(query, date_range, key)
    elif backend == "exa":
        key = config.get("EXA_API_KEY")
        if not key:
            raise RuntimeError("EXA_API_KEY is required when web_backend='exa'")
        items, artifact = exa_search(query, date_range, key)
    elif backend == "serper":
        key = config.get("SERPER_API_KEY")
        if not key:
            raise RuntimeError("SERPER_API_KEY is required when web_backend='serper'")
        items, artifact = serper_search(query, date_range, key)
    elif backend == "parallel":
        key = config.get("PARALLEL_API_KEY")
        if not key:
            raise RuntimeError("PARALLEL_API_KEY is required when web_backend='parallel'")
        items, artifact = parallel_search(query, date_range, key)
    elif backend == "parallel-mcp":
        items, artifact = parallel_mcp.search(
            query, date_range, config.get("PARALLEL_API_KEY")
        )
    elif backend == "googlenews":
        items, artifact = googlenews_search(query, date_range)
    elif backend == "tinyfish":
        key = config.get("TINYFISH_API_KEY")
        if not key:
            raise RuntimeError("TINYFISH_API_KEY is required when web_backend='tinyfish'")
        items, artifact = tinyfish_search(query, date_range, key)
    elif backend == "keyless":
        items, artifact = web_search_keyless.keyless_search(query, date_range, config)
    elif backend != "none":
        raise ValueError(f"Unsupported web backend: {backend!r}")
    else:
        return [], {}
    if items and not _reddit_excluded(config):
        # Reddit enrichment is a best-effort secondary fetch on already-retrieved
        # web results. Isolate its HTTP failures in a throwaway capture sink so a
        # reddit.com fetch failure (e.g. a 403 on a datacenter IP) is not
        # attributed to the web/grounding source itself — which would otherwise
        # discard the successfully retrieved results and report the source failed.
        with http.capture_failures():
            items = _enrich_reddit_items(items)
    return items, artifact


def _reddit_excluded(config: dict) -> bool:
    """Return True when EXCLUDE_SOURCES contains 'reddit'.

    Respects the same suppression knob the pipeline uses for source gating,
    so a user who set EXCLUDE_SOURCES=reddit doesn't get Reddit content
    smuggled back in via web-search URLs.
    """
    raw = (config.get("EXCLUDE_SOURCES") or "").split(",")
    return any(s.strip().lower() == "reddit" for s in raw)


def _enrich_reddit_items(items: list[dict]) -> list[dict]:
    """Enrich web search results that are Reddit URLs with thread body and comments.

    Claude Code's WebFetch blocks reddit.com, so the model can't retrieve
    Reddit content from web search results. This fetches it via the public
    JSON API (reddit.com/.../.json) which bypasses that restriction.

    Callers should gate this with EXCLUDE_SOURCES=reddit handling (see
    `_reddit_excluded`) so a user who explicitly excluded Reddit doesn't
    get Reddit content via web-search URLs.
    """
    from . import reddit_enrich
    from .reddit_enrich import RedditRateLimitError

    for item in items:
        url = item.get("url", "")
        if "reddit.com" not in url or "/comments/" not in url:
            continue
        try:
            thread_data = reddit_enrich.fetch_thread_data(url, timeout=8)
            if not thread_data:
                continue
            parsed = reddit_enrich.parse_thread_data(thread_data)
            # selftext lives under parsed["submission"], not at the top level
            selftext = (parsed.get("submission") or {}).get("selftext", "")
            if selftext:
                item["snippet"] = selftext[:2000]
            comments = parsed.get("comments", [])
            top = reddit_enrich.get_top_comments(comments)
            if top:
                item["top_comments"] = [
                    {"score": c.get("score", 0), "excerpt": (c.get("body") or "")[:200]}
                    for c in top[:5]
                ]
            item["enriched_via"] = "reddit_json_api"
        except RedditRateLimitError as exc:
            # Stop iterating to avoid flooding more 429s
            sys.stderr.write(f"[Web] Reddit rate-limited, halting enrichment: {exc}\n")
            break
        except Exception as exc:
            sys.stderr.write(f"[Web] Reddit enrichment failed for {url}: {exc}\n")
    return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_date(value: object) -> str | None:
    if value is None:
        return None
    parsed = dates.parse_date(str(value).strip())
    if not parsed:
        return None
    return parsed.date().isoformat()


def _serper_date_param(iso_date: str) -> str:
    """Convert YYYY-MM-DD to MM/DD/YYYY for Serper tbs parameter."""
    parts = iso_date.split("-")
    return f"{parts[1]}/{parts[2]}/{parts[0]}"


def _in_date_range(pub_date: str | None, date_range: tuple[str, str]) -> bool:
    if not pub_date:
        return False
    return date_range[0] <= pub_date <= date_range[1]


def _domain(url: str) -> str:
    return urlparse(url).netloc.strip().lower()
