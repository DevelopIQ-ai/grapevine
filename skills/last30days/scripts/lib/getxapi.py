"""GetXAPI public X search. Requires GETXAPI_KEY, never an X session cookie."""
from __future__ import annotations

import os
import re
from datetime import date, timedelta
from urllib.parse import urlencode

from . import http, xquik
from .x_api import is_own_post

BASE_URL = "https://api.getxapi.com/twitter/tweet/advanced_search"
DEFAULT_MAX_PAGES = 10
# Per-lane page budget by run depth: deeper runs spend more pages before giving
# up on a cursor. LAST30DAYS_GETXAPI_MAX_PAGES overrides all of them.
DEPTH_MAX_PAGES = {"quick": 5, "default": 10, "deep": 20}
# Each topic query fans out to both GetXAPI product lanes: Latest (chronological
# firehose) and Top (engagement-ranked). Top surfaces high-signal posts that sit
# outside the recent tail Latest returns.
TOPIC_PRODUCTS = ("Latest", "Top")


def _max_pages(depth="default"):
    """Per-lane page budget; LAST30DAYS_GETXAPI_MAX_PAGES overrides."""
    try:
        return max(1, int(os.environ.get("LAST30DAYS_GETXAPI_MAX_PAGES", "")))
    except ValueError:
        return DEPTH_MAX_PAGES.get(depth, DEFAULT_MAX_PAGES)


def _search(query, from_date, to_date, token, limit, topic, product="Latest", prefix="GX", depth="default", gate=None, ledger=None):
    """Bound page spend, preserve partial evidence, and never echo provider errors.

    ``gate`` (x_research.DailyGate) bounds daily spend and short-circuits every
    lane while a provider 429/5xx latch is active. ``ledger``
    (x_research.Ledger) dedupes post ids across runs and records cursor state
    per product so digs accumulate instead of restarting.
    """
    if gate is not None:
        blocked = gate.check()
        if blocked:
            return [], f"GetXAPI {blocked}"
    items, seen_ids, seen_cursors = [], set(), set()
    ledger_seen = ledger.seen_ids(query) if ledger is not None else set()
    cursor = None
    # The engine owns the date window even if a planner supplied operators.
    query = re.sub(r"\b(?:since|until):\S+", "", query).strip()
    # Engine dates are inclusive; X until: is exclusive. Include the final day.
    until = (date.fromisoformat(to_date) + timedelta(days=1)).isoformat()
    params = {"q": f"{query} since:{from_date} until:{until}", "product": product}
    for _ in range(_max_pages(depth)):
        if cursor:
            params["cursor"] = cursor
        try:
            if gate is not None:
                gate.charge()
            response = http.get(BASE_URL + "?" + urlencode(params),
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=30, retries=2)
        except http.HTTPError as exc:
            status = getattr(exc, "status_code", None)
            # Latch on 429 only: http.get already retries transient 5xx, and a
            # persistent outage surfaces as a fatal error that halts the lanes.
            if gate is not None and status == 429:
                gate.latch()
            kind = {401: "auth failed", 403: "auth failed", 402: "payment required",
                    429: "rate limited"}.get(status, "request failed")
            return items, f"GetXAPI {kind} (HTTP {status})"
        except Exception as exc:
            return items, f"GetXAPI request failed ({type(exc).__name__})"
        if not isinstance(response, dict) or not isinstance(response.get("tweets"), list):
            return items, "GetXAPI invalid response schema"
        page_ids = []
        for tweet in response["tweets"]:
            if not isinstance(tweet, dict):
                continue
            post_id = str(tweet.get("id") or "")
            author = tweet.get("author") or {}
            if not isinstance(author, dict):
                continue
            handle = str(author.get("userName") or author.get("username") or "").lstrip("@")
            if not post_id.isdigit() or not re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle) or post_id in seen_ids:
                continue
            normalized = dict(tweet, author=dict(author, username=handle))
            item = xquik._parse_tweet(normalized, len(items), topic, id_prefix=prefix)
            if item and (not item['date'] or from_date <= item['date'] <= to_date):
                seen_ids.add(post_id)
                page_ids.append(post_id)
                item['post_id'] = post_id
                if post_id in ledger_seen:
                    item['previously_seen'] = True
                # The shared Xquik parser truncates to 500 characters. Jev needs
                # all returned evidence; downstream consumers bound their own input.
                item['text'] = str(tweet.get('text') or '').strip()
                items.append(item)
            if len(items) >= limit:
                if ledger is not None:
                    ledger.record_page(query, product, page_ids,
                                       response.get("next_cursor"), True)
                return items, None
        if ledger is not None:
            ledger.record_page(query, product, page_ids,
                               response.get("next_cursor"),
                               bool(response.get("has_more")))
        if not response.get("has_more"):
            return items, None
        cursor = response.get("next_cursor")
        if not isinstance(cursor, str) or not cursor.strip() or cursor in seen_cursors:
            return items, "GetXAPI pagination missing or repeated cursor"
        seen_cursors.add(cursor)
    return items, "GetXAPI page limit reached; partial results"


def _is_fatal(error):
    """Fatal errors (auth, rate-limit, transport, schema) halt all further spend.
    Lane-local exhaustion — page limit or a stuck cursor — only ends that lane."""
    return "page limit" not in error and "pagination" not in error


def search_x(topic, from_date, to_date, depth="default", token="", gate=None, ledger=None):
    """Search topics with upstream query expansion and depth-dependent limits."""
    if not token:
        return {"items": [], "error": "No GETXAPI_KEY configured"}
    from . import x_research
    if gate is None:
        gate = x_research.DailyGate()
    if ledger is None:
        ledger = x_research.Ledger()
    cfg = xquik.DEPTH_CONFIG.get(depth, xquik.DEPTH_CONFIG["default"])
    items, seen, errors = [], set(), []
    queries = [topic] if os.environ.get('LAST30DAYS_GETXAPI_EXACT_QUERY') == '1' else xquik.expand_xquik_queries(topic, depth)
    # Split each query's item budget across the Latest and Top product lanes.
    per_product_limit = max(1, -(-cfg['limit'] // len(TOPIC_PRODUCTS)))
    for query in queries:
        if ledger.enabled:
            ledger.record_query_run(query)
        for product in TOPIC_PRODUCTS:
            found, error = _search(query, from_date, to_date, token,
                                   per_product_limit, topic, product=product,
                                   depth=depth, gate=gate, ledger=ledger)
            for item in found:
                if item['post_id'] not in seen:
                    seen.add(item['post_id'])
                    item['id'] = f"GX{len(items) + 1}"
                    items.append(item)
            if error:
                errors.append(error)
                if _is_fatal(error):
                    # Do not spend more after auth, rate-limit, or transport failure.
                    return {"items": items, "error": "; ".join(errors)}
    return {"items": items, **({"error": "; ".join(errors)} if errors else {})}


def search_exact(query, from_date, to_date, *, limit=40, token="", depth="default", gate=None, ledger=None):
    """Run one already-formed query through both product lanes (dig follow-ups).

    Unlike ``search_x`` this does no query expansion — the caller owns the
    query text. Used by the LLM-steered dig loop in lib/x_research.py.
    """
    if not token:
        return {"items": [], "error": "No GETXAPI_KEY configured"}
    from . import x_research
    if gate is None:
        gate = x_research.DailyGate()
    if ledger is None:
        ledger = x_research.Ledger()
    if ledger.enabled:
        ledger.record_query_run(query)
    items, seen, errors = [], set(), []
    per_product_limit = max(1, -(-limit // len(TOPIC_PRODUCTS)))
    for product in TOPIC_PRODUCTS:
        found, error = _search(query, from_date, to_date, token,
                               per_product_limit, query, product=product,
                               prefix="GXD", depth=depth,
                               gate=gate, ledger=ledger)
        for item in found:
            if item['post_id'] not in seen:
                seen.add(item['post_id'])
                items.append(item)
        if error:
            errors.append(error)
            if _is_fatal(error):
                break
    return {"items": items, **({"error": "; ".join(errors)} if errors else {})}


def _handles(handles, topic, from_date, to_date, count_per, token, mentions, depth="default"):
    if not token:
        return []
    from . import x_research
    gate = x_research.DailyGate()
    ledger = x_research.Ledger()
    items, seen = [], set()
    for raw in handles:
        handle = str(raw).strip().lstrip('@')
        if not re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle):
            continue
        query = f"@{handle}" if mentions else f"from:{handle}"
        if ledger.enabled:
            ledger.record_query_run(query)
        found, error = _search(query, from_date, to_date, token, count_per, topic,
                               prefix="GXA" if mentions else "GXF", depth=depth,
                               gate=gate, ledger=ledger)
        for item in found:
            if mentions and is_own_post(item['url'], handle):
                continue
            if item['post_id'] not in seen:
                seen.add(item['post_id'])
                item['id'] = f"{'GXA' if mentions else 'GXF'}{len(items) + 1}"
                items.append(item)
        if error:
            break
    return items


def search_handles(handles, topic, from_date, to_date, *, count_per=8, token="", depth="default"):
    """Posts authored by a person, without requiring topic words in each post."""
    return _handles(handles, topic, from_date, to_date, count_per, token, False, depth)


def search_mentions(handles, from_date, to_date, *, topic="", count_per=5, token="", depth="default"):
    """Posts mentioning a person, excluding that person's own posts."""
    return _handles(handles, topic, from_date, to_date, count_per, token, True, depth)
