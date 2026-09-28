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


def _search(query, from_date, to_date, token, limit, topic, product="Latest", prefix="GX", depth="default"):
    """Bound page spend, preserve partial evidence, and never echo provider errors."""
    items, seen_ids, seen_cursors = [], set(), set()
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
            response = http.get(BASE_URL + "?" + urlencode(params),
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=30, retries=2)
        except http.HTTPError as exc:
            status = getattr(exc, "status_code", None)
            kind = {401: "auth failed", 403: "auth failed", 402: "payment required",
                    429: "rate limited"}.get(status, "request failed")
            return items, f"GetXAPI {kind} (HTTP {status})"
        except Exception as exc:
            return items, f"GetXAPI request failed ({type(exc).__name__})"
        if not isinstance(response, dict) or not isinstance(response.get("tweets"), list):
            return items, "GetXAPI invalid response schema"
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
                item['post_id'] = post_id
                # The shared Xquik parser truncates to 500 characters. Jev needs
                # all returned evidence; downstream consumers bound their own input.
                item['text'] = str(tweet.get('text') or '').strip()
                items.append(item)
            if len(items) >= limit:
                return items, None
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


def search_x(topic, from_date, to_date, depth="default", token=""):
    """Search topics with upstream query expansion and depth-dependent limits."""
    if not token:
        return {"items": [], "error": "No GETXAPI_KEY configured"}
    cfg = xquik.DEPTH_CONFIG.get(depth, xquik.DEPTH_CONFIG["default"])
    items, seen, errors = [], set(), []
    queries = [topic] if os.environ.get('LAST30DAYS_GETXAPI_EXACT_QUERY') == '1' else xquik.expand_xquik_queries(topic, depth)
    # Split each query's item budget across the Latest and Top product lanes.
    per_product_limit = max(1, -(-cfg['limit'] // len(TOPIC_PRODUCTS)))
    for query in queries:
        for product in TOPIC_PRODUCTS:
            found, error = _search(query, from_date, to_date, token,
                                   per_product_limit, topic, product=product,
                                   depth=depth)
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


def _handles(handles, topic, from_date, to_date, count_per, token, mentions, depth="default"):
    if not token:
        return []
    items, seen = [], set()
    for raw in handles:
        handle = str(raw).strip().lstrip('@')
        if not re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle):
            continue
        query = f"@{handle}" if mentions else f"from:{handle}"
        found, error = _search(query, from_date, to_date, token, count_per, topic,
                               prefix="GXA" if mentions else "GXF", depth=depth)
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
