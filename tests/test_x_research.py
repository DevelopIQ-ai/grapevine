"""Tests for lib/x_research.py — the pixie GetXAPI researcher port.

Covers the daily capacity gate (budget + retry latch), the cross-run JSON
ledger (seen ids, cursor state, eviction), and the LLM-steered dig loop
(planner drives follow-up queries until coverage or budget stops it).
"""
import json
import time
from unittest.mock import patch

import pytest

from lib import env, getxapi, http, x_research
from lib import discovery_providers


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    """Point the ledger/gate files at a tmp dir and restore CONFIG_DIR after."""
    monkeypatch.setattr(env, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("LAST30DAYS_GETXAPI_DAILY_BUDGET", raising=False)
    monkeypatch.delenv("LAST30DAYS_X_LEDGER", raising=False)
    monkeypatch.delenv("LAST30DAYS_X_LEDGER_MAX_QUERIES", raising=False)
    monkeypatch.delenv("LAST30DAYS_X_LEDGER_MAX_IDS", raising=False)
    monkeypatch.delenv("LAST30DAYS_X_DIG_QUERIES", raising=False)
    # Dig tests swap in fake classifiers explicitly; keep Jev off by default
    # so a stray real API key in the environment can't trigger network calls.
    monkeypatch.setenv("LAST30DAYS_X_DIG_JEV", "0")
    yield


def tweet(post_id="111", handle="alice"):
    return {
        "id": post_id,
        "text": "interesting post",
        "author": {"userName": handle, "name": "Alice", "description": "builder",
                   "followers": 9001, "location": "SF"},
        "createdAt": "Fri Sep 18 10:00:00 +0000 2026",
        "likeCount": 7,
    }


# ---------------------------------------------------------------------------
# DailyGate
# ---------------------------------------------------------------------------

def test_gate_charges_and_persists(tmp_path):
    gate = x_research.DailyGate()
    assert gate.check() is None
    gate.charge()
    gate.charge()
    state = json.loads((tmp_path / x_research.GATE_FILE).read_text())
    assert state["calls"] == 2


def test_gate_budget_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("LAST30DAYS_GETXAPI_DAILY_BUDGET", "2")
    gate = x_research.DailyGate()
    gate.charge()
    gate.charge()
    assert gate.check() == "daily budget reached (2 calls/day)"


def test_gate_run_cap_blocks_across_instances(tmp_path, monkeypatch):
    x_research.DailyGate._run_calls = 0
    monkeypatch.setenv("LAST30DAYS_MAX_X_CALLS", "2")
    gate = x_research.DailyGate()
    gate.charge()
    gate.charge()
    assert gate.check() == "run call cap reached (2 calls this run)"
    # getxapi builds a fresh gate per call site; the cap must still hold.
    assert x_research.DailyGate().check() == "run call cap reached (2 calls this run)"
    x_research.DailyGate._run_calls = 0


def test_gate_latch_blocks_until_retry_window(tmp_path):
    gate = x_research.DailyGate()
    gate.latch()
    assert "rate limited" in (gate.check() or "")
    # A fresh gate object reads the same latched state file.
    assert "rate limited" in (x_research.DailyGate().check() or "")
    # Expired latch clears.
    past = {"day": x_research.datetime.now(x_research.UTC).date().isoformat(),
            "calls": 1, "retry_until": time.time() - 10}
    (tmp_path / x_research.GATE_FILE).write_text(json.dumps(past))
    assert x_research.DailyGate().check() is None


def test_gate_disabled_without_config_dir(monkeypatch):
    monkeypatch.setattr(env, "CONFIG_DIR", None)
    gate = x_research.DailyGate()
    gate.charge()
    gate.latch()
    assert gate.check() is None


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def test_ledger_roundtrip_and_dedupe(tmp_path):
    ledger = x_research.Ledger()
    ledger.record_query_run("AI Agents ")
    ledger.record_page("AI Agents ", "Latest", ["1", "2"], "cur-1", True)
    ledger.record_page("AI Agents ", "Latest", ["2", "3"], "cur-2", False)
    fresh = x_research.Ledger()  # reload from disk — cross-run behavior
    key = x_research.normalize_query("ai   agents")
    assert fresh.seen_ids("ai   agents") == {"1", "2", "3"}
    entry = fresh._entry(key)
    assert entry["runs"] == 1
    assert entry["products"]["Latest"]["has_more"] is False


def test_ledger_disabled_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("LAST30DAYS_X_LEDGER", "0")
    ledger = x_research.Ledger()
    assert ledger.enabled is False
    ledger.record_page("q", "Latest", ["1"], None, False)
    assert ledger.seen_ids("q") == set()
    assert not (tmp_path / x_research.LEDGER_FILE).exists()


def test_ledger_evicts_oldest_queries(tmp_path, monkeypatch):
    monkeypatch.setenv("LAST30DAYS_X_LEDGER_MAX_QUERIES", "3")
    ledger = x_research.Ledger()
    for i in range(5):
        ledger.record_query_run(f"q{i}")
    queries = ledger._data["queries"]
    assert len(queries) == 3
    assert "q4" in queries and "q0" not in queries


def test_ledger_ids_cap_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LAST30DAYS_X_LEDGER_MAX_IDS", "3")
    ledger = x_research.Ledger()
    ledger.record_page("q", "Latest", ["1", "2", "3", "4", "5"], None, False)
    assert x_research.Ledger().seen_ids("q") == {"3", "4", "5"}


def test_ledger_caps_zero_means_unbounded(tmp_path, monkeypatch):
    monkeypatch.setenv("LAST30DAYS_X_LEDGER_MAX_QUERIES", "0")
    monkeypatch.setenv("LAST30DAYS_X_LEDGER_MAX_IDS", "0")
    ledger = x_research.Ledger()
    for i in range(6):
        ledger.record_query_run(f"q{i}")
    assert len(ledger._data["queries"]) == 6


def test_normalize_query():
    assert x_research.normalize_query("  AI   Agents\n") == "ai agents"
    assert x_research.normalize_query(None) == ""


# ---------------------------------------------------------------------------
# getxapi integration: gate + ledger through search_x
# ---------------------------------------------------------------------------

def test_search_x_gate_blocks_all_lanes(monkeypatch):
    monkeypatch.setenv("LAST30DAYS_GETXAPI_DAILY_BUDGET", "0")  # budget off
    gate = x_research.DailyGate()
    gate.latch()  # simulate a recent 429
    with patch.object(http, "get") as mock_get:
        result = getxapi.search_x("AI agents", "2026-08-19", "2026-09-19",
                                  depth="quick", token="dummy", gate=gate)
    assert mock_get.call_count == 0
    assert "rate limited" in result["error"]


def test_search_x_ledger_marks_previously_seen(tmp_path):
    ledger = x_research.Ledger()
    ledger.record_page("ai agents", "Latest", ["111"], None, False)
    page = {"tweets": [tweet("111"), tweet("222")], "has_more": False}
    with patch.object(http, "get", return_value=page):
        result = getxapi.search_x("AI agents", "2026-08-19", "2026-09-19",
                                  depth="quick", token="dummy", ledger=ledger)
    by_id = {i["post_id"]: i for i in result["items"]}
    assert by_id["111"]["previously_seen"] is True
    assert "previously_seen" not in by_id["222"]
    # The page was recorded into the ledger too.
    assert x_research.Ledger().seen_ids("ai agents") >= {"111", "222"}


def test_search_x_charges_gate(tmp_path):
    page = {"tweets": [tweet()], "has_more": False}
    with patch.object(http, "get", return_value=page):
        getxapi.search_x("AI agents", "2026-08-19", "2026-09-19",
                         depth="quick", token="dummy")
    state = json.loads((tmp_path / x_research.GATE_FILE).read_text())
    # one expanded query x Latest+Top lanes on quick depth
    assert state["calls"] >= 2


def test_search_x_author_fields(tmp_path):
    page = {"tweets": [tweet()], "has_more": False}
    with patch.object(http, "get", return_value=page):
        result = getxapi.search_x("AI agents", "2026-08-19", "2026-09-19",
                                  depth="quick", token="dummy")
    item = result["items"][0]
    assert item["author_name"] == "Alice"
    assert item["author_bio"] == "builder"
    assert item["author_followers"] == 9001
    assert item["author_location"] == "SF"


# ---------------------------------------------------------------------------
# dig(): LLM-steered follow-up rounds
# ---------------------------------------------------------------------------

class FakePlanner:
    def __init__(self, environ=None):
        self.calls = 0

    def plan(self, objective, filters, feedback, timeout):
        self.calls += 1
        if self.calls == 1:
            return {
                "action": "search",
                "reason": "chase",
                "coverage_summary": "needs more",
                "queries": ["alt phrasing", "from:bob", "alt phrasing"],
                "usage": {},
            }
        return {"action": "stop", "reason": "done",
                "coverage_summary": "covered", "queries": [], "usage": {}}


def test_dig_runs_followups_and_dedupes(tmp_path, monkeypatch):
    monkeypatch.setenv("LAST30DAYS_X_DIG_SEEDS", "0")
    interim = [{"post_id": "1", "text": "hit", "url": "https://x.com/a/status/1",
                "author_handle": "a"}]
    monkeypatch.setattr(discovery_providers, "Planner", FakePlanner)
    dig_page = {"items": [
        {"post_id": "1", "url": "https://x.com/a/status/1", "text": "dup"},
        {"post_id": "9", "url": "https://x.com/b/status/9", "text": "new",
         "author_handle": "b"},
    ]}
    with patch.object(getxapi, "search_exact", return_value=dig_page) as mock_exact:
        items, warnings, stats = x_research.dig(
            "AI agents", interim, ["AI agents"],
            from_date="2026-08-19", to_date="2026-09-19", depth="deep",
            token="dummy", rounds=3,
        )
    assert stats["rounds_run"] == 2
    assert stats["queries_run"] == 2  # repeat query dropped
    assert stats["new_items"] == 1
    assert items[0]["post_id"] == "9" and items[0]["dig_round"] == 1
    assert warnings == []
    # The planner saw interim hits in feedback.
    assert mock_exact.call_count == 2


def test_dig_queries_per_round_from_env(tmp_path, monkeypatch):
    class ManyQueriesPlanner(FakePlanner):
        def plan(self, objective, filters, feedback, timeout):
            self.calls += 1
            self.seen_request = feedback["queries_requested"]
            if self.calls == 1:
                return {"action": "search", "reason": "chase",
                        "coverage_summary": "needs more",
                        "queries": ["q1", "q2", "q3", "q4"], "usage": {}}
            return {"action": "stop", "reason": "done",
                    "coverage_summary": "covered", "queries": [], "usage": {}}

    monkeypatch.setenv("LAST30DAYS_X_DIG_QUERIES", "2")
    monkeypatch.setenv("LAST30DAYS_X_DIG_SEEDS", "0")
    monkeypatch.setattr(discovery_providers, "Planner", ManyQueriesPlanner)
    dig_page = {"items": [{"post_id": "9", "url": "https://x.com/b/9",
                           "text": "new"}]}
    with patch.object(getxapi, "search_exact", return_value=dig_page) as mock:
        items, _, stats = x_research.dig(
            "AI agents", [], ["AI agents"],
            from_date="2026-08-19", to_date="2026-09-19", depth="deep",
            token="dummy", rounds=2,
        )
    assert stats["queries_run"] == 2  # 4 offered, capped at env knob
    assert mock.call_count == 2


def test_dig_jev_classifies_and_drops_irrelevant(tmp_path, monkeypatch):
    class FakeJev:
        def __init__(self, environ=None):
            pass

        def classify(self, objective, criteria, candidate, timeout):
            text = candidate.get("text", "")
            relevant = "on-topic" in text
            return {"probabilities": {"c0": 0.9 if relevant else 0.05},
                    "evidence_sufficient": 0.9, "usage": {}, "model": "fake"}

    monkeypatch.setenv("LAST30DAYS_X_DIG_JEV", "1")
    monkeypatch.setattr(discovery_providers, "Planner", FakePlanner)
    monkeypatch.setattr(discovery_providers, "Jev", FakeJev)
    dig_page = {"items": [
        {"post_id": "9", "url": "https://x.com/b/9", "text": "on-topic new"},
        {"post_id": "8", "url": "https://x.com/c/8", "text": "crypto spam"},
    ]}
    with patch.object(getxapi, "search_exact", return_value=dig_page):
        items, warnings, stats = x_research.dig(
            "AI agents", [], ["AI agents"],
            from_date="2026-08-19", to_date="2026-09-19", depth="deep",
            token="dummy", rounds=2,
        )
    assert stats["jev_rejected"] == 1
    assert [i["post_id"] for i in items] == ["9"]
    assert items[0]["jev_score"] == 0.9
    assert warnings == []


def test_dig_jev_fail_open_on_provider_error(tmp_path, monkeypatch):
    class BrokenJev:
        def __init__(self, environ=None):
            pass

        def classify(self, objective, criteria, candidate, timeout):
            raise discovery_providers.ProviderError("down")

    monkeypatch.setenv("LAST30DAYS_X_DIG_JEV", "1")
    monkeypatch.setattr(discovery_providers, "Planner", FakePlanner)
    monkeypatch.setattr(discovery_providers, "Jev", BrokenJev)
    dig_page = {"items": [
        {"post_id": "9", "url": "https://x.com/b/9", "text": "new"},
        {"post_id": "8", "url": "https://x.com/c/8", "text": "also new"},
    ]}
    with patch.object(getxapi, "search_exact", return_value=dig_page):
        items, warnings, stats = x_research.dig(
            "AI agents", [], ["AI agents"],
            from_date="2026-08-19", to_date="2026-09-19", depth="deep",
            token="dummy", rounds=1,
        )
    # both new items kept when the classifier is down (post ids dedupe across
    # the round's two queries, so 2 unique items reach the classifier)
    assert stats["new_items"] == 2
    assert any("classifier degraded" in w for w in warnings)


def test_dig_stops_on_gate(tmp_path, monkeypatch):
    gate = x_research.DailyGate()
    gate.latch()
    monkeypatch.setattr(discovery_providers, "Planner", FakePlanner)
    items, warnings, stats = x_research.dig(
        "AI agents", [], ["AI agents"],
        from_date="2026-08-19", to_date="2026-09-19", depth="deep",
        token="dummy", rounds=2, gate=gate,
    )
    assert items == [] and stats["queries_run"] == 0
    assert warnings and "rate limited" in warnings[0]


def test_dig_without_planner_key(tmp_path, monkeypatch):
    for name in ("DISCOVERY_PLANNER_API_KEY", "AI_GATEWAY_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    items, warnings, stats = x_research.dig(
        "AI agents", [], [],
        from_date="2026-08-19", to_date="2026-09-19", depth="deep",
        token="dummy", rounds=2,
    )
    assert items == [] and stats["queries_run"] == 0
    assert warnings and "skipped" in warnings[0]


# ---------------------------------------------------------------------------
# dig_source — the generalized loop driving non-X lanes (HN, web)
# ---------------------------------------------------------------------------


class TwoRoundPlanner:
    """Round 1: two broad queries. Round 2: stop."""

    def __init__(self, environ=None):
        self.calls = 0
        self.feedback_notes = []

    def plan(self, objective, filters, feedback, timeout):
        self.calls += 1
        self.feedback_notes.append(feedback.get("context_note") or "")
        if self.calls == 1:
            return {"action": "search", "reason": "chase",
                    "coverage_summary": "needs more",
                    "queries": ["substrate", "show hn sandboxes"], "usage": {}}
        return {"action": "stop", "reason": "done",
                "coverage_summary": "covered", "queries": [], "usage": {}}


def _hn_items():
    return [
        {"id": "hn1", "title": "Show HN: Substrate", "url": "https://s.io",
         "hn_url": "https://news.ycombinator.com/item?id=hn1",
         "author": "bob", "engagement": {"points": 4}},
        {"id": "hn2", "title": "Show HN: DiscoBox", "url": "https://d.io",
         "hn_url": "https://news.ycombinator.com/item?id=hn2",
         "author": "cat", "engagement": {"points": 2}},
    ]


def test_dig_source_runs_queries_dedupes_and_tags_round(tmp_path, monkeypatch):
    monkeypatch.setenv("LAST30DAYS_X_DIG_SEEDS", "0")
    monkeypatch.setattr(discovery_providers, "Planner", TwoRoundPlanner)
    interim = [{"id": "hn1", "url": "https://s.io"}]
    calls = []

    def search(query):
        calls.append(query)
        return {"items": _hn_items()}

    items, warnings, stats = x_research.dig_source(
        "hackernews", "sandbox launches", interim, ["sandbox"],
        search_fn=search, rounds=3,
    )
    assert stats["rounds_run"] == 2 and stats["queries_run"] == 2
    # hn1 deduped against interim; hn2 kept with dig_round stamped.
    assert [i["id"] for i in items] == ["hn2"]
    assert items[0]["dig_round"] == 1
    assert warnings == []


def test_dig_source_hn_context_note_not_x(tmp_path, monkeypatch):
    planner = TwoRoundPlanner()
    monkeypatch.setattr(discovery_providers, "Planner",
                        lambda environ=None: planner)
    items, _, _ = x_research.dig_source(
        "hackernews", "sandbox launches", [], [],
        search_fn=lambda q: {"items": _hn_items()}, rounds=1,
    )
    assert items
    assert planner.feedback_notes and "Algolia" in planner.feedback_notes[0]


def test_dig_source_jev_drops_off_topic(tmp_path, monkeypatch):
    class FakeJev:
        def __init__(self, environ=None):
            pass

        def classify(self, objective, criteria, item, timeout):
            drop = "unrelated" in item["text"].lower()
            return {"probabilities": {"c0": 0.1 if drop else 0.9},
                    "evidence_sufficient": 0.9}

    monkeypatch.delenv("LAST30DAYS_X_DIG_JEV", raising=False)
    monkeypatch.setattr(discovery_providers, "Planner", TwoRoundPlanner)
    monkeypatch.setattr(discovery_providers, "Jev", FakeJev)
    items, warnings, stats = x_research.dig_source(
        "grounding", "sandbox launches", [], [],
        search_fn=lambda q: {"items": [
            {"id": "w1", "title": "Substrate sandbox runtime",
             "url": "https://a.io"},
            {"id": "w2", "title": "unrelated cooking blog",
             "url": "https://b.io"},
        ]},
        rounds=1,
    )
    assert [i["id"] for i in items] == ["w1"]
    assert items[0]["jev_score"] == 0.9
    assert stats["jev_rejected"] >= 1


def test_dig_source_seeds_mine_entity_names(tmp_path, monkeypatch):
    monkeypatch.delenv("LAST30DAYS_X_DIG_SEEDS", raising=False)
    monkeypatch.setattr(discovery_providers, "Planner", TwoRoundPlanner)
    interim = [
        {"id": "a", "title": "Prime Sandboxes goes GA", "url": "https://p.io"},
        {"id": "b", "title": "Prime Sandboxes vs Edera compared",
         "url": "https://e.io"},
    ]
    calls = []

    def search(query):
        calls.append(query)
        return {"items": []}

    x_research.dig_source(
        "hackernews", "sandbox launches", interim, ["sandbox"],
        search_fn=search, rounds=1,
    )
    # "Prime Sandboxes" recurs across interim hits -> deterministic seed query,
    # independent of what the planner chose to ask.
    assert "Prime Sandboxes" in calls


def test_dig_source_judge_drops_low_scores(tmp_path, monkeypatch):
    class FakeJudge:
        MAX_CANDIDATES = 40

        def __init__(self, environ=None):
            pass

        def judge(self, objective, candidates, timeout):
            return {
                c["i"]: 90 if "substrate" in (c["title"] + c["text"]).lower() else 10
                for c in candidates
            }

    monkeypatch.delenv("LAST30DAYS_X_DIG_JUDGE", raising=False)
    monkeypatch.setattr(discovery_providers, "Planner", TwoRoundPlanner)
    monkeypatch.setattr(discovery_providers, "Judge", FakeJudge)
    items, warnings, stats = x_research.dig_source(
        "grounding", "sandbox launches", [], [],
        search_fn=lambda q: {"items": [
            {"id": "w1", "title": "Substrate sandbox runtime",
             "url": "https://a.io"},
            {"id": "w2", "title": "celebrity gossip",
             "url": "https://b.io"},
        ]},
        rounds=1,
    )
    assert [i["id"] for i in items] == ["w1"]
    assert items[0]["judge_score"] == 90
    assert stats["judge_rejected"] == 1


def test_dig_source_ledger_marks_previously_seen(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery_providers, "Planner", TwoRoundPlanner)
    ledger = x_research.Ledger({})
    search = lambda q: {"items": _hn_items()}
    first, _, _ = x_research.dig_source(
        "hackernews", "sandboxes", [], [], search_fn=search,
        rounds=1, ledger=ledger,
    )
    assert not any(i.get("previously_seen") for i in first)
    # Second run, fresh interim: same ids re-fetched are flagged.
    second, _, _ = x_research.dig_source(
        "hackernews", "sandboxes", [], [], search_fn=search,
        rounds=1, ledger=ledger,
    )
    assert second and all(i.get("previously_seen") for i in second)
