"""Tests for the multi-source dig lane wiring in _run_multi_source_dig.

Covers which lanes activate given `available` and LAST30DAYS_X_DIG_SOURCES,
and that each lane's search_fn delegates to the right source adapter.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from lib import env, pipeline, schema, x_research


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "CONFIG_DIR", tmp_path)
    yield


def _plan() -> schema.QueryPlan:
    return schema.QueryPlan(
        intent="what happened",
        freshness_mode="recent",
        cluster_mode="default",
        raw_topic="test topic",
        subqueries=[
            schema.SubQuery(
                label="q1",
                search_query="test topic",
                ranking_query="test topic",
                sources=["x"],
            )
        ],
        source_weights={},
    )


ALL_SOURCES = [
    "hackernews", "grounding", "github", "bluesky", "youtube",
    "arxiv", "techmeme",
]


def _capture_lanes(monkeypatch, available, dig_sources=None,
                   enumerate_flag=None):
    """Run _run_multi_source_dig with dig_source stubbed; return {src: fn}."""
    captured: dict[str, object] = {}

    def fake_dig_source(source, query, interim, tried, *, search_fn, rounds,
                        ledger, seed_items):
        captured[source] = search_fn
        return [], [], {"rounds_run": 0, "queries_run": 0}

    monkeypatch.setattr(x_research, "dig_source", fake_dig_source)
    config = {"_x_dig_rounds": 1}
    if dig_sources is not None:
        config["LAST30DAYS_X_DIG_SOURCES"] = dig_sources
    if enumerate_flag is not None:
        config["LAST30DAYS_X_DIG_ENUMERATE"] = enumerate_flag
    pipeline._run_multi_source_dig(
        topic="test topic",
        bundle=schema.RetrievalBundle(),
        plan=_plan(),
        config=config,
        depth="default",
        date_range=("2026-09-01", "2026-09-29"),
        web_backend="none",
        available=list(available),
        mock=False,
    )
    return captured


def test_all_diggable_sources_get_lanes(monkeypatch):
    lanes = _capture_lanes(monkeypatch, ALL_SOURCES)
    # Keyless lanes are always present; keyed/binary lanes join via
    # available; enumeration lanes follow their base source's availability.
    assert set(lanes) == {
        "hackernews", "grounding", "googlenews", "reddit",
        "github", "bluesky", "youtube", "arxiv", "techmeme",
        "hackernews_enum", "github_enum",
    }


def test_enum_lanes_disabled_by_flag(monkeypatch):
    lanes = _capture_lanes(monkeypatch, ALL_SOURCES, enumerate_flag="0")
    assert "hackernews_enum" not in lanes
    assert "github_enum" not in lanes


def test_enum_lanes_filter_to_base_source(monkeypatch):
    lanes = _capture_lanes(
        monkeypatch, ALL_SOURCES, dig_sources="hackernews_enum")
    assert set(lanes) == {"hackernews_enum"}


def test_enum_lane_enumerates_once(monkeypatch):
    lanes = _capture_lanes(
        monkeypatch, ["hackernews"], dig_sources="hackernews_enum")
    assert "hackernews_enum" in lanes
    fn = lanes["hackernews_enum"]
    fake = {"hits": [{"objectID": "1", "title": "Show HN: x"}]}
    parsed = [{"id": "1", "title": "Show HN: x", "url": "u"}]
    with patch.object(
        pipeline.hackernews, "enumerate_algolia_window", return_value=fake
    ) as mock_enum, patch.object(
        pipeline.hackernews, "parse_hackernews_response", return_value=parsed
    ):
        first = fn("anything")
        second = fn("anything else")
    mock_enum.assert_called_once()
    assert first["items"] == parsed
    assert second == {"items": []}


def test_gated_lanes_skip_when_unavailable(monkeypatch):
    lanes = _capture_lanes(monkeypatch, ["hackernews"])
    assert set(lanes) == {
        "hackernews", "googlenews", "reddit", "hackernews_enum",
    }


def test_lane_filter_restricts_sources(monkeypatch):
    lanes = _capture_lanes(monkeypatch, ALL_SOURCES, dig_sources="github")
    assert set(lanes) == {"github"}


def test_lane_filter_accepts_news_alias(monkeypatch):
    lanes = _capture_lanes(monkeypatch, ALL_SOURCES, dig_sources="news")
    assert set(lanes) == {"googlenews"}


def _run_fn_test(monkeypatch, lane, available, module_name, func_name,
                 response, parse_name=None, parse_result=None):
    """Activate one lane, invoke its search_fn, assert adapter delegation."""
    lanes = _capture_lanes(monkeypatch, available, dig_sources=lane)
    assert lane in lanes
    module = getattr(pipeline, module_name)
    with patch.object(module, func_name, return_value=response) as mock_fn:
        if parse_name:
            with patch.object(module, parse_name,
                              return_value=parse_result or []) as mock_parse:
                out = lanes[lane]("some dig query")
            mock_parse.assert_called_once()
        else:
            out = lanes[lane]("some dig query")
        mock_fn.assert_called_once()
    return out


def test_github_lane_searches_and_parses(monkeypatch):
    monkeypatch.setattr(pipeline.github, "resolve_token", lambda t=None: None)
    out = _run_fn_test(
        monkeypatch, "github", ["github"], "github", "search_github",
        {"items": [{"id": 1}]}, "parse_github_response", [{"id": "g1"}],
    )
    assert out == {"items": [{"id": "g1"}], "error": None}


def test_bluesky_lane_searches_and_parses(monkeypatch):
    out = _run_fn_test(
        monkeypatch, "bluesky", ["bluesky"], "bluesky", "search_bluesky",
        {"posts": [{}]}, "parse_bluesky_response", [{"id": "b1"}],
    )
    assert out == {"items": [{"id": "b1"}], "error": None}


def test_youtube_lane_searches_and_parses(monkeypatch):
    out = _run_fn_test(
        monkeypatch, "youtube", ["youtube"], "youtube_yt", "search_youtube",
        {"items": [{"id": "v1"}]}, "parse_youtube_response", [{"id": "v1"}],
    )
    assert out == {"items": [{"id": "v1"}], "error": None}


def test_arxiv_lane_searches_and_parses(monkeypatch):
    out = _run_fn_test(
        monkeypatch, "arxiv", ["arxiv"], "arxiv", "search_arxiv",
        {"results": [{}]}, "parse_arxiv_response", [{"id": "a1"}],
    )
    assert out == {"items": [{"id": "a1"}], "error": None}


def test_techmeme_lane_searches_and_parses(monkeypatch):
    out = _run_fn_test(
        monkeypatch, "techmeme", ["techmeme"], "techmeme", "search_techmeme",
        {"results": [{}]}, "parse_techmeme_response", [{"id": "t1"}],
    )
    assert out == {"items": [{"id": "t1"}], "error": None}


def test_search_fn_failure_returns_error_not_raise(monkeypatch):
    lanes = _capture_lanes(monkeypatch, ["github"], dig_sources="github")
    monkeypatch.setattr(
        pipeline.github, "search_github",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(pipeline.github, "resolve_token", lambda t=None: None)
    out = lanes["github"]("query")
    assert out["items"] == []
    assert "boom" in out["error"]
