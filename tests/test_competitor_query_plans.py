"""Host-written query plans must reach every comparison pipeline."""
import json
import sys
from unittest import mock

import pytest
import last30days as cli
from lib import capabilities, env, fanout, pipeline


def plan(topic):
    return {"intent": "opinion", "freshness_mode": "recent", "cluster_mode": "debate",
            "subqueries": [{"search_query": topic, "ranking_query": topic,
                            "sources": ["reddit", "hackernews"]}]}


def test_keyless_comparison_threads_distinct_plans_to_actual_cli_closures():
    plans = {"ChatGPT": {"query_plan": plan("ChatGPT")},
             "Grok": {"query_plan": plan("Grok")}}
    captured = {}

    def run(**kwargs):
        assert capabilities.check(kwargs["config"],
                                  plan_provided=kwargs.get("external_plan") is not None,
                                  agent_rerank=bool(kwargs["config"].get("_agent_rerank"))) == []
        captured[kwargs["topic"]] = kwargs["external_plan"]
        return type("Report", (), {"artifacts": {}})()

    def invoke(**kwargs):
        kwargs["main_runner"]()
        for entity in kwargs["competitors"]:
            kwargs["competitor_runner"](entity)
        return []  # Stop before rendering; assertions cover the real CLI closures.

    with mock.patch.object(env, "get_config", return_value={"LAST30DAYS_NATIVE_SEARCH": "1"}), \
         mock.patch.object(pipeline, "run", side_effect=run), \
         mock.patch.object(fanout, "run_competitor_fanout", side_effect=invoke), \
         mock.patch.object(sys, "argv", ["last30days", "ChatGPT vs Grok", "--agent-rerank",
                                        "--competitors-plan", json.dumps(plans)]):
        cli.main()
    assert captured == {entity: entry["query_plan"] for entity, entry in plans.items()}


@pytest.mark.parametrize("invalid", [None, [], {}, {"subqueries": []}])
def test_invalid_entity_query_plan_fails_before_source_work(invalid):
    with pytest.raises(SystemExit) as error:
        cli.parse_competitors_plan(json.dumps({"Grok": {"query_plan": invalid}}))
    assert error.value.code == 2


def test_existing_targeting_only_plan_stays_compatible():
    assert cli.parse_competitors_plan('{"Grok":{"subreddits":["grok"]}}')["grok"]["subreddits"] == ["grok"]
