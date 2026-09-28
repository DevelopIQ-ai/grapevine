"""Live recall benchmark: did a run surface every known-good item?

Spec JSON: {"topic", "from_date", "to_date", "args", "expected":
[{"label", "match"}]}. Each `match` is a case-insensitive substring checked
against every emitted item's text, title, url, author_handle, and post_id.

Usage:
    python3 tests/eval/recall_bench.py tests/eval/recall/sandbox-launches.json
    python3 tests/eval/recall_bench.py spec.json --dig 0   # dig-off baseline
    python3 tests/eval/recall_bench.py spec.json --lift    # dig-off vs dig-on
        lift on any topic, no answer key needed

Prints a hit/miss table and recall score. Exits 1 below --floor (default 0).
--lift mode runs the spec twice (dig disabled, then dig enabled per spec args)
and reports how many extra Jev-passing items the dig surfaced — an evergreen
measure that works on any topic without a hand-built answer key.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "skills" / "last30days" / "scripts" / "last30days.py"
CONFIG_ENV = Path.home() / ".config" / "last30days" / ".env"


def _load_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def _haystack(item: dict) -> str:
    parts = [
        str(item.get("text") or ""),
        str(item.get("title") or ""),
        str(item.get("url") or ""),
        str(item.get("author_handle") or ""),
        str(item.get("author") or ""),
        str(item.get("post_id") or ""),
    ]
    return "\n".join(parts).lower()


def _find_items(report) -> list[dict]:
    if isinstance(report, dict):
        for key in ("items", "results"):
            if isinstance(report.get(key), list):
                return report[key]
        # report JSON may nest sections; walk for lists of dicts with text/url
        found: list[dict] = []
        for value in report.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                found.extend(value)
            elif isinstance(value, dict):
                found.extend(_find_items(value))
        return found
    return []


def _run_engine(spec: dict, spec_stem: str, engine_args: list[str],
                timeout: int) -> tuple[list[dict], str]:
    plan = {"raw_topic": spec["topic"], "subqueries": spec["subqueries"],
            "freshness_mode": spec.get("freshness_mode", "strict_recent")}
    for key in ("intent", "cluster_mode"):
        if key in spec:
            plan[key] = spec[key]
    plan_path = Path(f"/tmp/recall-bench-{spec_stem}.plan.json")
    plan_path.write_text(json.dumps(plan, indent=1))

    cmd = [sys.executable, str(ENGINE), spec["topic"], "--emit", "json",
           "--no-browser-cookies", "--agent-rerank",
           "--plan", str(plan_path), *engine_args]
    env = _load_env(CONFIG_ENV)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=env, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(2)

    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        # emit=json may print logs before the JSON; take the last {..} block
        start = proc.stdout.find("{")
        report = json.loads(proc.stdout[start:]) if start >= 0 else {}
    return _find_items(report), proc.stderr


def _item_key(item: dict) -> str:
    return str(item.get("post_id") or item.get("url") or
               item.get("text") or item.get("title") or "")


def _strip_dig(engine_args: list[str]) -> list[str]:
    out = [a for i, a in enumerate(engine_args)
           if a != "--x-dig" and
           (i == 0 or engine_args[i - 1] != "--x-dig")]
    return out + ["--x-dig", "0"]


def _lift(spec: dict, stem: str, timeout: int) -> int:
    base_args = _strip_dig(list(spec.get("args") or []))
    dig_args = list(spec.get("args") or [])
    if "--x-dig" not in dig_args:
        dig_args += ["--x-dig", "2"]

    base_items, base_err = _run_engine(spec, stem, base_args, timeout)
    dig_items, dig_err = _run_engine(spec, stem, dig_args, timeout)

    base_keys = {_item_key(it) for it in base_items}
    added = [it for it in dig_items if _item_key(it) not in base_keys]
    jev_passed = [it for it in added
                  if float(it.get("jev_score") or 0) > 0.2]
    print(f"baseline items: {len(base_items)}")
    print(f"dig items:      {len(dig_items)}")
    print(f"dig added:      {len(added)} posts the baseline missed "
          f"({len(jev_passed)} passed the relevance judge)")
    for line in dig_err.splitlines():
        if "dig" in line.lower():
            print(" ", line)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("--dig", type=int, default=None,
                    help="override spec's --x-dig rounds")
    ap.add_argument("--lift", action="store_true",
                    help="run dig-off vs dig-on; no answer key needed")
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    engine_args = list(spec.get("args") or [])
    if args.dig is not None:
        engine_args = _strip_dig(engine_args)[:-2] + ["--x-dig", str(args.dig)]

    if args.lift:
        return _lift(spec, args.spec.stem, args.timeout)

    items, stderr = _run_engine(spec, args.spec.stem, engine_args,
                                args.timeout)
    haystacks = [_haystack(it) for it in items]

    hits = misses = 0
    for exp in spec["expected"]:
        needle = exp["match"].lower()
        found = any(needle in h for h in haystacks)
        mark = "HIT " if found else "MISS"
        print(f"{mark} {exp['label']}")
        hits += found
        misses += not found

    total = hits + misses
    recall = hits / total if total else 0.0
    print(f"\nrecall {hits}/{total} = {recall:.0%}  "
          f"({len(items)} items emitted)")
    for line in stderr.splitlines():
        if "dig" in line.lower():
            print(" ", line)
    return 0 if recall >= args.floor else 1


if __name__ == "__main__":
    raise SystemExit(main())
