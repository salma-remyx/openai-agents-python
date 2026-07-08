"""Remyx `helps?` benchmark for the Causal Minimal Tool Filtering change.

Deterministic, no LLM / no API keys. Measures the feature's actual claim: the
causal filter should expose only the *minimal frontier* of tools needed to make
progress toward a goal, instead of all tools.

  tool_precision = relevant exposed / total exposed   (target ↑ — the filter's job)
  tool_recall    = relevant exposed / relevant needed  (guardrail — must stay 1.0)

On the PR branch the causal filter is present → precision high, recall 1.0.
On `main` the filter is absent (import fails) → all tools exposed → precision low.
Remyx runs this on both and reports the delta.

Run by the Remyx eval runner as: python eval/tool_filter_bench.py [--variant ...]
Emits a single JSON line on stdout (the runner parses it into quality metrics).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

# The filesystem tool set exercised by the example (mirrors the PR's tests).
FS_TOOLS = ["list_directory", "read_file", "write_file", "get_file_info"]

# (established_facts, goal_facts, the minimal tools that should be exposed).
# Grounded in the PR's own unit tests (FILESYSTEM_TOOL_CONTRACTS):
SCENARIOS = [
    (set(),                     {"file_contents_known"}, {"list_directory"}),
    ({"target_path_known"},     {"file_contents_known"}, {"read_file"}),
    (set(),                     {"target_path_known"},   {"list_directory"}),
]


@dataclass
class FakeTool:
    name: str


@dataclass
class WorkflowState:
    established_facts: set = field(default_factory=set)
    goal_facts: set = field(default_factory=set)


def _load_filter():
    """The causal filter (PR branch), or None on baseline where it doesn't exist."""
    try:
        from examples.mcp.tool_filter_example.main import build_causal_filter
        return build_causal_filter()
    except Exception as exc:  # baseline (main): feature not present
        print(f"[bench] no causal filter present ({exc}) — baseline (expose all)", file=sys.stderr)
        return None


def _visible(tool_filter, facts, goal) -> set:
    if tool_filter is None:
        return set(FS_TOOLS)  # baseline: no filtering, all tools exposed
    from agents import Agent, RunContextWrapper
    from agents.mcp.util import ToolFilterContext
    payload = WorkflowState(established_facts=set(facts), goal_facts=set(goal))
    ctx = ToolFilterContext(
        run_context=RunContextWrapper(context=payload),
        agent=Agent(name="remyx-bench"),
        server_name="filesystem",
    )
    return {name for name in FS_TOOLS if tool_filter(ctx, FakeTool(name=name))}


def main() -> int:
    tool_filter = _load_filter()
    precisions, recalls = [], []
    for facts, goal, needed in SCENARIOS:
        vis = _visible(tool_filter, facts, goal)
        hit = len(vis & needed)
        precisions.append(hit / max(1, len(vis)))
        recalls.append(hit / len(needed))
    metrics = {
        "tool_precision": round(sum(precisions) / len(precisions), 4),
        "tool_recall": round(sum(recalls) / len(recalls), 4),
    }
    print(json.dumps(metrics))  # runner parses this line into quality metrics
    return 0


if __name__ == "__main__":
    sys.exit(main())
