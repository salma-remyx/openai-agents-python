"""Deterministic benchmark: tool-exposure precision of CMTF causal filtering.

Runs a scripted 3-step filesystem workflow (discover path -> read file -> goal
met) and measures, at each step, how many *extraneous* (non-necessary) tools
are exposed alongside the one causally-necessary tool. Falls back to "no
filtering" (all tools always exposed -- the pre-CMTF baseline behaviour) when
the causal filter module is unavailable, so it can run unmodified on both
`main` and the PR head.

Metrics:
  - avg_extraneous_tool_rate (target, lower is better): mean fraction of the
    tool menu that is exposed but not causally necessary at each step.
  - necessary_tool_coverage (guardrail, must not regress below 1.0): mean
    fraction of steps where the causally-necessary tool remains visible.
"""

from __future__ import annotations

import json
import os
import sys

# Put repo root on sys.path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ALL_TOOLS = ["list_directory", "read_file", "write_file", "get_file_info"]

# (established_facts, goal_facts, necessary_tool_or_None)
SCENARIOS = [
    (frozenset(), frozenset({"file_contents_known"}), "list_directory"),
    (frozenset({"target_path_known"}), frozenset({"file_contents_known"}), "read_file"),
    (frozenset({"file_contents_known"}), frozenset({"file_contents_known"}), None),
]

_FALLBACK_CONTRACTS = [
    ("list_directory", frozenset(), frozenset({"target_path_known"})),
    ("read_file", frozenset({"target_path_known"}), frozenset({"file_contents_known"})),
    ("write_file", frozenset({"content_ready"}), frozenset({"file_written"})),
]


def _fallback_frontier(facts: frozenset, goal: frozenset) -> set:
    """Baseline behaviour: no causal filtering available -> expose everything."""
    return set(ALL_TOOLS)


try:
    from examples.mcp.tool_filter_example.causal_tool_filter import (  # type: ignore
        CausalState,
        ToolContract,
        compute_frontier,
    )

    try:
        from examples.mcp.tool_filter_example.main import (  # type: ignore
            FILESYSTEM_TOOL_CONTRACTS as _CONTRACTS,
        )
    except Exception:
        _CONTRACTS = [
            ToolContract(name=name, preconditions=pre, effects=eff)
            for name, pre, eff in _FALLBACK_CONTRACTS
        ]

    def _frontier(facts: frozenset, goal: frozenset) -> set:
        state = CausalState(facts=frozenset(facts), goal=frozenset(goal))
        return compute_frontier(_CONTRACTS, state)

except Exception:
    _frontier = _fallback_frontier  # type: ignore


def main() -> None:
    extraneous_rates = []
    coverages = []

    total_tools = len(ALL_TOOLS)

    for facts, goal, necessary in SCENARIOS:
        exposed = set(_frontier(facts, goal)) & set(ALL_TOOLS)
        # also allow uncontracted/unknown tool names returned by fallback
        exposed = {t for t in ALL_TOOLS if t in exposed}

        if necessary is not None:
            covered = 1.0 if necessary in exposed else 0.0
            coverages.append(covered)
            extraneous_count = len(exposed - {necessary})
        else:
            # goal already satisfied: ideally nothing is exposed
            extraneous_count = len(exposed)

        extraneous_rates.append(extraneous_count / total_tools)

    avg_extraneous_tool_rate = sum(extraneous_rates) / len(extraneous_rates)
    necessary_tool_coverage = (
        sum(coverages) / len(coverages) if coverages else 1.0
    )

    print(
        json.dumps(
            {
                "avg_extraneous_tool_rate": avg_extraneous_tool_rate,
                "necessary_tool_coverage": necessary_tool_coverage,
            }
        )
    )


if __name__ == "__main__":
    main()