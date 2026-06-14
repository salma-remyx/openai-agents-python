"""Tests for the CMTF causal tool filter wired into the MCP tool-filter example.

These exercise the example call site (``examples.mcp.tool_filter_example.main``)
to prove the integration: ``build_causal_filter`` produces a real
``ToolFilterCallable`` whose per-tool decisions match the causal next-step
frontier as the run state advances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents import Agent, RunContextWrapper
from agents.mcp.util import ToolFilterContext

# The example module under test is intentionally a non-new call site.
from examples.mcp.tool_filter_example.causal_tool_filter import (
    CausalState,
    ToolContract,
    compute_frontier,
    create_causal_tool_filter,
)
from examples.mcp.tool_filter_example.main import build_causal_filter


@dataclass
class FakeTool:
    """Minimal stand-in for an MCP tool (the filter only reads ``name``)."""

    name: str


@dataclass
class WorkflowState:
    """Run-context payload exposing CMTF facts the default resolver reads."""

    established_facts: set[str] = field(default_factory=set)
    goal_facts: set[str] = field(default_factory=set)


def _context(payload: Any) -> ToolFilterContext:
    return ToolFilterContext(
        run_context=RunContextWrapper(context=payload),
        agent=Agent(name="filter-test-agent"),
        server_name="filesystem",
    )


def _visible(tool_filter: Any, payload: Any, names: list[str]) -> set[str]:
    ctx = _context(payload)
    return {name for name in names if tool_filter(ctx, FakeTool(name=name))}


FS_TOOLS = ["list_directory", "read_file", "write_file", "get_file_info"]


def test_build_causal_filter_exposes_only_enabled_frontier_tool() -> None:
    """With no path known yet, only list_directory advances toward the goal."""

    tool_filter = build_causal_filter()
    payload = WorkflowState(established_facts=set(), goal_facts={"file_contents_known"})

    visible = _visible(tool_filter, payload, FS_TOOLS)

    # read_file is relevant but premature (its precondition is unmet); write_file
    # is irrelevant to the goal; get_file_info has no contract -> hidden.
    assert visible == {"list_directory"}


def test_frontier_advances_after_path_is_discovered() -> None:
    """Once a path is known, read_file becomes the minimal next-step tool."""

    tool_filter = build_causal_filter()
    payload = WorkflowState(
        established_facts={"target_path_known"},
        goal_facts={"file_contents_known"},
    )

    visible = _visible(tool_filter, payload, FS_TOOLS)

    # list_directory's effect is already satisfied, so it drops off the frontier.
    assert visible == {"read_file"}


def test_no_causal_state_falls_back_to_exposing_everything() -> None:
    """Without facts/goal on the context, the filter must not over-prune."""

    tool_filter = build_causal_filter()

    visible = _visible(tool_filter, object(), FS_TOOLS)

    # No causal information available -> expose all contracted/uncontracted tools.
    assert visible == set(FS_TOOLS)


def test_goal_already_satisfied_exposes_nothing() -> None:
    """When the goal is met, no tool is causally necessary."""

    tool_filter = build_causal_filter()
    payload = WorkflowState(
        established_facts={"file_contents_known"},
        goal_facts={"file_contents_known"},
    )

    assert _visible(tool_filter, payload, FS_TOOLS) == set()


def test_expose_uncontracted_opt_in() -> None:
    """Uncontracted tools can be opted back in without affecting the frontier."""

    tool_filter = create_causal_tool_filter(
        [ToolContract(name="list_directory", effects={"target_path_known"})],
        expose_uncontracted=True,
    )
    payload = WorkflowState(goal_facts={"target_path_known"})

    visible = _visible(tool_filter, payload, ["list_directory", "mystery_tool"])
    assert visible == {"list_directory", "mystery_tool"}


def test_compute_frontier_is_minimal_over_a_chain() -> None:
    """A multi-step contract chain exposes exactly one frontier tool per step."""

    contracts = [
        ToolContract(name="a", effects={"x"}),
        ToolContract(name="b", preconditions={"x"}, effects={"y"}),
        ToolContract(name="c", preconditions={"y"}, effects={"z"}),
    ]
    goal = frozenset({"z"})

    assert compute_frontier(contracts, CausalState(facts=frozenset(), goal=goal)) == {"a"}
    assert compute_frontier(contracts, CausalState(facts=frozenset({"x"}), goal=goal)) == {"b"}
    assert compute_frontier(contracts, CausalState(facts=frozenset({"y"}), goal=goal)) == {"c"}
