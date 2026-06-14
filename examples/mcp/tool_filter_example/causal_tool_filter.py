"""Causal Minimal Tool Filtering (CMTF) for MCP tool menus.

Adapted from "ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable
LLM Agents" (https://arxiv.org/abs/2606.06284v1).

Semantic relevance is not enough to decide which tools to expose: a tool can be
related to the task yet be unnecessary or premature at the current step. CMTF
instead selects tools by *causal sufficiency*. Each tool carries a lightweight
precondition-effect contract, and the filter exposes only the minimal next-step
frontier -- tools whose preconditions are already satisfied and whose effects lie
on a causal path from the current state toward the user goal.

This module is training-free and deterministic. It plugs into the SDK's existing
dynamic tool-filter contract (``ToolFilterCallable`` consumed by
``MCPServer._apply_dynamic_tool_filter``) by returning a per-tool predicate that
answers, for each tool, "is this tool on the causal frontier right now?".
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agents.mcp.util import ToolFilterCallable, ToolFilterContext


@dataclass
class ToolContract:
    """A lightweight precondition-effect contract for a single tool.

    Attributes:
        name: The MCP tool name this contract describes.
        preconditions: Facts that must already hold for the tool to be usable.
        effects: Facts the tool establishes when it succeeds.
    """

    name: str
    preconditions: Collection[str] = field(default_factory=frozenset)
    effects: Collection[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.preconditions = frozenset(self.preconditions)
        self.effects = frozenset(self.effects)


@dataclass(frozen=True)
class CausalState:
    """The causal state CMTF conditions on for a single filtering decision.

    Attributes:
        facts: Facts already established in the current run.
        goal: Facts the user ultimately wants established.
    """

    facts: frozenset[str] = field(default_factory=frozenset)
    goal: frozenset[str] = field(default_factory=frozenset)


StateResolver = Callable[[ToolFilterContext], "CausalState | None"]


def compute_frontier(contracts: Iterable[ToolContract], state: CausalState) -> set[str]:
    """Return the names of tools on the causal next-step frontier.

    A tool is on the frontier when (a) all of its preconditions are already
    satisfied by ``state.facts`` and (b) at least one of its effects is still
    *needed* to reach the goal. "Needed" is computed by backward reachability:
    start from the unmet goal facts, then repeatedly pull in the preconditions of
    any tool that can produce a needed fact. This keeps the frontier minimal --
    relevant-but-premature tools (whose preconditions are unmet) and
    relevant-but-unnecessary tools (whose effects are not on a goal path) are
    both excluded.
    """

    normalized = [
        (contract.name, frozenset(contract.preconditions), frozenset(contract.effects))
        for contract in contracts
    ]
    facts = set(state.facts)
    needed = set(state.goal) - facts

    # Backward pass: expand the set of facts we still need to establish.
    changed = True
    while changed:
        changed = False
        for _name, preconditions, effects in normalized:
            if effects & needed:
                new_needs = preconditions - facts - needed
                if new_needs:
                    needed |= new_needs
                    changed = True

    frontier: set[str] = set()
    for name, preconditions, effects in normalized:
        enabled = preconditions <= facts
        advances_goal = bool(effects & needed)
        if enabled and advances_goal:
            frontier.add(name)
    return frontier


def _coerce_facts(payload: Any, *keys: str) -> frozenset[str] | None:
    """Pull a set of fact tokens out of a run-context payload by attr or key."""

    for key in keys:
        value: Any = None
        if isinstance(payload, Mapping):
            value = payload.get(key)
        else:
            value = getattr(payload, key, None)
        if value is not None:
            return frozenset(value)
    return None


def default_state_resolver(context: ToolFilterContext) -> CausalState | None:
    """Read a :class:`CausalState` from ``run_context.context``.

    The application is expected to expose ``established_facts``/``facts`` and
    ``goal_facts``/``goal`` on its run-context object (a dataclass, an object
    with attributes, or a mapping). Returns ``None`` when no causal state is
    available so the caller can fall back to exposing everything.
    """

    run_context = getattr(context, "run_context", None)
    payload = getattr(run_context, "context", None) if run_context is not None else None
    if payload is None:
        return None

    facts = _coerce_facts(payload, "established_facts", "facts")
    goal = _coerce_facts(payload, "goal_facts", "goal")
    if facts is None and goal is None:
        return None
    return CausalState(facts=facts or frozenset(), goal=goal or frozenset())


def create_causal_tool_filter(
    contracts: Iterable[ToolContract],
    *,
    state_resolver: StateResolver | None = None,
    expose_uncontracted: bool = False,
) -> ToolFilterCallable:
    """Build a dynamic ``ToolFilterCallable`` implementing CMTF.

    Args:
        contracts: Precondition-effect contracts for the tools to govern.
        state_resolver: Resolves the current :class:`CausalState` from the
            filter context. Defaults to :func:`default_state_resolver`. When the
            resolver returns ``None`` the filter exposes all tools (no causal
            information is available, so it does not over-prune).
        expose_uncontracted: Whether tools without a contract are exposed. CMTF
            is conservative by default and hides tools it cannot reason about.

    Returns:
        A predicate ``(ToolFilterContext, MCPTool) -> bool`` suitable for an
        MCP server's ``tool_filter`` argument.
    """

    contract_list = list(contracts)
    contract_names = {contract.name for contract in contract_list}
    resolver = state_resolver or default_state_resolver

    def tool_filter(context: ToolFilterContext, tool: Any) -> bool:
        state = resolver(context)
        if state is None:
            # No causal state available; do not over-prune.
            return True

        tool_name = getattr(tool, "name", None)
        if tool_name not in contract_names:
            return expose_uncontracted

        return tool_name in compute_frontier(contract_list, state)

    return tool_filter
