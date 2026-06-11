"""Dynamic, relevance-based gating of MCP tools to shrink the per-turn "MCP/Tools Tax".

Eagerly injecting every MCP tool schema on every turn inflates the prompt (and the
KV cache) by thousands of tokens even when only a few tools are relevant. This module
implements a lightweight *tool attention* gate: each tool is scored for relevance to
the current task signal, and only tools above a threshold are exposed to the model.
Excluded tools never have their schema injected, which is the "lazy schema loading"
the gate buys us at no protocol cost.

It plugs into the existing ``tool_filter`` contract: ``attention_tool_filter`` returns a
``ToolFilterCallable`` that the SDK already calls per tool in
``_MCPServerWithClientSession._apply_dynamic_tool_filter``. No SDK change is required.

Scoring is intentionally dependency-free (lexical attention over token overlap) so the
example stays runnable without embedding models. Swap ``score_fn`` for an embedding-based
scorer when you want semantic gating.

Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema Loading
for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows" (arXiv:2604.21816).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from agents.mcp.util import ToolFilterCallable, ToolFilterContext

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens, also splitting snake/camel case."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return _TOKEN_RE.findall(spaced.lower())


def tool_relevance_score(query: str, tool_name: str, tool_description: str) -> float:
    """Return an attention-style relevance score in ``[0, 1]``.

    Each query token contributes its best match against the tool's name and description.
    Name matches are weighted higher than description matches because a tool's name is the
    strongest signal of intent. Prefix/substring overlaps count as partial matches so that
    ``"listing"`` still attends to ``"list_directory"``.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    name_tokens = set(_tokenize(tool_name))
    desc_tokens = set(_tokenize(tool_description or ""))

    total = 0.0
    for q in query_tokens:
        best = 0.0
        for token, weight in ((name_tokens, 1.0), (desc_tokens, 0.6)):
            for t in token:
                if q == t:
                    best = max(best, weight)
                elif len(q) >= 4 and (q.startswith(t) or t.startswith(q)):
                    best = max(best, weight * 0.5)
        total += best

    # Normalize by query length so longer queries are not penalized.
    return min(1.0, total / len(query_tokens))


def default_query_provider(context: ToolFilterContext) -> str:
    """Derive the task signal used to gate tools from the run context and agent.

    Resolution order:
    1. ``run_context.context["tool_query"]`` or a ``tool_query`` attribute, if present.
    2. The raw run context when it is itself a string.
    3. The requesting agent's name and instructions as a fallback signal.
    """
    raw = getattr(context.run_context, "context", None)
    if isinstance(raw, dict):
        from_dict = raw.get("tool_query")
        if isinstance(from_dict, str):
            return from_dict
    explicit = getattr(raw, "tool_query", None)
    if isinstance(explicit, str):
        return explicit
    if isinstance(raw, str):
        return raw

    agent = context.agent
    parts = [getattr(agent, "name", "") or ""]
    instructions = getattr(agent, "instructions", None)
    if isinstance(instructions, str):
        parts.append(instructions)
    return " ".join(p for p in parts if p)


def attention_tool_filter(
    *,
    min_score: float = 0.15,
    always_include: Sequence[str] = (),
    query_provider: Callable[[ToolFilterContext], str] = default_query_provider,
    score_fn: Callable[[str, str, str], float] = tool_relevance_score,
) -> ToolFilterCallable:
    """Build a ``ToolFilterCallable`` that gates MCP tools by relevance to the task.

    Args:
        min_score: Minimum relevance score (``[0, 1]``) for a tool to be exposed.
        always_include: Tool names that bypass the gate and are always exposed.
        query_provider: Extracts the task signal from the filter context.
        score_fn: ``(query, tool_name, tool_description) -> score`` relevance scorer.

    Returns:
        A per-tool callable compatible with ``MCPServer(tool_filter=...)``. Tools scoring
        below ``min_score`` are dropped so their schema is never sent to the model.
    """
    always = set(always_include)

    def _gate(context: ToolFilterContext, tool: Any) -> bool:
        if tool.name in always:
            return True
        query = query_provider(context)
        score = score_fn(query, tool.name, getattr(tool, "description", "") or "")
        return score >= min_score

    return _gate
