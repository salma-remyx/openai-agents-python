# MCP Tool Filter Example

Python port of the JS `examples/mcp/tool-filter-example.ts`. It shows how to:

- Run the filesystem MCP server locally via `npx`.
- Apply a static tool filter so only specific tools are exposed to the model.
- Observe that blocked tools are not available.
- Enable `require_approval="always"` and auto-approve interruptions in code so the HITL path is exercised.

Run it with:

```bash
uv run python examples/mcp/tool_filter_example/main.py
```

## Causal Minimal Tool Filtering (CMTF)

`causal_tool_filter.py` adds a *dynamic* alternative to the static allow/block
list — adapted from
[ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents](https://arxiv.org/abs/2606.06284v1).

Instead of a fixed allowlist, each tool carries a lightweight precondition-effect
contract, and the filter exposes only the *causal next-step frontier*: tools
whose preconditions are already satisfied and whose effects lie on a path toward
the user goal. Relevant-but-premature tools (unmet preconditions) and
relevant-but-unnecessary tools (off the goal path) are hidden, which reduces
wrong-tool calls and token cost. It is training-free and plugs into the existing
`tool_filter` callable contract.

Enable it by setting an environment variable before running:

```bash
MCP_TOOL_FILTER=causal uv run python examples/mcp/tool_filter_example/main.py
```

Prerequisites:

- `npx` available on your PATH.
- `OPENAI_API_KEY` set for the model calls.
