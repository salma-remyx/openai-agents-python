"""Tests for the attention-based MCP tool gate and its wiring into the example call site.

These cover the paper-derived dynamic tool gating: tools that do not attend to the task
signal are dropped so their schemas never reach the model (eliminating the "MCP Tax").
"""

import pytest
from mcp import Tool as MCPTool

from agents import Agent
from agents.mcp import ToolFilterContext
from agents.run_context import RunContextWrapper

# Import from the existing (non-new) example call site to prove the wiring, plus the
# capability module it now depends on.
from examples.mcp.tool_filter_example.attention_tool_gate import (
    attention_tool_filter,
    tool_relevance_score,
)
from examples.mcp.tool_filter_example.main import build_attention_filter

from .helpers import FakeMCPServer


def _make_context(query: str | None = None) -> ToolFilterContext:
    return ToolFilterContext(
        run_context=RunContextWrapper(context={"tool_query": query} if query else None),
        agent=Agent(name="MCP Assistant", instructions="Read and list files."),
        server_name="filesystem",
    )


def test_relevance_score_orders_tools_by_attention() -> None:
    query = "list and read files in the directory"
    read_score = tool_relevance_score(query, "read_file", "Read a file from disk")
    list_score = tool_relevance_score(query, "list_directory", "List directory entries")
    weather_score = tool_relevance_score(query, "get_weather", "Return the local forecast")

    assert read_score > weather_score
    assert list_score > weather_score
    assert weather_score < 0.15


def test_filter_gates_irrelevant_tools() -> None:
    gate = attention_tool_filter(min_score=0.15)
    context = _make_context("read the contents of a file")

    read_tool = MCPTool(name="read_file", description="Read a file", inputSchema={})
    weather_tool = MCPTool(name="get_weather", description="Forecast", inputSchema={})

    assert gate(context, read_tool) is True
    assert gate(context, weather_tool) is False


def test_always_include_bypasses_gate() -> None:
    gate = attention_tool_filter(min_score=0.99, always_include=["list_directory"])
    context = _make_context("completely unrelated request")
    tool = MCPTool(name="list_directory", description="List dir", inputSchema={})

    assert gate(context, tool) is True


@pytest.mark.asyncio
async def test_example_filter_drops_unrelated_tools_end_to_end() -> None:
    """Exercise the exact filter built by the example through the real filter pipeline."""
    server = FakeMCPServer(
        tool_filter=build_attention_filter(),
        server_name="filesystem",
    )
    server.add_tool("read_file", {})
    server.add_tool("list_directory", {})
    server.add_tool("get_weather", {})

    agent = Agent(name="MCP Assistant", instructions="Read and list files in the directory.")
    run_context = RunContextWrapper(context=None)

    tools = await server.list_tools(run_context, agent)
    names = {tool.name for tool in tools}

    # list_directory is always-included; read_file attends to the task; weather is gated out.
    assert "list_directory" in names
    assert "read_file" in names
    assert "get_weather" not in names
