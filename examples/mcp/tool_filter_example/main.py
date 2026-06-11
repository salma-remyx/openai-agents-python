import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any, cast

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServerStdio
from agents.mcp.util import ToolFilterCallable

if __package__ in (None, ""):
    # Allow running this file directly as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.mcp.tool_filter_example.attention_tool_gate import attention_tool_filter


def build_attention_filter() -> ToolFilterCallable:
    """Gate MCP tools by relevance to the task instead of a hand-maintained allowlist.

    Tools whose name/description do not attend to the agent's task signal are dropped, so
    their schemas are never injected into the prompt. This shrinks the per-turn "MCP Tax"
    while still letting relevant tools (read/list) through. See ``attention_tool_gate``.
    """
    return attention_tool_filter(
        min_score=0.15,
        always_include=["list_directory"],
    )


async def run_with_auto_approval(agent: Agent[Any], message: str) -> str | None:
    """Run and auto-approve interruptions."""

    result = await Runner.run(agent, message)
    while result.interruptions:
        state = result.to_state()
        for interruption in result.interruptions:
            print(f"Approving a tool call... (name: {interruption.name})")
            state.approve(interruption, always_approve=True)
        result = await Runner.run(agent, state)
    return cast(str | None, result.final_output)


async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(current_dir, "sample_files")
    target_path = os.path.join(samples_dir, "test.txt")

    async with MCPServerStdio(
        name="Filesystem Server with filter",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", samples_dir],
            "cwd": samples_dir,
        },
        require_approval="always",
        tool_filter=build_attention_filter(),
    ) as server:
        agent = Agent(
            name="MCP Assistant",
            instructions=(
                "Read and list files in the allowed directory. "
                "Use only the available filesystem tools. "
                "All file paths should be absolute paths inside the allowed directory. "
                "If a user asks for an action that requires an unavailable tool, "
                "explicitly explain that it is blocked by the tool filter."
            ),
            mcp_servers=[server],
        )
        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Tool Filter Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            result = await run_with_auto_approval(
                agent, f"List the files in this allowed directory: {samples_dir}"
            )
            print(result)

            blocked_result = await run_with_auto_approval(
                agent,
                (
                    f'Create a file at "{target_path}" with the text "hello". '
                    "If you cannot, explain that write operations are blocked by the tool filter."
                ),
            )
            print("\nAttempting to write a file (should be blocked):")
            print(blocked_result)


if __name__ == "__main__":
    if not shutil.which("npx"):
        raise RuntimeError("npx is required. Install it with `npm install -g npx`.")

    asyncio.run(main())
