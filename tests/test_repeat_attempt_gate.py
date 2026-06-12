"""Tests for the PROJECTMEM-style pre-action gate and its wiring.

These exercise the new ``examples.basic.repeat_attempt_gate`` capability *and*
its integration into the existing ``examples.basic.tool_guardrails`` example so
the wiring edit is covered, not just the new module in isolation.
"""

from __future__ import annotations

import pytest

from agents import Agent, ToolInputGuardrailData, ToolOutputGuardrailData
from agents.tool_context import ToolContext
from examples.basic.repeat_attempt_gate import (
    AttemptLog,
    action_key,
    make_outcome_recorder,
    make_repeat_attempt_gate,
)


def _input_data(tool_name: str, args: str) -> ToolInputGuardrailData:
    return ToolInputGuardrailData(
        context=ToolContext(
            context=None,
            tool_name=tool_name,
            tool_call_id="call_1",
            tool_arguments=args,
        ),
        agent=Agent(name="test"),
    )


def _output_data(tool_name: str, args: str, output: object) -> ToolOutputGuardrailData:
    return ToolOutputGuardrailData(
        context=ToolContext(
            context=None,
            tool_name=tool_name,
            tool_call_id="call_1",
            tool_arguments=args,
        ),
        agent=Agent(name="test"),
        output=output,
    )


def test_action_key_normalizes_argument_order():
    assert action_key("t", '{"a": 1, "b": 2}') == action_key("t", '{"b": 2, "a": 1}')
    assert action_key("t", "not json") == action_key("t", "  not json  ")


def test_last_failure_tracks_outcomes():
    log = AttemptLog()
    log.record("build", '{"target": "x"}', "failed", note="boom")
    assert log.last_failure("build", '{"target": "x"}') is not None
    # A later success on the same action clears the failure.
    log.record("build", '{"target": "x"}', "succeeded")
    assert log.last_failure("build", '{"target": "x"}') is None


@pytest.mark.asyncio
async def test_gate_blocks_repeated_failed_action():
    log = AttemptLog()
    gate = make_repeat_attempt_gate(log)

    # First attempt of an action is allowed and logged.
    first = await gate.run(_input_data("send_email", '{"to": "a@b.com"}'))
    assert first.behavior["type"] == "allow"

    # Record that this exact action failed.
    log.record("send_email", '{"to": "a@b.com"}', "failed", note="smtp down")

    # The next identical attempt is rejected by the pre-action gate.
    blocked = await gate.run(_input_data("send_email", '{"to": "a@b.com"}'))
    assert blocked.behavior["type"] == "reject_content"
    assert blocked.output_info["repeat"] is True

    # A different action is unaffected.
    other = await gate.run(_input_data("send_email", '{"to": "z@b.com"}'))
    assert other.behavior["type"] == "allow"


@pytest.mark.asyncio
async def test_outcome_recorder_classifies_results():
    log = AttemptLog()
    recorder = make_outcome_recorder(log)

    await recorder.run(_output_data("send_email", '{"to": "a@b.com"}', "Email sent to a@b.com"))
    await recorder.run(_output_data("send_email", '{"to": "x@b.com"}', "Error: SMTP failure"))

    assert log.events[0].status == "succeeded"
    assert log.events[1].status == "failed"


@pytest.mark.asyncio
async def test_tool_guardrails_example_is_wired():
    """The non-new example module wires the gate onto its send_email tool."""
    from examples.basic import tool_guardrails

    input_names = [g.get_name() for g in tool_guardrails.send_email.tool_input_guardrails]
    output_names = [g.get_name() for g in tool_guardrails.send_email.tool_output_guardrails]
    assert "repeat_attempt_gate" in input_names
    assert "attempt_outcome_recorder" in output_names

    # Drive the shared log through the example's wired guardrails end to end.
    log = tool_guardrails.attempt_log
    start = len(log.events)
    tool_guardrails.attempt_log.record(
        "send_email", '{"to": "a@b.com"}', "failed", note="smtp down"
    )
    blocked = await tool_guardrails.repeat_attempt_gate.run(
        _input_data("send_email", '{"to": "a@b.com"}')
    )
    assert blocked.behavior["type"] == "reject_content"
    assert len(log.events) > start
