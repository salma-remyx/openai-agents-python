"""Tests for contract-driven guardrails and their wiring into the output_guardrails example.

The integration check imports the existing example call site
(`examples.agent_patterns.output_guardrails`) and exercises the contract-compiled
guardrail attached to its agent, proving the wiring -- not just the new module in isolation.
"""

from __future__ import annotations

import asyncio

from agents import Agent, RunContextWrapper
from examples.agent_patterns.contract_guardrails import ContractReport, TaskContract

# The non-new call-site module the integration edits.
from examples.agent_patterns.output_guardrails import (
    MessageOutput,
    agent,
    support_reply_contract,
)


def _message(response: str = "ok", reasoning: str = "ok") -> MessageOutput:
    return MessageOutput(response=response, reasoning=reasoning, user_name=None)


def test_contract_compiles_to_named_guardrail() -> None:
    contract = TaskContract(name="demo", goal="be helpful", output_contract=MessageOutput)
    guardrail = contract.as_output_guardrail()
    assert guardrail.get_name() == "demo:output"


def test_check_output_reports_quality_failures() -> None:
    report = support_reply_contract.check_output(_message(response="call 650-1234"))
    assert isinstance(report, ContractReport)
    assert not report.passed
    assert "no_area_code_in_response" in {c.label for c in report.failures}


def test_check_output_passes_clean_message() -> None:
    report = support_reply_contract.check_output(_message())
    assert report.passed
    assert report.failures == []


def test_output_contract_validates_wrong_type() -> None:
    contract = TaskContract(name="demo", goal="g", output_contract=MessageOutput)
    report = contract.check_output("not a MessageOutput")
    assert not report.passed
    assert "output_contract" in {c.label for c in report.failures}


def test_input_boundary_tripwire() -> None:
    contract = TaskContract(
        name="demo",
        goal="g",
        input_boundaries=[("no_secrets", lambda text: "secret" not in text.lower())],
    )
    assert contract.check_input("please share the SECRET").passed is False
    assert contract.check_input("hello there").passed is True


def test_example_agent_uses_contract_guardrail() -> None:
    # The example agent must carry exactly the guardrail compiled from the contract.
    assert isinstance(agent, Agent)
    assert len(agent.output_guardrails) == 1
    compiled = agent.output_guardrails[0]
    assert compiled.get_name() == "support_reply:output"

    ctx: RunContextWrapper = RunContextWrapper(context=None)
    tripped = asyncio.run(compiled.run(ctx, agent, _message(response="my number is 650-123-4567")))
    assert tripped.output.tripwire_triggered is True
    assert isinstance(tripped.output.output_info, ContractReport)

    clean = asyncio.run(compiled.run(ctx, agent, _message(response="The capital is Sacramento.")))
    assert clean.output.tripwire_triggered is False
