"""Tests for the PROJECTMEM-style pre-action gate and its wiring.

These exercise the integration into the existing ``examples/basic/tool_guardrails.py``
call site, not just the standalone module, so they prove the gate is actually
attached to a tool the example agent uses.
"""

from __future__ import annotations

from types import SimpleNamespace

from examples.basic.action_memory_gate import (
    ActionMemory,
    action_signature,
    make_pre_action_gate,
)

# Import the existing (non-new) example module that wires the gate in.
from examples.basic.tool_guardrails import action_memory, apply_fix, pre_action_gate


def _run_gate(gate, tool_name: str, tool_arguments: str):
    data = SimpleNamespace(
        context=SimpleNamespace(tool_name=tool_name, tool_arguments=tool_arguments),
        agent=None,
    )
    return gate.guardrail_function(data)


def test_signature_is_order_independent() -> None:
    a = action_signature("apply_fix", '{"file": "client.py", "description": "retry"}')
    b = action_signature("apply_fix", '{"description": "retry", "file": "client.py"}')
    assert a == b


def test_gate_is_wired_into_apply_fix_tool() -> None:
    # The example must actually attach the gate to the tool, not just define it.
    assert pre_action_gate in (apply_fix.tool_input_guardrails or [])
    # And the example pre-seeds the shared log with a known failure to govern.
    assert any(e.type.value == "failure" for e in action_memory.events)


def test_gate_blocks_repeated_failed_fix_via_example_wiring() -> None:
    # This is the exact action the example pre-seeded as a prior failure.
    output = _run_gate(
        pre_action_gate,
        "apply_fix",
        '{"file": "client.py", "description": "retry the request"}',
    )
    assert output.behavior["type"] == "reject_content"
    assert output.output_info["gate"] == "repeated_failed_fix"


def test_gate_allows_and_records_novel_action() -> None:
    memory = ActionMemory()
    gate = make_pre_action_gate(memory)
    output = _run_gate(gate, "apply_fix", '{"file": "server.py", "description": "raise timeout"}')
    assert output.behavior["type"] == "allow"
    # The allowed attempt is logged so the agent's history stays complete.
    assert any(e.type.value == "attempt" for e in memory.events)


def test_fix_clears_prior_failure() -> None:
    memory = ActionMemory()
    sig = action_signature("apply_fix", '{"file": "client.py"}')
    memory.record_failure(sig, summary="first try failed")
    assert memory.has_failed(sig) is True
    memory.record_fix(sig, summary="second try worked")
    assert memory.has_failed(sig) is False


def test_gate_blocks_known_fragile_file() -> None:
    memory = ActionMemory()
    memory.mark_fragile_file("migrations/0001_init.py", summary="hand-edited schema")
    gate = make_pre_action_gate(memory)
    output = _run_gate(
        gate, "apply_fix", '{"file": "migrations/0001_init.py", "description": "tweak"}'
    )
    assert output.behavior["type"] == "reject_content"
    assert output.output_info["gate"] == "fragile_file"


def test_persistence_round_trip(tmp_path) -> None:
    log = tmp_path / "actions.jsonl"
    mem = ActionMemory(log_path=log)
    sig = action_signature("apply_fix", '{"file": "a.py"}')
    mem.record_failure(sig, summary="nope")
    # A fresh memory pointed at the same log replays the recorded events.
    reloaded = ActionMemory(log_path=log)
    assert reloaded.has_failed(sig) is True
