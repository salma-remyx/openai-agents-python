"""A deterministic "pre-action gate" backed by an append-only attempt log.

Adapted from PROJECTMEM: A Local-First, Event-Sourced Memory and Judgment Layer
for AI Coding Agents (https://arxiv.org/abs/2606.12329v1).

The paper's core idea is *Memory-as-Governance*: an agent records its work as an
append-only log of typed events (attempts, fixes, outcomes), and a deterministic
gate consults that log *before* the next action — warning or blocking when the
agent is about to repeat a fix that already failed. Instead of merely answering
the agent, memory acts on its next action.

This module ports that result onto the SDK's existing tool-guardrail call site,
which is itself a pre-action gate: a ``tool_input_guardrail`` runs immediately
before a function tool executes. We keep an in-session ``AttemptLog`` of every
tool call and its outcome; the input gate refuses a tool call whose normalized
arguments already failed, and a companion output gate records each outcome back
into the log.

Intentionally out of scope (not needed to deliver the result): the paper's MCP
server, CLI, plain-text on-disk projection, and cross-session provenance trail.
The value here is the deterministic don't-repeat-a-failed-attempt gate, which
slots directly into the SDK's per-tool guardrail hooks.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from agents import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrail,
    ToolOutputGuardrailData,
    tool_input_guardrail,
    tool_output_guardrail,
)

AttemptStatus = Literal["attempted", "succeeded", "failed"]


def action_key(tool_name: str, tool_arguments: str | None) -> str:
    """Build a stable identity for a tool call from its name and arguments.

    Arguments are parsed as JSON and re-serialized with sorted keys so that
    semantically identical calls (e.g. reordered keys or differing whitespace)
    collapse to the same key. Unparseable arguments fall back to their raw text.
    """
    raw = tool_arguments or ""
    try:
        normalized = json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        normalized = raw.strip()
    return f"{tool_name}({normalized})"


@dataclass
class AttemptEvent:
    """One typed entry in the append-only attempt log."""

    tool_name: str
    key: str
    status: AttemptStatus
    note: str = ""


@dataclass
class AttemptLog:
    """An append-only, in-session event log of tool attempts and outcomes.

    Mirrors PROJECTMEM's event log: entries are only ever appended, never
    mutated, so the log doubles as a provenance trail of what the agent tried.
    """

    events: list[AttemptEvent] = field(default_factory=list)

    def record(
        self,
        tool_name: str,
        tool_arguments: str | None,
        status: AttemptStatus,
        note: str = "",
    ) -> AttemptEvent:
        """Append an event and return it."""
        event = AttemptEvent(
            tool_name=tool_name,
            key=action_key(tool_name, tool_arguments),
            status=status,
            note=note,
        )
        self.events.append(event)
        return event

    def last_failure(self, tool_name: str, tool_arguments: str | None) -> AttemptEvent | None:
        """Return the most recent *failed* attempt matching this exact action.

        Returns ``None`` if the action has never failed, or if a later attempt
        of the same action succeeded (the agent is allowed to move on once a
        previously failing action starts working).
        """
        key = action_key(tool_name, tool_arguments)
        failure: AttemptEvent | None = None
        for event in self.events:
            if event.key != key:
                continue
            if event.status == "failed":
                failure = event
            elif event.status == "succeeded":
                failure = None
        return failure

    def summary(self) -> str:
        """Render a compact, AI-readable projection of the log."""
        if not self.events:
            return "No attempts recorded yet."
        lines = [
            f"{e.status.upper():9} {e.key}" + (f" — {e.note}" if e.note else "")
            for e in self.events
        ]
        return "\n".join(lines)


def _looks_like_failure(output: Any) -> bool:
    """Heuristically decide whether a tool result represents a failed attempt."""
    text = str(output).lower()
    markers = ("error", "failed", "failure", "traceback", "exception", "non-zero exit")
    return any(marker in text for marker in markers)


def make_repeat_attempt_gate(
    log: AttemptLog,
    *,
    block: bool = True,
) -> ToolInputGuardrail[Any]:
    """Build the pre-action gate.

    The returned guardrail runs before a tool executes. If the exact same action
    (tool name + normalized arguments) already failed earlier in the session, it
    short-circuits the call so the agent does not burn another turn repeating a
    known-bad fix.

    Args:
        log: The shared attempt log to consult.
        block: When ``True`` (default), repeats are rejected via
            ``reject_content`` so the model is told the call was not run. When
            ``False``, the gate allows the call but attaches a warning in
            ``output_info`` for softer, advisory governance.
    """

    @tool_input_guardrail(name="repeat_attempt_gate")
    def gate(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        ctx = data.context
        prior = log.last_failure(ctx.tool_name, ctx.tool_arguments)
        if prior is None:
            log.record(ctx.tool_name, ctx.tool_arguments, "attempted")
            return ToolGuardrailFunctionOutput(output_info={"repeat": False})

        detail = f" ({prior.note})" if prior.note else ""
        message = (
            f"🛑 Skipped: '{ctx.tool_name}' was already attempted with these exact "
            f"arguments and failed{detail}. Try a different approach instead of "
            f"repeating the same fix."
        )
        info = {"repeat": True, "key": prior.key, "prior_note": prior.note}
        if block:
            return ToolGuardrailFunctionOutput.reject_content(message=message, output_info=info)
        return ToolGuardrailFunctionOutput(output_info={**info, "warning": message})

    return gate


def make_outcome_recorder(
    log: AttemptLog,
    *,
    is_failure: Callable[[Any], bool] = _looks_like_failure,
) -> ToolOutputGuardrail[Any]:
    """Build the companion output gate that records each attempt's outcome.

    After a tool runs, this records ``succeeded`` or ``failed`` into the log so
    the pre-action gate can govern future calls. The failure heuristic is
    overridable for tools with a structured success/error convention.
    """

    @tool_output_guardrail(name="attempt_outcome_recorder")
    def recorder(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        ctx = data.context
        failed = is_failure(data.output)
        status: AttemptStatus = "failed" if failed else "succeeded"
        note = str(data.output)[:120] if failed else ""
        log.record(ctx.tool_name, ctx.tool_arguments, status, note=note)
        return ToolGuardrailFunctionOutput(output_info={"recorded": status})

    return recorder


__all__ = [
    "AttemptEvent",
    "AttemptLog",
    "AttemptStatus",
    "action_key",
    "make_outcome_recorder",
    "make_repeat_attempt_gate",
]
