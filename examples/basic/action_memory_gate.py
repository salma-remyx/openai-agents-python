"""Event-sourced action memory with a deterministic pre-action gate.

Adapted from PROJECTMEM: A Local-First, Event-Sourced Memory and Judgment
Layer for AI Coding Agents (arXiv:2606.12329).

PROJECTMEM records development as an append-only log of typed events and adds a
deterministic *pre-action gate* that warns an agent before it repeats a fix that
already failed or edits a known-fragile file. The paper frames this as
"Memory-as-Governance": memory that does not merely answer the agent but acts on
its next action.

This module implements that core idea as a small, dependency-free layer that
plugs into the Agents SDK through the existing ``tool_input_guardrail`` hook --
the SDK's "before a tool runs" extension point. A sandbox or coding agent that
carries an :class:`ActionMemory` therefore stops re-attempting actions that
history already shows do not work, instead of burning a turn re-deriving the same
dead end.

Scope: this delivers the in-session governance behavior -- a typed event log, a
deterministic action signature, and the gate. The paper's full MCP server, CLI
surface, and offline-summary projections are intentionally out of scope; they are
not needed to demonstrate the result in this repo.

Contributed via Remyx Recommendation (https://engine.remyx.ai).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from agents import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
    tool_input_guardrail,
)


class EventType(str, Enum):
    """Typed events mirroring PROJECTMEM's event vocabulary."""

    ATTEMPT = "attempt"
    FAILURE = "failure"
    FIX = "fix"
    FRAGILE_FILE = "fragile_file"
    NOTE = "note"


@dataclass
class ActionEvent:
    """A single immutable entry in the append-only action log."""

    type: EventType
    signature: str
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data


def action_signature(tool_name: str, tool_arguments: str | None) -> str:
    """Canonicalize a tool call into a stable, comparable signature.

    Argument key order and insignificant whitespace are normalized so that two
    logically identical tool calls map to the same signature regardless of how
    the model happened to serialize them. This determinism is what lets the gate
    recognize a repeated action without consulting a model.
    """
    parsed: Any
    try:
        parsed = json.loads(tool_arguments) if tool_arguments else {}
    except (json.JSONDecodeError, TypeError):
        parsed = {"__raw__": tool_arguments}
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return f"{tool_name}::{canonical}"


class ActionMemory:
    """Append-only, event-sourced log of agent actions.

    The log is the source of truth; projections (failed signatures, fragile
    files) are derived deterministically from it, mirroring PROJECTMEM's
    event-sourced design. An optional ``log_path`` persists events as JSON lines
    so memory survives across sessions.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        self._events: list[ActionEvent] = []
        self._log_path = Path(log_path) if log_path else None
        if self._log_path is not None and self._log_path.exists():
            self._load()

    def _load(self) -> None:
        assert self._log_path is not None
        for line in self._log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            self._events.append(
                ActionEvent(
                    type=EventType(raw["type"]),
                    signature=raw.get("signature", ""),
                    summary=raw.get("summary", ""),
                    metadata=raw.get("metadata", {}),
                )
            )

    def record(self, event: ActionEvent) -> ActionEvent:
        """Append an event and persist it if a log path is configured."""
        self._events.append(event)
        if self._log_path is not None:
            with self._log_path.open("a") as fh:
                fh.write(json.dumps(event.to_dict()) + "\n")
        return event

    # Convenience recorders for the paper's typed events. ----------------------
    def record_attempt(self, signature: str, summary: str = "", **metadata: Any) -> ActionEvent:
        return self.record(ActionEvent(EventType.ATTEMPT, signature, summary, metadata))

    def record_failure(self, signature: str, summary: str = "", **metadata: Any) -> ActionEvent:
        return self.record(ActionEvent(EventType.FAILURE, signature, summary, metadata))

    def record_fix(self, signature: str, summary: str = "", **metadata: Any) -> ActionEvent:
        return self.record(ActionEvent(EventType.FIX, signature, summary, metadata))

    def mark_fragile_file(self, path: str, summary: str = "") -> ActionEvent:
        return self.record(ActionEvent(EventType.FRAGILE_FILE, path, summary))

    # Deterministic projections over the log. ----------------------------------
    def failed_signatures(self) -> set[str]:
        return {e.signature for e in self._events if e.type == EventType.FAILURE}

    def succeeded_signatures(self) -> set[str]:
        return {e.signature for e in self._events if e.type == EventType.FIX}

    def fragile_files(self) -> set[str]:
        return {e.signature for e in self._events if e.type == EventType.FRAGILE_FILE}

    def has_failed(self, signature: str) -> bool:
        """True if this exact action failed before and was not since fixed.

        A later :meth:`record_fix` for the same signature clears the warning, so
        an approach that eventually works stops being blocked.
        """
        return (
            signature in self.failed_signatures() and signature not in self.succeeded_signatures()
        )

    @property
    def events(self) -> tuple[ActionEvent, ...]:
        return tuple(self._events)


def _files_in_arguments(tool_arguments: str | None) -> list[str]:
    """Best-effort extraction of file paths from a tool call's arguments."""
    try:
        parsed = json.loads(tool_arguments) if tool_arguments else {}
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    files: list[str] = []
    for key, value in parsed.items():
        if isinstance(value, str) and ("file" in key.lower() or "path" in key.lower()):
            files.append(value)
    return files


def make_pre_action_gate(
    memory: ActionMemory, *, record_attempts: bool = True
) -> ToolInputGuardrail[Any]:
    """Build a deterministic pre-action gate bound to ``memory``.

    The returned object is a regular :class:`ToolInputGuardrail`, so it attaches
    to any function tool via ``tool.tool_input_guardrails`` exactly like the
    guardrails already shown in ``tool_guardrails.py``. On every tool call it:

    1. Rejects the call if its action signature matches a recorded, unfixed
       failure (the "don't repeat a failed fix" rule).
    2. Rejects the call if it targets a file marked fragile.
    3. Otherwise records the attempt and allows execution.
    """

    @tool_input_guardrail(name="pre_action_gate")
    def pre_action_gate(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        ctx = data.context
        signature = action_signature(ctx.tool_name, ctx.tool_arguments)

        if memory.has_failed(signature):
            return ToolGuardrailFunctionOutput.reject_content(
                message=(
                    f"🛑 Blocked by action-memory gate: this exact action "
                    f"(`{ctx.tool_name}`) already failed earlier in the project and "
                    "was not since fixed. Try a different approach instead of "
                    "repeating it."
                ),
                output_info={"gate": "repeated_failed_fix", "signature": signature},
            )

        fragile = memory.fragile_files()
        for path in _files_in_arguments(ctx.tool_arguments):
            if path in fragile:
                return ToolGuardrailFunctionOutput.reject_content(
                    message=(
                        f"⚠️ Blocked by action-memory gate: `{path}` is flagged as "
                        "fragile. Confirm the change is intended before editing it."
                    ),
                    output_info={"gate": "fragile_file", "path": path},
                )

        if record_attempts:
            memory.record_attempt(signature, summary=f"Attempted {ctx.tool_name}.")
        return ToolGuardrailFunctionOutput.allow(
            output_info={"gate": "clear", "signature": signature}
        )

    return pre_action_gate
