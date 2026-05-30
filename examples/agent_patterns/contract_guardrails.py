"""Compile inspectable task contracts into Agents SDK guardrails.

Adapted from "Contractual Skills: A GovernSpec Design Framework for Enterprise AI
Agents" (arXiv:2605.22634). The paper argues that enterprise skills should make a task's
*goal, input boundaries, output contract, and quality criteria* inspectable rather than
burying them in free-form prose, and frames such contracts as a governance layer that
sits on top of -- and ultimately compiles down to -- runtime guardrails.

This module realizes that idea for this SDK. A :class:`TaskContract` is a small, readable
declaration of what a task accepts and what its output must satisfy. It compiles into the
SDK's existing :class:`~agents.guardrail.InputGuardrail` and
:class:`~agents.guardrail.OutputGuardrail` primitives, so the contract is enforced by code
the ``Runner`` already calls -- no new runtime is introduced. Each check produces a
structured :class:`ContractReport` in the guardrail's ``output_info`` so callers can see
exactly which clause passed or failed (the paper's "checkability" benefit), which gives a
retry policy a structured basis for deciding whether output is acceptable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    RunContextWrapper,
    TResponseInputItem,
)

__all__ = [
    "ContractCheck",
    "ContractReport",
    "TaskContract",
]


@dataclass
class ContractCheck:
    """The outcome of a single contract clause."""

    label: str
    passed: bool
    detail: str = ""


@dataclass
class ContractReport:
    """A structured, inspectable record of how an input or output met a contract."""

    contract: str
    stage: str  # "input" or "output"
    checks: list[ContractCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[ContractCheck]:
        return [check for check in self.checks if not check.passed]


def _input_to_text(value: str | list[TResponseInputItem]) -> str:
    """Flatten guardrail input into a single text blob for boundary checks."""
    if isinstance(value, str):
        return value

    parts: list[str] = []
    for item in value:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for chunk in content:
                text = chunk.get("text") if isinstance(chunk, dict) else None
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def _serialize_output(output: Any) -> str:
    """Best-effort string view of an agent output for substring screening."""
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except TypeError:
        return str(output)


@dataclass
class TaskContract:
    """A readable task contract that compiles into runtime guardrails.

    Attributes mirror the contractual-skill fields from the paper:

    - ``goal``: a human-readable statement of intent (the inspectable summary).
    - ``input_boundaries``: ``(label, predicate)`` pairs over the input text; a predicate
      returning ``False`` marks the input as out of bounds.
    - ``output_contract``: a pydantic model the output must validate against (the
      "output contract / expected schema" from the suggested experiment).
    - ``quality_criteria``: ``(label, predicate)`` pairs over the validated output object.
    - ``forbidden_output_substrings``: substrings that must not appear in the output.
    """

    name: str
    goal: str
    input_boundaries: list[tuple[str, Callable[[str], bool]]] = field(default_factory=list)
    output_contract: type[BaseModel] | None = None
    quality_criteria: list[tuple[str, Callable[[Any], bool]]] = field(default_factory=list)
    forbidden_output_substrings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Render the contract as inspectable text (progressive-loading friendly)."""
        lines = [f"Contract: {self.name}", f"Goal: {self.goal}"]
        if self.input_boundaries:
            lines.append("Input boundaries:")
            lines += [f"  - {label}" for label, _ in self.input_boundaries]
        if self.output_contract is not None:
            lines.append(f"Output contract: {self.output_contract.__name__}")
        if self.quality_criteria:
            lines.append("Quality criteria:")
            lines += [f"  - {label}" for label, _ in self.quality_criteria]
        if self.forbidden_output_substrings:
            lines.append(f"Forbidden in output: {', '.join(self.forbidden_output_substrings)}")
        return "\n".join(lines)

    def check_input(self, value: str | list[TResponseInputItem]) -> ContractReport:
        text = _input_to_text(value)
        report = ContractReport(contract=self.name, stage="input")
        for label, predicate in self.input_boundaries:
            try:
                ok = bool(predicate(text))
                detail = "" if ok else "input outside declared boundary"
            except Exception as exc:  # noqa: BLE001 - report instead of crashing the run
                ok, detail = False, f"boundary check errored: {exc}"
            report.checks.append(ContractCheck(label=label, passed=ok, detail=detail))
        return report

    def check_output(self, output: Any) -> ContractReport:
        report = ContractReport(contract=self.name, stage="output")

        validated: Any = output
        if self.output_contract is not None:
            if isinstance(output, self.output_contract):
                report.checks.append(ContractCheck(label="output_contract", passed=True))
            else:
                try:
                    payload = output.model_dump() if isinstance(output, BaseModel) else output
                    validated = self.output_contract.model_validate(payload)
                    report.checks.append(ContractCheck(label="output_contract", passed=True))
                except (ValidationError, TypeError, ValueError) as exc:
                    report.checks.append(
                        ContractCheck(
                            label="output_contract",
                            passed=False,
                            detail=f"does not satisfy {self.output_contract.__name__}: {exc}",
                        )
                    )

        for label, predicate in self.quality_criteria:
            try:
                ok = bool(predicate(validated))
                detail = "" if ok else "quality criterion not met"
            except Exception as exc:  # noqa: BLE001 - report instead of crashing the run
                ok, detail = False, f"criterion errored: {exc}"
            report.checks.append(ContractCheck(label=label, passed=ok, detail=detail))

        if self.forbidden_output_substrings:
            serialized = _serialize_output(output)
            for needle in self.forbidden_output_substrings:
                present = needle in serialized
                report.checks.append(
                    ContractCheck(
                        label=f"forbid:{needle}",
                        passed=not present,
                        detail="forbidden substring present" if present else "",
                    )
                )
        return report

    def as_input_guardrail(self) -> InputGuardrail[Any]:
        """Compile the input boundaries into an :class:`InputGuardrail`."""

        async def _guardrail(
            context: RunContextWrapper[Any],
            agent: Agent[Any],
            value: str | list[TResponseInputItem],
        ) -> GuardrailFunctionOutput:
            report = self.check_input(value)
            return GuardrailFunctionOutput(
                output_info=report,
                tripwire_triggered=not report.passed,
            )

        return InputGuardrail(guardrail_function=_guardrail, name=f"{self.name}:input")

    def as_output_guardrail(self) -> OutputGuardrail[Any]:
        """Compile the output contract and quality criteria into an :class:`OutputGuardrail`."""

        async def _guardrail(
            context: RunContextWrapper[Any],
            agent: Agent[Any],
            output: Any,
        ) -> GuardrailFunctionOutput:
            report = self.check_output(output)
            return GuardrailFunctionOutput(
                output_info=report,
                tripwire_triggered=not report.passed,
            )

        return OutputGuardrail(guardrail_function=_guardrail, name=f"{self.name}:output")
