from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, Field

from agents import (
    Agent,
    OutputGuardrailTripwireTriggered,
    Runner,
)
from examples.agent_patterns.contract_guardrails import TaskContract

"""
This example shows how to use output guardrails.

Output guardrails are checks that run on the final output of an agent.
They can be used to do things like:
- Check if the output contains sensitive data
- Check if the output is a valid response to the user's message

Instead of hand-writing the guardrail, we declare the task's acceptance criteria as a
`TaskContract` -- goal, output contract (schema), quality criteria, and forbidden content --
and compile it into an output guardrail. This follows the "contractual skills" framework
(see contract_guardrails.py), keeping the agent's contract inspectable in one place.

In this example, the contract forbids leaking a phone number in the response or reasoning.
"""


# The agent's output type
class MessageOutput(BaseModel):
    reasoning: str = Field(description="Thoughts on how to respond to the user's message")
    response: str = Field(description="The response to the user's message")
    user_name: str | None = Field(description="The name of the user who sent the message, if known")


support_reply_contract = TaskContract(
    name="support_reply",
    goal="Answer the user without leaking sensitive contact details.",
    output_contract=MessageOutput,
    quality_criteria=[
        ("no_area_code_in_response", lambda out: "650" not in out.response),
        ("no_area_code_in_reasoning", lambda out: "650" not in out.reasoning),
    ],
)


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    output_type=MessageOutput,
    output_guardrails=[support_reply_contract.as_output_guardrail()],
)


async def main():
    # This should be ok
    await Runner.run(agent, "What's the capital of California?")
    print("First message passed")

    # This should trip the guardrail
    try:
        result = await Runner.run(
            agent, "My phone number is 650-123-4567. Where do you think I live?"
        )
        print(
            f"Guardrail didn't trip - this is unexpected. Output: {json.dumps(result.final_output.model_dump(), indent=2)}"
        )

    except OutputGuardrailTripwireTriggered as e:
        report = e.guardrail_result.output.output_info
        failed = ", ".join(check.label for check in report.failures)
        print(f"Guardrail tripped. Contract '{report.contract}' failed clauses: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
