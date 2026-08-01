"""The agent loop: generate -> parse -> (dispatch tool -> observe) -> repeat.

This is a single-conversation-turn ReAct loop (Yao et al., 2022 style):
the model produces one Thought/Action/Action-Input block, we run the tool
ourselves and hand the result back as an Observation, and the model
continues from there until it emits a Final Answer or we hit
max_iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .. import guardrails
from ..backends.base import LLMBackend, build_llama3_prompt
from ..config import Config
from ..memory import LongTermMemory
from ..tools.base import ToolRegistry
from .parser import AgentStep, parse_step
from .prompts import render_system_prompt

StepCallback = Callable[[AgentStep, Optional[str]], None]
TokenCallback = Callable[[str], None]


@dataclass
class AgentResult:
    final_answer: str
    iterations: int
    scratchpad: str
    blocked: bool = False


class Agent:
    def __init__(
        self,
        backend: LLMBackend,
        tools: ToolRegistry,
        cfg: Config,
        long_term_memory: Optional[LongTermMemory] = None,
        session_id: str = "unknown",
    ):
        self.backend = backend
        self.tools = tools
        self.cfg = cfg
        self.long_term_memory = long_term_memory
        self.session_id = session_id
        self.system_prompt = render_system_prompt(tools.render_for_prompt())

    def run(
        self,
        user_query: str,
        on_step: Optional[StepCallback] = None,
        on_token: Optional[TokenCallback] = None,
    ) -> AgentResult:
        # Guardrail checkpoint 1: runs before any generation happens at all.
        gate = guardrails.check_input(user_query)
        if not gate.allowed:
            return AgentResult(final_answer=gate.reason, iterations=0, scratchpad="", blocked=True)

        scratchpad = ""
        max_iterations = self.cfg.agent.max_iterations

        for iteration in range(1, max_iterations + 1):
            prompt = build_llama3_prompt(self.system_prompt, user_query, scratchpad)
            stop = ["Observation:", "<|eot_id|>"]

            if on_token is not None:
                raw = ""
                for chunk in self.backend.generate_stream(
                    prompt,
                    stop=stop,
                    max_tokens=self.cfg.model.max_new_tokens,
                    temperature=self.cfg.model.temperature,
                    top_p=self.cfg.model.top_p,
                ):
                    raw += chunk
                    on_token(chunk)
                raw = raw.strip()
            else:
                raw = self.backend.generate(
                    prompt,
                    stop=stop,
                    max_tokens=self.cfg.model.max_new_tokens,
                    temperature=self.cfg.model.temperature,
                    top_p=self.cfg.model.top_p,
                ).strip()

            if not raw:
                break

            step = parse_step(raw)
            scratchpad += raw + "\n"

            if step.is_final:
                # Guardrail checkpoint 2: runs on every final answer before
                # it's returned, regardless of how it was reached.
                answer = guardrails.check_output(step.final_answer)
                if self.long_term_memory is not None:
                    self.long_term_memory.remember(user_query, answer, self.session_id)
                if on_step:
                    on_step(step, None)
                return AgentResult(final_answer=answer, iterations=iteration, scratchpad=scratchpad)

            observation = self._dispatch(step)
            scratchpad += f"Observation: {observation}\n"

            if on_step:
                on_step(step, observation)

        fallback = guardrails.check_output(
            "I wasn't able to reach a final answer within the iteration limit. "
            "Here is my reasoning so far:\n\n" + scratchpad
        )
        return AgentResult(final_answer=fallback, iterations=max_iterations, scratchpad=scratchpad)

    def _dispatch(self, step: AgentStep) -> str:
        if step.action is None:
            return "Error: no Action was specified."
        try:
            tool = self.tools.get(step.action)
        except KeyError:
            return f"Error: unknown tool '{step.action}'. Available tools: {', '.join(self.tools.names())}"

        try:
            return tool.run(**(step.action_input or {}))
        except TypeError as e:
            return f"Error: invalid arguments for '{step.action}': {e}"
        except Exception as e:  # noqa: BLE001 - surfaced to the model as an Observation, not raised
            return f"Error while running '{step.action}': {e}"
