"""Parses one model continuation into a structured AgentStep.

The model is instructed (see prompts.py) to produce either:

    Thought: ...
    Action: <tool name>
    Action Input: <json>

or:

    Thought: ...
    Final Answer: ...

This parser is intentionally forgiving: models -- especially small
fine-tunes -- drift from exact formatting under sampling. It tries a
strict regex first, then falls back to looser line-based matching before
giving up.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentStep:
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    final_answer: Optional[str] = None
    raw: str = ""

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


class ParseError(Exception):
    pass


_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\}|\S.*)", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def parse_step(text: str) -> AgentStep:
    text = text.strip()

    final_match = _FINAL_RE.search(text)
    if final_match:
        thought_match = _THOUGHT_RE.search(text)
        thought = thought_match.group(1).strip() if thought_match else ""
        return AgentStep(thought=thought, final_answer=final_match.group(1).strip(), raw=text)

    action_match = _ACTION_RE.search(text)
    input_match = _ACTION_INPUT_RE.search(text)
    thought_match = _THOUGHT_RE.search(text)

    if not action_match:
        # Model produced neither a Final Answer nor an Action -- treat the
        # whole thing as a final answer rather than crashing the loop.
        return AgentStep(thought="", final_answer=text, raw=text)

    action_name = action_match.group(1).strip()
    action_input = {}
    if input_match:
        raw_input = input_match.group(1).strip().rstrip("`")
        action_input = _parse_action_input(raw_input)

    thought = thought_match.group(1).strip() if thought_match else ""
    return AgentStep(thought=thought, action=action_name, action_input=action_input, raw=text)


def _parse_action_input(raw_input: str) -> dict:
    try:
        parsed = json.loads(raw_input)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except json.JSONDecodeError:
        pass

    # Loose fallback: try to salvage a JSON object embedded in extra text.
    brace_match = re.search(r"\{.*\}", raw_input, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: pass the raw string through under a generic key so the
    # tool call still happens rather than silently dropping the input.
    return {"raw": raw_input}
