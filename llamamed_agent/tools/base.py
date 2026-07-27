"""Tool base class and registry.

A Tool declares its name, a description (shown to the model), and a
JSON-schema-like `parameters` dict (shown to the model so it knows what
to put in "Action Input"). `run` receives already-validated kwargs and
must return a string: this string becomes the "Observation" fed back to
the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class Tool(ABC):
    name: str = "tool"
    description: str = ""
    parameters: Dict = {}  # JSON-schema "properties" style dict

    @abstractmethod
    def run(self, **kwargs) -> str:
        raise NotImplementedError

    def schema_block(self) -> str:
        params = ", ".join(
            f'{p}: {info.get("type", "string")}' for p, info in self.parameters.items()
        ) or "no arguments"
        return f"- {self.name}({params}): {self.description}"


class ToolRegistry:
    def __init__(self, tools: List[Tool]):
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def render_for_prompt(self) -> str:
        return "\n".join(t.schema_block() for t in self._tools.values())
