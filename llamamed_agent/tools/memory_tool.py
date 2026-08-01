from __future__ import annotations

from .base import Tool
from ..memory import LongTermMemory


class RecallMemoryTool(Tool):
    name = "recall_memory"
    description = (
        "Searches past conversations (across all sessions) for similar questions this agent "
        "has already answered. Use this before redoing a calculation or a literature search "
        "you may have already done in a previous session."
    )
    parameters = {
        "query": {"type": "string", "description": "What to search past turns for"},
    }

    def __init__(self, memory: LongTermMemory):
        self.memory = memory

    def run(self, query: str) -> str:
        hits = self.memory.recall(query, top_k=3)
        if not hits:
            return "No related past conversation found."
        return "\n\n---\n\n".join(hits)
