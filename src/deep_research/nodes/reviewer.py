"""Reviewer node that critiques synthesized reports."""
from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from ..prompts import REVIEWER_PROMPT
from ..retry import ainvoke_with_retry
from ..state import ResearchState, ReviewResult


def create_reviewer_node(model: BaseChatModel):
    parser = StrOutputParser()
    chain = REVIEWER_PROMPT | model | parser

    async def _node(state: ResearchState) -> Dict[str, Any]:
        if not state.get("draft_report"):
            return {"needs_revision": True}

        critique_raw = await ainvoke_with_retry(
            chain,
            {
                "query": state["query"],
                "report": state.get("draft_report") or "",
            },
        )
        try:
            parsed: ReviewResult = json.loads(critique_raw)
        except json.JSONDecodeError:
            parsed = ReviewResult(approved=False, critique=critique_raw, next_action="Revise and retry")

        needs_revision = not parsed.get("approved", False)
        message = AIMessage(content=json.dumps(parsed, indent=2), name="reviewer")
        messages = list(state.get("messages", [])) + [message]
        loop_count = state.get("loop_count", 0) + 1
        return {"review": parsed, "needs_revision": needs_revision, "messages": messages, "loop_count": loop_count}

    return _node


__all__ = ["create_reviewer_node"]
