"""Synthesizer node that writes the final report."""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from ..prompts import SYNTHESIZER_PROMPT
from ..state import ResearchState


def create_synthesizer_node(model: BaseChatModel):
    parser = StrOutputParser()
    chain = SYNTHESIZER_PROMPT | model | parser

    async def _node(state: ResearchState) -> Dict[str, Any]:
        findings_text = "\n".join(f"- {finding.source}: {finding.content}" for finding in state.get("findings", []))
        report = await chain.ainvoke(
            {
                "query": state["query"],
                "scope": state.get("scope") or "",
                "findings": findings_text or "No findings yet",
            }
        )
        messages = list(state.get("messages", [])) + [AIMessage(content=report, name="synthesizer")]
        return {"draft_report": report, "needs_revision": False, "messages": messages}

    return _node


__all__ = ["create_synthesizer_node"]
