"""Synthesizer node that writes the final report."""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from ..prompts import SYNTHESIZER_PROMPT
from ..retry import ainvoke_with_retry
from ..state import ResearchState


def _clip_text(value: str, max_len: int) -> str:
    """Clamp long strings to avoid exceeding model context limits."""

    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def create_synthesizer_node(model: BaseChatModel):
    parser = StrOutputParser()
    chain = SYNTHESIZER_PROMPT | model | parser

    async def _node(state: ResearchState) -> Dict[str, Any]:
        findings = state.get("findings", [])
        # Limit how much evidence we pass to the model to avoid context explosions
        trimmed = []
        for finding in findings[-12:]:  # only the most recent dozen
            source = _clip_text(finding.source, 120)
            content = _clip_text(finding.content, 800)
            trimmed.append(f"- {source}: {content}")
        findings_text = "\n".join(trimmed)
        scope = _clip_text(state.get("scope") or "", 500)
        report = await ainvoke_with_retry(
            chain,
            {
                "query": state["query"],
                "scope": scope,
                "findings": findings_text or "No findings yet",
            },
        )
        messages = list(state.get("messages", [])) + [AIMessage(content=report, name="synthesizer")]
        return {"draft_report": report, "needs_revision": False, "messages": messages}

    return _node


__all__ = ["create_synthesizer_node"]
