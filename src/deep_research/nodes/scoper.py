"""Scoper node implementation."""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from ..prompts import SCOPER_PROMPT
from ..retry import ainvoke_with_retry
from ..state import ResearchState


def create_scoper_node(model: BaseChatModel):
    """Return an async callable that enriches scope/context."""

    parser = StrOutputParser()
    chain = SCOPER_PROMPT | model | parser

    async def _node(state: ResearchState) -> Dict[str, Any]:
        history = "\n".join(message.content for message in state.get("messages", [])[-4:])
        scope = await ainvoke_with_retry(
            chain,
            {
                "query": state["query"],
                "scope": state.get("scope") or "(not defined yet)",
                "history": history or "(no prior conversation)",
            },
        )
        messages = list(state.get("messages", [])) + [AIMessage(content=scope, name="scoper")]
        return {"scope": scope, "messages": messages}

    return _node


__all__ = ["create_scoper_node"]
