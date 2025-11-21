"""Planner node implementation."""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from ..prompts import PLANNER_PROMPT
from ..retry import ainvoke_with_retry
from ..state import ResearchState


def _parse_plan(raw: str) -> List[str]:
    steps: List[str] = []
    for line in raw.splitlines():
        line = line.strip("- ")
        if not line:
            continue
        steps.append(line)
    return steps or ["Review question and perform general reconnaissance"]


def create_planner_node(model: BaseChatModel):
    parser = StrOutputParser()
    chain = PLANNER_PROMPT | model | parser

    async def _node(state: ResearchState) -> Dict[str, Any]:
        findings_preview = "\n".join(f"- {finding.source}" for finding in state.get("findings", [])[-3:]) or "none yet"
        result = await ainvoke_with_retry(
            chain,
            {
                "query": state["query"],
                "scope": state.get("scope") or "",
                "findings": findings_preview,
            },
        )
        plan = _parse_plan(result)
        messages = list(state.get("messages", [])) + [AIMessage(content="\n".join(plan), name="planner")]
        return {"plan": plan, "messages": messages}

    return _node


__all__ = ["create_planner_node"]
