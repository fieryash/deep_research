"""Prompt templates shared by LangGraph nodes."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SCOPER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Scoper in a deep research team."
            "Clarify the research question, capture the key sub-questions, "
            "and highlight knowledge gaps. Respond with markdown headings.",
        ),
        (
            "human",
            "Research question: {query}\n"
            "Current scope (if any): {scope}\n"
            "Conversation summary: {history}",
        ),
    ]
)

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Planner agent. Produce an ordered list of steps that the "
            "research team should execute. Include tooling suggestions when possible.",
        ),
        (
            "human",
            "Question: {query}\nScope: {scope}\nKey prior findings: {findings}",
        ),
    ]
)

RESEARCHER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Researcher agent. Combine browsing, MCP tools, and reasoning "
            "to produce fresh findings. Each finding should include the source "
            "and a concise quote."
            "If you cannot access tools you may reason hypothetically but state so.",
        ),
        (
            "human",
            "Question: {query}\nPlan focus: {plan}\nExisting findings: {findings}",
        ),
    ]
)

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Synthesizer agent. Turn the findings into a final report that "
            "answers the research question. Structure it with sections and highlight "
            "open questions. The audience is an executive technical reader.",
        ),
        (
            "human",
            "Question: {query}\nScope: {scope}\nFindings: {findings}",
        ),
    ]
)

REVIEWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Reviewer agent. Provide actionable critique on the draft report "
            "and decide whether another research loop is needed. Respond with JSON.",
        ),
        (
            "human",
            "Question: {query}\nDraft report: {report}",
        ),
    ]
)

__all__ = [
    "SCOPER_PROMPT",
    "PLANNER_PROMPT",
    "RESEARCHER_PROMPT",
    "SYNTHESIZER_PROMPT",
    "REVIEWER_PROMPT",
]
