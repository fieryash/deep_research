"""State definitions shared across the LangGraph workflow."""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


@dataclass
class Finding:
    """Single research snippet that we can persist later."""

    source: str
    content: str
    confidence: float = 0.5
    captured_at: datetime = field(default_factory=datetime.utcnow)


class ReviewResult(TypedDict, total=False):
    approved: bool
    critique: str
    next_action: str


class ResearchState(TypedDict):
    """Global state object that LangGraph threads through nodes."""

    query: str
    scope: Optional[str]
    plan: List[str]
    findings: List[Finding]
    draft_report: Optional[str]
    review: Optional[ReviewResult]
    needs_revision: bool
    loop_count: int
    messages: Annotated[List[BaseMessage], operator.add]


__all__ = ["Finding", "ResearchState", "ReviewResult"]
