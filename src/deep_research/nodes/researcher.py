"""Researcher node that orchestrates web + MCP tools."""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from ..prompts import RESEARCHER_PROMPT
from ..retry import ainvoke_with_retry
from ..state import Finding, ResearchState
from ..tools import MCPToolGateway, TavilySearchResult, TavilySearchTool


class ResearcherNode:
    def __init__(
        self,
        model: BaseChatModel,
        search_tool: TavilySearchTool | None,
        mcp_gateway: MCPToolGateway | None,
    ) -> None:
        self._model = model
        self._search_tool = search_tool
        self._mcp_gateway = mcp_gateway
        self._parser = StrOutputParser()
        self._chain = RESEARCHER_PROMPT | model | self._parser

    async def _search_web(self, query: str, limit: int = 3) -> List[TavilySearchResult]:
        if not self._search_tool:
            return []
        try:
            return await self._search_tool.search(query, max_results=limit)
        except Exception as exc:  # pragma: no cover - network failures
            return [
                TavilySearchResult(
                    url="tavily:error",
                    content=f"Search failed for '{query}': {exc}",
                    score=0.0,
                )
            ]

    async def _call_mcp(self, query: str) -> List[Finding]:
        findings: List[Finding] = []
        if not self._mcp_gateway:
            return findings
        try:
            await self._mcp_gateway.start()
        except Exception:
            return findings

        for tool_name in self._mcp_gateway.available_tools:
            if "search" not in tool_name.lower() and "rag" not in tool_name.lower():
                continue
            try:
                payload = await self._mcp_gateway.invoke(tool_name, {"query": query})
            except Exception as exc:  # pragma: no cover - mcp/io errors
                findings.append(Finding(source=f"mcp:{tool_name}", content=str(exc), confidence=0.2))
                continue
            findings.append(Finding(source=f"mcp:{tool_name}", content=payload, confidence=0.7))
        return findings

    async def __call__(self, state: ResearchState) -> Dict[str, Any]:
        query = state["query"]
        plan_steps = state.get("plan") or ["survey open web"]

        evidence_blobs: List[str] = []
        new_findings: List[Finding] = list(state.get("findings", []))

        for step in plan_steps:
            combined_query = f"{query} :: {step}"
            for result in await self._search_web(combined_query):
                evidence_blobs.append(f"{result.url}: {result.content}")
                new_findings.append(
                    Finding(source=result.url, content=result.content, confidence=min(1.0, result.score))
                )

        for finding in await self._call_mcp(query):
            evidence_blobs.append(f"{finding.source}: {finding.content}")
            new_findings.append(finding)

        research_summary = await ainvoke_with_retry(
            self._chain,
            {
                "query": query,
                "plan": "\n".join(plan_steps),
                "findings": "\n\n".join(evidence_blobs) or "No external evidence gathered yet",
            },
        )

        new_findings.append(Finding(source="researcher", content=research_summary, confidence=0.8))
        messages = list(state.get("messages", [])) + [AIMessage(content=research_summary, name="researcher")]
        return {"findings": new_findings, "messages": messages}


def create_researcher_node(
    model: BaseChatModel,
    search_tool: TavilySearchTool | None,
    mcp_gateway: MCPToolGateway | None,
) -> ResearcherNode:
    return ResearcherNode(model=model, search_tool=search_tool, mcp_gateway=mcp_gateway)


__all__ = ["create_researcher_node", "ResearcherNode"]
