"""LangGraph assembly for the Deep Research pipeline."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from .config import AppConfig
from .llm import init_models
from .nodes.planner import create_planner_node
from .nodes.researcher import create_researcher_node
from .nodes.reviewer import create_reviewer_node
from .nodes.scoper import create_scoper_node
from .nodes.synthesizer import create_synthesizer_node
from .persistence import ResearchLogger
from .state import ResearchState
from .tools import MCPToolGateway, TavilySearchTool


@dataclass
class ResearchRunResult:
    run_id: str
    state: ResearchState
    log_path: str


class DeepResearchPipeline:
    """High-level orchestrator that wraps the LangGraph workflow."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_dirs()
        self.models = init_models(config)
        self.search_tool = self._build_search_tool()
        self.mcp_gateway = MCPToolGateway(config.mcp_servers) if config.mcp_servers else None
        self.logger = ResearchLogger(config.persistence_dir)
        self.graph = self._build_graph()

    def _build_search_tool(self) -> TavilySearchTool | None:
        if self.config.search.provider == "tavily":
            return TavilySearchTool(self.config.search.tavily_api_key)
        return None

    def _build_graph(self):
        workflow = StateGraph(ResearchState)

        scoper = create_scoper_node(self.models["summarizer"])
        planner = create_planner_node(self.models["summarizer"])
        researcher = create_researcher_node(
            model=self.models["researcher"],
            search_tool=self.search_tool,
            mcp_gateway=self.mcp_gateway,
        )
        synthesizer = create_synthesizer_node(self.models["synthesizer"])
        reviewer = create_reviewer_node(self.models["reviewer"])

        workflow.add_node("scope", scoper)
        workflow.add_node("plan", planner)
        workflow.add_node("research", researcher)
        workflow.add_node("synth", synthesizer)
        workflow.add_node("review", reviewer)

        workflow.add_edge(START, "scope")
        workflow.add_edge("scope", "plan")
        workflow.add_edge("plan", "research")
        workflow.add_edge("research", "synth")
        workflow.add_edge("synth", "review")

        def review_router(state: ResearchState) -> str:
            if state.get("needs_revision"):
                max_loops = self.config.max_research_loops
                if state.get("loop_count", 0) < max_loops:
                    return "revise"
            return "complete"

        workflow.add_conditional_edges(
            "review",
            review_router,
            {
                "revise": "research",
                "complete": END,
            },
        )

        return workflow.compile()

    async def run(self, query: str, *, scope: str | None = None, metadata: Dict[str, Any] | None = None) -> ResearchRunResult:
        run_id = uuid4().hex
        base_messages = [HumanMessage(content=f"Research question: {query}")]
        if metadata:
            base_messages.append(HumanMessage(content=str(metadata)))
        initial_state: ResearchState = {
            "query": query,
            "scope": scope,
            "plan": [],
            "findings": [],
            "draft_report": None,
            "review": None,
            "needs_revision": False,
            "loop_count": 0,
            "messages": base_messages,
        }
        final_state = await self.graph.ainvoke(initial_state)
        log_path = str(self.logger.log(run_id, final_state))
        return ResearchRunResult(run_id=run_id, state=final_state, log_path=log_path)

    async def run_stream(self, query: str, *, scope: str | None = None, metadata: Dict[str, Any] | None = None):
        run_id = uuid4().hex
        base_messages = [HumanMessage(content=f"Research question: {query}")]
        if metadata:
            base_messages.append(HumanMessage(content=str(metadata)))
        initial_state: ResearchState = {
            "query": query,
            "scope": scope,
            "plan": [],
            "findings": [],
            "draft_report": None,
            "review": None,
            "needs_revision": False,
            "loop_count": 0,
            "messages": base_messages,
        }
        
        final_state = initial_state
        async for chunk in self.graph.astream(initial_state):
            for node, update in chunk.items():
                if isinstance(update, dict):
                    final_state.update(update)
                yield {"type": "update", "node": node, "content": update}
        
        log_path = str(self.logger.log(run_id, final_state))
        yield {"type": "result", "data": ResearchRunResult(run_id=run_id, state=final_state, log_path=log_path)}

    async def shutdown(self) -> None:
        if self.mcp_gateway:
            await self.mcp_gateway.shutdown()


__all__ = ["DeepResearchPipeline", "ResearchRunResult"]
