"""Tooling integrations (Tavily, MCP servers, etc.)."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import MCPServerConfig

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
except Exception:  # pragma: no cover - optional dependency during bootstrap
    TavilyClient = None  # type: ignore


@dataclass
class TavilySearchResult:
    url: str
    content: str
    score: float


class TavilySearchTool:
    """Thin async wrapper around Tavily's search API."""

    def __init__(self, api_key: Optional[str]) -> None:
        self.api_key = api_key
        self._client = TavilyClient(api_key=api_key) if TavilyClient and api_key else None

    async def search(self, query: str, *, max_results: int = 5) -> List[TavilySearchResult]:
        if not self._client:
            raise RuntimeError(
                "Tavily client unavailable. Install tavily-python and set TAVILY_API_KEY or switch providers."
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.search(query=query, max_results=max_results),
        )
        results = []
        for item in response.get("results", []):
            results.append(
                TavilySearchResult(
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                )
            )
        return results


def _render_tool_result(result: Any) -> str:
    """Convert MCP tool results into a string."""

    try:
        import mcp.types as types  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("mcp package required to render tool results") from exc

    chunks: List[str] = []

    if isinstance(result, types.CallToolResult):
        if result.isError and result.error:
            return f"Error: {result.error.message}"
        if result.content:
            for block in result.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    chunks.append(getattr(block, "text", ""))
                elif block_type == "image":
                    chunks.append(f"[image] {getattr(block, 'uri', '')}")
                else:
                    chunks.append(str(block))
        if result.structuredContent:
            chunks.append(json.dumps(result.structuredContent, indent=2))
    else:
        if isinstance(result, dict):
            return json.dumps(result, indent=2)
        chunks.append(str(result))

    return "\n".join(chunk for chunk in chunks if chunk)


class MCPToolGateway:
    """Manages MCP sessions and exposes higher-level helpers."""

    def __init__(self, servers: List[MCPServerConfig]):
        self._servers = servers
        self._stack: AsyncExitStack | None = None
        self._sessions: Dict[str, Any] = {}
        self._tools: Dict[str, Any] = {}
        self._started = False

    @property
    def available_tools(self) -> List[str]:
        return list(self._tools.keys())

    async def start(self) -> None:
        if self._started or not self._servers:
            return
        try:
            from mcp.client.session import ClientSession  # type: ignore
            from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore
            from mcp.client.sse import sse_client  # type: ignore
            from mcp.client.websocket import websocket_client  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("mcp package missing; MCP tools disabled (%s)", exc)
            return

        self._stack = AsyncExitStack()

        for server in self._servers:
            try:
                if server.transport == "stdio":
                    if not server.command:
                        raise ValueError(f"MCP server '{server.name}' missing command for stdio transport")
                    params = StdioServerParameters(
                        command=server.command,
                        args=server.args,
                        env=server.env or None,
                    )
                    read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
                elif server.transport == "sse":
                    if not server.sse_url:
                        raise ValueError(f"MCP server '{server.name}' missing sse_url")
                    read_stream, write_stream = await self._stack.enter_async_context(
                        sse_client(server.sse_url, headers=server.env or None)
                    )
                elif server.transport == "websocket":
                    if not server.websocket_url:
                        raise ValueError(f"MCP server '{server.name}' missing websocket_url")
                    read_stream, write_stream = await self._stack.enter_async_context(
                        websocket_client(server.websocket_url, headers=server.env or None)
                    )
                else:  # pragma: no cover - validated by pydantic
                    raise ValueError(f"Unknown transport {server.transport}")

                session = ClientSession(read_stream, write_stream)
                session = await self._stack.enter_async_context(session)
                await session.initialize()
                self._sessions[server.name] = session

                tool_result = await session.list_tools()
                for tool in tool_result.tools:
                    qualified_name = f"{server.name}:{tool.name}"
                    self._tools[qualified_name] = (server.name, tool)
            except Exception as exc:
                logger.exception("Failed to connect to MCP server %s: %s", server.name, exc)

        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            return
        try:
            if self._stack:
                await self._stack.aclose()
        finally:
            self._sessions.clear()
            self._tools.clear()
            self._stack = None
            self._started = False

    async def invoke(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        if not self._started:
            await self.start()

        if not self._tools:
            raise RuntimeError("No MCP tools available")

        qualified_name = self._resolve_name(tool_name)
        server_name, tool = self._tools[qualified_name]
        session = self._sessions[server_name]

        result = await session.call_tool(tool.name, arguments or {})
        return _render_tool_result(result)

    def _resolve_name(self, raw: str) -> str:
        if raw in self._tools:
            return raw

        matches = [name for name in self._tools if name.endswith(f":{raw}")]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"Tool '{raw}' not found in MCP registry")
        raise KeyError(f"Tool name '{raw}' is ambiguous: {matches}")


__all__ = ["TavilySearchTool", "TavilySearchResult", "MCPToolGateway"]
