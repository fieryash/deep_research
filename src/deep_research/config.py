"""Application configuration and settings helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """LLM model configuration to keep roles separate."""

    summarizer_model: str = Field(
        default="google/gemini-2.0-flash-exp", description="Model used for summarizing snippets."
    )
    researcher_model: str = Field(
        default="google/gemini-2.0-flash-exp", description="Model that runs the research agent."
    )
    synthesizer_model: str = Field(
        default="google/gemini-2.0-flash-exp", description="Model that writes the final report."
    )
    reviewer_model: str = Field(
        default="google/gemini-2.0-flash-exp", description="Model that critiques the report."
    )


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server we want to talk to."""

    name: str
    transport: Literal["stdio", "sse", "websocket"] = "stdio"
    command: Optional[str] = Field(default=None, description="Command to launch the server.")
    args: List[str] = Field(default_factory=list)
    sse_url: Optional[AnyHttpUrl] = Field(default=None)
    websocket_url: Optional[AnyHttpUrl] = Field(default=None)
    env: Dict[str, str] = Field(default_factory=dict)


class SearchConfig(BaseModel):
    provider: Literal["tavily", "mcp", "duckduckgo"] = "tavily"
    tavily_api_key: Optional[str] = None


class AppConfig(BaseSettings):
    """Top-level settings for the Deep Research Studio app."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["dev", "prod"] = Field(default="dev")
    models: ModelConfig = Field(default_factory=ModelConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)
    persistence_dir: Path = Field(default=Path("data/logs"))
    cache_dir: Path = Field(default=Path(".cache"))
    max_research_loops: int = Field(default=2, ge=0, le=6)

    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)
    google_cx: Optional[str] = Field(default=None)

    def ensure_dirs(self) -> None:
        """Create cache/persistence paths if missing."""

        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


__all__ = ["AppConfig", "ModelConfig", "MCPServerConfig", "SearchConfig"]
