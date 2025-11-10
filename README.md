# Deep Research Studio

LangGraph + MCP reference implementation for multi-agent deep research with a Streamlit UI. Inspired by LangChain's Open Deep Research, but structured for customization.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env  # fill in API keys + MCP server definitions
```

Run the CLI:

```bash
python -m deep_research "How is LangGraph different from LangChain Expression Language?"
```

Or launch the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Configuration

Set env vars in `.env`:

- `OPENAI_API_KEY` – required for default models.
- `TAVILY_API_KEY` – enable web search.
- `MCP_SERVERS` – JSON array of Model Context Protocol servers to auto-connect, e.g. `[{"name": "fs", "transport": "stdio", "command": "mcp-filesystem", "args": ["--root", "./notes"]}]`.
- `MAX_RESEARCH_LOOPS` – limit review/refine cycles (default 2).

## Project layout

- `src/deep_research/` – LangGraph nodes, state, persistence, tool adapters.
- `streamlit_app.py` – Streamlit frontend that wraps the pipeline.
- `data/logs/` – JSON logs per run.

## Next steps

- Extend `MCPToolGateway` with custom tool routing.
- Swap in different LLM providers via `AppConfig`.
- Add RAG/vector store persistence for findings in `persistence.py`.
