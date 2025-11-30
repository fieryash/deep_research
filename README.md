# Deep Research Studio

_High-signal insights from a five-agent LangGraph, the Model Context Protocol, and a Streamlit cockpit._

Deep Research Studio is a batteries-included reference app for running multi-step research investigations. It blends LangGraph's deterministic control flow, MCP tool routing, and pragmatic UI touches so you can answer "what's really going on?" questions in minutes instead of hours.

## Why you'll like it

- **Purpose-built crew** - Scoper -> Planner -> Researcher -> Synthesizer -> Reviewer agents keep context tight and reports tight.
- **Bring your own tools** - Drop in Tavily web search, MCP filesystem/RAG servers, or your own MCP endpoints without touching the graph logic.
- **Human-friendly UX** - Choose between a fast CLI (`python -m deep_research ...`) or the Streamlit "mission control" UI for live status updates.
- **Always-on logging** - Every run is serialized to `data/logs/<run_id>.json`, so you can diff findings between iterations or feed them to downstream analytics.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
pip install -e .
cp .env.example .env             # then add your API keys + MCP servers
```

Smoke-test the CLI:

```bash
python -m deep_research "How is LangGraph different from LangChain Expression Language?"
```

Prefer a GUI? Launch Streamlit:

```bash
streamlit run streamlit_app.py
```

The Streamlit app streams node completions ("scope", "plan", "research", "synth", "review") and renders the final report, latest findings, reviewer JSON, and a link to the saved log.

---

## Configuration guide

All config flows through `deep_research.config.AppConfig`, which reads `.env` (via `pydantic-settings`). Key knobs:

| Variable | Why it matters |
| --- | --- |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` | Required if any of the agent roles point to those providers. Default models use Gemini (`google/gemini-2.0-flash-exp`). |
| `TAVILY_API_KEY` | Enables the async Tavily search tool used by the Researcher node. |
| `MCP_SERVERS` | JSON describing MCP transports (`stdio`, `sse`, or `websocket`). Example:<br>`[{"name":"fs","transport":"stdio","command":"mcp-filesystem","args":["--root","./notes"]}]` |
| `MAX_RESEARCH_LOOPS` | Reviewer-driven safety rail (default `2`). Controls how many times we loop back for revisions. |
| `PERSISTENCE_DIR` / `CACHE_DIR` | Override locations for JSON logs and scratch artifacts (defaults: `data/logs` and `.cache`). |

Changing models: defaults point to Gemini (`google/gemini-2.0-flash-exp`). To switch to OpenAI, set env vars like:

```bash
MODELS__SUMMARIZER_MODEL=openai/gpt-4o-mini
MODELS__RESEARCHER_MODEL=openai/gpt-4o-mini
MODELS__SYNTHESIZER_MODEL=openai/gpt-4o-mini
MODELS__REVIEWER_MODEL=openai/gpt-4o-mini
```

Need to verify Gemini access or enumerate models? Use the helper scripts:

- `python list_models.py` - calls `google.generativeai` to list models exposing `generateContent`.
- `python verify_gemini.py` - sanity checks `AppConfig`, confirms Gemini is selected, and (optionally) instantiates the LangChain wrappers.

---

## Usage

### CLI (fastest loop)

```bash
python -m deep_research "Will MCP replace REST APIs?" --scope "Focus on developer tooling"
```

You'll get:

- Run + log identifiers for audit trails.
- The synthesized report.
- Reviewer feedback JSON (approved flag, critique, next recommended action).

### Streamlit mission control

1. `streamlit run streamlit_app.py`
2. Enter a research question + optional scope in the sidebar form.
3. Watch live progress ticks as each node completes.
4. Download or copy the report, glance at the six most recent findings, and inspect reviewer feedback without leaving the page.

The UI caches the pipeline via `st.cache_resource`, so reloading settings is a single button press.

---

## Architecture at a glance

```
Human prompt
   |
   |-> Scoper      - reframes the ask, highlights gaps, drops markdown summaries.
   |-> Planner     - produces an ordered to-do list with tool hints.
   |-> Researcher  - executes Tavily + MCP tool calls, writes evidence snippets.
   |-> Synthesizer - turns findings into an exec-ready report.
   \-> Reviewer    - critiques, toggles `needs_revision`, loops or ends.
```

- **LangGraph core** (`src/deep_research/graph.py`) wires these nodes with `StateGraph`, uses `HumanMessage` threads, and exposes both `run` and `run_stream` for CLI/UI parity.
- **State contract** (`state.py`) keeps everything typed (`TypedDict` + dataclasses) so it's obvious what each node may read/write.
- **Prompts** live in `prompts.py`, making it painless to iterate on agent instructions without editing node logic.
- **Model bootstrap** (`llm.py`) lets each role specify its own model ID (OpenAI or Gemini out of the box) via `ModelConfig`.
- **Persistence** lives in `persistence.py`, converting dataclasses, messages, and timestamps into JSON for `ResearchLogger`.

---

## Integrations & customization

- **Web search** - `TavilySearchTool` asynchronously batches queries built from the plan, adds confidence scores, and feeds structured snippets to the researcher agent. Swap the provider by extending `SearchConfig`.
- **MCP tooling** - `MCPToolGateway` opens stdio/SSE/WebSocket sessions, auto-lists tools, and exposes `gateway.invoke(<tool>, {...})`. Drop in your filesystem, RAG, or custom automations without changing the LangGraph wiring.
- **Logging/observability** - Every run saves the complete `ResearchState` to JSON (including findings, critiques, and loop counts). Point `PERSISTENCE_DIR` somewhere durable to keep an audit trail.
- **Agent tuning** - Adjust prompt tone, models, or temperature per role inside `prompts.py` and `llm.py`. The graph itself stays untouched.

---

## Project layout

- `src/deep_research/` - all LangGraph nodes, prompts, state definitions, and utility layers.
- `streamlit_app.py` - Streamlit UI with async progress streaming and cached pipeline.
- `list_models.py` / `verify_gemini.py` - standalone scripts for Gemini model discovery and smoke testing.
- `data/logs/` - persisted run histories (auto-created).
- `.cache/` - stash for any future tool artifacts.

---

## Local development

```bash
pip install -e ".[dev]"      # pulls pytest + ruff
ruff check src               # lint
pytest                       # when you add tests
```

Tips:

- Use `.env.example` as your baseline, then layer secrets via direnv or your shell of choice.
- Running `python -m deep_research ...` inside VS Code's debugger provides a nice REPL-style loop.
- Want more aggressive revision loops? Bump `MAX_RESEARCH_LOOPS` or teach the reviewer agent to be stricter.

---

## Troubleshooting

- **"No MCP tools available"** - ensure `mcp` is installed and your `MCP_SERVERS` entries include either `command` (stdio) or a valid `sse_url`/`websocket_url`.
- **Tavily errors** - double-check `TAVILY_API_KEY`; the researcher node gracefully degrades but will note the failure in findings.
- **Model init exceptions** - make sure each model family's API key is set before switching the corresponding `ModelConfig` field.
- **Streamlit not updating** - click "Reload settings" in the sidebar to bust the cached pipeline after changing `.env`.

Now go run a question that deserves a thoughtful, multi-source answer. 
