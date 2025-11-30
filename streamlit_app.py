"""Streamlit UI for the Deep Research Studio."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from deep_research.config import AppConfig
from deep_research.graph import DeepResearchPipeline, ResearchRunResult


@st.cache_resource(show_spinner=False)
def load_pipeline() -> DeepResearchPipeline:
    # On Streamlit Cloud, secrets live in .streamlit/secrets.toml. Copy any flat entries into
    # the environment so AppConfig picks them up the same way as .env.
    for key, value in getattr(st, "secrets", {}).items():
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))
    config = AppConfig()
    return DeepResearchPipeline(config)


async def run_pipeline_async(pipeline: DeepResearchPipeline, question: str, scope: Optional[str]) -> ResearchRunResult:
    container = st.status("Starting research...", expanded=True)
    result = None
    async for event in pipeline.run_stream(question, scope=scope):
        if event["type"] == "update":
            node = event["node"]
            container.write(f"Completed step: **{node}**")
            if node == "plan":
                container.markdown("Plan generated.")
            elif node == "research":
                container.markdown("Research gathered.")
            elif node == "synth":
                container.markdown("Draft report written.")
            elif node == "review":
                container.markdown("Review completed.")
        elif event["type"] == "result":
            result = event["data"]
    
    container.update(label="Research Complete!", state="complete", expanded=False)
    return result


def run_pipeline(pipeline: DeepResearchPipeline, question: str, scope: Optional[str]) -> ResearchRunResult:
    return asyncio.run(run_pipeline_async(pipeline, question, scope))


def render_result(result: ResearchRunResult) -> None:
    st.subheader("Final Report")
    st.markdown(result.state.get("draft_report") or "_No report generated._")

    st.subheader("Findings")
    for finding in result.state.get("findings", [])[-6:]:
        st.markdown(f"- **{finding.source}** — {finding.content[:400]}...")

    st.subheader("Reviewer Feedback")
    st.json(result.state.get("review") or {})
    st.caption(f"Run saved to {result.log_path}")


st.set_page_config(page_title="Deep Research Studio", page_icon="🔬", layout="wide")
st.title("🔬 Deep Research Studio")

with st.sidebar:
    st.header("Configuration")
    st.write("Values pulled from your .env file via AppConfig.")
    if st.button("Reload settings"):
        st.cache_resource.clear()
        st.rerun()

pipeline = load_pipeline()

if pipeline.search_tool is None and pipeline.config.search.provider == "tavily":
    st.warning("TAVILY_API_KEY missing; Tavily web search disabled.")

with st.form("research-form"):
    question = st.text_area("What should we investigate?", height=120)
    scope = st.text_input("Optional scope or constraints")
    submitted = st.form_submit_button("Run Deep Research")

if submitted:
    if not question.strip():
        st.error("Please enter a research question.")
    else:
        result = run_pipeline(pipeline, question.strip(), scope.strip() or None)
        render_result(result)
