"""Streamlit UI for the Deep Research Studio."""
from __future__ import annotations

import asyncio
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
    config = AppConfig()
    return DeepResearchPipeline(config)


def run_pipeline(pipeline: DeepResearchPipeline, question: str, scope: Optional[str]) -> ResearchRunResult:
    return asyncio.run(pipeline.run(question, scope=scope))


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
        st.experimental_rerun()

pipeline = load_pipeline()

with st.form("research-form"):
    question = st.text_area("What should we investigate?", height=120)
    scope = st.text_input("Optional scope or constraints")
    submitted = st.form_submit_button("Run Deep Research")

if submitted:
    if not question.strip():
        st.error("Please enter a research question.")
    else:
        with st.spinner("Working through LangGraph pipeline..."):
            result = run_pipeline(pipeline, question.strip(), scope.strip() or None)
        render_result(result)
