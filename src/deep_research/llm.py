"""Utility helpers for initializing chat models based on config."""
from __future__ import annotations

from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import AppConfig


def _build_openai(model_id: str, api_key: str | None) -> BaseChatModel:
    if "/" in model_id:
        _, name = model_id.split("/", maxsplit=1)
    else:
        name = model_id
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing but required for OpenAI models")
    return ChatOpenAI(model=name, api_key=api_key, temperature=0.2)


def init_models(config: AppConfig) -> Dict[str, BaseChatModel]:
    """Instantiate chat models for each agent role."""

    models = config.models
    providers = {
        "summarizer": models.summarizer_model,
        "researcher": models.researcher_model,
        "synthesizer": models.synthesizer_model,
        "reviewer": models.reviewer_model,
    }

    registry: Dict[str, BaseChatModel] = {}
    for role, model_id in providers.items():
        if model_id.startswith("openai"):
            registry[role] = _build_openai(model_id, config.openai_api_key)
        else:
            raise ValueError(f"Unsupported model id '{model_id}' for role {role}")
    return registry


__all__ = ["init_models"]
