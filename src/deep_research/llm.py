"""Utility helpers for initializing chat models based on config."""
from __future__ import annotations

from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .config import AppConfig


def _build_openai(model_id: str, api_key: str | None) -> BaseChatModel:
    if "/" in model_id:
        _, name = model_id.split("/", maxsplit=1)
    else:
        name = model_id
    api_key = api_key.strip() if api_key else None
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing but required for OpenAI models")
    return ChatOpenAI(model=name, api_key=api_key, temperature=0.2)


def _build_google(model_id: str, api_key: str | None) -> BaseChatModel:
    if "/" in model_id:
        _, name = model_id.split("/", maxsplit=1)
    else:
        name = model_id
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing but required for Google models")
    return ChatGoogleGenerativeAI(
        model=name, 
        google_api_key=api_key, 
        temperature=0.2,
        # Rely on our own tenacity wrapper instead of the client's internal retry loop
        max_retries=0,
        timeout=60
    )


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
        elif model_id.startswith("google"):
            registry[role] = _build_google(model_id, config.google_api_key)
        else:
            raise ValueError(f"Unsupported model id '{model_id}' for role {role}")
    return registry


__all__ = ["init_models"]
