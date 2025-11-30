"""Persistence helpers for logging research runs."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from .state import ResearchState


def _to_serializable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, BaseMessage):
        # Flatten LangChain messages so logs remain JSON serializable
        return {
            "type": value.type,
            "content": value.content,
            "name": getattr(value, "name", None),
            "additional_kwargs": getattr(value, "additional_kwargs", {}),
        }
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


class ResearchLogger:
    """Writes each run into a JSONL file for downstream analysis."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def log(self, run_id: str, state: ResearchState) -> Path:
        payload = _to_serializable(state)
        payload["run_id"] = run_id
        payload["timestamp"] = datetime.utcnow().isoformat()
        out_file = self.root / f"{run_id}.json"
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_file


__all__ = ["ResearchLogger"]
