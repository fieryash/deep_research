"""CLI entrypoint for ad-hoc runs."""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from .config import AppConfig
from .graph import DeepResearchPipeline


async def _async_main(args: argparse.Namespace) -> None:
    config = AppConfig()
    pipeline = DeepResearchPipeline(config)
    try:
        result = await pipeline.run(args.query, scope=args.scope)
    finally:
        await pipeline.shutdown()

    print(f"Run ID: {result.run_id}")
    print(f"Log path: {result.log_path}")
    print("\nFinal Report:\n")
    print(result.state.get("draft_report") or "<missing>")
    print("\nReview:\n")
    print(json.dumps(result.state.get("review"), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deep Research Studio CLI")
    parser.add_argument("query", help="Research question to investigate")
    parser.add_argument("--scope", help="Optional scope preset", default=None)
    args = parser.parse_args(argv)
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
