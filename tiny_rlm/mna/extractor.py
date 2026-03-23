"""M&A extraction orchestrator — wraps RLMEngine with M&A-specific prompts."""

import asyncio
from typing import Any

from tiny_rlm.config import RLMConfig, load_config
from tiny_rlm.engine import RLMEngine
from tiny_rlm.mna.models import ExtractionResult, MnATransaction, parse_extraction
from tiny_rlm.mna.prompts import build_mna_root_prompt
from tiny_rlm.mna.schema import load_schema


class MnAExtractor:
    """Extract M&A transactions from document text using RLM."""

    def __init__(
        self,
        config: RLMConfig | None = None,
        config_path: str = "rlm_config.yaml",
        schema_path: str | None = None,
        verbose: bool = True,
    ):
        self.config = config or load_config(config_path)
        self.schema = load_schema(schema_path)
        self.engine = RLMEngine(self.config, verbose=verbose)
        self._system_prompt = build_mna_root_prompt(self.schema)

    async def extract_async(self, document_text: str) -> ExtractionResult:
        """Extract M&A transactions from document text (async)."""
        raw_result = await self.engine.run(
            context=document_text,
            query="Extract all M&A transactions from this document. "
            "Return a list of transaction dicts with all available fields.",
            system_prompt_override=self._system_prompt,
        )

        transactions = parse_extraction(raw_result)
        usage = self.engine.tracker.summary()

        return ExtractionResult(
            transactions=transactions,
            raw_result=raw_result,
            usage=usage,
        )

    def extract(self, document_text: str) -> ExtractionResult:
        """Extract M&A transactions from document text (sync wrapper)."""
        return asyncio.run(self.extract_async(document_text))
