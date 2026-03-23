"""Tests for tiny_rlm.prompts and tiny_rlm.mna.prompts."""

import pytest

from tiny_rlm.prompts import LEAF_SYSTEM_PROMPT, ROOT_SYSTEM_PROMPT, STEP0_INSPECT_CODE
from tiny_rlm.mna.prompts import build_field_extraction_prompt, build_mna_root_prompt
from tiny_rlm.mna.schema import load_schema


class TestBasePrompts:
    def test_root_prompt_has_llm_query(self):
        assert "llm_query" in ROOT_SYSTEM_PROMPT

    def test_root_prompt_has_final(self):
        assert "FINAL" in ROOT_SYSTEM_PROMPT

    def test_root_prompt_has_context(self):
        assert "context" in ROOT_SYSTEM_PROMPT

    def test_root_prompt_has_asyncio_gather(self):
        assert "asyncio.gather" in ROOT_SYSTEM_PROMPT

    def test_leaf_prompt_no_llm_query(self):
        assert "do NOT have access to llm_query" in LEAF_SYSTEM_PROMPT.lower() or \
               "You do NOT have access to llm_query()" in LEAF_SYSTEM_PROMPT

    def test_leaf_prompt_has_final(self):
        assert "FINAL" in LEAF_SYSTEM_PROMPT

    def test_step0_inspect_code_prints_type(self):
        assert "type(context)" in STEP0_INSPECT_CODE

    def test_step0_inspect_code_prints_length(self):
        assert "len(context)" in STEP0_INSPECT_CODE


class TestBuildMnaRootPrompt:
    def test_contains_repl_instructions(self):
        schema = load_schema()
        prompt = build_mna_root_prompt(schema)
        assert "llm_query" in prompt
        assert "FINAL" in prompt
        assert "context" in prompt

    def test_contains_mna_rules(self):
        schema = load_schema()
        prompt = build_mna_root_prompt(schema)
        assert "M&A" in prompt or "m&a" in prompt.lower()

    def test_contains_field_groups(self):
        schema = load_schema()
        prompt = build_mna_root_prompt(schema)
        assert "DATES" in prompt or "dates" in prompt.lower()
        assert "PARTICIPANTS" in prompt or "participants" in prompt.lower()
        assert "CONSIDERATION" in prompt or "consideration" in prompt.lower()

    def test_contains_strategy(self):
        schema = load_schema()
        prompt = build_mna_root_prompt(schema)
        assert "asyncio.gather" in prompt

    def test_custom_schema(self, sample_schema):
        schema = load_schema(sample_schema)
        prompt = build_mna_root_prompt(schema)
        assert "test extractor" in prompt.lower()
        assert "FINAL" in prompt


class TestBuildFieldExtractionPrompt:
    def test_includes_group_name(self):
        prompt = build_field_extraction_prompt(
            "DATES",
            ["deal_announcement_date", "completion_date"],
            {"deal_announcement_date": "The date of announcement", "completion_date": "Close date"},
        )
        assert "DATES" in prompt

    def test_includes_field_names(self):
        prompt = build_field_extraction_prompt(
            "CONSIDERATION",
            ["consideration_size"],
            {"consideration_size": "Total deal value"},
        )
        assert "consideration_size" in prompt

    def test_includes_descriptions(self):
        prompt = build_field_extraction_prompt(
            "ADVISORS",
            ["acquiror_financial_advisors"],
            {"acquiror_financial_advisors": "Banks advising the acquiror"},
        )
        assert "Banks advising" in prompt

    def test_missing_description_handled(self):
        prompt = build_field_extraction_prompt(
            "TEST",
            ["unknown_field"],
            {},
        )
        assert "unknown_field" in prompt
        assert "No description" in prompt

    def test_contains_rules(self):
        prompt = build_field_extraction_prompt("X", ["f"], {"f": "desc"})
        assert "FINAL" in prompt
        assert "null" in prompt
