"""Shared fixtures for tiny-rlm tests."""

import pytest

from tiny_rlm.config import RLMConfig


@pytest.fixture
def default_config():
    return RLMConfig()


@pytest.fixture
def tight_budget_config():
    return RLMConfig(
        max_money_spent=0.001,
        max_completion_tokens=100,
        max_prompt_tokens=100,
    )


@pytest.fixture
def sample_schema(tmp_path):
    """Write a minimal mna.yaml and return its path."""
    schema_content = """\
name: "mna_transaction"
version: "v1.0.0-test"
description: "Test schema"

llm:
  service_model: "test-model"
  seed: 42

prompt:
  system_prompt: |
    You are a test extractor.
  template: |
    Extract from: {{ text }}

schema:
  field_metadata:
    enable_citations: true
    enable_confidence_scoring: true

  fields:
    deal_announcement_date:
      type: DateField
      description: "The announcement date."
    deal_type:
      type: enum
      enum: [acquisition, merger of equals]
      description: "The deal type."
    acquirors:
      type: array
      items:
        type: object
        properties:
          name:
            type: EntityField
            description: "Acquiror name."
      description: "List of acquirors."
    target:
      type: object
      properties:
        name:
          type: EntityField
          description: "Target name."
      description: "Target entity."
    consideration_size:
      type: NumericField
      description: "Total deal value."
"""
    p = tmp_path / "test_mna.yaml"
    p.write_text(schema_content)
    return str(p)
