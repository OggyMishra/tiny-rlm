"""Load and parse the M&A extraction schema from mna.yaml."""

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_SCHEMA_PATH = Path(__file__).parent.parent.parent / "schema" / "mna.yaml"


def load_schema(path: str | Path | None = None) -> dict:
    """Load the full schema dict from mna.yaml."""
    p = Path(path) if path else _DEFAULT_SCHEMA_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def get_system_prompt(schema: dict | None = None) -> str:
    """Extract the expert system prompt from the schema."""
    schema = schema or load_schema()
    return schema["prompt"]["system_prompt"]


def get_template(schema: dict | None = None) -> str:
    """Extract the extraction template from the schema."""
    schema = schema or load_schema()
    return schema["prompt"]["template"]


def get_fields(schema: dict | None = None) -> dict[str, Any]:
    """Get the full fields dict (field_name -> field_def)."""
    schema = schema or load_schema()
    return schema["schema"]["fields"]


def get_field_names(schema: dict | None = None) -> list[str]:
    """Get flat list of top-level field names."""
    return list(get_fields(schema).keys())


def get_field_descriptions(schema: dict | None = None) -> dict[str, str]:
    """Get field_name -> description mapping."""
    fields = get_fields(schema)
    result = {}
    for name, defn in fields.items():
        result[name] = defn.get("description", "")
    return result


def get_field_groups(schema: dict | None = None) -> dict[str, list[str]]:
    """Group fields by category based on YAML comments/structure.

    Returns dict like:
        {"dates": [...], "deal_features": [...], "participants": [...],
         "consideration": [...], "funding": [...], "advisors": [...],
         "termination": [...]}
    """
    fields = get_fields(schema)
    names = list(fields.keys())

    # Field boundaries based on schema structure
    groups: dict[str, list[str]] = {
        "dates": [],
        "deal_features": [],
        "participants": [],
        "consideration": [],
        "funding": [],
        "advisors": [],
        "termination": [],
    }

    date_fields = {
        "undisturbed_target_share_price_date", "deal_rumor_date", "bid_date",
        "letter_of_intent_date", "agreement_signing_date", "deal_announcement_date",
        "expected_completion_timeframe", "revision_date", "cancellation_date",
        "completion_date",
    }
    feature_fields = {
        "deal_type", "deal_status", "deal_attitude", "deal_summary", "deal_rationale",
    }
    participant_fields = {
        "acquirors", "acquiror_ultimate_parents", "target", "sellers",
        "seller_ultimate_parents",
    }
    funding_fields = {
        "source_of_funding_summary", "cash_on_hand_component_of_deal_funding",
        "bridge_loan_amount", "debt_funding_term_loan_amount",
        "debt_funding_senior_notes_amount", "debt_funding_subordinated_notes_amount",
        "debt_funding_convertible_notes_amount",
        "equity_funding_private_placement_amount",
        "equity_funding_public_placement_amount",
    }
    advisor_fields = {
        "acquiror_financial_advisors", "seller_financial_advisors",
        "acquiror_legal_advisors", "seller_legal_advisors",
        "acquiror_auditors", "seller_auditors",
    }
    termination_fields = {
        "termination_fee_paid_by_acquiror", "termination_fee_paid_by_target_seller",
    }

    for name in names:
        if name in date_fields:
            groups["dates"].append(name)
        elif name in feature_fields:
            groups["deal_features"].append(name)
        elif name in participant_fields:
            groups["participants"].append(name)
        elif name in funding_fields:
            groups["funding"].append(name)
        elif name in advisor_fields:
            groups["advisors"].append(name)
        elif name in termination_fields:
            groups["termination"].append(name)
        else:
            groups["consideration"].append(name)

    return groups
