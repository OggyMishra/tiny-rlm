"""Pydantic output models for M&A transaction extraction."""

from typing import Any

from pydantic import BaseModel, Field


class EntityInfo(BaseModel):
    name: str | None = None
    legal_name: str | None = None
    entity_type: str | None = None
    ticker: str | None = None
    country: str | None = None
    state_province: str | None = None


class PricePerShare(BaseModel):
    price: str | None = None
    source: str | None = None


class MnATransaction(BaseModel):
    """A single M&A transaction extracted from a document."""

    # Dates
    undisturbed_target_share_price_date: str | None = None
    deal_rumor_date: str | None = None
    bid_date: str | None = None
    letter_of_intent_date: str | None = None
    agreement_signing_date: str | None = None
    deal_announcement_date: str | None = None
    expected_completion_timeframe: str | None = None
    revision_date: str | None = None
    cancellation_date: str | None = None
    completion_date: str | None = None

    # Deal features
    deal_type: str | None = None
    deal_status: str | None = None
    deal_attitude: str | None = None
    deal_summary: str | None = None
    deal_rationale: str | None = None

    # Participants
    acquirors: list[EntityInfo] = Field(default_factory=list)
    acquiror_ultimate_parents: list[EntityInfo] = Field(default_factory=list)
    target: EntityInfo | None = None
    sellers: list[EntityInfo] = Field(default_factory=list)
    seller_ultimate_parents: list[EntityInfo] = Field(default_factory=list)

    # Consideration
    target_shares_outstanding_pre_announcement: str | None = None
    target_shares_outstanding_pre_announcement_type: str | None = None
    pct_shares_owned_by_acquiror_pre_announcement: str | None = None
    pct_shares_acquired: str | None = None
    pct_shares_owned_post_deal_completion: str | None = None
    announced_purchase_price_per_share: list[PricePerShare] = Field(default_factory=list)
    number_of_target_shares_sought: str | None = None
    number_of_acquiror_issued_shares: str | None = None
    consideration_size: str | None = None
    consideration_size_min: str | None = None
    consideration_size_max: str | None = None
    contingent_payment_amount: str | None = None
    contingent_payment_amount_min: str | None = None
    contingent_payment_amount_max: str | None = None
    contingent_payment_timeframe: str | None = None
    adjusted_consideration_size: str | None = None
    stock_consideration_amount: str | None = None
    stock_consideration_min: str | None = None
    stock_consideration_max: str | None = None
    stock_consideration_pct: str | None = None
    cash_consideration_amount: str | None = None
    cash_consideration_min: str | None = None
    cash_consideration_max: str | None = None
    cash_consideration_pct: str | None = None
    target_debt_assumed: str | None = None
    current_portion_of_long_term_debt_assumed: str | None = None
    short_term_debt_assumed: str | None = None
    long_term_debt_assumed: str | None = None
    capital_leases_assumed: str | None = None
    less_cash_assumed: str | None = None
    target_net_debt_assumed: str | None = None
    preferred_equity_assumed: str | None = None
    minority_interest_assumed: str | None = None
    other_adjustments_to_enterprise_value: str | None = None
    deal_value: str | None = None
    total_enterprise_value_revenue: str | None = None
    total_enterprise_value_ebitda: str | None = None
    total_enterprise_value_ebit: str | None = None
    total_enterprise_value_ffo: str | None = None
    price_to_book_value: str | None = None
    price_to_tangible_book_value: str | None = None
    price_to_earnings: str | None = None

    # Funding
    source_of_funding_summary: str | None = None
    cash_on_hand_component_of_deal_funding: str | None = None
    bridge_loan_amount: str | None = None
    debt_funding_term_loan_amount: str | None = None
    debt_funding_senior_notes_amount: str | None = None
    debt_funding_subordinated_notes_amount: str | None = None
    debt_funding_convertible_notes_amount: str | None = None
    equity_funding_private_placement_amount: str | None = None
    equity_funding_public_placement_amount: str | None = None

    # Advisors
    acquiror_financial_advisors: list[str] = Field(default_factory=list)
    seller_financial_advisors: list[str] = Field(default_factory=list)
    acquiror_legal_advisors: list[str] = Field(default_factory=list)
    seller_legal_advisors: list[str] = Field(default_factory=list)
    acquiror_auditors: list[str] = Field(default_factory=list)
    seller_auditors: list[str] = Field(default_factory=list)

    # Termination
    termination_fee_paid_by_acquiror: str | None = None
    termination_fee_paid_by_target_seller: str | None = None


class ExtractionResult(BaseModel):
    """Result of M&A extraction from a single document."""

    transactions: list[MnATransaction] = Field(default_factory=list)
    raw_result: Any = None
    usage: dict | None = None


def parse_extraction(raw: Any) -> list[MnATransaction]:
    """Parse raw LLM extraction output into validated MnATransaction list.

    Handles various formats the LLM might return:
    - list of dicts
    - single dict (wraps in list)
    - dict with a 'transactions' key
    """
    if raw is None:
        return []

    # If string, try to parse as JSON
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []

    # Unwrap if dict with 'transactions' key
    if isinstance(raw, dict):
        if "transactions" in raw:
            raw = raw["transactions"]
        else:
            raw = [raw]

    if not isinstance(raw, list):
        return []

    transactions = []
    for item in raw:
        if isinstance(item, dict):
            flat = _flatten_metadata(item)
            _coerce_entity_fields(flat)
            _coerce_numeric_to_str(flat)
            try:
                transactions.append(MnATransaction.model_validate(flat))
            except Exception as e:
                import sys
                print(f"[parse_extraction] Validation failed: {e}", file=sys.stderr)
                print(f"[parse_extraction] Keys: {list(flat.keys())}", file=sys.stderr)
                continue

    return transactions


def _coerce_entity_fields(flat: dict) -> None:
    """Fix common entity field format issues in-place.

    Handles:
    - Single dict where list expected: {acquirors: {...}} -> [{...}]
    - Strings in entity lists: ["Sunoco LP"] -> [{"name": "Sunoco LP"}]
    - target as list: ["Parkland"] -> {"name": "Parkland"}
    """
    entity_list_fields = ("acquirors", "acquiror_ultimate_parents",
                          "sellers", "seller_ultimate_parents")

    for field in entity_list_fields:
        if field not in flat:
            continue
        val = flat[field]
        if isinstance(val, dict):
            flat[field] = [val]
        elif isinstance(val, list):
            coerced = []
            for item in val:
                if isinstance(item, str):
                    coerced.append({"name": item})
                elif isinstance(item, dict):
                    coerced.append(item)
                else:
                    coerced.append(item)
            flat[field] = coerced

    # target: list -> first item, string -> EntityInfo
    if "target" in flat:
        val = flat["target"]
        if isinstance(val, list):
            if val:
                first = val[0]
                flat["target"] = {"name": first} if isinstance(first, str) else first
            else:
                flat["target"] = None
        elif isinstance(val, str):
            flat["target"] = {"name": val}


# Fields that are str | None in MnATransaction but LLMs often return as int/float.
_STR_FIELDS = {
    name
    for name, field_info in MnATransaction.model_fields.items()
    if field_info.annotation in (str | None,)  # noqa: E721
    or str(field_info.annotation) == "str | None"
}


def _coerce_numeric_to_str(flat: dict) -> None:
    """Convert int/float values to str for fields typed as str | None."""
    for key, val in flat.items():
        if key in _STR_FIELDS and isinstance(val, (int, float)):
            flat[key] = str(val)


_MODEL_FIELDS = set(MnATransaction.model_fields.keys())

# Groups the LLM may nest fields under (lowercase normalized).
# These are NOT model fields — they are category containers.
_GROUP_KEYS = {
    "dates", "deal_features", "features", "participants", "consideration",
    "funding", "advisors", "termination", "deal_dates", "deal_consideration",
    "sources_of_funding", "deal_participants",
}


def _normalize_key(key: str) -> str:
    """Normalize a key: 'Title Case Key' or 'Title_Case' -> snake_case."""
    return key.lower().strip().replace(" ", "_")


def _is_group_key(key: str, nkey: str, val: dict) -> bool:
    """Check if a key represents a group container, not a model field."""
    if nkey in _GROUP_KEYS:
        return True
    # If the key is NOT a model field, but its children ARE model fields → it's a group
    if nkey not in _MODEL_FIELDS:
        child_keys = {_normalize_key(k) for k in val}
        if child_keys & _MODEL_FIELDS:
            return True
    return False


def _flatten_metadata(data: dict) -> dict:
    """Flatten grouped + metadata-wrapped LLM output to flat field dict.

    Handles:
    - Grouped: {"dates": {"field": ...}, "Consideration": {"field": ...}}
    - Metadata: {"field": {"raw_value": x, "confidence_score": 0.9, "citations": [...]}}
    - Title-case keys: {"Target": {...}, "Acquirors": [...]}
    - Combined: {"Dates": {"field": {"raw_value": x}}}
    """
    result = {}
    for key, val in data.items():
        nkey = _normalize_key(key)

        if isinstance(val, dict):
            if "raw_value" in val:
                result[nkey] = val["raw_value"]
            elif "value" in val and ("confidence" in str(val.keys()).lower()
                                     or "citations" in val or "confidence_score" in val):
                result[nkey] = val["value"]
            elif _is_group_key(key, nkey, val):
                result.update(_flatten_metadata(val))
            elif nkey in ("target",):
                result[nkey] = _flatten_metadata(val)
            else:
                result[nkey] = val
        elif isinstance(val, list):
            flattened_list = []
            for item in val:
                if isinstance(item, dict):
                    if "raw_value" in item and len(item) <= 3:
                        flattened_list.append(item["raw_value"])
                    elif "value" in item and len(item) <= 3:
                        flattened_list.append(item["value"])
                    else:
                        flattened_list.append(_flatten_metadata(item))
                else:
                    flattened_list.append(item)
            result[nkey] = flattened_list
        else:
            result[nkey] = val
    return result
