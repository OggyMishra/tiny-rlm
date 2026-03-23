"""Tests for tiny_rlm.mna.models — Pydantic models and parse_extraction."""

import pytest

from tiny_rlm.mna.models import (
    EntityInfo,
    ExtractionResult,
    MnATransaction,
    PricePerShare,
    parse_extraction,
    _flatten_metadata,
    _coerce_entity_fields,
)


class TestEntityInfo:
    def test_defaults_to_none(self):
        e = EntityInfo()
        assert e.name is None
        assert e.ticker is None

    def test_with_values(self):
        e = EntityInfo(name="Acme Corp", ticker="ACME", country="US")
        assert e.name == "Acme Corp"
        assert e.ticker == "ACME"


class TestMnATransaction:
    def test_minimal_transaction(self):
        t = MnATransaction(deal_type="acquisition")
        assert t.deal_type == "acquisition"
        assert t.acquirors == []
        assert t.target is None

    def test_full_transaction(self):
        t = MnATransaction(
            deal_type="acquisition",
            deal_status="announced",
            target=EntityInfo(name="TargetCo"),
            acquirors=[EntityInfo(name="AcquirorCo")],
            consideration_size="$1 billion",
        )
        assert t.target.name == "TargetCo"
        assert len(t.acquirors) == 1
        assert t.consideration_size == "$1 billion"

    def test_model_dump_excludes_none(self):
        t = MnATransaction(deal_type="acquisition")
        data = t.model_dump(exclude_none=True)
        assert "deal_type" in data
        assert "deal_rumor_date" not in data


class TestParseExtraction:
    def test_none_input(self):
        assert parse_extraction(None) == []

    def test_empty_string(self):
        assert parse_extraction("") == []

    def test_invalid_json_string(self):
        assert parse_extraction("not json at all") == []

    def test_single_dict(self):
        raw = {"deal_type": "acquisition", "consideration_size": "$500M"}
        result = parse_extraction(raw)
        assert len(result) == 1
        assert result[0].deal_type == "acquisition"
        assert result[0].consideration_size == "$500M"

    def test_list_of_dicts(self):
        raw = [
            {"deal_type": "acquisition"},
            {"deal_type": "merger of equals"},
        ]
        result = parse_extraction(raw)
        assert len(result) == 2
        assert result[0].deal_type == "acquisition"
        assert result[1].deal_type == "merger of equals"

    def test_dict_with_transactions_key(self):
        raw = {
            "transactions": [
                {"deal_type": "acquisition", "deal_status": "completed"}
            ]
        }
        result = parse_extraction(raw)
        assert len(result) == 1
        assert result[0].deal_status == "completed"

    def test_json_string_input(self):
        import json
        raw = json.dumps({"deal_type": "acquisition"})
        result = parse_extraction(raw)
        assert len(result) == 1
        assert result[0].deal_type == "acquisition"

    def test_non_list_non_dict_returns_empty(self):
        assert parse_extraction(42) == []

    def test_with_entity_objects(self):
        raw = {
            "acquirors": [{"name": "Sunoco LP", "ticker": "SUN"}],
            "target": {"name": "Parkland", "ticker": "PKI"},
        }
        result = parse_extraction(raw)
        assert len(result) == 1
        assert result[0].acquirors[0].name == "Sunoco LP"
        assert result[0].target.name == "Parkland"


class TestFlattenMetadata:
    def test_raw_value_flattened(self):
        data = {"deal_type": {"raw_value": "acquisition", "confidence_score": 0.9}}
        flat = _flatten_metadata(data)
        assert flat["deal_type"] == "acquisition"

    def test_value_with_confidence_flattened(self):
        data = {"deal_type": {"value": "acquisition", "confidence_score": 0.9}}
        flat = _flatten_metadata(data)
        assert flat["deal_type"] == "acquisition"

    def test_value_with_citations_flattened(self):
        data = {"deal_type": {"value": "acquisition", "citations": [1, 2]}}
        flat = _flatten_metadata(data)
        assert flat["deal_type"] == "acquisition"

    def test_grouped_structure_flattened(self):
        data = {
            "dates": {
                "deal_announcement_date": "2025-05-05",
                "completion_date": "2025-12-01",
            }
        }
        flat = _flatten_metadata(data)
        assert flat["deal_announcement_date"] == "2025-05-05"
        assert flat["completion_date"] == "2025-12-01"

    def test_nested_metadata_in_groups(self):
        data = {
            "dates": {
                "deal_announcement_date": {"raw_value": "May 5", "confidence_score": 0.95},
            }
        }
        flat = _flatten_metadata(data)
        assert flat["deal_announcement_date"] == "May 5"

    def test_plain_values_preserved(self):
        data = {"deal_type": "acquisition", "deal_status": "announced"}
        flat = _flatten_metadata(data)
        assert flat["deal_type"] == "acquisition"
        assert flat["deal_status"] == "announced"

    def test_list_with_metadata_items(self):
        data = {
            "acquiror_financial_advisors": [
                {"raw_value": "Goldman Sachs"},
                {"raw_value": "Barclays"},
            ]
        }
        flat = _flatten_metadata(data)
        assert flat["acquiror_financial_advisors"] == ["Goldman Sachs", "Barclays"]

    def test_title_case_keys_normalized(self):
        data = {"Deal Type": "acquisition", "Deal Status": "announced"}
        flat = _flatten_metadata(data)
        assert flat["deal_type"] == "acquisition"
        assert flat["deal_status"] == "announced"


class TestCoerceEntityFields:
    def test_single_dict_to_list(self):
        flat = {"acquirors": {"name": "Acme"}}
        _coerce_entity_fields(flat)
        assert flat["acquirors"] == [{"name": "Acme"}]

    def test_string_items_to_entity(self):
        flat = {"acquirors": ["Acme Corp", "Beta Inc"]}
        _coerce_entity_fields(flat)
        assert flat["acquirors"] == [{"name": "Acme Corp"}, {"name": "Beta Inc"}]

    def test_target_list_to_first(self):
        flat = {"target": [{"name": "TargetCo"}]}
        _coerce_entity_fields(flat)
        assert flat["target"] == {"name": "TargetCo"}

    def test_target_string_to_entity(self):
        flat = {"target": "TargetCo"}
        _coerce_entity_fields(flat)
        assert flat["target"] == {"name": "TargetCo"}

    def test_target_empty_list(self):
        flat = {"target": []}
        _coerce_entity_fields(flat)
        assert flat["target"] is None

    def test_target_string_list_to_first(self):
        flat = {"target": ["TargetCo"]}
        _coerce_entity_fields(flat)
        assert flat["target"] == {"name": "TargetCo"}

    def test_dict_items_preserved(self):
        flat = {"acquirors": [{"name": "A", "ticker": "AA"}]}
        _coerce_entity_fields(flat)
        assert flat["acquirors"] == [{"name": "A", "ticker": "AA"}]

    def test_sellers_coerced(self):
        flat = {"sellers": "SellerCo"}
        # sellers is a string, not list — shouldn't crash
        _coerce_entity_fields(flat)
        # String not in expected format, left as-is (not a list or dict)

    def test_missing_fields_ignored(self):
        flat = {"deal_type": "acquisition"}
        _coerce_entity_fields(flat)  # Should not raise
        assert flat["deal_type"] == "acquisition"


class TestExtractionResult:
    def test_defaults(self):
        r = ExtractionResult()
        assert r.transactions == []
        assert r.raw_result is None
        assert r.usage is None

    def test_with_data(self):
        txn = MnATransaction(deal_type="acquisition")
        r = ExtractionResult(
            transactions=[txn],
            usage={"prompt_tokens": 100, "cost": 0.01},
        )
        assert len(r.transactions) == 1
        assert r.usage["cost"] == 0.01
