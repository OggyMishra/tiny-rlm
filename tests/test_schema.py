"""Tests for tiny_rlm.mna.schema — schema loading and field utilities."""

import pytest

from tiny_rlm.mna.schema import (
    get_field_descriptions,
    get_field_groups,
    get_field_names,
    get_fields,
    get_system_prompt,
    get_template,
    load_schema,
)


class TestLoadSchema:
    def test_load_from_default(self):
        schema = load_schema()
        assert schema["name"] == "mna_transaction"
        assert "fields" in schema["schema"]

    def test_load_from_custom_path(self, sample_schema):
        schema = load_schema(sample_schema)
        assert schema["name"] == "mna_transaction"
        assert schema["version"] == "v1.0.0-test"

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_schema(str(tmp_path / "nope.yaml"))


class TestGetSystemPrompt:
    def test_returns_string(self):
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_extraction_rules(self):
        prompt = get_system_prompt()
        assert "extract" in prompt.lower()

    def test_from_custom_schema(self, sample_schema):
        schema = load_schema(sample_schema)
        prompt = get_system_prompt(schema)
        assert "test extractor" in prompt.lower()


class TestGetTemplate:
    def test_returns_string(self):
        tpl = get_template()
        assert isinstance(tpl, str)
        assert "{{ text }}" in tpl


class TestGetFields:
    def test_returns_dict(self):
        fields = get_fields()
        assert isinstance(fields, dict)
        assert len(fields) > 50  # 86 fields expected

    def test_known_fields_present(self):
        fields = get_fields()
        assert "deal_type" in fields
        assert "acquirors" in fields
        assert "consideration_size" in fields
        assert "target" in fields


class TestGetFieldNames:
    def test_returns_list(self):
        names = get_field_names()
        assert isinstance(names, list)
        assert "deal_type" in names

    def test_count_matches_fields(self):
        names = get_field_names()
        fields = get_fields()
        assert len(names) == len(fields)


class TestGetFieldDescriptions:
    def test_returns_dict_of_strings(self):
        descs = get_field_descriptions()
        assert isinstance(descs, dict)
        for k, v in descs.items():
            assert isinstance(v, str)

    def test_descriptions_non_empty(self):
        descs = get_field_descriptions()
        assert all(len(v) > 0 for v in descs.values())


class TestGetFieldGroups:
    def test_returns_expected_groups(self):
        groups = get_field_groups()
        expected = {"dates", "deal_features", "participants", "consideration",
                    "funding", "advisors", "termination"}
        assert set(groups.keys()) == expected

    def test_all_fields_assigned(self):
        groups = get_field_groups()
        all_grouped = []
        for fields in groups.values():
            all_grouped.extend(fields)
        all_names = get_field_names()
        assert set(all_grouped) == set(all_names)

    def test_dates_group_has_date_fields(self):
        groups = get_field_groups()
        assert "deal_announcement_date" in groups["dates"]
        assert "completion_date" in groups["dates"]

    def test_participants_group(self):
        groups = get_field_groups()
        assert "acquirors" in groups["participants"]
        assert "target" in groups["participants"]

    def test_no_empty_groups(self):
        groups = get_field_groups()
        for name, fields in groups.items():
            assert len(fields) > 0, f"Group '{name}' is empty"
