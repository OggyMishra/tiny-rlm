"""Tests for tiny_rlm.config — RLMConfig, UsageTracker, load_config."""

import pytest

from tiny_rlm.config import RLMConfig, Usage, UsageTracker, load_config


class TestRLMConfig:
    def test_defaults(self):
        cfg = RLMConfig()
        assert cfg.max_depth == 2
        assert cfg.max_calls_per_subagent == 15
        assert cfg.truncate_len == 5000
        assert cfg.max_money_spent == 5.0
        assert cfg.max_completion_tokens == 200000
        assert cfg.max_prompt_tokens == 500000

    def test_custom_values(self):
        cfg = RLMConfig(max_depth=5, max_money_spent=10.0)
        assert cfg.max_depth == 5
        assert cfg.max_money_spent == 10.0

    def test_load_config_from_yaml(self, tmp_path):
        yaml_content = "primary_agent: test-model\nmax_depth: 3\nmax_money_spent: 1.5\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_content)
        cfg = load_config(str(p))
        assert cfg.primary_agent == "test-model"
        assert cfg.max_depth == 3
        assert cfg.max_money_spent == 1.5
        # Defaults preserved for unspecified fields
        assert cfg.truncate_len == 5000

    def test_load_config_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert cfg == RLMConfig()

    def test_load_config_ignores_unknown_keys(self, tmp_path):
        yaml_content = "primary_agent: m\nunknown_key: 999\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_content)
        cfg = load_config(str(p))
        assert cfg.primary_agent == "m"
        assert not hasattr(cfg, "unknown_key")


class TestUsageTracker:
    def test_initial_state(self):
        tracker = UsageTracker()
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.cost == 0.0

    def test_track_accumulates(self):
        tracker = UsageTracker()
        tracker.track(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.01))
        tracker.track(Usage(prompt_tokens=200, completion_tokens=100, total_tokens=300, cost=0.02))
        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 150
        assert tracker.total_tokens == 450
        assert abs(tracker.cost - 0.03) < 1e-9

    def test_track_with_none_cost(self):
        tracker = UsageTracker()
        tracker.track(Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=None))
        assert tracker.cost == 0.0
        assert tracker.prompt_tokens == 10

    def test_summary(self):
        tracker = UsageTracker()
        tracker.track(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.01))
        s = tracker.summary()
        assert s == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.01,
        }

    def test_check_budget_money_exceeded(self, tight_budget_config):
        tracker = UsageTracker()
        tracker.track(Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=1.0))
        with pytest.raises(RuntimeError, match="Budget exceeded"):
            tracker.check_budget(tight_budget_config)

    def test_check_budget_completion_tokens_exceeded(self, tight_budget_config):
        tracker = UsageTracker()
        tracker.track(Usage(prompt_tokens=0, completion_tokens=200, total_tokens=200, cost=0.0))
        with pytest.raises(RuntimeError, match="Completion token budget exceeded"):
            tracker.check_budget(tight_budget_config)

    def test_check_budget_prompt_tokens_exceeded(self, tight_budget_config):
        tracker = UsageTracker()
        tracker.track(Usage(prompt_tokens=200, completion_tokens=0, total_tokens=200, cost=0.0))
        with pytest.raises(RuntimeError, match="Prompt token budget exceeded"):
            tracker.check_budget(tight_budget_config)

    def test_check_budget_within_limits(self, default_config):
        tracker = UsageTracker()
        tracker.track(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.01))
        tracker.check_budget(default_config)  # Should not raise
