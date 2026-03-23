"""RLM configuration and usage tracking."""

import threading
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class RLMConfig:
    primary_agent: str = "as-agent-gemini-3-flash"
    sub_agent: str = "as-agent-gemini-3-flash"
    max_depth: int = 2
    max_calls_per_subagent: int = 15
    truncate_len: int = 5000
    max_money_spent: float = 5.0
    max_completion_tokens: int = 200000
    max_prompt_tokens: int = 500000


def load_config(path: str | Path = "rlm_config.yaml") -> RLMConfig:
    path = Path(path)
    if not path.exists():
        return RLMConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    valid = {k: v for k, v in data.items() if k in RLMConfig.__dataclass_fields__}
    return RLMConfig(**valid)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None


class UsageTracker:
    """Thread-safe cumulative usage tracker across all LLM calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cost = 0.0

    def track(self, usage: Usage) -> None:
        with self._lock:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            if usage.cost is not None:
                self.cost += usage.cost

    def check_budget(self, config: RLMConfig) -> None:
        with self._lock:
            if self.cost > config.max_money_spent:
                raise RuntimeError(
                    f"Budget exceeded: ${self.cost:.4f} spent, "
                    f"limit is ${config.max_money_spent}"
                )
            if self.completion_tokens > config.max_completion_tokens:
                raise RuntimeError(
                    f"Completion token budget exceeded: "
                    f"{self.completion_tokens:,} used, "
                    f"limit is {config.max_completion_tokens:,}"
                )
            if self.prompt_tokens > config.max_prompt_tokens:
                raise RuntimeError(
                    f"Prompt token budget exceeded: "
                    f"{self.prompt_tokens:,} used, "
                    f"limit is {config.max_prompt_tokens:,}"
                )

    def summary(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cost": self.cost,
            }
