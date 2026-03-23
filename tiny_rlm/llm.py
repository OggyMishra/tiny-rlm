"""LiteLLM-based LLM client."""

import os

from dotenv import load_dotenv
import litellm

from tiny_rlm.config import Usage

load_dotenv()

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LLMClient:
    """Thin wrapper around LiteLLM completion."""

    def __init__(self, model: str = "gemini/gemini-2.5-flash"):
        self._api_base = os.getenv("LITELLM_BASE_URL")
        self._api_key = os.getenv("LITELLM_API_KEY")
        # When using a LiteLLM proxy, use "openai/" prefix so litellm SDK
        # routes through the proxy's /chat/completions endpoint.
        # The model name should match a model group on the proxy.
        if self._api_base:
            self.model = f"openai/{model}" if not model.startswith("openai/") else model
        else:
            self.model = model

    def _call_kwargs(self) -> dict:
        """Build kwargs for litellm.completion() with proxy config."""
        kw: dict = {}
        if self._api_base:
            kw["api_base"] = self._api_base
        if self._api_key:
            kw["api_key"] = self._api_key
        return kw

    def complete(self, messages: list[dict[str, str]] | str, **kwargs) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resp = litellm.completion(
            model=self.model, messages=messages, **self._call_kwargs(), **kwargs
        )
        return resp.choices[0].message.content

    def complete_with_usage(
        self, messages: list[dict[str, str]] | str, **kwargs
    ) -> tuple[str, Usage]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resp = litellm.completion(
            model=self.model, messages=messages, **self._call_kwargs(), **kwargs
        )
        content = resp.choices[0].message.content
        u = resp.usage
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
            cost=cost,
        )
        return content, usage
