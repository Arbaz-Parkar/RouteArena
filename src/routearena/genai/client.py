"""Small OpenAI-compatible chat client used only for optional explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from routearena.config import Settings, settings

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_TOKENS = 450


class LLMError(RuntimeError):
    """A safe, user-facing error from optional language-model setup or requests."""


@dataclass
class LLMClient:
    """Call an OpenAI-compatible chat endpoint without exposing provider details."""

    model: str
    _client: Any

    @classmethod
    def from_settings(cls, config: Settings = settings) -> LLMClient:
        """Create the SDK client from environment-backed settings, or explain what is missing."""
        if not config.llm_model or not config.llm_api_key:
            raise LLMError("Set LLM_MODEL and LLM_API_KEY in .env to enable model answers.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "Install the project dependencies to add the optional OpenAI SDK."
            ) from exc

        options: dict[str, Any] = {
            "api_key": config.llm_api_key,
            "timeout": DEFAULT_TIMEOUT_SECONDS,
            "max_retries": DEFAULT_MAX_RETRIES,
        }
        if config.llm_base_url:
            options["base_url"] = config.llm_base_url
        return cls(model=config.llm_model, _client=OpenAI(**options))

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        """Return one plain-text completion, converting SDK failures to a safe message."""
        try:
            request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": DEFAULT_MAX_TOKENS,
            }
            if json_mode:
                request["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**request)
        except Exception as exc:  # Provider SDKs expose several transport-specific errors.
            raise LLMError(
                "The configured model could not answer. Check the provider URL, model name, "
                "API key, and that the provider is running."
            ) from exc

        content = response.choices[0].message.content if response.choices else None
        if not isinstance(content, str) or not content.strip():
            raise LLMError("The configured model returned an empty answer.")
        return content.strip()


def model_is_configured(config: Settings = settings) -> bool:
    """Return whether the minimum settings for a chat request are present."""
    return bool(config.llm_model and config.llm_api_key)
