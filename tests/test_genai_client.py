"""Offline tests for the optional language-model wrapper and grounding check."""

from types import SimpleNamespace

import pytest

from routearena.config import Settings
from routearena.genai.client import LLMClient, LLMError, model_is_configured
from routearena.genai.grounding import check_numeric_grounding


class _FakeCompletions:
    def __init__(self, content: str = "A grounded answer.") -> None:
        self.content = content
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def test_model_configuration_is_optional() -> None:
    assert not model_is_configured(Settings())
    assert model_is_configured(Settings(llm_model="local-model", llm_api_key="ollama"))


def test_chat_wrapper_uses_low_temperature_and_token_limit() -> None:
    completions = _FakeCompletions()
    client = LLMClient(
        "local-model",
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    assert client.complete("system", "question") == "A grounded answer."
    assert completions.request["temperature"] == 0.2
    assert completions.request["max_tokens"] > 0
    assert len(completions.request["messages"]) == 2
    client.complete("system", "return JSON", json_mode=True)
    assert completions.request["response_format"] == {"type": "json_object"}


def test_chat_wrapper_turns_provider_failure_into_safe_error() -> None:
    class BrokenCompletions:
        def create(self, **_request):
            raise RuntimeError("private provider detail")

    client = LLMClient(
        "local-model",
        SimpleNamespace(chat=SimpleNamespace(completions=BrokenCompletions())),
    )
    with pytest.raises(LLMError, match="configured model could not answer") as error:
        client.complete("system", "question")
    assert "private provider detail" not in str(error.value)


def test_grounding_allows_rounding_and_rejects_invented_values() -> None:
    assert check_numeric_grounding("The route is 12.3 minutes.", "Computed route: 12.34 min.").grounded
    result = check_numeric_grounding("The route is 12.3 minutes and has 6 stops.", "Computed route: 12.34 min.")
    assert not result.grounded
    assert result.unsupported_numbers == ("6",)
