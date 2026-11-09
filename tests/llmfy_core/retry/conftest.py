"""Fixtures for the retry tests: a scripted model and a fake clock."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest

from llmfy.llmfy_core.llms.base_ai_model import BaseAIModel
from llmfy.llmfy_core.model_backend import ModelBackend
from llmfy.llmfy_core.responses.ai_response import AIResponse
from llmfy.llmfy_core.retry import retry_runner
from llmfy.llmfy_core.service_provider import ServiceProvider


class ScriptedModel(BaseAIModel):
    """Answers each call with the next scripted outcome.

    `script` items are an `AIResponse` (returned) or an exception (raised).
    `streams` items are one list per stream call; each element is a chunk to
    yield or an exception to raise at that point. When a script runs out the
    model answers `AIResponse(content="ok")` (or an empty stream).
    """

    def __init__(
        self,
        script: list[AIResponse | Exception] | None = None,
        streams: list[list[AIResponse | Exception]] | None = None,
    ) -> None:
        self.backend = ModelBackend.OPENAI_CHAT
        self.provider = ServiceProvider.OPENAI
        self.model_name = "scripted-model"
        self.script = list(script or [])
        self.streams = list(streams or [])
        self.calls: list[list[dict[str, Any]]] = []

    def _next(self) -> AIResponse:
        outcome = self.script.pop(0) if self.script else AIResponse(content="ok")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def _next_stream(self) -> list[AIResponse | Exception]:
        return self.streams.pop(0) if self.streams else []

    def generate(self, messages, tools=None, **kwargs) -> AIResponse:
        self.calls.append(messages)
        return self._next()

    async def agenerate(self, messages, tools=None, **kwargs) -> AIResponse:
        self.calls.append(messages)
        return self._next()

    def generate_stream(self, messages, tools=None, **kwargs) -> Generator[AIResponse, Any, None]:
        self.calls.append(messages)
        for item in self._next_stream():
            if isinstance(item, Exception):
                raise item
            yield item

    async def agenerate_stream(
        self, messages, tools=None, **kwargs
    ) -> AsyncGenerator[AIResponse, Any]:
        self.calls.append(messages)
        for item in self._next_stream():
            if isinstance(item, Exception):
                raise item
            yield item


class FakeClock:
    """Replaces the retry runner's clock and sleeps: sleeping advances time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.sleep(seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(retry_runner, "_monotonic", fake.monotonic)
    monkeypatch.setattr(retry_runner, "_sleep", fake.sleep)
    monkeypatch.setattr(retry_runner, "_async_sleep", fake.async_sleep)
    return fake
