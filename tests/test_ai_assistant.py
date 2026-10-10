from types import SimpleNamespace

import pytest

from services import ai_assistant, analytics
from utils.formatting import format_inr


@pytest.fixture
def ctx(dataset):
    return analytics.build_ai_context(dataset["txns"], dataset["daily"], dataset["profile"])


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_status_without_key(no_key):
    ok, reason = ai_assistant.ai_status()
    assert not ok and "ANTHROPIC_API_KEY" in reason


def test_demo_mode_quotes_verified_numbers(no_key, ctx):
    ans = ai_assistant.answer("Why did my sales decline this week?", ctx)
    wk = ctx["last_7_vs_previous_7"]
    assert "Demo mode" in ans
    assert format_inr(wk["sales"]) in ans and format_inr(wk["prev_sales"]) in ans
    assert "What the data shows" in ans and "Possible reasons" in ans


@pytest.mark.parametrize("q", ["Which days generate the most sales?",
                               "Are my transaction counts or average transaction values declining?",
                               "What should I investigate to improve sales?",
                               "Why are payments failing?", "hello"])
def test_demo_mode_answers_every_question(no_key, ctx, q):
    assert len(ai_assistant.answer(q, ctx)) > 50


def test_sensitive_question_is_blocked_before_any_api_call(monkeypatch, ctx):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(ai_assistant, "_client", lambda: pytest.fail("API must not be called"))
    ans = ai_assistant.answer("my otp is 482913, check my payment", ctx)
    assert "not processed" in ans


def test_missing_data_message(no_key):
    assert "don't have any sales data" in ai_assistant.answer("sales?", {"data_available": False})


class _FakeStream:
    def __init__(self, chunks, stop="end_turn"):
        self.text_stream = iter(chunks)
        self._stop = stop

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self._stop)


def test_claude_request_shape(monkeypatch, ctx):
    calls = {}

    def stream(**kwargs):
        calls.update(kwargs)
        return _FakeStream(["Sales ", "fell."])

    fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(ai_assistant, "_client", lambda: fake)
    history = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "q1"},
               {"role": "assistant", "content": "a1"}]
    ans = ai_assistant.answer("Why?", ctx, history)
    assert ans == "Sales fell."
    assert calls["model"] == ai_assistant.MODEL
    assert calls["fallbacks"] == "default"
    assert calls["messages"][0]["role"] == "user" and calls["messages"][-1]["content"] == "Why?"
    assert "VERIFIED_DATA" in calls["system"][1]["text"]


def test_refusal_is_handled(monkeypatch, ctx):
    fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
        stream=lambda **k: _FakeStream([], stop="refusal"))))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(ai_assistant, "_client", lambda: fake)
    assert "can't help" in ai_assistant.answer("Why?", ctx)
