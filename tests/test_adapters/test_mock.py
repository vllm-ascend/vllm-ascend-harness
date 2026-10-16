"""Unit tests for MockAdapter."""

import pytest

from vllm_ascend_harness.adapters.base import GenerationRequest
from vllm_ascend_harness.adapters.mock import MockAdapter


@pytest.fixture
def adapter():
    return MockAdapter()


class TestMockGenerate:
    def test_generate_returns_text(self, adapter):
        result = adapter.generate(GenerationRequest(prompt="hello world"))
        assert "[mock] hello world" == result.text

    def test_generate_token_counts(self, adapter):
        result = adapter.generate(GenerationRequest(prompt="hello world"))
        assert result.prompt_tokens == 2
        # "[mock] hello world".split() → ["[mock]", "hello", "world"] → 3 tokens
        assert result.completion_tokens == 3

    def test_generate_custom_prefix(self):
        adapter = MockAdapter(prefix="[test] ")
        result = adapter.generate(GenerationRequest(prompt="hi"))
        assert result.text == "[test] hi"

    def test_generate_finish_reason(self, adapter):
        result = adapter.generate(GenerationRequest(prompt="test"))
        assert result.finish_reason == "stop"

    def test_generate_metadata(self, adapter):
        result = adapter.generate(GenerationRequest(prompt="test"))
        assert result.metadata["adapter"] == "mock"
        assert result.metadata["model"] == "mock-model"


class TestMockStream:
    def test_stream_yields_tokens(self, adapter):
        request = GenerationRequest(prompt="hello world")
        results = list(adapter.stream(request))
        # "[mock] hello world" → ["[mock]", "hello", "world"] → 3 tokens
        assert len(results) == 3
        assert results[-1].finish_reason == "stop"
        assert all(r.metadata["streaming"] is True for r in results)

    def test_stream_first_token_has_prompt_tokens(self, adapter):
        results = list(adapter.stream(GenerationRequest(prompt="a b c")))
        assert results[0].prompt_tokens == 3
        assert all(r.prompt_tokens == 0 for r in results[1:])


class TestMockLifecycle:
    def test_health_check_when_open(self, adapter):
        assert adapter.health_check() is True

    def test_health_check_after_close(self, adapter):
        adapter.close()
        assert adapter.health_check() is False

    def test_generate_after_close_raises(self, adapter):
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            adapter.generate(GenerationRequest(prompt="x"))

    def test_stream_after_close_raises(self, adapter):
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            list(adapter.stream(GenerationRequest(prompt="x")))
