"""Unit tests for VllmAscendAdapter.

These tests verify the adapter's behavior without requiring vllm
or Ascend NPU hardware to be installed.
"""

import pytest

from vllm_ascend_harness.adapters.base import GenerationRequest
from vllm_ascend_harness.adapters.vllm_ascend import (
    VllmAscendAdapter,
    _request_to_sampling_params,
)


class TestSamplingParamsMapping:
    def test_basic_mapping(self):
        req = GenerationRequest(prompt="hi", max_tokens=128, temperature=0.7, top_p=0.9)
        params = _request_to_sampling_params(req)
        assert params["max_tokens"] == 128
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9

    def test_stop_included_when_set(self):
        req = GenerationRequest(prompt="hi", stop=["END"])
        params = _request_to_sampling_params(req)
        assert params["stop"] == ["END"]

    def test_stop_omitted_when_none(self):
        req = GenerationRequest(prompt="hi")
        params = _request_to_sampling_params(req)
        assert "stop" not in params


class TestVllmAscendAdapterInit:
    def test_import_error_without_vllm(self):
        with pytest.raises(ImportError, match="vllm is required"):
            VllmAscendAdapter(model="test-model")

    def test_import_error_message_mentions_mock(self):
        with pytest.raises(ImportError, match="MockAdapter"):
            VllmAscendAdapter(model="test-model")


class TestVllmAscendAdapterWithMockEngine:
    """Test generate/health_check/close using an injected mock engine."""

    def _make_adapter_with_mock_engine(self):
        adapter = VllmAscendAdapter.__new__(VllmAscendAdapter)
        adapter.model = "test-model"
        adapter._llm_kwargs = {}
        adapter._engine = _MockVllmEngine()
        adapter._mock_mode = False
        adapter._SamplingParams = _MockSamplingParams
        return adapter

    def test_generate_with_mock_engine(self):
        adapter = self._make_adapter_with_mock_engine()
        result = adapter.generate(GenerationRequest(prompt="hello"))
        assert result.text == "mock output"
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0

    def test_health_check_with_engine(self):
        adapter = self._make_adapter_with_mock_engine()
        assert adapter.health_check() is True

    def test_health_check_without_engine(self):
        adapter = VllmAscendAdapter.__new__(VllmAscendAdapter)
        adapter._engine = None
        assert adapter.health_check() is False

    def test_close_releases_engine(self):
        adapter = self._make_adapter_with_mock_engine()
        adapter.close()
        assert adapter._engine is None

    def test_generate_without_engine_raises(self):
        adapter = VllmAscendAdapter.__new__(VllmAscendAdapter)
        adapter._engine = None
        with pytest.raises(RuntimeError, match="not initialized"):
            adapter.generate(GenerationRequest(prompt="x"))


class _MockSamplingParams:
    """Minimal mock of vllm.SamplingParams."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _MockVllmEngine:
    """Minimal mock of vllm.LLM for testing the adapter logic."""

    def generate(self, prompts, sampling_params, use_tqdm=False):
        class _Output:
            class _Completion:
                text = "mock output"
                token_ids = [1, 2, 3]
                finish_reason = "stop"

            outputs = [_Completion()]
            prompt_token_ids = [10, 20, 30]
            request_id = "test-001"

        return [_Output()]
