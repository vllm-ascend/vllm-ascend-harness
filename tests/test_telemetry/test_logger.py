"""Unit tests for logging utilities."""

import json
import logging

from vllm_ascend_harness.telemetry.logger import JsonFormatter, get_logger


class TestGetLogger:
    def test_basic_logger(self):
        logger = get_logger("test.basic", level=logging.DEBUG)
        assert logger.name == "test.basic"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 1

    def test_logger_has_handler(self):
        logger = get_logger("test.handler")
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_logger_no_duplicate_handlers(self):
        logger1 = get_logger("test.dup")
        n_handlers = len(logger1.handlers)
        logger2 = get_logger("test.dup")
        assert len(logger2.handlers) == n_handlers

    def test_json_format(self):
        logger = get_logger("test.json", json_format=True)
        assert any(
            isinstance(h.formatter, JsonFormatter) for h in logger.handlers
        )

    def test_file_rotation(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = get_logger(
            "test.file", log_file=log_file, max_bytes=1024, backup_count=2
        )
        logger.info("test message")
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "test message" in content

    def test_json_output_is_valid_json(self, tmp_path):
        log_file = tmp_path / "json.log"
        logger = get_logger("test.jsonfile", json_format=True, log_file=log_file)
        logger.info("structured log")
        content = log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.strip())
        assert parsed["message"] == "structured log"
        assert parsed["level"] == "INFO"

    def test_logger_propagate_false(self):
        logger = get_logger("test.nopropagate")
        assert logger.propagate is False


class TestJsonFormatter:
    def test_format_basic(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"

    def test_format_with_exception(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exc_info" in parsed
        assert "ValueError" in parsed["exc_info"]
