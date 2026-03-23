"""Tests for tiny_rlm.engine — helper functions and RLMEngine."""

import pytest

from tiny_rlm.engine import _extract_repl_block, _truncate_last_n


class TestExtractReplBlock:
    def test_single_repl_block(self):
        text = 'Some text\n```repl\nprint("hello")\n```\nMore text'
        assert _extract_repl_block(text) == 'print("hello")'

    def test_multiple_repl_blocks(self):
        text = '```repl\nx = 1\n```\ntext\n```repl\ny = 2\n```'
        result = _extract_repl_block(text)
        assert "x = 1" in result
        assert "y = 2" in result

    def test_no_repl_block(self):
        text = "Just plain text with no code blocks"
        assert _extract_repl_block(text) is None

    def test_python_block_not_matched(self):
        text = '```python\nprint("hello")\n```'
        assert _extract_repl_block(text) is None

    def test_empty_repl_block(self):
        text = "```repl\n\n```"
        result = _extract_repl_block(text)
        assert result is not None  # Matches but may be empty string

    def test_multiline_code(self):
        text = '```repl\nfor i in range(3):\n    print(i)\n```'
        result = _extract_repl_block(text)
        assert "for i in range(3):" in result
        assert "print(i)" in result


class TestTruncateLastN:
    def test_short_text_shows_full(self):
        result = _truncate_last_n("hello", 100)
        assert "FULL OUTPUT SHOWN" in result
        assert "hello" in result

    def test_long_text_truncated(self):
        text = "a" * 200
        result = _truncate_last_n(text, 50)
        assert "TRUNCATED" in result
        assert "Last 50 chars" in result
        assert result.endswith("a" * 50)

    def test_empty_text(self):
        result = _truncate_last_n("", 100)
        assert "EMPTY OUTPUT" in result

    def test_exact_length(self):
        text = "a" * 100
        result = _truncate_last_n(text, 100)
        assert "FULL OUTPUT SHOWN" in result

    def test_one_over_limit(self):
        text = "a" * 101
        result = _truncate_last_n(text, 100)
        assert "TRUNCATED" in result
