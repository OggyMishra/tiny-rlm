"""Tests for tiny_rlm.repl — REPLEnv sandboxed execution."""

import asyncio

from tiny_rlm.repl import REPLEnv


class TestREPLBasicExecution:
    def test_simple_print(self):
        repl = REPLEnv()
        result = repl.execute('print("hello")')
        assert result.stdout == "hello\n"
        assert result.stderr == ""

    def test_variable_persistence(self):
        repl = REPLEnv()
        repl.execute("x = 42")
        result = repl.execute("print(x)")
        assert result.stdout == "42\n"

    def test_multi_line_code(self):
        repl = REPLEnv()
        result = repl.execute("for i in range(3):\n    print(i)")
        assert result.stdout == "0\n1\n2\n"

    def test_error_captured_in_stderr(self):
        repl = REPLEnv()
        result = repl.execute("1 / 0")
        assert "Error" in result.stderr
        assert "division by zero" in result.stderr

    def test_execution_time_tracked(self):
        repl = REPLEnv()
        result = repl.execute("x = 1")
        assert result.execution_time >= 0


class TestREPLSandbox:
    def test_eval_blocked(self):
        repl = REPLEnv()
        result = repl.execute('eval("1+1")')
        assert "Error" in result.stderr

    def test_exec_blocked(self):
        repl = REPLEnv()
        result = repl.execute('exec("x=1")')
        assert "Error" in result.stderr

    def test_input_blocked(self):
        repl = REPLEnv()
        result = repl.execute('input("prompt")')
        assert "Error" in result.stderr

    def test_import_allowed(self):
        repl = REPLEnv()
        result = repl.execute("import json\nprint(json.dumps({'a': 1}))")
        assert '{"a": 1}' in result.stdout

    def test_re_import_allowed(self):
        repl = REPLEnv()
        result = repl.execute("import re\nprint(re.findall(r'\\d+', 'abc123def456'))")
        assert "['123', '456']" in result.stdout


class TestREPLFinal:
    def test_final_sets_result(self):
        repl = REPLEnv()
        repl.execute('FINAL({"answer": 42})')
        assert repl.final_set is True
        assert repl.final_result == {"answer": 42}

    def test_final_not_set_initially(self):
        repl = REPLEnv()
        assert repl.final_set is False
        assert repl.final_result is None

    def test_final_with_string(self):
        repl = REPLEnv()
        repl.execute('FINAL("done")')
        assert repl.final_result == "done"

    def test_final_with_list(self):
        repl = REPLEnv()
        repl.execute("FINAL([1, 2, 3])")
        assert repl.final_result == [1, 2, 3]

    def test_final_with_none(self):
        repl = REPLEnv()
        repl.execute("FINAL(None)")
        assert repl.final_set is True
        assert repl.final_result is None


class TestREPLContext:
    def test_load_context_string(self):
        repl = REPLEnv()
        repl.load_context("hello world")
        result = repl.execute("print(context)")
        assert "hello world" in result.stdout

    def test_load_context_preserves_type(self):
        repl = REPLEnv()
        repl.load_context("test document text")
        result = repl.execute("print(type(context).__name__)")
        assert "str" in result.stdout

    def test_context_with_special_chars(self):
        repl = REPLEnv()
        text = 'He said "hello" and \'goodbye\''
        repl.load_context(text)
        result = repl.execute("print(len(context))")
        assert result.stdout.strip() == str(len(text))

    def test_context_with_newlines(self):
        repl = REPLEnv()
        repl.load_context("line1\nline2\nline3")
        result = repl.execute("print(context.count('\\n'))")
        assert result.stdout.strip() == "2"


class TestREPLAsync:
    def test_execute_async_simple(self):
        repl = REPLEnv()
        result = asyncio.run(repl.execute_async('print("async hello")'))
        assert result.stdout == "async hello\n"

    def test_execute_async_variable_persistence(self):
        repl = REPLEnv()
        repl.execute("x = 10")
        result = asyncio.run(repl.execute_async("print(x * 2)"))
        assert result.stdout == "20\n"

    def test_execute_async_final(self):
        repl = REPLEnv()
        asyncio.run(repl.execute_async("FINAL(99)"))
        assert repl.final_set is True
        assert repl.final_result == 99

    def test_execute_async_with_await(self):
        """Test that await syntax works when llm_query is provided."""

        async def mock_llm_query(prompt: str):
            return {"extracted": "test_value"}

        repl = REPLEnv(llm_query_fn=mock_llm_query)
        repl.load_context("doc text")
        result = asyncio.run(
            repl.execute_async(
                'result = await llm_query("extract from context")\nprint(result)'
            )
        )
        assert "test_value" in result.stdout

    def test_execute_async_error_handling(self):
        repl = REPLEnv()
        result = asyncio.run(repl.execute_async("raise ValueError('boom')"))
        assert "Error" in result.stderr
        assert "boom" in result.stderr


class TestREPLGetVariable:
    def test_get_existing_variable(self):
        repl = REPLEnv()
        repl.execute("my_var = [1, 2, 3]")
        assert repl.get_variable("my_var") == [1, 2, 3]

    def test_get_nonexistent_variable(self):
        repl = REPLEnv()
        assert repl.get_variable("missing") is None
