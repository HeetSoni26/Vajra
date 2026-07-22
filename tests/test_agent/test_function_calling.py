"""Tests for JSON extraction, parsing, and syntax repair in FunctionParser."""

from vajra_agent.function_calling import FunctionParser


def test_parse_markdown_json_block():
    text = """
I will use the file_tool to read the file.
```json
{
  "tool": "file_tool",
  "arguments": {
    "action": "read",
    "path": "config.py"
  }
}
```
"""
    call = FunctionParser.parse(text)
    assert call is not None
    assert call.tool_name == "file_tool"
    assert call.arguments["action"] == "read"
    assert call.arguments["path"] == "config.py"


def test_parse_raw_json_string():
    text = '{"tool": "python_tool", "arguments": {"code": "print(1)"}}'
    call = FunctionParser.parse(text)
    assert call is not None
    assert call.tool_name == "python_tool"
    assert call.arguments["code"] == "print(1)"


def test_parse_trailing_comma_repair():
    text = '{"tool": "shell_tool", "arguments": {"command": "dir",}}'
    call = FunctionParser.parse(text)
    assert call is not None
    assert call.tool_name == "shell_tool"


def test_parse_plain_text_returns_none():
    text = "The answer is 42."
    call = FunctionParser.parse(text)
    assert call is None
