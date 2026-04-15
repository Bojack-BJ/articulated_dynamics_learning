from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?$"
)


class YAMLSubsetError(ValueError):
    """Raised when a config file uses YAML outside the supported subset."""


def load_command_config(path: str | Path) -> dict[str, Any]:
    """Load a CLI config file.

    The loader accepts JSON directly and a small YAML subset that is enough for
    the project's CLI use cases: nested mappings, simple lists, strings,
    booleans, nulls, integers, and floats.
    """

    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _parse_yaml_subset(text)
    if not isinstance(data, dict):
        raise YAMLSubsetError(f"Config root must be a mapping: {config_path}")
    return data


def expand_config_argv(argv: list[str], parser: argparse.ArgumentParser) -> list[str]:
    """Expand a YAML/JSON config file into normal argparse-style argv tokens."""

    if not argv:
        return argv

    config_path: str | None = None
    trailing_args: list[str] = []
    if argv[0] in {"--config", "-c"}:
        if len(argv) < 2:
            raise YAMLSubsetError("--config expects a YAML or JSON file path")
        config_path = argv[1]
        trailing_args = argv[2:]
    elif Path(argv[0]).suffix.lower() in {".yaml", ".yml", ".json"}:
        config_path = argv[0]
        trailing_args = argv[1:]
    else:
        return argv

    config = load_command_config(config_path)
    synthesized = config_to_argv(config, parser)
    # Later CLI flags win, so trailing args are appended after config-derived args.
    return synthesized + trailing_args


def config_to_argv(config: dict[str, Any], parser: argparse.ArgumentParser) -> list[str]:
    command = config.get("command")
    if not isinstance(command, str) or not command:
        raise YAMLSubsetError("Config must include a non-empty 'command' field")

    args_payload = dict(config.get("args", {})) if isinstance(config.get("args", {}), dict) else None
    if args_payload is None:
        raise YAMLSubsetError("Config field 'args' must be a mapping when provided")
    for key, value in config.items():
        if key not in {"command", "args", "description"}:
            args_payload.setdefault(key, value)

    subparser = _resolve_subparser(parser, command)
    return [command, *_command_args_to_argv(subparser, args_payload)]


def _resolve_subparser(parser: argparse.ArgumentParser, command: str) -> argparse.ArgumentParser:
    subparsers_action = next(
        (
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ),
        None,
    )
    if subparsers_action is None or command not in subparsers_action.choices:
        raise YAMLSubsetError(f"Unknown command in config: {command}")
    return subparsers_action.choices[command]


def _command_args_to_argv(subparser: argparse.ArgumentParser, args_payload: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    consumed_keys: set[str] = set()

    positional_actions = [
        action
        for action in subparser._actions
        if not action.option_strings and action.dest != "help"
    ]
    for action in positional_actions:
        if action.dest not in args_payload:
            raise YAMLSubsetError(f"Missing positional argument '{action.dest}' in config")
        _append_value_tokens(argv, args_payload[action.dest])
        consumed_keys.add(action.dest)

    optional_lookup: dict[str, argparse.Action] = {}
    for action in subparser._actions:
        if not action.option_strings:
            continue
        optional_lookup[action.dest] = action
        optional_lookup[action.dest.replace("_", "-")] = action
        for option in action.option_strings:
            if option.startswith("--"):
                optional_lookup[option[2:]] = action

    for raw_key, value in args_payload.items():
        if raw_key in consumed_keys:
            continue
        action = optional_lookup.get(raw_key) or optional_lookup.get(raw_key.replace("_", "-"))
        if action is None:
            raise YAMLSubsetError(f"Unknown config key for command '{subparser.prog}': {raw_key}")
        flag = next((option for option in action.option_strings if option.startswith("--")), action.option_strings[0])
        _append_option_tokens(argv, action, flag, value)

    return argv


def _append_option_tokens(argv: list[str], action: argparse.Action, flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(action, argparse._StoreTrueAction):
        if bool(value):
            argv.append(flag)
        return
    if isinstance(action, argparse._StoreFalseAction):
        if not bool(value):
            argv.append(flag)
        return

    argv.append(flag)
    _append_value_tokens(argv, value)


def _append_value_tokens(argv: list[str], value: Any) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            argv.append(str(item))
        return
    argv.append(str(value))


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    parser = _YAMLSubsetParser(text)
    data = parser.parse()
    if not isinstance(data, dict):
        raise YAMLSubsetError("YAML config root must be a mapping")
    return data


class _YAMLSubsetParser:
    """Very small indentation-based YAML parser for CLI configs.

    It intentionally supports only the subset the project needs for command
    configuration files. Unsupported YAML features should fail loudly instead of
    being parsed incorrectly.
    """

    def __init__(self, text: str) -> None:
        self.lines = self._prepare_lines(text)
        self.index = 0

    def parse(self) -> Any:
        if not self.lines:
            return {}
        indent, _ = self.lines[0]
        value = self._parse_block(indent)
        if self.index != len(self.lines):
            raise YAMLSubsetError("Trailing YAML content could not be parsed")
        return value

    def _prepare_lines(self, text: str) -> list[tuple[int, str]]:
        prepared: list[tuple[int, str]] = []
        for raw_line in text.splitlines():
            line = _strip_comment(raw_line).rstrip()
            if not line.strip():
                continue
            if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]:
                raise YAMLSubsetError("Tabs are not supported in YAML indentation")
            indent = len(line) - len(line.lstrip(" "))
            prepared.append((indent, line.lstrip(" ")))
        return prepared

    def _parse_block(self, expected_indent: int) -> Any:
        if self.index >= len(self.lines):
            raise YAMLSubsetError("Unexpected end of YAML input")
        indent, content = self.lines[self.index]
        if indent != expected_indent:
            raise YAMLSubsetError(f"Unexpected indentation: expected {expected_indent}, got {indent}")
        if content.startswith("- "):
            return self._parse_sequence(expected_indent)
        return self._parse_mapping(expected_indent)

    def _parse_mapping(self, expected_indent: int) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        while self.index < len(self.lines):
            indent, content = self.lines[self.index]
            if indent < expected_indent:
                break
            if indent != expected_indent:
                raise YAMLSubsetError(f"Unexpected indentation inside mapping: {indent}")
            if content.startswith("- "):
                raise YAMLSubsetError("Cannot mix sequence items directly into a mapping block")
            key, separator, rest = content.partition(":")
            if separator != ":":
                raise YAMLSubsetError(f"Expected key:value entry, got: {content}")
            key = key.strip()
            rest = rest.lstrip(" ")
            self.index += 1
            if not rest:
                if self.index < len(self.lines) and self.lines[self.index][0] > indent:
                    value = self._parse_block(self.lines[self.index][0])
                else:
                    value = None
            else:
                value = _parse_yaml_scalar(rest)
            mapping[key] = value
        return mapping

    def _parse_sequence(self, expected_indent: int) -> list[Any]:
        items: list[Any] = []
        while self.index < len(self.lines):
            indent, content = self.lines[self.index]
            if indent < expected_indent:
                break
            if indent != expected_indent or not content.startswith("- "):
                raise YAMLSubsetError(f"Unexpected indentation inside sequence: {indent}")
            item_text = content[2:].lstrip(" ")
            self.index += 1
            if not item_text:
                if self.index < len(self.lines) and self.lines[self.index][0] > indent:
                    value = self._parse_block(self.lines[self.index][0])
                else:
                    value = None
            else:
                value = _parse_yaml_scalar(item_text)
            items.append(value)
        return items


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for index, char in enumerate(line):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "'" and not in_double and not escaped:
            in_single = not in_single
        elif char == '"' and not in_single and not escaped:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
        escaped = False
    return line


def _parse_yaml_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(item) for item in _split_inline_list(inner)]
    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        return text[1:-1]

    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if _INTEGER_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _split_inline_list(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    bracket_depth = 0
    for char in text:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "[" and not in_single and not in_double:
            bracket_depth += 1
        elif char == "]" and not in_single and not in_double:
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "," and not in_single and not in_double and bracket_depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        items.append("".join(current).strip())
    return items
