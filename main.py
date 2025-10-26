#!/usr/bin/env python3
"""
OpenRouter CLI chat with function‑calling (MCP).

Features
--------
* Two tools:
    - search_gutenberg_books – search Project Gutenberg (Gutendex API)
    - python_execute        – run a short Python snippet in a sandbox
* Strong system prompt with concrete usage examples.
* When the model calls `python_execute` without a `code` argument the script
  tries to locate a *valid* Python snippet in the conversation history
  (using ast.parse).  If none is found a helpful error is returned.
* Optional `--provider` flag to force a specific OpenRouter backend.
* No regex‑based shortcuts – the assistant decides when to call a tool.

Usage
-----
    export OPENROUTER_API_KEY="sk-..."
    python openrouter_cli_tool.py \
        --model openai/gpt-4o                # any function‑calling model
        --provider deepinfra                  # optional, force a provider
        --no-tools                           # optional, disable tools
"""

import argparse
import ast
import json
import os
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, List

import requests
from openai import OpenAI
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CONFIG_PATH = Path.home() / ".openrouter_chat.json"


def load_config() -> Dict[str, Any]:
    """Load optional JSON config from ~/.openrouter_chat.json."""
    if CONFIG_PATH.is_file():
        try:
            return json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
        except Exception as e:
            print(f"⚠️  Could not read config: {e}", file=sys.stderr)
    return {}


# ----------------------------------------------------------------------
# ARGPARSE
# ----------------------------------------------------------------------
def build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OpenRouter CLI chat with function calling (MCP)."
    )
    p.add_argument(
        "--api-key",
        default=defaults.get("api_key") or os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var).",
    )
    p.add_argument(
        "--model",
        default=defaults.get("model", "openai/gpt-4o"),
        help="Model ID, e.g. openai/gpt-4o",
    )
    p.add_argument(
        "--provider",
        default=defaults.get("provider", ""),
        help=(
            "Provider slug (openrouter, groq, deepinfra, etc.). "
            "If given, final model ID becomes <provider>/<model>."
        ),
    )
    p.add_argument(
        "--referer",
        default=defaults.get("referer"),
        help="Optional HTTP‑Referer header for OpenRouter rankings.",
    )
    p.add_argument(
        "--title",
        default=defaults.get("title"),
        help="Optional X‑Title header for OpenRouter rankings.",
    )
    p.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="Base URL of the OpenRouter API (normally leave default).",
    )
    p.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable tool calling even if the model supports it.",
    )
    return p


# ----------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ----------------------------------------------------------------------
def search_gutenberg_books(search_terms: List[str]) -> str:
    """Search Project Gutenberg via the public Gutendex API."""
    base = "https://gutendex.com/books"
    lines = []

    for term in search_terms:
        try:
            resp = requests.get(base, params={"search": term}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            books = data.get("results", [])[:5]  # up to 5 results per term
            titles = [b.get("title", "Untitled") for b in books]
            lines.append(f'🔎 "{term}" → ' + ", ".join(titles))
        except Exception as exc:
            lines.append(f'⚠️  error searching "{term}": {exc}')

    return "\n".join(lines) or "No books found."


def python_execute(code: str) -> str:
    """
    Execute a short Python snippet in a sandbox.
    Only a tiny whitelist of built‑ins is exposed.
    Returns stdout or a concise traceback on error.
    """
    import io
    import contextlib
    import traceback

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "print": print,
    }

    # Debug line – will appear in the tool output.
    debug_line = f"Executing Python code:\n{code}\n"

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            print(debug_line, end="")          # show the debug line first
            exec(code, {"__builtins__": safe_builtins}, {})
        result = stdout.getvalue().strip()
        return result if result else "(no output)"
    except Exception:
        tb = traceback.format_exc()
        # Show only the last line of the traceback (the actual error message).
        return f"❌ python_execute raised an exception:\n{tb.splitlines()[-2:]}"


# ----------------------------------------------------------------------
# TOOL REGISTRY (name → implementation + JSON schema)
# ----------------------------------------------------------------------
tool_registry = {
    "search_gutenberg_books": {
        "func": search_gutenberg_books,
        "description": "Search for books in the Project Gutenberg library",
        "parameters": {
            "type": "object",
            "properties": {
                "search_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search terms to find books",
                }
            },
            "required": ["search_terms"],
        },
    },
    "python_execute": {
        "func": python_execute,
        "description": "Execute a short Python snippet (no network, no file I/O).",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "A short piece of Python code. Use `print()` for output. "
                        "The code must be self‑contained and cannot import external modules."
                    ),
                }
            },
            "required": ["code"],
        },
    },
}


def build_function_definitions() -> List[Dict[str, Any]]:
    """Convert the registry into the OpenAI‑compatible function list."""
    defs = []
    for name, meta in tool_registry.items():
        defs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta["description"],
                    "parameters": meta["parameters"],
                },
            }
        )
    return defs


# ----------------------------------------------------------------------
# Helper: try to find a *valid* Python snippet in the conversation history
# ----------------------------------------------------------------------
def extract_code_from_user(messages: List[Dict[str, Any]]) -> str | None:
    """
    Scan past user messages (most recent first) and return the first one
    that parses successfully with `ast.parse`.  Returns None if nothing
    parses.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        txt = msg.get("content", "")
        try:
            ast.parse(txt, mode="exec")
            return txt
        except Exception:
            continue
    return None


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------
class Spinner:
    """Simple tqdm spinner while waiting for the remote model."""

    def __enter__(self):
        self.tq = tqdm(total=0, bar_format="⏳ {desc}", leave=False, colour="cyan")
        self.tq.set_description_str("Calling model…")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tq.clear()
        self.tq.close()


def wrap_print(text: str, prefix: str = "🤖"):
    """Wrap long output to 80 columns and keep a nice prefix."""
    lines = textwrap.wrap(text, width=80)
    for i, line in enumerate(lines):
        print(f"{prefix if i == 0 else ' ' * len(prefix)} {line}")


# ----------------------------------------------------------------------
# MAIN chat loop
# ----------------------------------------------------------------------
def chat_loop(
    client: OpenAI,
    model: str,
    extra_headers: Dict[str, str],
    enable_tools: bool,
) -> None:
    """Interactive REPL that can call the registered tools."""
    messages: List[Dict[str, Any]] = []

    # ------------------- System prompt (explicit, with examples) -------------------
    messages.append(
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "When the user provides a Python snippet you **must** call the function "
                "`python_execute` with a single argument `code` containing that snippet. "
                "If the user merely describes a calculation without giving code, answer "
                "normally or ask for the snippet – do not call the tool with empty code. "
                "When the user asks for book titles you **must** call "
                "`search_gutenberg_books` with an array of search terms. "
                "If the request does not need a tool, answer directly.\n\n"
                "Correct usage examples (the assistant should follow this pattern):\n"
                "User: execute 2**8 in python\n"
                "Assistant: (calls `python_execute` with arguments {\"code\": \"print(2**8)\"})\n"
                "User: give me three romance books from Gutenberg\n"
                "Assistant: (calls `search_gutenberg_books` with arguments "
                "{\"search_terms\": [\"romance\"]})\n\n"
                "Available tools:\n"
                + "\n".join(
                    f"- {name}: {meta['description']}"
                    for name, meta in tool_registry.items()
                )
            ),
        }
    )
    # ---------------------------------------------------------------------------

    print("\n💬  OpenRouter chat – type your message, `exit` to quit.")
    print("   (Tools are enabled)" if enable_tools else "   (Tools are disabled)")

    tools_supported = enable_tools

    while True:
        # ----------------------- Get user input -----------------------
        try:
            user_input = input("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋  Bye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("👋  Bye!")
            break
        if not user_input:
            continue

        # Add the user message to history
        messages.append({"role": "user", "content": user_input})

        # ----------------------- First request -----------------------
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "extra_headers": extra_headers,
            "temperature": 0.7,
        }
        if tools_supported:
            request_kwargs["tools"] = build_function_definitions()
            request_kwargs["tool_choice"] = "auto"

        try:
            with Spinner():
                resp = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            print(f"\n❌ API error: {exc}\n")
            # discard the user message so they can retry
            messages.pop()
            continue

        choice = resp.choices[0]

        # ----------------------- Did the model request a tool? -----------------------
        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            func_name = tool_call.function.name
            raw_args = tool_call.function.arguments

            # Parse the JSON arguments (may be empty)
            try:
                args = json.loads(raw_args) if raw_args else {}
            except Exception:
                args = {}

            # -------------------------------------------------------------
            # Special handling for python_execute when `code` is missing.
            # -------------------------------------------------------------
            if func_name == "python_execute" and not args.get("code"):
                inferred = extract_code_from_user(messages)
                if inferred:
                    args["code"] = inferred
                else:
                    # Friendly fallback – tell the user we need proper code.
                    tool_result = (
                        "❌ The assistant tried to call `python_execute` but no valid "
                        "Python code was supplied. Please provide a short Python snippet "
                        "(e.g. `print(2**8)`)."
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": tool_result,
                        }
                    )
                    wrap_print(tool_result, prefix="🤖")
                    # Skip the second request – we already have an answer.
                    continue
            # -------------------------------------------------------------

            print(f"\n🛠️  Model wants to run `{func_name}` with args {args}")

            # ----------------------- Run the local implementation -----------------------
            if func_name not in tool_registry:
                tool_result = f"❌ Unknown tool `{func_name}`."
            else:
                try:
                    tool_result = tool_registry[func_name]["func"](**args)
                except Exception as exc:
                    tb = traceback.format_exc()
                    tool_result = f"❌ Error while executing `{func_name}`:\n{tb}"

            # Insert the tool result back into the conversation.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": tool_result,
                }
            )

            # ----------------------- Second request (model sees tool output) -----------------------
            try:
                with Spinner():
                    follow_up = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        extra_headers=extra_headers,
                        temperature=0.7,
                    )
                final_answer = follow_up.choices[0].message.content.strip()
                wrap_print(final_answer)
                messages.append({"role": "assistant", "content": final_answer})
            except Exception as exc:
                print(f"\n❌ Follow‑up API error: {exc}\n")
            continue

        # ----------------------- Plain text reply (no tool) -----------------------
        answer = choice.message.content.strip()
        wrap_print(answer)
        messages.append({"role": "assistant", "content": answer})


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args()

    # Build the final model identifier (optional provider prefix)
    if args.provider:
        model_id = f"{args.provider}/{args.model}"
    else:
        model_id = args.model

    if not args.api_key:
        parser.error(
            "OpenRouter API key required (use --api-key, env var, or config file)."
        )

    extra_headers = {}
    if args.referer:
        extra_headers["HTTP-Referer"] = args.referer
    if args.title:
        extra_headers["X-Title"] = args.title

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    try:
        chat_loop(
            client,
            model=model_id,
            extra_headers=extra_headers,
            enable_tools=not args.no_tools,
        )
    except KeyboardInterrupt:
        print("\n👋  Bye!")


if __name__ == "__main__":
    main()