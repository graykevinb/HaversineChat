#!/usr/bin/env python3
"""
OpenRouter CLI chat with tool (function) calling support.
All Groq‑related work‑arounds have been removed.

Example usage:
    export OPENROUTER_API_KEY="sk-..."
    python openrouter_cli_tool.py \
        --model google/gemini-2.0-flash-001 \
        --provider deepinfra   # optional – you can also embed the provider in the model ID
"""

import argparse
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
    """Read optional JSON config from ~/.openrouter_chat.json."""
    if CONFIG_PATH.is_file():
        try:
            return json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
        except Exception as e:
            print(f"⚠️  Could not read config: {e}", file=sys.stderr)
    return {}


# ----------------------------------------------------------------------
# CLI ARGUMENTS
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
        default=defaults.get("model", "google/gemini-2.0-flash-001"),
        help="Model ID, e.g. google/gemini-2.0-flash-001",
    )
    p.add_argument(
        "--provider",
        default=defaults.get("provider", ""),
        help=(
            "Provider slug (openrouter, groq, deepinfra, etc.). "
            "If given, the final model identifier becomes <provider>/<model>."
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
    """
    Very small wrapper around the free Gutendex API
    (https://gutendex.com).  For each search term it returns the
    first few titles that match.
    """
    base = "https://gutendex.com/books"
    all_titles = []

    for term in search_terms:
        try:
            resp = requests.get(base, params={"search": term}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            books = data.get("results", [])[:5]  # take up to 5 per term
            titles = [b.get("title", "Untitled") for b in books]
            all_titles.append(f'🔎 "{term}" → ' + ", ".join(titles))
        except Exception as exc:
            all_titles.append(f'⚠️  error searching "{term}": {exc}')

    # Return a single string that the LLM can read easily
    return "\n".join(all_titles) or "No books found."

def python_execute(code: str) -> str:
    """
    Execute a short Python snippet in a sandbox and return its stdout.
    The sandbox only exposes a tiny whitelist of built‑ins.
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

    # ------------------------------------------------------------------
    # Your explicit debugging line – this will appear in the tool result
    # ------------------------------------------------------------------
    debug_line = f"Executing Python code:\n{code}\n"
    # ------------------------------------------------------------------

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            # Print the debug line first
            print(debug_line, end="")          # `end=""` so we don’t add an extra newline
            # Then actually run the user code
            exec(code, {"__builtins__": safe_builtins}, {})
        result = stdout.getvalue().strip()
        return result if result else "(no output)"
    except Exception:
        tb = traceback.format_exc()
        return f"❌ python_execute raised an exception:\n{tb.splitlines()[-2:]}"

# ----------------------------------------------------------------------
# TOOL REGISTRY (name → implementation + schema)
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
    },
    # You can add more tools here, following the same pattern.
}


def build_function_definitions() -> List[Dict[str, Any]]:
    """
    Turn the `tool_registry` into the JSON structure OpenRouter expects:
    [
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        },
        …
    ]
    """
    definitions = []
    for name, meta in tool_registry.items():
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta["description"],
                    "parameters": meta["parameters"],
                },
            }
        )
    return definitions


# ----------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------
class Spinner:
    """Simple tqdm‑based spinner while we wait for the remote model."""

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
# CHAT LOOP
# ----------------------------------------------------------------------
def chat_loop(
    client: OpenAI,
    model: str,
    extra_headers: Dict[str, str],
    enable_tools: bool,
) -> None:
    messages: List[Dict[str, Any]] = []

    # --------------------------------------------------------------
    # System prompt – you can keep the one you already have
    # --------------------------------------------------------------
    messages.append(
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "When the user asks for a Python calculation or any code, you MUST call the function "
                "`python_execute` with a single argument `code` (a short Python snippet). "
                "When the user asks for book titles, call `search_gutenberg_books`. "
                "If the request does not need a tool, answer normally.\n\n"
                "Available tools:\n"
                + "\n".join(
                    f"- {name}: {meta['description']}"
                    for name, meta in tool_registry.items()
                )
            ),
        }
    )

    print("\n💬  OpenRouter chat – type your message, `exit` to quit.")
    print("   (Tools are enabled)" if enable_tools else "   (Tools are disabled)")

    tools_supported = enable_tools

    while True:
        # ----------------------------------------------------------
        # get user input
        # ----------------------------------------------------------
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

        messages.append({"role": "user", "content": user_input})

        # ----------------------------------------------------------
        # first request (with tools if enabled)
        # ----------------------------------------------------------
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
            messages.pop()          # drop the last user message
            continue

        choice = resp.choices[0]

        # ----------------------------------------------------------
        # Did the model ask for a tool?
        # ----------------------------------------------------------
        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]          # only one per turn
            func_name = tool_call.function.name
            raw_args   = tool_call.function.arguments

            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}

            print(f"\n🛠️  Model wants to run `{func_name}` with args {args}")

            if func_name not in tool_registry:
                tool_result = f"❌ Unknown tool `{func_name}`."
            else:
                try:
                    tool_result = tool_registry[func_name]["func"](**args)
                except Exception as exc:
                    tb = traceback.format_exc()
                    tool_result = f"❌ Error while executing `{func_name}`:\n{tb}"

            # ------------------------------------------------------
            # **Correct** way to add the tool result to the conversation
            # ------------------------------------------------------
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": tool_result,
                }
            )
            # (no extra empty assistant message is needed)

            # ------------------------------------------------------
            # second request – model now sees the tool output
            # ------------------------------------------------------
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

        # ----------------------------------------------------------
        # No tool call – plain text response
        # ----------------------------------------------------------
        answer = choice.message.content.strip()
        wrap_print(answer)
        messages.append({"role": "assistant", "content": answer})


# ----------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Build the final model identifier (provider optional)
    # --------------------------------------------------------------
    if args.provider:
        model_id = f"{args.provider}/{args.model}"
    else:
        model_id = args.model
    # --------------------------------------------------------------

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