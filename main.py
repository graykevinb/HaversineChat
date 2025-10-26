#!/usr/bin/env python3
"""
OpenRouter CLI chat with two tools:

1. search_gutenberg_books – search Project Gutenberg (Gutendex API)
2. python_execute        – execute any Python snippet (no sandbox)

If the model tries to call `python_execute` without a `code` argument,
the assistant automatically asks the model to supply the missing code,
then runs the tool once the code is received.
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
# CONFIG (optional user config file)
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
# ARGUMENT PARSER
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
    Execute any Python snippet. **No sandbox** – the code runs with full
    Python built‑ins and can import any module available in the environment.
    """
    import io
    import contextlib

    debug_line = f"Executing Python code:\n{code}\n"

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            print(debug_line, end="")          # show the snippet first
            exec(code, {}, {})                # unrestricted exec
        result = stdout.getvalue().strip()
        return result if result else "(no output)"
    except Exception:
        tb = traceback.format_exc()
        # Return only the last line of the traceback (the actual error message)
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
        "description": "Execute any Python snippet (no sandbox).",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "A Python snippet. Use `print()` for output if you want "
                        "something returned. The snippet runs with full access "
                        "to the Python environment."
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
# UI helpers
# ----------------------------------------------------------------------
class Spinner:
    """Simple tqdm spinner while we wait for the remote model."""

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
    """Interactive REPL that can call the two registered tools."""
    messages: List[Dict[str, Any]] = []

    # ------------------- System prompt (explicit, with examples) -------------------
    messages.append(
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "When the user wants to run Python code you **must** call the function "
                "`python_execute` with a single argument `code` containing the exact snippet. "
                "If the user only describes a calculation, ask the model to provide the "
                "code before calling the tool. "
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
    # -------------------------------------------------------------------------

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

        # Record the user message
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
            # Remove the user message so they can retry
            messages.pop()
            continue

        choice = resp.choices[0]

        # ----------------------- Did the model request a tool? -----------------------
        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            func_name = tool_call.function.name
            raw_args = tool_call.function.arguments

            # Parse arguments JSON (may be empty)
            try:
                args = json.loads(raw_args) if raw_args else {}
            except Exception:
                args = {}

            # -----------------------------------------------------------------
            # Special handling for python_execute when `code` is missing.
            # -----------------------------------------------------------------
            if func_name == "python_execute" and not args.get("code"):
                # Ask the model again for the missing code.
                # We do *not* add the tool call to the history; we just prompt.
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I need the exact Python code you want me to run. "
                            "Please provide the snippet (use `print()` if you want output)."
                        ),
                    }
                )
                # Go back to the top of the loop – a new request will be sent.
                continue
            # -----------------------------------------------------------------

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

            # Insert the tool result back into the conversation history.
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
            model=args.model,
            extra_headers=extra_headers,
            enable_tools=not args.no_tools,
        )
    except KeyboardInterrupt:
        print("\n👋  Bye!")


if __name__ == "__main__":
    main()