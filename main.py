#!/usr/bin/env python3
"""
openrouter_chat.py

A tiny command‑line chat client for OpenRouter‑hosted LLMs.
Works with any model that follows the OpenAI chat‑completion API.

Usage examples
--------------
# Using environment variable for the key
export OPENROUTER_API_KEY="sk-..."
python openrouter_chat.py --model openai/gpt-4o

# Or passing the key on the command line (not recommended for shared shells)
python openrouter_chat.py \
    --api-key "$OPENROUTER_API_KEY" \
    --model anthropic/claude-3.5-sonnet \
    --referer "https://my‑site.example" \
    --title "My Demo Chat"

# Store defaults in ~/.openrouter_chat.json to avoid typing them each run
{
  "api_key": "sk-...",
  "model": "openai/gpt-4o",
  "referer": "https://my‑site.example",
  "title": "My Demo Chat"
}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from tqdm import tqdm


# ----------------------------------------------------------------------
# Helper: load a tiny json config file (if it exists)
# ----------------------------------------------------------------------
def load_config() -> Dict[str, Any]:
    cfg_path = Path.home() / ".openrouter_chat.json"
    if cfg_path.is_file():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not read config {cfg_path}: {e}", file=sys.stderr)
    return {}


# ----------------------------------------------------------------------
# Helper: simple spinner while waiting for the API
# ----------------------------------------------------------------------
class Spinner:
    def __enter__(self):
        self._spinner = tqdm(
            total=0,
            bar_format="⏳ {desc}",
            leave=False,
            colour="cyan",
        )
        self._spinner.set_description_str("Waiting for model…")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._spinner.clear()
        self._spinner.close()


# ----------------------------------------------------------------------
# Main chat function
# ----------------------------------------------------------------------
def chat_loop(
    client: OpenAI,
    model: str,
    extra_headers: Dict[str, str],
) -> None:
    """
    Interactive REPL that sends user messages to OpenRouter
    and prints the assistant's reply.
    """
    # Keep full conversation history as required by the API
    messages: List[Dict[str, str]] = []

    print("\n💬  OpenRouter chat – type your message and press Enter.")
    print("   (type 'exit' or press Ctrl‑C to quit)\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋  Bye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("👋  Bye!")
            break

        if not user_input:
            continue  # ignore empty lines

        # Append user's message
        messages.append({"role": "user", "content": user_input})

        # Call the API
        try:
            with Spinner():
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_headers=extra_headers,
                    temperature=0.7,  # you can expose this as an arg later
                )
        except Exception as exc:
            # Print a concise error, but keep the conversation state so you can retry
            print(f"\n❌  API error: {exc}\n")
            # optionally remove the last user message so we can resend later
            messages.pop()
            continue

        # Extract assistant response
        assistant_msg = completion.choices[0].message
        response_text = assistant_msg.content.strip()
        print(f"\n🤖 {assistant_msg.role.capitalize()}: {response_text}\n")

        # Append assistant's reply to the history
        messages.append({"role": assistant_msg.role, "content": response_text})


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_argparser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI chat client for OpenRouter (OpenAI‑compatible API)."
    )
    parser.add_argument(
        "--api-key",
        default=defaults.get("api_key") or os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        default=defaults.get("model", "openai/gpt-4o"),
        help="Model name on OpenRouter, e.g. openai/gpt-4o",
    )
    parser.add_argument(
        "--referer",
        default=defaults.get("referer"),
        help="Optional HTTP‑Referer header for OpenRouter rankings.",
    )
    parser.add_argument(
        "--title",
        default=defaults.get("title"),
        help="Optional X‑Title header for OpenRouter rankings.",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="Base URL of the OpenRouter API (normally leave default).",
    )
    return parser


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    # Load any saved defaults from ~/.openrouter_chat.json
    config_defaults = load_config()

    parser = build_argparser(config_defaults)
    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        parser.error(
            "OpenRouter API key not supplied. Use --api-key, set OPENROUTER_API_KEY env var, or put it in ~/.openrouter_chat.json."
        )

    # Build extra headers dict (only include if values are provided)
    extra_headers = {}
    if args.referer:
        extra_headers["HTTP-Referer"] = args.referer
    if args.title:
        extra_headers["X-Title"] = args.title

    # Initialise the OpenAI‑compatible client that talks to OpenRouter
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Start the interactive chat
    try:
        chat_loop(client, args.model, extra_headers)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()