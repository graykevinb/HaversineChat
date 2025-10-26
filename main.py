#!/usr/bin/env python3
"""
OpenRouter CLI chat with extended tools:

1. search_gutenberg_books – search Project Gutenberg (Gutendex API)
2. python_execute        – execute any Python snippet (no sandbox)
3. fetch_url             – fetch the raw content of a URL (GET request)
4. google_search         – run a Google Custom Search query (requires API key)

Features
--------
* Strong system prompt with concrete usage examples.
* If the model calls `python_execute` without a `code` argument,
  the assistant asks the model to supply the missing snippet.
* When `fetch_url` is used, the assistant automatically converts HTML → plain text
  and then asks the model to **summarise** the page.  If the model still says
  nothing, the cleaned‑up text is shown in a fenced block so you never get an empty
  reply.
* **New:** Results from `google_search` are rendered immediately as markdown.
* Rich UI: markdown, syntax‑highlighted code, **LaTeX → Unicode** rendering.
* Google Search API key and Custom Search Engine ID are taken from environment
  variables `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`.
"""

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import requests
from openai import OpenAI
from tqdm import tqdm
from rich.console import Console
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from sympy import pretty
from sympy.parsing.latex import parse_latex

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
# OPTIONAL IMPORT HELPER – ensures bs4 is available
# ----------------------------------------------------------------------
def _ensure_bs4():
    """Install beautifulsoup4 on‑the‑fly if it is missing."""
    try:
        import bs4  # noqa: F401
    except ImportError:  # pragma: no cover
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
        import bs4  # noqa: F401


_ensure_bs4()

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
    """Execute any Python snippet (no sandbox)."""
    import io
    import contextlib

    debug_line = f"Executing Python code:\n{code}\n"

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            print(debug_line, end="")  # echo the snippet first
            exec(code, {}, {})  # unrestricted exec
        result = stdout.getvalue().strip()
        return result if result else "(no output)"
    except Exception:
        tb = traceback.format_exc()
        return f"❌ python_execute raised an exception:\n{tb.splitlines()[-2:]}"


def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch the raw HTML of a URL via GET."""
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "OpenRouterChat/1.0"},
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        return f"⚠️  Failed to fetch {url!r}: {exc}"


def html_to_text(html: str, max_chars: int = 2000) -> str:
    """
    Convert HTML → clean plain‑text.
    - Removes script / style / noscript / svg / head / header / footer tags.
    - Strips HTML comments.
    - Collapses whitespace.
    - Truncates to *max_chars* characters (adds an ellipsis if trimmed).
    """
    from bs4 import BeautifulSoup, Comment

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:               # lxml not installed or parsing failed
        soup = BeautifulSoup(html, "html.parser")

    for tag_name in [
        "script",
        "style",
        "noscript",
        "svg",
        "head",
        "header",
        "footer",
        "meta",
        "link",
        "iframe",
        "form",
        "input",
        "button",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def google_search(query: str, num_results: int = 5) -> str:
    """
    Perform a Google Custom Search.
    Requires the env vars ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_ID``.
    Returns a short, readable list of result titles and snippets.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        return (
            "❌ Google Search is not configured. Please set the environment variables "
            "`GOOGLE_API_KEY` and `GOOGLE_CSE_ID`."
        )

    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": min(max(num_results, 1), 10),  # Google caps at 10 per request
    }

    try:
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return "🔎 No results found."

        lines = []
        for i, item in enumerate(items, 1):
            title = item.get("title", "No title")
            snippet = re.sub(r"\s+", " ", item.get("snippet", "")).strip()
            link = item.get("link", "")
            lines.append(f"{i}. {title}\n   {snippet}\n   {link}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"⚠️  Google Search failed: {exc}"


# ----------------------------------------------------------------------
# TOOL REGISTRY
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
    "fetch_url": {
        "func": fetch_url,
        "description": "Fetch the raw HTML of a URL (GET request).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The URL to fetch.",
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                    "default": 10,
                    "description": "Request timeout in seconds.",
                },
            },
            "required": ["url"],
        },
    },
    "google_search": {
        "func": google_search,
        "description": "Run a Google Custom Search query (requires API key).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string.",
                },
                "num_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Maximum number of results to return (Google caps at 10).",
                },
            },
            "required": ["query"],
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
# LaTeX → Unicode renderer (SymPy)
# ----------------------------------------------------------------------
def _pretty_latex(expr: str) -> str:
    """Convert LaTeX → pretty Unicode; fall back to raw LaTeX on error."""
    try:
        sym_expr = parse_latex(expr)
        return pretty(sym_expr, use_unicode=True)
    except Exception:
        return expr


def render_latex_in_text(text: str) -> str:
    """Replace LaTeX delimiters with pretty‑printed Unicode."""
    def repl_display(m):
        expr = m.group(1).strip()
        return "\n" + _pretty_latex(expr) + "\n"

    def repl_inline(m):
        expr = m.group(1).strip()
        return _pretty_latex(expr)

    text = re.sub(r'\\\[(.+?)\\\]', repl_display, text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.+?)\$\$', repl_display, text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', repl_inline, text)
    text = re.sub(r'\$(.+?)\$', repl_inline, text)
    return text


# ----------------------------------------------------------------------
# UI helpers (Rich + Prompt‑Toolkit)
# ----------------------------------------------------------------------
console = Console()
session = PromptSession()   # replaces built‑in input()


def rich_print(text: str, prefix: str = "🤖"):
    """
    Render markdown (including LaTeX) and code fences.
    If the text looks like raw HTML, treat it as plain text.
    """
    text = render_latex_in_text(text)

    # Simple HTML detection – if we see any <tag>, just print as plain text.
    if re.search(r'<[^>]+>', text):
        lines = text.splitlines()
        if lines:
            console.print(f"{prefix} {lines[0]}")
            if len(lines) > 1:
                console.print("\n".join(lines[1:]))
        else:
            console.print(f"{prefix} (empty)")
        return

    # Normal markdown rendering
    lines = text.splitlines()
    if not lines:
        return
    console.print(f"{prefix} {lines[0]}")
    if len(lines) > 1:
        console.print(Markdown("\n".join(lines[1:]), code_theme="monokai"))


# ----------------------------------------------------------------------
# Helper: deterministic fallback summary for fetch_url
# ----------------------------------------------------------------------
def _fallback_summary(
    text: str, client: OpenAI, model: str, extra_headers: dict
) -> str:
    """
    Ask the LLM once more with an explicit “summarise in 2‑3 sentences”
    prompt.  If that still returns nothing, return a concise raw excerpt.
    """
    prompt = (
        "Please give a concise 2‑sentence summary of the following page content. "
        "Do not add any commentary, just the summary.\n\n"
        f"=== PAGE CONTENT ===\n{text}\n=== END ==="
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_headers=extra_headers,
            temperature=0.0,
        )
        summary = resp.choices[0].message.content
        if summary and summary.strip():
            return summary.strip()
    except Exception:
        pass
    # Last‑ditch fallback
    return (
        f"(No summary could be generated – here is the raw excerpt)\n"
        f"```text\n{text}\n```"
    )


# ----------------------------------------------------------------------
# MAIN chat loop
# ----------------------------------------------------------------------
def chat_loop(
    client: OpenAI,
    model: str,
    extra_headers: Dict[str, str],
    enable_tools: bool,
) -> None:
    messages: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------------
    # System prompt (robust, with checklist & explicit examples)
    # ----------------------------------------------------------------------
    SYSTEM_PROMPT = """You are a helpful, truthful assistant that can call functions (tools) when they are the **best way** to satisfy the user’s request.

    ### General rules
    1. **Never fabricate a tool call.**  
    - Call a tool **only** if the user explicitly asks for an action that the tool can perform (e.g. “run this Python code”, “search Gutenberg”, “fetch a web page”, “search Google”, **or ask for a website summary / any information from a URL**).  
    - If the request can be answered directly from your knowledge, reply without a tool.

    2. **Always obey the function signature.**  
    - Use the exact parameter names and types defined in the function schema.  
    - If a required argument is missing, **first ask the user** for that argument; do **not** call the tool with an empty value.

    3. **One tool per turn.**  
    - After a tool call the model receives the result in a `tool` message and must then produce the final answer (or ask a follow‑up clarification).

    4. **Never expose raw HTML to the user.**  
    - For `fetch_url` convert the response to plain‑text with `html_to_text` before showing it.  
    - If the model fails to summarise the page, you must return a short fallback excerpt yourself inside a fenced code block.

    ### Tool‑use checklist (run this mental checklist before you decide)
    - **python_execute** – Does the user want *code* to be run or a numeric result?  
    - If only a description is given, ask “Please provide the exact Python snippet (use `print()` for the output).”
    - **search_gutenberg_books** – Does the user ask for book titles, authors or genres from Project Gutenberg?  
    - Build an **array** of search terms; never send a single string.
    - **fetch_url** – Does the user:
    * request the **contents of a web page**,  
    * ask for a **summary**,  
    * want to **extract specific information**,  
    * or any other **related task** that requires reading the page?  
    - Provide the URL as the `url` argument; optional `timeout` only if the user requests a longer wait.
    - **google_search** – Does the user want a web‑search result set?  
    - Supply `query`; include `num_results` only if the user specifies a limit (1‑10).

    ### Concrete usage examples (the assistant must follow this exact pattern)

    **User:** `run 2**8 in python`  
    **Assistant:** *(calls `python_execute` with `{ "code": "print(2**8)" }`)*  

    **User:** `show me three romance books from Gutenberg`  
    **Assistant:** *(calls `search_gutenberg_books` with `{ "search_terms": ["romance"] }`)*  

    **User:** `fetch the front page of https://example.com`  
    **Assistant:** *(calls `fetch_url` with `{ "url": "https://example.com" }`)*  

    **User:** `summarise the article at https://news.ycombinator.com/item?id=4000000`  
    **Assistant:** *(calls `fetch_url` with `{ "url": "https://news.ycombinator.com/item?id=4000000" }`)*  

    **User:** `search for Python tutorials on Google`  
    **Assistant:** *(calls `google_search` with `{ "query": "Python tutorials" }`)*  

    ### Available tools
    """  # the list of tools will be appended program‑matically below

    # Dynamically add the up‑to‑date tool list (keeps descriptions in sync)
    SYSTEM_PROMPT += "\n".join(
        f"- {name}: {meta['description']}"
        for name, meta in tool_registry.items()
    )

    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    console.print("\n💬  OpenRouter chat – type your message, `exit` to quit.")
    console.print(
        "   (Tools are enabled)" if enable_tools else "   (Tools are disabled)"
    )

    tools_supported = enable_tools

    while True:
        # ----------------------- Get user input -----------------------
        try:
            user_input = session.prompt("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n👋  Bye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            console.print("👋  Bye!")
            break
        if not user_input:
            continue

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
            with tqdm(total=0, bar_format="⏳ {desc}", leave=False, colour="cyan") as sp:
                sp.set_description_str("Calling model…")
                resp = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            console.print(f"\n❌ API error: {exc}\n")
            messages.pop()          # discard user message so they can retry
            continue

        choice = resp.choices[0]

        # ----------------------- Did the model request a tool? -----------------------
        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            func_name = tool_call.function.name
            raw_args = tool_call.function.arguments

            # Parse arguments (may be empty)
            try:
                args = json.loads(raw_args) if raw_args else {}
            except Exception:
                args = {}

            # ---- If python_execute is missing `code`, ask the model for it ----
            if func_name == "python_execute" and not args.get("code"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I need the exact Python code you want me to run. "
                            "Please provide the snippet (use `print()` if you want output)."
                        ),
                    }
                )
                continue
            # ----------------------------------------------------------------

            console.print(f"\n🛠️  Model wants to run `{func_name}` with args {args}")

            # ----------------------- Run the tool -----------------------
            if func_name not in tool_registry:
                tool_result = f"❌ Unknown tool `{func_name}`."
            else:
                try:
                    tool_result = tool_registry[func_name]["func"](**args)
                except Exception as exc:
                    tb = traceback.format_exc()
                    tool_result = f"❌ Error while executing `{func_name}`:\n{tb}"

            # For fetch_url we want the *plain‑text* version that the model will see.
            model_visible_result = tool_result
            if func_name == "fetch_url":
                model_visible_result = html_to_text(tool_result)

            # ------------------------------------------------------------------
            # Special handling for tools that should **immediately** be shown
            # ------------------------------------------------------------------
            if func_name == "google_search":
                # Show the Google results straight away (markdown rendering)
                rich_print(model_visible_result, prefix="🔍")
                messages.append(
                    {"role": "assistant", "content": model_visible_result}
                )
                continue  # back to top of loop – no follow‑up LLM call needed

            # Insert tool result into the conversation (the model sees the cleaned text)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": model_visible_result,
                }
            )

            # ----------------------- Second request (model sees tool output) -----------------------
            try:
                with tqdm(total=0, bar_format="⏳ {desc}", leave=False, colour="cyan") as sp:
                    sp.set_description_str("Calling model (after tool)…")
                    follow_up = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        extra_headers=extra_headers,
                        temperature=0.7,
                    )
                final_answer = follow_up.choices[0].message.content

                # --------------------------------------------------------------
                # Special handling for fetch_url: guarantee a summary
                # --------------------------------------------------------------
                if func_name == "fetch_url":
                    final_answer = follow_up.choices[0].message.content
                    if not final_answer or not final_answer.strip():
                        final_answer = _fallback_summary(
                            model_visible_result, client, model, extra_headers
                        )
                    if not final_answer or not final_answer.strip():
                        fallback = (
                            f"(Tool `{func_name}` result)\n"
                            f"```text\n{model_visible_result}\n```"
                        )
                        rich_print(fallback)
                        messages.append({"role": "assistant", "content": fallback})
                        continue

                    final_answer = final_answer.strip()
                    rich_print(final_answer)
                    messages.append({"role": "assistant", "content": final_answer})
                    continue  # go back to top of loop

                # ----------------------- Non‑fetch tool paths -----------------------
                if final_answer is None or not final_answer.strip():
                    fallback = (
                        f"(Tool `{func_name}` result)\n"
                        f"```text\n{model_visible_result}\n```"
                    )
                    rich_print(fallback)
                    messages.append({"role": "assistant", "content": fallback})
                else:
                    final_answer = final_answer.strip()
                    rich_print(final_answer)
                    messages.append({"role": "assistant", "content": final_answer})

            except Exception as exc:
                console.print(f"\n❌ Follow‑up API error: {exc}\n")
            continue

        # ----------------------- Plain text reply (no tool) -----------------------
        answer = choice.message.content.strip()
        rich_print(answer)
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
        console.print("\n👋  Bye!")


if __name__ == "__main__":
    main()