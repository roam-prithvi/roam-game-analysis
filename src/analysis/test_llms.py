"""
Simple CLI script that queries three different LLM providers
(Google Gemini, OpenAI GPT and Anthropic Claude) with a single prompt.

Environment variables required (already loaded in src.__init__):
• GEMINI_API_KEY
• OPENAI_API_KEY
• ANTHROPIC_API_KEY

Install dependencies:
    pip install google-generativeai openai anthropic
"""

from __future__ import annotations

import sys
from typing import Callable, Dict

try:
    from google import generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover – handled at runtime
    genai = None  # type: ignore

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

try:
    import anthropic  # type: ignore
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore

# Import API keys that were loaded in src.__init__
from src import ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY


PromptFn = Callable[[str], str]


def _call_openai(prompt: str) -> str:
    """Query OpenAI ChatCompletion (gpt-3.5-turbo) and return the reply text."""
    if openai is None:
        return "⚠️ openai python package not installed."

    # The v1 SDK uses `client = openai.OpenAI(...)` whereas v0 still uses the
    # global helpers.  We try to support both while keeping the code simple.
    try:
        # New SDK (>= 1.0)
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  # type: ignore[attr-defined]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except AttributeError:
        # Legacy SDK (< 1.0)
        openai.api_key = OPENAI_API_KEY  # type: ignore[attr-defined]
        response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["choices"][0]["message"]["content"].strip()


def _call_gemini(prompt: str) -> str:
    """Query Google Gemini-Pro and return the reply text."""
    if genai is None:
        return "⚠️ google-generativeai package not installed."

    # Configure the client (idempotent)
    genai.configure(api_key=GEMINI_API_KEY)  # type: ignore[arg-type]
    # should be 150 req/min according to https://ai.google.dev/gemini-api/docs/rate-limits#tier-1
    model = genai.GenerativeModel("gemini-2.5-pro-preview-06-05")  # type: ignore[attr-defined]
    response = model.generate_content(prompt)  # type: ignore[attr-defined]
    return (response.text or "").strip()


def _call_anthropic(prompt: str) -> str:
    """Query Anthropic Claude-3 Haiku and return the reply text."""
    if anthropic is None:
        return "⚠️ anthropic package not installed."

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)  # type: ignore[attr-defined]
    response = client.messages.create(  # type: ignore[attr-defined]
        model="claude-3-haiku-20240307",
        max_tokens=32,
        messages=[{"role": "user", "content": prompt}],
    )

    # `response.content` is a list of content blocks — concatenate them.
    chunks = getattr(response, "content", [])  # type: ignore[attr-defined]
    text_parts = [getattr(c, "text", str(c)) for c in chunks]
    return "".join(text_parts).strip()


def main() -> None:
    """Entry-point callable that runs the demo prompt against all providers."""

    prompt: str = "Please say exactly: this is a test"

    callers: Dict[str, PromptFn] = {
        "Gemini Pro 2.5 (Google)": _call_gemini,
        "GPT-3.5 (OpenAI)": _call_openai,
        "Claude-3 (Anthropic)": _call_anthropic,
    }

    for name, fn in callers.items():
        print(f"\n— {name} —")
        try:
            reply: str = fn(prompt)
        except Exception as exc:  # pragma: no cover – keep demo resilient
            print(f"❌ Error: {exc}", file=sys.stderr)
            continue
        print(reply)


if __name__ == "__main__":
    main()
