"""
Smoke test: verify that the Anthropic API key is configured correctly
and the SDK can make a successful call to Claude.

Usage:
    python scripts/test_anthropic_connection.py

Reads ANTHROPIC_API_KEY from .env via python-dotenv.
"""

import os
import sys
import time

from anthropic import Anthropic
from dotenv import load_dotenv


def main() -> int:
    # Load environment variables from .env (if present)
    load_dotenv()

    # Verify the key is actually loaded
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Check your .env file.")
        return 1
    if not api_key.startswith("sk-ant-"):
        print("ERROR: ANTHROPIC_API_KEY has unexpected format.")
        return 1
    print(f"Key loaded successfully (length {len(api_key)} chars, starts with {api_key[:15]}...)")

    # Make a minimal API call
    client = Anthropic()  # SDK reads ANTHROPIC_API_KEY from environment automatically

    start_time = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: 'FinDirector API connection OK'",
            }
        ],
    )
    elapsed = time.time() - start_time

    response_text = response.content[0].text
    print(f"\nResponse: {response_text}")
    print(f"\nLatency: {elapsed:.2f}s")
    print(f"Tokens — input: {response.usage.input_tokens}, output: {response.usage.output_tokens}")

    # Cost estimate (Sonnet 4.5: $3/M input, $15/M output as of 2025)
    cost = (response.usage.input_tokens / 1_000_000) * 3.0 + (response.usage.output_tokens / 1_000_000) * 15.0
    print(f"Estimated cost: ${cost:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
