"""
Claude API client for envaudit.
Adapted from multiver's opus_client.py.
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class ClaudeResponse:
    """Parsed Claude response."""
    content: str
    parsed_json: Optional[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float
    model: str


# Pricing per 1M tokens (as of March 2026)
PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}


class ClaudeClient:
    """Synchronous Claude client with cost tracking."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required: pip install anthropic")

        self.model = model
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.total_cost = 0.0
        self.total_calls = 0

    def query(self, prompt: str, system: Optional[str] = None,
              temperature: float = 0.3) -> ClaudeResponse:
        """Send a query to Claude and parse the response."""
        t0 = time.time()

        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        latency = time.time() - t0
        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        pricing = PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        self.total_cost += cost
        self.total_calls += 1

        # Try to parse JSON from the response
        parsed = _extract_json(content)

        return ClaudeResponse(
            content=content,
            parsed_json=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_seconds=latency,
            model=self.model,
        )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from Claude's response, handling markdown code blocks."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from ```json ... ``` blocks
    import re
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
