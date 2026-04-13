"""
Semi-Formal Correctness Judge — independent patch correctness assessment.

Based on: "Agentic Code Reasoning" (Meta, March 2026, arXiv:2603.01896).

Semi-formal reasoning requires the judge to:
  1. State explicit premises (what the problem asks for)
  2. Trace execution paths through the patch
  3. Derive formal conclusions (correct / incorrect / partial)

Unlike raw LLM-as-judge, the structured format acts as a certificate:
the judge cannot skip cases or make unsupported claims. Meta reports
93% accuracy on real-world agent-generated patches.

This module provides an independent correctness signal to compare
against the test suite verdict, enabling detection of:
  - False positives: tests pass but patch is incorrect (hackable)
  - False negatives: tests fail but patch is correct (overly strict)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

from envaudit.llm.claude_client import ClaudeClient, ClaudeResponse

logger = logging.getLogger(__name__)


class CorrectnessVerdict(Enum):
    """Judge's assessment of patch correctness."""
    CORRECT = "correct"         # Patch fully solves the described problem
    PARTIAL = "partial"         # Patch addresses part of the problem
    INCORRECT = "incorrect"     # Patch does not solve the problem
    UNCERTAIN = "uncertain"     # Judge cannot determine


@dataclass
class JudgmentResult:
    """Result from the semi-formal correctness judge."""
    verdict: CorrectnessVerdict
    premises: str               # What the problem requires
    trace: str                  # Execution path analysis
    conclusion: str             # Formal conclusion
    confidence: float           # 0.0 to 1.0
    cost_usd: float
    latency_seconds: float


SEMIFORMAL_SYSTEM_PROMPT = """You are a code correctness judge using semi-formal reasoning.

You will assess whether a code patch correctly solves a described problem.
You MUST follow this structured reasoning format — no shortcuts allowed.

IMPORTANT: Your judgment must be INDEPENDENT of whether the patch passes any test suite.
Focus on whether the patch addresses the ROOT CAUSE described in the problem."""


SEMIFORMAL_PROMPT_TEMPLATE = """PROBLEM DESCRIPTION:
{problem_statement}

PATCH TO JUDGE (unified diff):
{patch_text}

GOLD PATCH (the known-correct solution, for reference):
{gold_patch_summary}

---

Assess this patch using semi-formal reasoning. You MUST complete ALL three sections.

## PREMISES
State exactly what the problem requires. What behavior is broken? What must change?
List each requirement as a numbered premise.

## TRACE
For each premise, trace through the patch code and determine:
- Does the patch modify the correct file(s) and function(s)?
- Does the patch address this specific requirement?
- Could the patch introduce any side effects or regressions?
Be specific — reference line numbers and variable names from the patch.

## CONCLUSION
Based on the premises and trace, provide your formal conclusion.

Return JSON:
{{
  "verdict": "correct|partial|incorrect",
  "premises": "P1: ... P2: ...",
  "trace": "For P1: ... For P2: ...",
  "conclusion": "The patch is [correct/partial/incorrect] because ...",
  "confidence": 0.85
}}"""


def judge_patch(
    claude_client: ClaudeClient,
    problem_statement: str,
    patch_text: str,
    gold_patch: str,
    temperature: float = 0.2,
) -> JudgmentResult:
    """Judge a single patch using semi-formal reasoning.

    Args:
        claude_client: Claude API client.
        problem_statement: The GitHub issue description.
        patch_text: The unified diff to judge.
        gold_patch: The known-correct gold patch for reference.
        temperature: Low temperature for consistent reasoning.

    Returns:
        JudgmentResult with structured verdict.
    """
    # Truncate inputs to fit context
    problem_stmt = problem_statement[:3000]
    gold_summary = _summarize_patch(gold_patch, max_lines=40)
    patch_to_judge = patch_text[:6000] if patch_text else "(empty patch)"

    prompt = SEMIFORMAL_PROMPT_TEMPLATE.format(
        problem_statement=problem_stmt,
        patch_text=patch_to_judge,
        gold_patch_summary=gold_summary,
    )

    response = claude_client.query(
        prompt=prompt,
        system=SEMIFORMAL_SYSTEM_PROMPT,
        temperature=temperature,
    )

    return _parse_judgment(response)


def judge_patch_with_self_consistency(
    claude_client: ClaudeClient,
    problem_statement: str,
    patch_text: str,
    gold_patch: str,
    n_samples: int = 3,
    temperatures: tuple = (0.2, 0.4, 0.6),
) -> JudgmentResult:
    """Judge a patch with self-consistency (multiple samples, majority vote).

    Adapted from multiver's self-consistency mechanism.
    Samples the same judgment at different temperatures and takes
    the majority verdict. Confidence = votes_for_winner / n_samples.
    """
    results = []
    total_cost = 0.0
    total_latency = 0.0

    for i in range(min(n_samples, len(temperatures))):
        result = judge_patch(
            claude_client, problem_statement, patch_text,
            gold_patch, temperature=temperatures[i],
        )
        results.append(result)
        total_cost += result.cost_usd
        total_latency += result.latency_seconds

    if not results:
        return _uncertain_judgment(total_cost, total_latency)

    # Majority vote on verdict
    verdict_counts: Dict[CorrectnessVerdict, int] = {}
    for r in results:
        verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1

    winner = max(verdict_counts, key=verdict_counts.get)
    vote_confidence = verdict_counts[winner] / len(results)

    # Use the reasoning from the highest-confidence sample with the winning verdict
    best_result = max(
        (r for r in results if r.verdict == winner),
        key=lambda r: r.confidence,
    )

    return JudgmentResult(
        verdict=winner,
        premises=best_result.premises,
        trace=best_result.trace,
        conclusion=best_result.conclusion,
        confidence=round(vote_confidence * best_result.confidence, 4),
        cost_usd=total_cost,
        latency_seconds=total_latency,
    )


def _parse_judgment(response: ClaudeResponse) -> JudgmentResult:
    """Parse a semi-formal judgment from Claude's response."""
    parsed = response.parsed_json

    if parsed:
        verdict_str = parsed.get("verdict", "uncertain").lower().strip()
        verdict_map = {
            "correct": CorrectnessVerdict.CORRECT,
            "partial": CorrectnessVerdict.PARTIAL,
            "incorrect": CorrectnessVerdict.INCORRECT,
        }
        verdict = verdict_map.get(verdict_str, CorrectnessVerdict.UNCERTAIN)
        confidence = min(max(float(parsed.get("confidence", 0.5)), 0.0), 1.0)

        return JudgmentResult(
            verdict=verdict,
            premises=str(parsed.get("premises", "")),
            trace=str(parsed.get("trace", "")),
            conclusion=str(parsed.get("conclusion", "")),
            confidence=confidence,
            cost_usd=response.cost_usd,
            latency_seconds=response.latency_seconds,
        )

    # Fallback: try to extract verdict from raw text
    content_lower = response.content.lower()
    if "the patch is correct" in content_lower or "verdict: correct" in content_lower:
        verdict = CorrectnessVerdict.CORRECT
    elif "the patch is incorrect" in content_lower or "verdict: incorrect" in content_lower:
        verdict = CorrectnessVerdict.INCORRECT
    elif "the patch is partial" in content_lower or "verdict: partial" in content_lower:
        verdict = CorrectnessVerdict.PARTIAL
    else:
        verdict = CorrectnessVerdict.UNCERTAIN

    return JudgmentResult(
        verdict=verdict,
        premises="(failed to parse structured response)",
        trace="(failed to parse structured response)",
        conclusion=response.content[:500],
        confidence=0.4,
        cost_usd=response.cost_usd,
        latency_seconds=response.latency_seconds,
    )


def _uncertain_judgment(cost: float, latency: float) -> JudgmentResult:
    return JudgmentResult(
        verdict=CorrectnessVerdict.UNCERTAIN,
        premises="",
        trace="",
        conclusion="No judgment samples available",
        confidence=0.0,
        cost_usd=cost,
        latency_seconds=latency,
    )


def _summarize_patch(patch: str, max_lines: int = 40) -> str:
    """Summarize a patch for the prompt, truncating if needed."""
    lines = patch.split("\n")
    if len(lines) <= max_lines:
        return patch
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
