"""
Test Augmenter — generates targeted test augmentations for weak tasks.

Based on:
  - HardTestGen (ICLR 2026): LLM-synthesized harder tests, +11.22pp precision.
  - ACECoder (ACL 2025): Fully automated test synthesis, +25% on HumanEval-plus.
  - EvolveCoder (March 2026): Solution-conditioned adversarial test refinement.

For each task flagged as FIX:
  1. Analyzes the SPECIFIC exploit patches that passed Docker
  2. Identifies what the test suite doesn't check
  3. Generates test code that blocks those exploits while passing with the gold patch
  4. Validates the augmentation: gold patch passes, exploits fail

The augmented tests are Python test functions in unittest/pytest format,
ready to be appended to the existing test file.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from envaudit.llm.claude_client import ClaudeClient, ClaudeResponse

logger = logging.getLogger(__name__)


@dataclass
class TestAugmentation:
    """A generated test augmentation for a specific task."""
    instance_id: str
    test_code: str                    # Python test function(s)
    target_exploits: List[str]        # Exploit patch_ids this test is designed to block
    rationale: str                    # Why this test blocks the exploits
    expected_gold_pass: bool          # Should the gold patch pass this test?
    confidence: float                 # 0.0 to 1.0
    cost_usd: float


@dataclass
class AugmentationResult:
    """Result from augmenting a single task."""
    instance_id: str
    augmentations: List[TestAugmentation]
    total_cost_usd: float
    n_exploits_targeted: int          # How many exploit patches are targeted


AUGMENTER_SYSTEM_PROMPT = """You are a test engineer strengthening a test suite that has been shown to accept incorrect solutions.

You will be given:
1. The problem description (what the code should do)
2. The existing test code (what's already tested)
3. The gold patch (the correct solution)
4. Specific exploit patches that INCORRECTLY PASS the existing tests

Your job: write additional test functions that would BLOCK the exploits while still PASSING with the gold patch.

RULES:
1. Generate 1-3 test functions in Python (unittest or pytest style, matching the existing test style)
2. Each test must pass when the gold patch is applied
3. Each test must fail for at least one of the exploit patches
4. Tests should check BEHAVIORS, not implementation details
5. Include diverse inputs — different values, edge cases, types
6. Do NOT modify existing tests — only add new ones
7. Include necessary imports"""


AUGMENTER_PROMPT_TEMPLATE = """PROBLEM DESCRIPTION:
{problem_statement}

EXISTING TEST CODE (from test_patch):
{existing_tests}

GOLD PATCH (the correct fix):
{gold_patch}

EXPLOIT PATCHES THAT INCORRECTLY PASS THE EXISTING TESTS:
{exploit_details}

---

The exploits above pass the existing tests but do NOT correctly solve the problem.
Write additional test functions that would BLOCK these exploits.

Think about:
- What inputs would distinguish the gold patch from each exploit?
- What edge cases does the existing test miss?
- What behaviors should the correct solution exhibit that the exploits don't?

Return JSON:
{{
  "augmentations": [
    {{
      "test_code": "def test_...(self):\\n    ...",
      "rationale": "This test blocks exploit X because ...",
      "targets": ["exploit_0", "exploit_1"],
      "confidence": 0.85
    }}
  ]
}}"""


class TestAugmenter:
    """Generates targeted test augmentations for weak tasks."""

    def __init__(self, claude_client: ClaudeClient):
        self.claude = claude_client

    def augment_task(
        self,
        instance_id: str,
        problem_statement: str,
        existing_test_patch: str,
        gold_patch: str,
        passed_exploits: List[Dict[str, Any]],
    ) -> AugmentationResult:
        """Generate test augmentations for a task with known exploits.

        Args:
            instance_id: Task identifier.
            problem_statement: The GitHub issue description.
            existing_test_patch: Current test code (unified diff).
            gold_patch: The correct solution (unified diff).
            passed_exploits: List of dicts with keys:
                - patch_id: str
                - patch_text: str (unified diff)
                - strategy_name: str
                - strategy_description: str
        """
        if not passed_exploits:
            return AugmentationResult(
                instance_id=instance_id,
                augmentations=[],
                total_cost_usd=0.0,
                n_exploits_targeted=0,
            )

        exploit_details = self._format_exploits(passed_exploits)

        prompt = AUGMENTER_PROMPT_TEMPLATE.format(
            problem_statement=problem_statement[:3000],
            existing_tests=existing_test_patch[:5000],
            gold_patch=gold_patch[:4000],
            exploit_details=exploit_details,
        )

        response = self.claude.query(
            prompt=prompt,
            system=AUGMENTER_SYSTEM_PROMPT,
            temperature=0.3,
        )

        augmentations = self._parse_augmentations(
            response, instance_id, passed_exploits,
        )

        return AugmentationResult(
            instance_id=instance_id,
            augmentations=augmentations,
            total_cost_usd=response.cost_usd,
            n_exploits_targeted=len(passed_exploits),
        )

    def _format_exploits(self, exploits: List[Dict[str, Any]]) -> str:
        """Format exploit patches for the prompt."""
        parts = []
        for i, exp in enumerate(exploits):
            patch_text = exp.get("patch_text", "")
            if len(patch_text) > 2000:
                patch_text = patch_text[:2000] + "\n... (truncated)"
            parts.append(
                f"EXPLOIT {i} ({exp.get('strategy_name', 'unknown')}):\n"
                f"Description: {exp.get('strategy_description', 'N/A')}\n"
                f"Patch:\n{patch_text}\n"
            )
        return "\n---\n".join(parts)

    def _parse_augmentations(
        self,
        response: ClaudeResponse,
        instance_id: str,
        exploits: List[Dict[str, Any]],
    ) -> List[TestAugmentation]:
        """Parse augmentations from Claude's response."""
        augmentations = []
        parsed = response.parsed_json

        if parsed and "augmentations" in parsed:
            for aug in parsed["augmentations"]:
                test_code = aug.get("test_code", "")
                if not test_code.strip():
                    continue

                targets = aug.get("targets", [])
                # Map target names to exploit patch_ids
                target_ids = []
                for t in targets:
                    # Handle both "exploit_0" format and index
                    if isinstance(t, str) and t.startswith("exploit_"):
                        idx = int(t.split("_")[1]) if "_" in t else 0
                        if idx < len(exploits):
                            target_ids.append(exploits[idx].get("patch_id", t))
                    else:
                        target_ids.append(str(t))

                augmentations.append(TestAugmentation(
                    instance_id=instance_id,
                    test_code=test_code,
                    target_exploits=target_ids or [e["patch_id"] for e in exploits],
                    rationale=aug.get("rationale", ""),
                    expected_gold_pass=True,
                    confidence=min(max(float(aug.get("confidence", 0.7)), 0.0), 1.0),
                    cost_usd=response.cost_usd / max(len(parsed["augmentations"]), 1),
                ))
        else:
            logger.warning("Failed to parse augmentations for %s", instance_id)

        return augmentations
