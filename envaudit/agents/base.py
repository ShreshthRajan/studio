"""
Base agent class adapted from multiver.
Provides common interface for all envaudit agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import time


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Issue:
    """A single finding from an audit agent."""
    type: str
    severity: Severity
    message: str
    confidence: float = 1.0
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    agent_source: Optional[str] = None
    exploit_patch: Optional[str] = None  # For hackability: the actual exploit code


@dataclass
class AgentResult:
    """Result from an audit agent."""
    agent_name: str
    verdict: str  # "PASS", "WARNING", "FAIL"
    score: float  # 0.0 to 1.0
    issues: List[Issue]
    metrics: Dict[str, Any]
    confidence: float
    execution_time: float
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class TaskData:
    """A single SWE-bench task to audit."""
    instance_id: str
    repo: str
    problem_statement: str
    test_patch: str  # Unified diff of test files
    gold_patch: str  # Unified diff of the gold fix
    fail_to_pass: List[str]  # Pytest node IDs that should go from fail -> pass
    pass_to_pass: List[str]  # Pytest node IDs that must keep passing
    is_verified: bool = False
    difficulty: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base class for all envaudit agents."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    async def analyze(self, task: TaskData) -> AgentResult:
        start = time.time()
        try:
            result = await self._analyze(task)
            result.execution_time = time.time() - start
            return result
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                verdict="ERROR",
                score=0.0,
                issues=[],
                metrics={"error": str(e)},
                confidence=0.0,
                execution_time=time.time() - start,
                error=str(e),
            )

    @abstractmethod
    async def _analyze(self, task: TaskData) -> AgentResult:
        pass
