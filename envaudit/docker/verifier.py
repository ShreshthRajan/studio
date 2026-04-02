"""
Docker-based exploit verification using SWE-bench harness.

Applies a generated exploit patch to the actual codebase inside a Docker
container and runs the FAIL_TO_PASS tests. If the tests pass, the exploit
is VERIFIED — the test suite is provably hackable.

Requires:
    - Docker Desktop installed and running
    - swebench package installed (pip install swebench)
    - ~120GB free disk for Docker images
    - macOS M-series: uses --namespace '' for local ARM builds
"""

import json
import subprocess
import tempfile
import os
import shutil
import logging
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of Docker-based exploit verification."""
    instance_id: str
    exploit_name: str
    resolved: bool  # True = exploit passed the tests (test suite IS hackable)
    error: Optional[str] = None
    log_path: Optional[str] = None


def is_docker_available() -> bool:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def verify_exploit(
    instance_id: str,
    exploit_patch: str,
    exploit_name: str = "exploit",
    timeout_seconds: int = 600,
    run_id: Optional[str] = None,
) -> VerificationResult:
    """Verify an exploit patch by running it through SWE-bench Docker harness.

    Args:
        instance_id: SWE-bench instance ID (e.g., "django__django-11099")
        exploit_patch: Unified diff string to apply as the "solution"
        exploit_name: Name for logging
        timeout_seconds: Max time for Docker eval (default 10 min)
        run_id: Unique run identifier

    Returns:
        VerificationResult with resolved=True if exploit passes tests
    """
    if not is_docker_available():
        return VerificationResult(
            instance_id=instance_id,
            exploit_name=exploit_name,
            resolved=False,
            error="Docker not available. Install Docker Desktop to enable verification.",
        )

    run_id = run_id or f"envaudit_{instance_id}_{exploit_name}"

    # Create temp directory for predictions
    tmpdir = tempfile.mkdtemp(prefix="envaudit_")
    pred_path = os.path.join(tmpdir, "predictions.jsonl")
    try:
        # Write prediction JSONL
        pred = {
            "instance_id": instance_id,
            "model_name_or_path": f"envaudit-{exploit_name}",
            "model_patch": exploit_patch,
        }
        with open(pred_path, "w") as f:
            f.write(json.dumps(pred) + "\n")

        # Build command — use Python 3.10+ for swebench (requires 3.10+ syntax)
        python_bin = _find_python310()
        cmd = [
            python_bin, "-m", "swebench.harness.run_evaluation",
            "--predictions_path", pred_path,
            "--instance_ids", instance_id,
            "--max_workers", "1",
            "--run_id", run_id,
        ]

        # macOS ARM: build images locally
        if platform.machine() in ("arm64", "aarch64"):
            cmd.extend(["--namespace", ""])

        logger.info(f"Running Docker verification for {instance_id} ({exploit_name})...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=tmpdir,
        )

        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            logger.warning(f"Docker eval failed for {instance_id}: {error_msg}")
            return VerificationResult(
                instance_id=instance_id,
                exploit_name=exploit_name,
                resolved=False,
                error=f"Harness returned code {result.returncode}: {error_msg}",
            )

        # Parse results — check if the instance was "resolved"
        resolved = _check_resolved(tmpdir, run_id, instance_id)

        return VerificationResult(
            instance_id=instance_id,
            exploit_name=exploit_name,
            resolved=resolved,
            log_path=tmpdir,
        )

    except subprocess.TimeoutExpired:
        return VerificationResult(
            instance_id=instance_id,
            exploit_name=exploit_name,
            resolved=False,
            error=f"Docker verification timed out after {timeout_seconds}s",
        )
    except Exception as e:
        return VerificationResult(
            instance_id=instance_id,
            exploit_name=exploit_name,
            resolved=False,
            error=str(e),
        )


def _find_python310() -> str:
    """Find a Python 3.10+ binary that has swebench installed."""
    import shutil
    import subprocess
    # Check candidates in order, verify swebench is importable
    candidates = [
        "/opt/homebrew/bin/python3.10",
        "/opt/homebrew/bin/python3.11",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.13",
    ]
    # Also check PATH
    for name in ["python3.10", "python3.11", "python3.12", "python3.13"]:
        p = shutil.which(name)
        if p and p not in candidates:
            candidates.append(p)

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            r = subprocess.run(
                [candidate, "-c", "import swebench; print('ok')"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and "ok" in r.stdout:
                logger.info(f"Using {candidate} for swebench harness")
                return candidate
        except Exception:
            continue

    logger.warning("No Python 3.10+ with swebench found, falling back to python3")
    return "python3"


def _check_resolved(tmpdir: str, run_id: str, instance_id: str) -> bool:
    """Check SWE-bench evaluation results for resolution status."""
    # SWE-bench writes results to multiple possible locations
    possible_paths = [
        Path(tmpdir) / run_id / f"{instance_id}.{run_id}.json",
        Path(tmpdir) / f"{run_id}.json",
        Path(tmpdir) / "results" / f"{run_id}.json",
    ]

    # Also check the standard swebench output format
    for root, dirs, files in os.walk(tmpdir):
        for f in files:
            if f.endswith(".json") and run_id in f:
                possible_paths.append(Path(root) / f)

    for path in possible_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                # Check various result formats
                if isinstance(data, dict):
                    if instance_id in data.get("resolved", []):
                        return True
                    if data.get(instance_id, {}).get("resolved", False):
                        return True
                    if data.get("resolved_instances"):
                        if instance_id in data["resolved_instances"]:
                            return True
            except (json.JSONDecodeError, KeyError):
                continue

    # Check stdout logs for "RESOLVED" keyword as fallback
    log_files = list(Path(tmpdir).rglob("*.log"))
    for lf in log_files:
        try:
            content = lf.read_text()
            if instance_id in content and "RESOLVED" in content:
                return True
        except Exception:
            continue

    return False
