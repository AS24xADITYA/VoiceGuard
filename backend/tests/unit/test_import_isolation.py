"""Test that the ai/ package does not import from app/.

Per 04 §1: ai/ must not import from api/, db/, or any web framework.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path


def test_ai_does_not_import_app():
    """Walk every .py file under ai/ and assert no import of app or web frameworks."""
    ai_root = Path(__file__).resolve().parent.parent.parent / "ai"
    assert ai_root.exists(), f"ai/ directory not found at {ai_root}"

    forbidden_modules = {"app", "fastapi", "starlette", "uvicorn"}
    violations = []

    for root, _dirs, files in os.walk(ai_root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            filepath = Path(root) / fname
            source = filepath.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(filepath))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in forbidden_modules:
                            violations.append(
                                f"{filepath.relative_to(ai_root)}:{node.lineno} "
                                f"imports '{alias.name}'"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top in forbidden_modules:
                            violations.append(
                                f"{filepath.relative_to(ai_root)}:{node.lineno} "
                                f"imports from '{node.module}'"
                            )

    assert not violations, (
        "ai/ must not import from app/ or web frameworks.\nViolations:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
