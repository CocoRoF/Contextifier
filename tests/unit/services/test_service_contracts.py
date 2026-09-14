# tests/unit/services/test_service_contracts.py
"""
Static contract check between handlers and the shared services.

Handlers call the services through ``self._image_service`` / ``self._tag_service``
and friends, and most of those call sites are wrapped in ``except Exception``
so that one bad image cannot abort a document. That safety net also hides a
typo: a call to a method that does not exist, or a keyword the service never
declared, degrades silently instead of failing.

That is not hypothetical — before this test existed, image extraction was dead
in seven handlers (``save_and_tag(image_bytes=…)`` against a parameter named
``image_data``) and page/slide/sheet tagging was dead in five
(``make_page_tag`` / ``page_tag`` against ``create_page_tag``), with a green
test suite throughout, because the test doubles were plain MagicMocks that
accepted anything.

This walks the AST of every module and verifies each service call against the
real class via ``inspect.signature``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Dict, List, Type

from contextifier.services.chart_service import ChartService
from contextifier.services.image_service import ImageService
from contextifier.services.metadata_service import MetadataService
from contextifier.services.table_service import TableService
from contextifier.services.tag_service import TagService

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "contextifier"

SERVICE_ATTRS: Dict[str, Type] = {
    "image_service": ImageService,
    "tag_service": TagService,
    "chart_service": ChartService,
    "table_service": TableService,
    "metadata_service": MetadataService,
}


def _service_for(attr_name: str) -> Type | None:
    """Map ``_image_service`` / ``image_service`` → the service class."""
    return SERVICE_ATTRS.get(attr_name.lstrip("_"))


def _receiver_name(node: ast.Attribute) -> str | None:
    """Return the attribute/variable name the call is made on."""
    owner = node.value
    if (
        isinstance(owner, ast.Attribute)
        and isinstance(owner.value, ast.Name)
        and owner.value.id == "self"
    ):
        return owner.attr
    if isinstance(owner, ast.Name):
        return owner.id
    return None


def _collect_violations() -> List[str]:
    violations: List[str] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(PACKAGE_ROOT.parent)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue

            receiver = _receiver_name(node.func)
            if receiver is None:
                continue
            service = _service_for(receiver)
            if service is None:
                continue

            method_name = node.func.attr
            if method_name.startswith("_"):
                continue  # private helpers are not part of the contract

            method = getattr(service, method_name, None)
            if method is None or not inspect.isfunction(method):
                violations.append(
                    f"{rel}:{node.lineno}: {receiver}.{method_name}() "
                    f"does not exist on {service.__name__}"
                )
                continue

            params = inspect.signature(method).parameters
            accepts_var_kw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            for keyword in node.keywords:
                if keyword.arg is None:  # **kwargs splat — cannot check statically
                    continue
                if keyword.arg not in params and not accepts_var_kw:
                    valid = ", ".join(p for p in params if p != "self")
                    violations.append(
                        f"{rel}:{node.lineno}: {receiver}.{method_name}("
                        f"{keyword.arg}=…) is not a parameter (valid: {valid})"
                    )

    return violations


def test_handlers_call_services_with_real_signatures() -> None:
    violations = _collect_violations()
    assert not violations, "Service contract violations:\n  " + "\n  ".join(violations)
