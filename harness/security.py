"""Deterministic boundaries; none depend on a model's classification."""

import ast
import math
import operator
import os
from pathlib import Path
import re
import stat


SECRET_PATTERNS = [
    re.compile(r"(?i)\b(?:api[_ -]?key|password|passwd|secret|access[_ -]?token|authorization|credential|private[_ -]?key)\b\s*(?:(?:is|equals)\s+|[:=]\s*|\s+)(?:bearer\s+)?(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|`[^`\r\n]*`|[^\r\n,;]+)"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|AKIA[A-Z0-9]{16})\b"),
    re.compile(r"-----BEGIN [\s\S]*?PRIVATE KEY-----[\s\S]*?(?:-----END [\s\S]*?PRIVATE KEY-----|$)"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}=*"),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@[^\s]+"),
]


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def redact(value):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if re.search(r"(?i)password|secret|token|api.?key|authorization", str(k))
                    and not str(k).endswith("tokens") else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for pattern in SECRET_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
    return value


def workspace_path(root: Path, value: str, *, must_exist=False) -> Path:
    """Resolve paths, including existing symlink parents, before checking ancestry."""
    root = root.resolve(strict=True)
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else root / path
    result = candidate.resolve(strict=must_exist)
    if not result.is_relative_to(root):
        raise PermissionError("Path escapes the selected workspace")
    relative = result.relative_to(root)
    if any(part in {".git", ".ssh", ".aws"} or part == ".env" or part.startswith(".env.")
           for part in (*relative.parts, *candidate.parts)):
        raise PermissionError("Credential files and Git metadata are protected")
    return result


def directory_fd(root: Path, selected: Path, *, create=False) -> int:
    """Walk canonical workspace directories with openat/O_NOFOLLOW on Linux/WSL."""
    relative = selected.relative_to(root)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in relative.parts:
            if create:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_fd(root: Path, selected: Path) -> int:
    parent = directory_fd(root, selected.parent)
    try:
        descriptor = os.open(selected.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("Only regular files can be read")
        return descriptor
    finally:
        os.close(parent)


def arithmetic(expression: str):
    if len(expression) > 256:
        raise ValueError("Expression is too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression is too complex")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                  ast.Mod: operator.mod, ast.Pow: operator.pow}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            result = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large")
            result = operations[type(node.op)](left, right)
        else:
            raise ValueError("Only arithmetic constants and operators are allowed")
        if isinstance(result, complex) or abs(result) > 10**100 or not math.isfinite(result):
            raise ValueError("Result is outside the supported finite range")
        return result

    return visit(tree.body)
