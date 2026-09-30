"""Code normalisation and the two representations.

Arush - this turns a piece of code into numbers so the computer can compare solutions. Two
ways, kept separate: AST features (an interpretable vector: how many loops, does it use a dict,
is it recursive...) which needs nothing extra, and a UniXcoder embedding (needs the optional
'embed' extra; heavy, loads only if you call it). `normalise` first strips comments and renames
local variables so that JUST renaming or reformatting never counts as a new idea - only a real
change of approach does.

* ``normalise``: strip comments/docstrings, format with black, canonicalise
  local identifiers, so that renaming and reformatting do not count as novelty.
* ``ast_features``: interpretable structural vector from Python's ``ast``.
* ``embed``: UniXcoder embedding (requires the ``embed`` extra; loads lazily).

Both representations are analysed separately; no composite is formed.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Iterable

import numpy as np

# ------------------------------------------------------------------ normalise

_BUILTIN_KEEP = {"self", "cls", "print", "len", "range", "sorted", "enumerate", "zip", "map", "filter",
                 "list", "dict", "set", "tuple", "min", "max", "sum", "abs", "int", "float", "str", "bool"}


class _Canon(ast.NodeTransformer):
    """Rename function-local names to v0, v1, ... in order of first appearance."""

    def __init__(self) -> None:
        self.map: dict[str, str] = {}

    def _name(self, n: str) -> str:
        if n in _BUILTIN_KEEP or n.startswith("__"):
            return n
        if n not in self.map:
            self.map[n] = f"v{len(self.map)}"
        return self.map[n]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._name(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._name(node.arg)
        return node


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def normalise(src: str, *, canonicalise_names: bool = True, use_black: bool = True) -> str:
    tree = ast.parse(src)
    tree = strip_docstrings(tree)
    if canonicalise_names:
        # keep the entry-point function name stable so tests still resolve it
        keep = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        canon = _Canon()
        canon.map.update({k: k for k in keep})
        tree = canon.visit(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)  # comments are dropped by unparse
    if use_black:
        try:
            import black  # noqa: WPS433
            out = black.format_str(out, mode=black.Mode(line_length=88))
        except Exception:  # noqa: BLE001 — black is cosmetic; never fail on it
            pass
    return out


# ------------------------------------------------------------------ AST features

_DATA_STRUCT_CALLS = {"dict": "dict", "set": "set", "list": "list", "deque": "deque", "heappush": "heap",
                      "heappop": "heap", "heapify": "heap", "defaultdict": "dict", "Counter": "dict", "sorted": "sort",
                      "sort": "sort", "bisect": "bisect", "bisect_left": "bisect", "bisect_right": "bisect"}

FEATURE_NAMES = [
    "n_for", "n_while", "n_if", "n_comprehension", "n_call", "n_return", "n_lambda", "n_try",
    "max_loop_depth", "max_nesting", "n_functions", "is_recursive", "n_lines",
    "uses_dict", "uses_set", "uses_list", "uses_deque", "uses_heap", "uses_sort", "uses_bisect",
    "n_slices", "n_subscripts", "n_binops", "n_compares", "n_bool_ops", "cyclomatic",
]


def _loop_depth(tree: ast.AST) -> int:
    best = 0

    def rec(node: ast.AST, depth: int) -> None:
        nonlocal best
        for child in ast.iter_child_nodes(node):
            d = depth + 1 if isinstance(child, (ast.For, ast.While, ast.comprehension)) else depth
            best = max(best, d)
            rec(child, d)

    rec(tree, 0)
    return best


def _nesting(tree: ast.AST) -> int:
    best = 0

    def rec(node: ast.AST, depth: int) -> None:
        nonlocal best
        for child in ast.iter_child_nodes(node):
            d = depth + 1 if isinstance(child, (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.FunctionDef)) else depth
            best = max(best, d)
            rec(child, d)

    rec(tree, 0)
    return best


def _cyclomatic(src: str, tree: ast.AST) -> int:
    try:
        from radon.complexity import cc_visit
        blocks = cc_visit(src)
        return max((b.complexity for b in blocks), default=1)
    except Exception:  # noqa: BLE001 — fall back to a simple decision-point count
        return 1 + sum(isinstance(n, (ast.If, ast.For, ast.While, ast.BoolOp, ast.Try, ast.comprehension)) for n in ast.walk(tree))


def ast_features(src: str) -> np.ndarray:
    tree = ast.parse(src)
    c = Counter(type(n).__name__ for n in ast.walk(tree))
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    fnames = {f.name for f in funcs}
    recursive = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in fnames for n in ast.walk(tree)
    )
    used: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
            if name in _DATA_STRUCT_CALLS:
                used.add(_DATA_STRUCT_CALLS[name])
        elif isinstance(n, ast.Dict):
            used.add("dict")
        elif isinstance(n, ast.Set):
            used.add("set")
        elif isinstance(n, ast.List):
            used.add("list")
    feats = [
        c["For"], c["While"], c["If"], c["ListComp"] + c["DictComp"] + c["SetComp"] + c["GeneratorExp"],
        c["Call"], c["Return"], c["Lambda"], c["Try"],
        _loop_depth(tree), _nesting(tree), len(funcs), int(recursive), src.count("\n") + 1,
        int("dict" in used), int("set" in used), int("list" in used), int("deque" in used),
        int("heap" in used), int("sort" in used), int("bisect" in used),
        c["Slice"], c["Subscript"], c["BinOp"], c["Compare"], c["BoolOp"], _cyclomatic(src, tree),
    ]
    return np.asarray(feats, dtype=float)


def ast_feature_matrix(sources: Iterable[str]) -> np.ndarray:
    return np.vstack([ast_features(s) for s in sources])


# ------------------------------------------------------------------ embeddings

_embedder = None


def embed(sources: list[str], model_name: str = "microsoft/unixcoder-base", batch_size: int = 16) -> np.ndarray:
    """UniXcoder mean-pooled embeddings, L2-normalised. Loads the model on first call."""
    global _embedder
    if _embedder is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(dev)
        _embedder = (tok, mdl, dev, torch)
    tok, mdl, dev, torch = _embedder
    out = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        out.append(pooled.cpu().numpy())
    return np.vstack(out)
