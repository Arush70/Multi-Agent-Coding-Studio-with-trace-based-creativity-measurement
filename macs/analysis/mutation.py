"""Mutation testing of the held-out tests (proposal 3.7: "mutation testing is applied to the
held-out suites of five tasks to confirm they discriminate").

A correct solution is changed in one small place at a time (a mutant); a good test suite fails
on ("kills") most mutants. The mutation score is killed / total. Implemented here with Python's
ast module rather than the mutmut tool the proposal names: mutmut mutates files in a package and
does not run on Windows, while this works on the single-function solutions the project scores,
through the same sandbox. Operators (the classic set):

    comparison   <  <=  >  >=  ==  !=  swapped with a neighbour (e.g. < -> <=, == -> !=)
    arithmetic   +  -  *  //  %  swapped (e.g. + -> -)
    constant     integer constants c -> c + 1
    boolean      and <-> or; True <-> False; `not x` -> x
    return       `return x` -> `return None`
"""
from __future__ import annotations

import ast
import copy

_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}


def _sites(tree: ast.AST) -> list[tuple[int, str]]:
    """(node index in ast.walk order, operator name) for every place a mutant can be made."""
    out = []
    for i, n in enumerate(ast.walk(tree)):
        if isinstance(n, ast.Compare):
            out += [(i, "comparison")] if type(n.ops[0]) in _CMP else []
        elif isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            out.append((i, "arithmetic"))
        elif isinstance(n, ast.Constant) and type(n.value) is int:
            out.append((i, "constant"))
        elif isinstance(n, ast.Constant) and type(n.value) is bool:
            out.append((i, "boolean"))
        elif isinstance(n, ast.BoolOp):
            out.append((i, "boolean"))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            out.append((i, "boolean"))
        elif isinstance(n, ast.Return) and n.value is not None:
            out.append((i, "return"))
    return out


class _Apply(ast.NodeTransformer):
    def __init__(self, target: int):
        self.target, self.i = target, -1
        self.index = {}

    def run(self, tree):
        self.index = {id(n): k for k, n in enumerate(ast.walk(tree))}
        return self.visit(tree)

    def visit(self, node):
        k = self.index.get(id(node))
        node = super().visit(node)
        if k != self.target:
            return node
        if isinstance(node, ast.Compare):
            node.ops = [_CMP[type(node.ops[0])]()] + node.ops[1:]
        elif isinstance(node, ast.BinOp):
            node.op = _BIN[type(node.op)]()
        elif isinstance(node, ast.Constant) and type(node.value) is bool:
            node.value = not node.value
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            node.value = node.value + 1
        elif isinstance(node, ast.BoolOp):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif isinstance(node, ast.UnaryOp):
            return node.operand
        elif isinstance(node, ast.Return):
            node.value = ast.Constant(value=None)
        return node


def mutants(code: str, limit: int = 40) -> list[tuple[str, str]]:
    """Up to `limit` (operator, mutant source) pairs, one change each, spread over the code."""
    tree = ast.parse(code)
    sites = _sites(tree)
    if len(sites) > limit:
        step = len(sites) / limit
        sites = [sites[int(k * step)] for k in range(limit)]
    out = []
    for idx, op in sites:
        t2 = copy.deepcopy(tree)
        t2 = _Apply(idx).run(t2)
        ast.fix_missing_locations(t2)
        try:
            src = ast.unparse(t2)
        except Exception:  # noqa: BLE001
            continue
        if src != ast.unparse(tree):
            out.append((op, src))
    return out


def mutation_score(task, code: str, *, backend: str = "subprocess", limit: int = 40) -> dict:
    """Run the task's held-out tests on every mutant of a correct solution."""
    from macs.analysis.score import score_code_heldout
    base = score_code_heldout(task, code, backend=backend)
    if not base.get("passed_all"):
        return {"error": "the solution is not correct on the held-out tests", "total": 0}
    rows = []
    for op, src in mutants(code, limit=limit):
        r = score_code_heldout(task, src, backend=backend)
        rows.append({"operator": op, "killed": not r.get("passed_all")})
    killed = sum(r["killed"] for r in rows)
    by_op = {}
    for r in rows:
        d = by_op.setdefault(r["operator"], [0, 0])
        d[0] += r["killed"]; d[1] += 1
    return {"total": len(rows), "killed": killed, "score": killed / len(rows) if rows else None,
            "by_operator": {k: {"killed": a, "total": b} for k, (a, b) in by_op.items()}}
