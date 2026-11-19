"""Read a plan back as plain text, and check totals against a figure from finance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tieback.plan import Plan, Step, _step_from


def _n(x: float | int | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.0f}" if float(x).is_integer() or abs(x) >= 1000 else f"{x:,.2f}"


def _mark(ok: bool) -> str:
    return "ok" if ok else "CHECK"


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def _step_lines(step: Step) -> list[str]:
    t = step.totals
    keys = " + ".join(f"{k.other} = {k.base}" for k in step.keys)
    head = f"{step.table}  [{step.kind}]  on {keys}"
    if step.window:
        head += f", {step.window.date} within [{step.window.start}, {step.window.end})"
    if step.kind == "group":
        head = head.replace("[group]", "[per group]")
    out = [head] + [f"    {r}" for r in step.reasons]
    rows_ok = t.get("base_rows_before") == t.get("base_rows_after")
    out.append(f"    rows: {_n(t.get('base_rows_before'))} before, {_n(t.get('base_rows_after'))} after"
               f"  {_mark(rows_ok)}" + ("" if rows_ok else "  (the join fanned out)"))
    if step.kind == "lookup":
        out.append(f"    matched {_n(t.get('base_rows_matched'))} rows, {_n(t.get('base_rows_unmatched'))} without a partner;"
                   f" {_n(t.get('other_rows_unused'))} of {_n(t.get('other_rows'))} {step.table} rows unused,"
                   f" {_n(t.get('other_keys_ambiguous'))} keys listed twice and skipped")
        return out
    unit = t.get("unit", "row")
    unit = "rows" if unit == "row" else unit
    parts = [t.get(f"units_{s}", 0) for s in ("attached", "no_key", "outside_window", "ambiguous")]
    ok = sum(parts) == t.get("units_in_file") and t.get("units_attached") == t.get("units_in_table")
    out.append(f"    {unit}: {_n(t.get('units_in_file'))} in file = {_n(parts[0])} attached + {_n(parts[1])} no matching key"
               f" + {_n(parts[2])} outside every term + {_n(parts[3])} ambiguous  {_mark(ok)}")
    for col, a in (t.get("amounts") or {}).items():
        unattached = a.get("no_key", 0) + a.get("outside_window", 0) + a.get("ambiguous", 0)
        ok = _close(a["in_file"], a["in_table"] + unattached) and _close(a.get("attached", 0), a["in_table"])
        out.append(f"    {col}: {_n(a['in_file'])} in file = {_n(a['in_table'])} in table + {_n(unattached)} unattached  {_mark(ok)}")
    return out


def report(plan: Plan) -> str:
    from tieback.checks import readable
    return readable(_report(plan))


def _report(plan: Plan) -> str:
    """The plan as an actuary would read it: each join, why that key, and
    whether every row, claim and dollar is accounted for."""
    t = plan.totals
    lines = [f"base: {plan.base} ({_n(t.get('base_rows'))} rows)"
             + (f", exposure {plan.exposure}" if plan.exposure else ", no exposure column found")]
    for step in plan.steps:
        lines += [""] + _step_lines(step)
    for u in plan.unresolved:
        lines += ["", f"{u['table']}  NOT JOINED", f"    {u['reason']}"]
    lines.append("")
    if t.get("exposure_in") is not None:
        ok = _close(t["exposure_in"], t["exposure_out"])
        lines.append(f"exposure: {_n(t['exposure_in'])} in, {_n(t['exposure_out'])} out  {_mark(ok)}")
    lines.append(f"rows: {_n(t.get('base_rows'))} in, {_n(t.get('rows'))} out  {_mark(t.get('base_rows') == t.get('rows'))}")
    if plan.findings:
        lines += ["", findings_text(plan.findings)]
    return "\n".join(lines)


_MARK = {"error": "ERROR", "warning": "REVIEW", "info": "FIXED"}


def findings_text(findings: list[dict], examples: int = 3) -> str:
    """The findings, most serious first: what is wrong, what it does to the
    numbers, a few examples and what to do."""
    n = {s: sum(1 for f in findings if f["severity"] == s) for s in ("error", "warning", "info")}
    out = [f"findings: {n['error']} errors, {n['warning']} to review, {n['info']} fixed"]
    for f in findings:
        out.append(f"  {_MARK[f['severity']]:<6}  {f['message']}")
        if f.get("impact"):
            out.append("          affects " + ", ".join(f"{k} {_n(v)}" for k, v in f["impact"].items()))
        ex = f.get("examples") or []
        for e in ex[:examples]:
            out.append("          e.g. " + ", ".join(f"{k}={v}" for k, v in e.items()))
        if len(ex) > examples:
            out.append(f"         and {len(ex) - examples} more")
        if f.get("action"):
            out.append(f"          -> {f['action']}")
    return "\n".join(out)


def save_plan(plan: Plan, path: str | Path) -> None:
    """Write the plan as JSON, so the next run is a replay anyone can audit."""
    Path(path).write_text(json.dumps(plan.as_dict(), indent=2, default=str))


def load_plan(path: str | Path) -> dict:
    """Read a saved plan, ready for `tieback.replay`."""
    d = json.loads(Path(path).read_text())
    for s in d["steps"]:
        _step_from(s)  # fail early on a malformed plan
    return d


@dataclass(frozen=True)
class TieOut:
    column: str
    total: float
    expected: float
    difference: float
    relative: float
    tolerance: float

    @property
    def ties(self) -> bool:
        return abs(self.relative) <= self.tolerance

    def __str__(self) -> str:
        verdict = "ties" if self.ties else "does not tie"
        return (f"{self.column}: {self.total:,.2f} against {self.expected:,.2f} expected, "
                f"{self.relative:+.3%} ({verdict} within {self.tolerance:.2%})")


def check_total(frame: pl.DataFrame, column: str, expected: float, tolerance: float = 0.001) -> TieOut:
    """Compare a column's total with a control figure (earned premium from the
    general ledger, paid losses from finance). Tolerance is relative."""
    if column not in frame.columns:
        raise ValueError(f"No column named {column!r}.")
    if expected == 0:
        raise ValueError("The expected total is zero; a relative tolerance means nothing against it.")
    total = float(frame[column].sum() or 0.0)
    diff = total - expected
    return TieOut(column, total, float(expected), diff, diff / expected, tolerance)
