"""tieback: join insurance extracts that were never meant to be joined, and
prove the result ties back to the totals.

    import tieback
    tables = tieback.read(["policies.csv", "vehicles.csv", "claims.csv"])
    book, plan = tieback.assemble(tables)
    print(tieback.report(plan))
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from tieback.keys import (
    JoinCandidate,
    JoinDiagnostics,
    JoinSpec,
    ReconciliationResult,
    detect_join_keys,
    execute_join,
    normalize_key,
)
from tieback.plan import KeyPair, Plan, Step, Window, assemble, parse_dates, replay
from tieback.report import TieOut, check_total, load_plan, report, save_plan

__version__ = "0.1.0"

__all__ = [
    "JoinCandidate", "JoinDiagnostics", "JoinSpec", "KeyPair", "Plan", "ReconciliationResult", "Step",
    "TieOut", "Window", "assemble", "check_total", "detect_join_keys", "execute_join", "load_plan",
    "normalize_key", "parse_dates", "read", "replay", "report", "save_plan",
]


def read(paths: list[str | Path]) -> dict[str, pl.DataFrame]:
    """Read CSV or Parquet files into the {name: frame} mapping `assemble` takes.
    Text dates are parsed later, by the plan. Ids that differ only by a prefix
    or leading zeros ('ON-0482113' and 482113) are matched by the plan too."""
    out: dict[str, pl.DataFrame] = {}
    for p in map(Path, paths):
        if p.name in out:
            raise ValueError(f"Two files are named {p.name}; rename one.")
        if p.suffix.lower() == ".parquet":
            out[p.name] = pl.read_parquet(p)
        else:
            out[p.name] = pl.read_csv(p, infer_schema_length=10_000)
    return out
