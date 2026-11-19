"""Command line.

    tieback sample/*.csv -o book.parquet --plan plan.json --findings findings.json
    tieback sample/*.csv --control EARNED_CAR_YRS=21130 --repair all
    tieback --replay plan.json sample/*.csv
    tieback demo ./sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tieback


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["demo"]:
        from tieback.demo import build
        out = Path(argv[1]) if len(argv) > 1 else Path("tieback_demo")
        for name, df in build(out, messy=True).items():
            print(f"{out / name}: {df.height:,} rows")
        print("\nFive extracts from five systems, with the problems real ones have.")
        print(f"Now run:  tieback {out}/*.csv")
        return 0

    ap = argparse.ArgumentParser(prog="tieback", description="Join insurance extracts and check they tie back to the totals.")
    ap.add_argument("files", nargs="+", help="CSV or Parquet extracts (two or more)")
    ap.add_argument("-o", "--out", help="write the joined table here (.parquet or .csv)")
    ap.add_argument("--plan", help="save the plan as JSON here")
    ap.add_argument("--replay", metavar="PLAN", help="run a saved plan instead of detecting keys, and list the rows it missed")
    ap.add_argument("--exposure", help="name of the exposure column, if it is not detected")
    ap.add_argument("--repair", choices=["safe", "all", "off"], default="safe",
                    help="correct mistyped keys: safe (case, spacing, swapped neighbours; default), all (also a changed character), off")
    ap.add_argument("--control", action="append", default=[], metavar="COLUMN=TOTAL",
                    help="a total from finance the table must tie to, e.g. EARNED_CAR_YRS=21130 (repeatable)")
    ap.add_argument("--findings", metavar="FILE", help="write every finding as JSON here")
    ap.add_argument("--version", action="version", version=f"tieback {tieback.__version__}")
    a = ap.parse_args(argv)

    tables = tieback.read(a.files)
    if a.replay:
        result = tieback.replay(tables, tieback.load_plan(a.replay))
        from tieback.checks import readable
        for name, grain in result["grain"].items():
            print(readable(f"{name}: {grain}"))
        for s in result["steps"]:
            for m in s["misses"]:
                print(f"  {s['table']}: {m['rows'].height:,} rows not used ({m['why'].replace('_', ' ')})")
        return 0

    hints = {name: a.exposure for name in tables} if a.exposure else None
    controls = {}
    for c in a.control:
        col, _, val = c.partition("=")
        try:
            controls[col] = float(val.replace(",", ""))
        except ValueError:
            ap.error(f"--control needs COLUMN=NUMBER, got {c!r}")
    repair = {"safe": True, "all": "all", "off": False}[a.repair]
    book, plan = tieback.assemble(tables, exposure_hints=hints, repair=repair, controls=controls)
    print(tieback.report(plan))
    if a.out:
        out = Path(a.out)
        book.write_csv(out) if out.suffix.lower() == ".csv" else book.write_parquet(out)
        print(f"\nwrote {out} ({book.height:,} rows, {book.width} columns)")
    if a.plan:
        tieback.save_plan(plan, a.plan)
        print(f"wrote {a.plan}")
    if a.findings:
        import json
        Path(a.findings).write_text(json.dumps(plan.findings, indent=2, default=str))
        print(f"wrote {a.findings} ({len(plan.findings)} findings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
