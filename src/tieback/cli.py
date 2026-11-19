"""Command line.

    tieback policies.csv vehicles.csv claims.csv -o book.parquet --plan plan.json
    tieback --replay plan.json policies.csv vehicles.csv claims.csv
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
        for name, df in build(out).items():
            print(f"{out / name}: {df.height:,} rows")
        print(f"\nNow run:  tieback {out}/policies.csv {out}/vehicles.csv {out}/claims.csv")
        return 0

    ap = argparse.ArgumentParser(prog="tieback", description="Join insurance extracts and check they tie back to the totals.")
    ap.add_argument("files", nargs="+", help="CSV or Parquet extracts (two or more)")
    ap.add_argument("-o", "--out", help="write the joined table here (.parquet or .csv)")
    ap.add_argument("--plan", help="save the plan as JSON here")
    ap.add_argument("--replay", metavar="PLAN", help="run a saved plan instead of detecting keys, and list the rows it missed")
    ap.add_argument("--exposure", help="name of the exposure column, if it is not detected")
    ap.add_argument("--version", action="version", version=f"tieback {tieback.__version__}")
    a = ap.parse_args(argv)

    tables = tieback.read(a.files)
    if a.replay:
        result = tieback.replay(tables, tieback.load_plan(a.replay))
        for name, grain in result["grain"].items():
            print(f"{name}: {grain}")
        for s in result["steps"]:
            for m in s["misses"]:
                print(f"  {s['table']}: {m['rows'].height:,} rows not used ({m['why'].replace('_', ' ')})")
        return 0

    hints = {name: a.exposure for name in tables} if a.exposure else None
    book, plan = tieback.assemble(tables, exposure_hints=hints)
    print(tieback.report(plan))
    if a.out:
        out = Path(a.out)
        book.write_csv(out) if out.suffix.lower() == ".csv" else book.write_parquet(out)
        print(f"\nwrote {out} ({book.height:,} rows, {book.width} columns)")
    if a.plan:
        tieback.save_plan(plan, a.plan)
        print(f"wrote {a.plan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
