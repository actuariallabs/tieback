"""Everything else that is wrong with the extracts, found after the join.

Each finding says what is wrong, how many rows, what it does to the numbers
(exposure or money affected), a few examples, and what to do. Severity:

  error    the numbers are wrong if nothing is done (double exposure, a table
           that did not join, a total that does not tie)
  warning  needs a decision (orphan claims, a spelling that splits a rating
           group, a negative amount that may be a recovery)
  info     done for you and listed for the record (corrected keys)

Pure compute: frames and the plan in, a list of dicts out.
"""

from __future__ import annotations

import re
from datetime import date

import polars as pl

from tieback.repair import _distance_one, _loose

_PAID = re.compile(r"paid", re.IGNORECASE)
_INCURRED = re.compile(r"incurred|incd", re.IGNORECASE)
_LOSS_DATE = re.compile(r"loss|accident|occur|incident", re.IGNORECASE)
_REPORT_DATE = re.compile(r"report|notif|fnol|opened", re.IGNORECASE)
# Findings whose examples are the whole list (each one needs a look).
_FULL = {"key_likely_typo", "key_suspect", "category_spellings"}


def _f(check: str, severity: str, message: str, *, table: str | None = None, column: str | None = None,
       count: int = 0, impact: dict | None = None, examples: list | None = None, action: str = "") -> dict:
    return {"check": check, "severity": severity, "table": table, "column": column, "count": int(count),
            "message": message, "impact": impact or {}, "examples": (examples or [])[:3] if check not in _FULL else (examples or []),
            "action": action}


def _money(x: float) -> str:
    return f"{x:,.0f}"


def _examples(df: pl.DataFrame, cols: list[str]) -> list[dict]:
    cols = [c for c in cols if c in df.columns]
    return [{k: (str(v) if v is not None else None) for k, v in r.items()} for r in df.select(cols).head(3).iter_rows(named=True)]


def run_checks(inputs: dict[str, pl.DataFrame], plan, book: pl.DataFrame, controls: dict[str, float]) -> list[dict]:
    out: list[dict] = []
    expo = plan.exposure
    events = {s.table: s for s in plan.steps if s.kind in ("rollup", "group")}
    key_cols = {(s.table, k.other) for s in plan.steps for k in s.keys} | {(plan.base, k.base) for s in plan.steps for k in s.keys}
    id_cols = {(s.table, s.count_of) for s in plan.steps if s.count_of}

    # ── Tables that did not join ─────────────────────────────────────────────
    for u in plan.unresolved:
        out.append(_f("not_joined", "error", f"{u['table']} could not be joined. {u['reason']}", table=u["table"],
                      count=inputs[u["table"]].height,
                      action="Check that it shares an id with the other files, or join it by hand and pass the result."))

    # ── Duplicate rows ───────────────────────────────────────────────────────
    for name, df in inputs.items():
        extra = df.height - df.unique().height
        if not extra:
            continue
        dup = df.filter(df.is_duplicated()).unique(maintain_order=True)
        impact, sev = {}, "warning"
        if name == plan.base and expo and expo in df.columns:
            sev = "error"
            impact[expo] = round(float(df[expo].sum() or 0) - float(df.unique()[expo].sum() or 0), 4)
        if name in events:
            sev = "error"
            for c in events[name].sums:
                impact[c] = round(float(df[c].sum() or 0) - float(df.unique()[c].sum() or 0), 2)
        out.append(_f("duplicate_rows", sev, f"{name} has {extra:,} row{'s' if extra > 1 else ''} that repeat another row exactly.",
                      table=name, count=extra, impact=impact, examples=_examples(dup, dup.columns[:5]),
                      action="Drop the repeats if the source system exported them twice."))

    # ── Overlapping periods and exposure that does not fit them ──────────────
    win = next((s for s in plan.steps if s.window), None)
    if win is not None and win.window.start in book.columns and win.window.end in book.columns:
        st, en = win.window.start, win.window.end
        ent = [k.base for k in win.keys if k.base in book.columns]
        bad = book.filter(pl.col(en) <= pl.col(st))
        if bad.height:
            out.append(_f("period_ends_before_start", "error", f"{bad.height:,} {plan.base} rows end on or before they start ({st} to {en}).",
                          table=plan.base, count=bad.height, examples=_examples(bad, ent + [st, en])))
        if ent:
            srt = book.sort(ent + [st]).with_columns(pl.col(en).shift(1).over(ent).alias("__prev_end"))
            ov = srt.filter(pl.col("__prev_end").is_not_null() & (pl.col(st) < pl.col("__prev_end")))
            if ov.height:
                days = ov.select((pl.min_horizontal(pl.col("__prev_end"), pl.col(en)) - pl.col(st)).dt.total_days().sum()).item() or 0
                out.append(_f("overlapping_periods", "error",
                              f"{ov.height:,} {plan.base} rows overlap an earlier period of the same " + " and ".join(ent)
                              + ": the overlap is exposure counted twice.",
                              table=plan.base, count=ov.height, impact={"overlap_years": round(days / 365.25, 2)},
                              examples=_examples(ov, ent + [st, en]),
                              action="Keep one row per period: usually a mid-term change exported as a second full term."))
        if expo and expo in book.columns:
            length = (pl.col(en) - pl.col(st)).dt.total_days() / 365.25
            over = book.filter(pl.col(expo) > length * 1.01 + 0.003)
            if over.height:
                out.append(_f("exposure_exceeds_period", "error",
                              f"{over.height:,} {plan.base} rows carry more {expo} than their period is long.",
                              table=plan.base, column=expo, count=over.height,
                              impact={expo: round(float(over.select(pl.col(expo) - length).sum().item() or 0), 3)},
                              examples=_examples(over, ent + [st, en, expo])))
    if expo and expo in book.columns:
        bad = book.filter(pl.col(expo).is_null() | (pl.col(expo) <= 0))
        if bad.height:
            out.append(_f("exposure_missing_or_zero", "warning", f"{bad.height:,} {plan.base} rows have no positive {expo}.",
                          table=plan.base, column=expo, count=bad.height, examples=_examples(bad, [expo])))

    # ── Amounts and dates in the event files ─────────────────────────────────
    today = date.today()
    for name, df in inputs.items():
        step = events.get(name)
        sums = step.sums if step else []
        for c in sums:
            neg = df.filter(pl.col(c) < 0)
            if neg.height:
                out.append(_f("negative_amount", "warning", f"{neg.height:,} {name} rows have a negative {c} (total {_money(float(neg[c].sum()))}).",
                              table=name, column=c, count=neg.height, impact={c: round(float(neg[c].sum()), 2)},
                              examples=_examples(neg, ([step.count_of] if step.count_of else []) + [c]),
                              action="Recoveries and reversals are negative on purpose; data-entry signs are not. Decide which."))
        paid = next((c for c in sums if _PAID.search(c)), None)
        inc = next((c for c in sums if _INCURRED.search(c)), None)
        if paid and inc:
            over = df.filter(pl.col(paid) > pl.col(inc) + 0.005)
            if over.height:
                out.append(_f("paid_exceeds_incurred", "warning", f"{over.height:,} {name} rows have {paid} above {inc}.",
                              table=name, count=over.height, impact={paid: round(float(over.select(pl.col(paid) - pl.col(inc)).sum().item()), 2)},
                              examples=_examples(over, ([step.count_of] if step and step.count_of else []) + [paid, inc]),
                              action="Incurred should include paid. Usually a stale case reserve or a mis-keyed amount."))
        dates = [c for c, t in df.schema.items() if t == pl.Date]
        loss = next((c for c in dates if _LOSS_DATE.search(c)), None)
        rep = next((c for c in dates if _REPORT_DATE.search(c)), None)
        if loss and rep:
            bad = df.filter(pl.col(rep) < pl.col(loss))
            if bad.height:
                out.append(_f("reported_before_loss", "warning", f"{bad.height:,} {name} rows were reported before the loss happened ({rep} < {loss}).",
                              table=name, count=bad.height, examples=_examples(bad, ([step.count_of] if step and step.count_of else []) + [loss, rep]),
                              action="Often day and month swapped, or the two dates swapped. The loss date places the claim in its term."))
        for c in dates:
            odd = df.filter((pl.col(c) < date(1950, 1, 1)) | (pl.col(c) > date(today.year + 1, today.month, 1)))
            if odd.height:
                out.append(_f("implausible_date", "warning", f"{odd.height:,} {name} rows have a {c} before 1950 or more than a year ahead.",
                              table=name, column=c, count=odd.height, examples=_examples(odd, [c])))

    # ── Spellings that split one category in two ─────────────────────────────
    for name, df in inputs.items():
        for c, t in df.schema.items():
            if t != pl.Utf8 or (name, c) in key_cols or (name, c) in id_cols:
                continue
            vc = df[c].drop_nulls().value_counts(sort=True)
            if vc.height < 2 or vc.height > 1_000 or vc.height > 0.5 * max(df.height, 1):
                continue
            vals = vc[c].to_list()
            counts = dict(zip(vals, vc["count"].to_list()))
            if sum(1 for v in vals if re.fullmatch(r"[\d.\-/ ]+", v)) > 0.5 * len(vals):
                continue
            mapping = {}
            for i, v in enumerate(vals):
                for u in vals[:i]:
                    if u in mapping:
                        continue
                    lv, lu = _loose(v), _loose(u)
                    same = lv == lu and v != u
                    near = (len(lu) >= 5 and _distance_one(lv, lu) is not None and counts[v] <= 0.1 * counts[u])
                    if same or near:
                        mapping[v] = u
                        break
            if mapping:
                rows = sum(counts[v] for v in mapping)
                out.append(_f("category_spellings", "warning",
                              f"{name}.{c} spells {len(mapping)} value{'s' if len(mapping) > 1 else ''} two ways: "
                              + "; ".join(f"'{v}' ({counts[v]:,}) looks like '{u}' ({counts[u]:,})" for v, u in list(mapping.items())[:4]) + ".",
                              table=name, column=c, count=rows, examples=[{"from": v, "to": u} for v, u in mapping.items()],
                              action="Each spelling becomes its own rating level. Map them to one before fitting."))

    # ── Attributes that disagree between files ───────────────────────────────
    for s in plan.steps:
        if s.kind != "lookup":
            continue
        label = re.sub(r"\.(csv|xlsx?|txt|parquet)$", "", s.table, flags=re.IGNORECASE)
        for c in s.added_columns:
            if not c.endswith(f"_{label}"):
                continue
            mine = c[: -len(label) - 1]
            if mine not in book.columns:
                continue
            both = book.filter(pl.col(mine).is_not_null() & pl.col(c).is_not_null())
            try:
                diff = both.filter(pl.col(mine).cast(pl.Utf8) != pl.col(c).cast(pl.Utf8))
            except Exception:
                continue
            if diff.height:
                out.append(_f("files_disagree", "warning",
                              f"{mine} differs between {plan.base} and {s.table} on {diff.height:,} rows.",
                              table=s.table, column=mine, count=diff.height, examples=_examples(diff, [mine, c]),
                              action="Decide which system is the source of truth for this field."))

    # ── What did not attach ──────────────────────────────────────────────────
    for s in plan.steps:
        t = s.totals
        if s.kind == "lookup":
            if t.get("base_rows_unmatched"):
                out.append(_f("rows_without_partner", "warning",
                              f"{t['base_rows_unmatched']:,} {plan.base} rows found no row in {s.table}: their {s.table} columns are empty.",
                              table=s.table, count=t["base_rows_unmatched"]))
            continue
        unit = t.get("unit", "row")
        lost = {k: t.get(f"units_{k}", 0) for k in ("no_key", "outside_window", "ambiguous")}
        n = sum(lost.values())
        if n:
            money = {c: round(a.get("no_key", 0) + a.get("outside_window", 0) + a.get("ambiguous", 0), 2)
                     for c, a in (t.get("amounts") or {}).items()}
            parts = [f"{lost['no_key']:,} with no matching key" if lost["no_key"] else "",
                     f"{lost['outside_window']:,} outside every period" if lost["outside_window"] else "",
                     f"{lost['ambiguous']:,} matching several rows" if lost["ambiguous"] else ""]
            losses = any(_PAID.search(c) or _INCURRED.search(c) or re.search(r"loss", c, re.I) for c in money)
            unit_s = "rows" if unit == "row" else f"{unit}s"
            out.append(_f("unattached_events", "warning",
                          f"{n:,} {unit_s} in {s.table} are not in the table: " + ", ".join(p for p in parts if p) + ".",
                          table=s.table, count=n, impact=money,
                          action=("Losses with no exposure: keep them in a separate load or chase the missing policies."
                                  if losses else "They have no exposure row to sit on.")
                                 + " `tieback --replay` lists every one."))

    # ── Keys: corrected, and suspected ───────────────────────────────────────
    for s in plan.steps:
        for r in s.repairs:
            out.append(_f("key_corrected", "info",
                          f"{r['table']}.{r['column']}: '{r['original']}' read as '{r['corrected']}' ({r['kind']}; {r['evidence']}).",
                          table=r["table"], column=r["column"], count=r["rows"]))
    groups: dict[tuple, list[dict]] = {}
    for sp in plan.suspects:
        groups.setdefault((sp["table"], sp["column"], bool(sp.get("likely"))), []).append(sp)
    for (table, column, likely), sps in groups.items():
        rows = sum(sp["rows"] for sp in sps)
        n = len(sps)
        if likely:
            out.append(_f("key_likely_typo", "warning",
                          f"{table}.{column}: {n} value{'s match' if n > 1 else ' matches'} nothing but "
                          f"{'sit' if n > 1 else 'sits'} one character from a real key "
                          "whose other columns and dates fit, and only that one. Not applied: a policy missing from the "
                          "extract can sit one digit from a real one by chance.",
                          table=table, column=column, count=rows,
                          examples=[{"value": sp["value"], "likely": sp["likely"]} for sp in sps],
                          action="Confirm in the source system, or run with repair='all' to apply them."))
        else:
            out.append(_f("key_suspect", "warning",
                          f"{table}.{column}: {n} value{'s match' if n > 1 else ' matches'} nothing and "
                          f"{'sit' if n > 1 else 'sits'} one keystroke from real keys "
                          "that the other columns cannot tell apart.",
                          table=table, column=column, count=rows,
                          examples=[{"value": sp["value"], "near": ", ".join(sp["candidates"])} for sp in sps],
                          action="Look them up in the source system."))
    # ── Control totals ───────────────────────────────────────────────────────
    for col, expected in (controls or {}).items():
        if col not in book.columns:
            out.append(_f("control_total", "error", f"No column {col} in the table to tie to {expected:,}.", column=col))
            continue
        total = float(book[col].sum() or 0.0)
        rel = (total - expected) / expected if expected else float("inf")
        ok = abs(rel) <= 0.001
        out.append(_f("control_total", "info" if ok else "error",
                      f"{col}: {total:,.2f} against {expected:,.2f} from finance, {rel:+.3%}" + (" (ties)" if ok else " (does not tie)"),
                      column=col, impact={col: round(total - expected, 2)}))

    for f in out:
        f["message"] = readable(f["message"])
        if f.get("column"):
            f["column"] = readable(f["column"])
    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(out, key=lambda f: order[f["severity"]])


def readable(text: str) -> str:
    """Part columns split out of a two-number id, as a person would say them."""
    return re.sub(r"·(\d)", r" (part \1)", text)
