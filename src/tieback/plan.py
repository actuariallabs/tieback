"""Assemble several extracts into one modelling table, with control totals.

A pricing submission rarely arrives as one file. The usual shape is three,
from systems never designed to be joined:

  policy admin       one row per policy TERM ('ON-0482113', term 1, term 2)
  vehicle schedule   one row per vehicle per term, carrying the exposure
  claims             one row per claim per COVERAGE, with a loss date

Pairwise key matching (`schema_reconciliation`) cannot assemble that:

- the same policy is 'ON-0482113' in one system and 482113 in another;
- a renewal repeats the policy number, so the number alone matches every
  term: the vehicle schedule identifies the term by its start date, and the
  claims file has no term at all, only a loss date that must fall inside one;
- a claim with three coverages is three rows; counting rows triples it.

The plan starts from the table that carries exposure (the rating grain).
Lookups attach first (exactly one partner row per base row: the policy
term's attributes onto each vehicle), then events roll up onto it (claims
counted once per claim, amounts summed). Each key is refined, column by
column and then by date window, until every row has exactly one partner.

Every step reports control totals, so nothing is duplicated or lost
silently: base rows and exposure are unchanged, and every claim and every
dollar in the claims file is either attached or listed as unattached with
its reason (no such policy, or outside every term).

Pure compute: frames in, frame and plan out.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import polars as pl

from tieback.keys import detect_join_keys, normalize_key
from tieback.repair import _TIERS as REPAIR_TIERS
from tieback.repair import APPLY_DEFAULT as REPAIR_DEFAULT
from tieback.repair import _with_value, find_repairs, rewrite
from tieback.repair import apply as apply_repairs

# Share of rows that may stay ambiguous (more than one partner) for a key to
# count as resolved. The ambiguous ones are never joined: they are reported.
_RESOLVED = 0.01
# A refinement must keep this share of the rows the key already matched.
_KEEP_MATCH = 0.90
# A lookup describes the base rows (every vehicle has a policy term), so it
# must cover most of them and carry no claim amounts. A table that matches
# only some rows (claimants), or holds paid or incurred amounts, holds events
# to count, even when no row happens to have two.
_LOOKUP_COVERAGE = 0.5
# Keys kept open at each depth of the refinement search.
_BEAM = 4
# Never materialise more candidate pairs than this while evaluating a key.
_MAX_PAIRS = 5_000_000
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d")
_EXPOSURE = re.compile(r"expos|earned|car_?y(ea)?rs|policy_?y(ea)?rs", re.IGNORECASE)
_AMOUNT = re.compile(r"amt|amount|paid|incurred|loss(?!_?date)|cost|reserve", re.IGNORECASE)
_CLAIM_ID = re.compile(r"claim|clm", re.IGNORECASE)
# Money that adds up: summed for events, split by exposure when it sits at a
# coarser grain than the exposure rows (premium per policy term, say).
_ADDITIVE = re.compile(r"prem|amt|amount|paid|incurred|loss(?!_?date)|cost|reserve|commission|fee|tax|recover|salvage|subro",
                       re.IGNORECASE)
# One id holding two numbers: '0482113/1' is policy 482113, term 1.
_COMPOSITE = r"^\D*?(\d+)[^\d\s]+(\d+)\D*$"
# Marks the part columns split out of a composite id. They are keys only and
# never reach the output.
_PART = "·"
# How a table joins, in order of preference: one partner per base row; one
# base row per event; one group of base rows per event.
_AIMS = ("lookup", "rollup", "group")


@dataclass
class KeyPair:
    base: str
    other: str
    form: str = "exact"   # "exact" | "digits" (prefixes and leading zeros dropped)
    example: tuple[str, str] | None = None


@dataclass
class Window:
    start: str            # base column: the term starts
    end: str              # base column: the term ends (exclusive)
    date: str             # other column: the event date
    share: float          # matched rows that fall in exactly one window


@dataclass
class Step:
    table: str
    kind: str             # "lookup" | "rollup"
    keys: list[KeyPair]
    window: Window | None = None
    count_of: str | None = None      # rollup: the id counted once (CLAIM_NO)
    count_column: str | None = None
    sums: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    added_columns: list[str] = field(default_factory=list)
    repairs: list[dict] = field(default_factory=list)    # key values corrected before this join
    allocated: list[str] = field(default_factory=list)   # amounts split by exposure across a group
    repair_stats: dict = field(default_factory=dict)     # unmatched keys, corrections, chance of a wrong one


@dataclass
class Plan:
    base: str
    exposure: str | None
    steps: list[Step]
    unresolved: list[dict]
    totals: dict = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)
    suspects: list[dict] = field(default_factory=list)   # mistyped keys that could not be told apart

    def as_dict(self) -> dict:
        return asdict(self)


def _evaluator(cur: pl.DataFrame, other: pl.DataFrame, keys: list[KeyPair], window: Window | None, aim: str):
    """For the repair search: how many of the given rows join with a candidate
    key in place, under every other key and the window of this step. Only the
    candidate's own rows on the far side are joined, so each check is small."""
    first = keys[0]
    far, far_col = (other, first.other) if aim == "lookup" else (cur, first.base)
    far_key = _norm(far, far_col, first.form)

    def evaluate(src: pl.DataFrame, col: str, rows: pl.Series, cand: str) -> int:
        value = rewrite(str(src[col][int(rows[0])]), first.form, cand)
        trial = _with_value(src, col, rows, value)
        if trial.height == 0 or far_key is None:
            return 0
        near = far.filter(far_key == cand)
        if near.height == 0:
            return 0
        if aim == "lookup":
            pairs = _pairs(trial, near, keys, None)
            return 0 if pairs is None else int((pairs.group_by("__b").len()["len"] == 1).sum())
        pairs = _pairs(near, trial, keys, window)
        if pairs is None or pairs.height == 0:
            return 0
        per = pairs.group_by("__o").agg(pl.len().alias("n"), pl.col("__g").n_unique().alias("g"))
        return int(((per["g"] == 1) if aim == "group" else (per["n"] == 1)).sum())
    return evaluate


def _known_ids(inputs: dict[str, pl.DataFrame], form: str, exclude: str, cache: dict) -> set[str]:
    """Every id-like value in the extracts other than `exclude`, normalised
    the same way as the key: a value found there is real, not a typo."""
    out: set[str] = set()
    for name, df in inputs.items():
        if name == exclude:
            continue
        key = (name, form)
        if key not in cache:
            vals: set[str] = set()
            for c, t in df.schema.items():
                if not (t == pl.Utf8 or t.is_integer()) or df[c].n_unique() < 0.2 * max(df.height, 1):
                    continue
                for f in {form, "exact", "digits"}:
                    k = normalize_key(df[c], f)
                    if k is not None:
                        vals |= set(k.drop_nulls().to_list())
            cache[key] = vals
        out |= cache[key]
    return out


# ── Preparation ───────────────────────────────────────────────────────────────


def date_format(col: pl.Series) -> str | None:
    """The strptime format a text column of dates is written in, or None."""
    if col.dtype != pl.Utf8:
        return None
    s = col.drop_nulls()
    if len(s) == 0 or not s.head(200).str.contains(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$|^\d{8}$").all():
        return None
    best, best_ok = None, 0.0
    for fmt in _DATE_FORMATS:
        ok = float(s.str.to_date(fmt, strict=False).is_not_null().mean() or 0.0)
        if ok > best_ok:
            best, best_ok = fmt, ok
    return best if best is not None and best_ok >= 0.95 else None


def parse_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Text columns that are dates become dates, so windows can compare them."""
    out = []
    for c, t in df.schema.items():
        if t == pl.Datetime:
            out.append(pl.col(c).dt.date())
            continue
        fmt = date_format(df[c])
        if fmt:
            out.append(pl.col(c).str.to_date(fmt, strict=False))
    return df.with_columns(out) if out else df


def split_composites(df: pl.DataFrame) -> pl.DataFrame:
    """Text ids that hold two numbers ('0482113/1', 'ON-482113-02') get their
    parts as extra key columns (`GL_REF·1`, `GL_REF·2`), so policy and term can
    each be matched. Dates are parsed before this and are not split."""
    adds = []
    for c, t in df.schema.items():
        if t != pl.Utf8 or _PART in c:
            continue
        s = df[c].drop_nulls().head(5_000)
        if len(s) == 0 or float(s.str.contains(_COMPOSITE).mean() or 0.0) < 0.95:
            continue
        parts = df[c].str.extract_groups(_COMPOSITE)
        adds += [parts.struct.field("1").cast(pl.Int64, strict=False).alias(f"{c}{_PART}1"),
                 parts.struct.field("2").cast(pl.Int64, strict=False).alias(f"{c}{_PART}2")]
    return df.with_columns(adds) if adds else df


def _prepare(df: pl.DataFrame) -> pl.DataFrame:
    return split_composites(parse_dates(df))


def _visible(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([c for c in df.columns if _PART not in c])


def exposure_column(df: pl.DataFrame, hint: str | None = None) -> str | None:
    if hint and hint in df.columns:
        return hint
    found = [c for c, t in df.schema.items() if t.is_numeric() and _EXPOSURE.search(c)]
    return next((c for c in found if df[c].dtype.is_float()), found[0] if found else None)


# ── Evaluating a key ──────────────────────────────────────────────────────────


@dataclass
class _Fit:
    keys: list[KeyPair]
    window: Window | None
    match_other: float     # share of other rows with at least one partner
    match_base: float      # share of base rows with at least one partner
    amb_other: float       # matched other rows with several base partners
    amb_base: float        # matched base rows with several other partners
    amb_group: float = 0.0  # matched other rows whose partners span several groups
    group_rows: float = 1.0  # base rows per matched other row, on average

    def placed(self, aim: str) -> float:
        """Share of rows with exactly one partner: what a refinement is
        judged on. Ambiguity alone would reward a key that matches fewer
        rows (a reported date, falling after the term, drops claims and so
        looks less ambiguous than the loss date)."""
        if aim == "lookup":
            return 0.0 if self.window is not None else self.match_base * (1 - self.amb_base)
        if aim == "group":
            return self.match_other * (1 - self.amb_group)
        return self.match_other * (1 - self.amb_other)

    def ambiguity(self, aim: str) -> float:
        """How far from one partner per row, for what the join is for: a
        lookup needs one partner per BASE row (and no window), a rollup one
        base row per event."""
        if aim == "lookup":
            return 1.0 if self.window is not None else self.amb_base
        if aim == "group":
            return self.amb_group
        return self.amb_other


# Normalised key columns, reused within one assemble run: the key search
# normalises the same columns many times. Frames are kept alive in the cache
# so an id is never reused while it is in use.
_NORM_CACHE: dict = {}


def _norm(df: pl.DataFrame, col: str, form: str) -> pl.Series | None:
    key = (id(df), col, form)
    hit = _NORM_CACHE.get(key)
    if hit is not None and hit[0] is df:
        return hit[1]
    s = df[col].cast(pl.Utf8) if df[col].dtype == pl.Date else normalize_key(df[col], form)
    if df.height >= 1_000:
        _NORM_CACHE[key] = (df, s)
    return s


def _keyed(df: pl.DataFrame, keys: list[KeyPair], side: str, extra: list[str], rid: str) -> pl.DataFrame | None:
    cols = []
    for i, k in enumerate(keys):
        s = _norm(df, getattr(k, side), k.form)
        if s is None:
            return None
        cols.append(s.alias(f"__k{i}"))
    frame = pl.DataFrame(cols).with_columns(pl.int_range(df.height, dtype=pl.UInt32).alias(rid))
    if extra:
        frame = frame.with_columns([df[c] for c in extra])
    return frame.drop_nulls([f"__k{i}" for i in range(len(keys))])


def _pairs(base: pl.DataFrame, other: pl.DataFrame, keys: list[KeyPair], window: Window | None) -> pl.DataFrame | None:
    """Every (base row, other row) pair that shares the key (and window)."""
    b = _keyed(base, keys, "base", [window.start, window.end] if window else [], "__b")
    o = _keyed(other, keys, "other", [window.date] if window else [], "__o")
    if b is None or o is None:
        return None
    on = [f"__k{i}" for i in range(len(keys))]
    # Bound the work before joining: pairs = sum over keys of the two counts.
    nb = b.group_by(on).len().rename({"len": "nb"})
    no = o.group_by(on).len().rename({"len": "no"})
    size = nb.join(no, on=on).select((pl.col("nb") * pl.col("no")).sum()).item()
    if size is None or size > _MAX_PAIRS:
        return None
    # A group: the base rows that share every key value (and the window), such
    # as the vehicles of one policy term. Events that match a whole group sit
    # at that grain, not at the grain of the base rows.
    b = b.with_columns(pl.struct(on + ([window.start, window.end] if window else [])).hash(seed=7).alias("__g"))
    pairs = o.join(b, on=on, how="inner")
    if window:
        d, s, e = pl.col(window.date), pl.col(window.start), pl.col(window.end)
        pairs = pairs.filter((s <= d) & (d < e))
    return pairs.select("__b", "__o", "__g")


def _fit(base, other, keys, window=None) -> _Fit | None:
    pairs = _pairs(base, other, keys, window)
    if pairs is None:
        return None
    per_o = pairs.group_by("__o").agg(pl.len().alias("len"), pl.col("__g").n_unique().alias("g"))
    per_b = pairs.group_by("__b").len()
    return _Fit(
        keys=keys, window=window,
        match_other=per_o.height / max(other.height, 1),
        match_base=per_b.height / max(base.height, 1),
        amb_other=float((per_o["len"] > 1).mean() or 0.0) if per_o.height else 0.0,
        amb_base=float((per_b["len"] > 1).mean() or 0.0) if per_b.height else 0.0,
        amb_group=float((per_o["g"] > 1).mean() or 0.0) if per_o.height else 0.0,
        group_rows=float(per_o["len"].mean() or 1.0) if per_o.height else 1.0,
    )


def _comparable(base: pl.DataFrame, other: pl.DataFrame, bc: str, oc: str) -> bool:
    """Cheap screen for a second key column: same kind of value, shared values."""
    bt, ot = base[bc].dtype, other[oc].dtype
    if bt in (pl.Float32, pl.Float64) or ot in (pl.Float32, pl.Float64):
        return False
    kind = lambda t: "date" if t == pl.Date else "int" if t.is_integer() else "text" if t == pl.Utf8 else None  # noqa: E731
    if kind(bt) is None or kind(bt) != kind(ot):
        return False
    bv = set(base[bc].drop_nulls().unique().to_list())
    ov = other[oc].drop_nulls()
    return len(ov) > 0 and float(ov.is_in(list(bv)).mean() or 0.0) >= 0.5


def _windows(base: pl.DataFrame, other: pl.DataFrame) -> list[tuple[str, str, str]]:
    starts_ends = []
    dates_b = [c for c, t in base.schema.items() if t == pl.Date]
    for s in dates_b:
        for e in dates_b:
            if s == e:
                continue
            ok = base.select(((pl.col(s) < pl.col(e)) | pl.col(s).is_null() | pl.col(e).is_null()).mean()).item()
            if ok is not None and ok >= 0.99:
                starts_ends.append((s, e))
    dates_o = [c for c, t in other.schema.items() if t == pl.Date]
    return [(s, e, d) for s, e in starts_ends for d in dates_o]


def _resolve(base: pl.DataFrame, other: pl.DataFrame, first: KeyPair, aim: str) -> _Fit | None:
    """Refine a key with more columns and a date window until each row has
    one partner: one per base row for a lookup, one per event for a rollup.

    A small beam, not a greedy walk: the refinement that helps most first
    (term number against vehicle number, say) can be a dead end that the
    right one (vehicle number, then loss date) is not. Of the keys that
    resolve, the one placing the most rows wins, then the shortest."""
    start = _fit(base, other, [first])
    if start is None:
        return None
    columns = [(bc, oc) for bc in base.columns for oc in other.columns
               if bc != first.base and oc != first.other and _comparable(base, other, bc, oc)]
    windows = _windows(base, other) if aim in ("rollup", "group") else []

    def refinements(fit: _Fit):
        used_b = {k.base for k in fit.keys}
        used_o = {k.other for k in fit.keys}
        for bc, oc in columns:
            if bc not in used_b and oc not in used_o:
                yield fit.keys + [KeyPair(bc, oc)], fit.window
        if fit.window is None:
            for s_, e, d in windows:
                yield fit.keys, Window(s_, e, d, 0.0)

    def size(fit: _Fit) -> int:
        return len(fit.keys) + (fit.window is not None)

    best_open, done = [start], []
    if start.ambiguity(aim) <= _RESOLVED:
        done.append(start)
    seen: set = set()
    for _ in range(3):
        frontier = []
        for fit in best_open:
            if fit.ambiguity(aim) <= _RESOLVED and aim != "group":
                continue
            for keys, window in refinements(fit):
                sig = (frozenset((k.base, k.other) for k in keys), window and (window.start, window.end, window.date))
                if sig in seen:
                    continue
                seen.add(sig)
                f = _fit(base, other, keys, window)
                if f is None or f.match_other < _KEEP_MATCH * start.match_other:
                    continue
                if aim == "group" and fit.ambiguity(aim) <= _RESOLVED:
                    # Already one group per event: refine only to a finer grain
                    # (policy -> policy term) that still keeps one group.
                    if f.ambiguity(aim) > _RESOLVED or f.group_rows >= 0.95 * fit.group_rows:
                        continue
                elif f.ambiguity(aim) >= 0.8 * fit.ambiguity(aim):
                    continue
                (done if f.ambiguity(aim) <= _RESOLVED else frontier).append(f)
        best_open = sorted(frontier, key=lambda f: -f.placed(aim))[:_BEAM]
        if not best_open:
            break
    pool = done or [start]
    if aim == "group":
        # The finest grain that still places nearly as many events: per policy
        # term rather than per policy, unless the term costs many events.
        top = max(f.placed(aim) for f in pool)
        near = [f for f in pool if f.placed(aim) >= 0.9 * top]
        fit = min(near, key=lambda f: (round(f.group_rows, 6), -round(f.placed(aim), 6), size(f)))
    else:
        fit = max(pool, key=lambda f: (round(f.placed(aim), 6), -size(f)))
    if fit.window is not None:
        fit.window.share = fit.placed(aim)
    return fit


def _per(x: float) -> str:
    return f"{x:.0%}" if x >= 0.995 or x <= 0.005 else f"{x:.1%}"


def _explain(base: pl.DataFrame, other: pl.DataFrame, fit: _Fit, aim: str, base_label: str, other_label: str) -> list[str]:
    """Why this key, in the order an actuary would check it."""
    first = fit.keys[0]
    out = [f"{first.other} ↔ {first.base}"
           + (f": '{first.example[1]}' and '{first.example[0]}' are the same policy once the prefix and "
              "leading zeros are dropped." if first.form == "digits" and first.example else ".")]
    alone = _fit(base, other, [first])
    if alone is not None and (len(fit.keys) > 1 or fit.window):
        if aim == "lookup":
            out.append(f"On its own it matches {_per(alone.amb_base)} of {base_label} rows to several "
                       f"{other_label} rows (renewals repeat the number).")
        else:
            out.append(f"On its own it matches {_per(alone.amb_other)} of {other_label} rows to several "
                       f"{base_label} rows (renewals and multi-unit policies repeat the number).")
    for k in fit.keys[1:]:
        out.append(f"{k.other} ↔ {k.base} narrows it.")
    if fit.window:
        w = fit.window
        out.append(f"{other_label} has no term, only dates: {w.date} between {w.start} and {w.end} "
                   "places each row in one term.")
        for c, t in other.schema.items():
            if t == pl.Date and c != w.date:
                alt = _fit(base, other, fit.keys, Window(w.start, w.end, c, 0.0))
                if alt is not None:
                    out.append(f"{c} instead would place {_per(alt.placed(aim))} of rows, against "
                               f"{_per(fit.placed(aim))} for {w.date}.")
    return out


# ── The plan ──────────────────────────────────────────────────────────────────


def _count_id(other: pl.DataFrame, used: set[str]) -> str | None:
    """The column that identifies one event across its rows (a claim number
    repeated once per coverage), or None if each row is one event."""
    options = []
    for c, t in other.schema.items():
        if c in used or _PART in c or t in (pl.Float32, pl.Float64, pl.Date, pl.Boolean) or not (t == pl.Utf8 or t.is_integer()):
            continue
        n = other[c].n_unique()
        if n < 0.2 * other.height:
            continue
        options.append((bool(_CLAIM_ID.search(c)), n, c))
    if not options:
        return None
    named, n, c = max(options)
    return c if named or n < other.height else None


def _sum_columns(other: pl.DataFrame, used: set[str]) -> list[str]:
    return [c for c, t in other.schema.items()
            if c not in used and _PART not in c and t.is_numeric() and _ADDITIVE.search(c) and not _EXPOSURE.search(c)]


def _no_dates(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([c for c, t in df.schema.items() if t != pl.Date])


def _label(name: str) -> str:
    return re.sub(r"\.(csv|xlsx?|txt)$", "", name, flags=re.IGNORECASE)


def _attach_lookup(cur: pl.DataFrame, other: pl.DataFrame, step: Step, label: str) -> pl.DataFrame:
    b = _keyed(cur, step.keys, "base", [], "__b")
    o = _keyed(other, step.keys, "other", [], "__o")
    on = [f"__k{i}" for i in range(len(step.keys))]
    dup = o.group_by(on).len().filter(pl.col("len") > 1).select(on)
    o_unique = o.join(dup, on=on, how="anti")
    pairs = b.join(o_unique, on=on, how="inner").select("__b", "__o")
    keep = [c for c in other.columns if c not in {k.other for k in step.keys} and _PART not in c]
    rename = {c: (f"{c}_{label}" if c in cur.columns else c) for c in keep}
    right = other.select(keep).rename(rename).with_columns(pl.int_range(other.height, dtype=pl.UInt32).alias("__o"))
    out = (cur.with_columns(pl.int_range(cur.height, dtype=pl.UInt32).alias("__b"))
              .join(pairs, on="__b", how="left").join(right, on="__o", how="left").drop("__b", "__o"))
    used_o = pairs["__o"].n_unique()
    step.added_columns = list(rename.values())
    step.totals = {
        "base_rows_before": cur.height, "base_rows_after": out.height,
        "base_rows_matched": pairs.height, "base_rows_unmatched": cur.height - pairs.height,
        "other_rows": other.height, "other_rows_used": used_o, "other_rows_unused": other.height - used_o,
        "other_keys_ambiguous": dup.height,
    }
    return out


def _rollup_status(cur: pl.DataFrame, other: pl.DataFrame, step: Step, group: bool = False):
    """Each event row's fate: attached to one row, or why not. Returns the
    events with `__o` and `__status`, the (row, event) pairs that attach, and
    the pairs that share the key whatever the date."""
    keyed_only = _pairs(cur, other, step.keys, None)
    pairs = _pairs(cur, other, step.keys, step.window)
    per_o = pairs.group_by("__o").agg(pl.len().alias("len"), pl.col("__g").n_unique().alias("g"))
    one = (pl.col("g") == 1) if group else (pl.col("len") == 1)
    single = per_o.filter(one).select("__o")
    assigned = pairs.join(single, on="__o", how="inner")

    o = other.with_columns(pl.int_range(other.height, dtype=pl.UInt32).alias("__o"))
    o = o.with_columns(
        pl.when(pl.col("__o").is_in(assigned["__o"].implode())).then(pl.lit("attached"))
        .when(pl.col("__o").is_in(per_o.filter(~one)["__o"].implode())).then(pl.lit("ambiguous"))
        .when(pl.col("__o").is_in(keyed_only["__o"].unique().implode())).then(pl.lit("outside_window"))
        .otherwise(pl.lit("no_key")).alias("__status")
    )
    return o, assigned, keyed_only


def _attach_rollup(cur: pl.DataFrame, other: pl.DataFrame, step: Step) -> pl.DataFrame:
    o, assigned, _ = _rollup_status(cur, other, step)
    unit = step.count_of
    count = (pl.col(unit).n_unique() if unit else pl.len()).alias(step.count_column)
    aggs = [count] + [pl.col(c).sum().alias(f"{c}_sum") for c in step.sums]
    rolled = o.join(assigned, on="__o", how="inner").group_by("__b").agg(aggs)

    out = (cur.with_columns(pl.int_range(cur.height, dtype=pl.UInt32).alias("__b"))
              .join(rolled, on="__b", how="left").drop("__b"))
    generated = [step.count_column] + [f"{c}_sum" for c in step.sums]
    out = out.with_columns([pl.col(c).fill_null(0) for c in generated])
    step.added_columns = generated

    def units(frame: pl.DataFrame) -> int:
        return frame[unit].n_unique() if unit else frame.height

    by = {s: o.filter(pl.col("__status") == s) for s in ("attached", "no_key", "outside_window", "ambiguous")}
    step.totals = {
        "base_rows_before": cur.height, "base_rows_after": out.height,
        "other_rows": other.height, "unit": unit or "row",
        "units_in_file": units(o),
        **{f"units_{s}": units(f) for s, f in by.items()},
        "units_in_table": int(out[step.count_column].sum()),
        "amounts": {
            c: {"in_file": float(o[c].sum() or 0.0),
                **{s: float(f[c].sum() or 0.0) for s, f in by.items()},
                "in_table": float(out[f"{c}_sum"].sum() or 0.0)}
            for c in step.sums
        },
    }
    return out


def assemble(tables: dict[str, pl.DataFrame], exposure_hints: dict[str, str | None] | None = None,
             count_column: str = "claim_count", *, repair: bool | str = True,
             controls: dict[str, float] | None = None) -> tuple[pl.DataFrame, Plan]:
    """Join every table onto the one that carries exposure. Returns the
    modelling table and the plan, with each step's control totals, the key
    corrections it made (`repair=False` to make none) and its findings.
    `controls` maps a column of the result to a total from finance to tie to."""
    if len(tables) < 2:
        raise ValueError("Assembling needs at least two tables.")
    _NORM_CACHE.clear()
    try:
        return _assemble(tables, exposure_hints, count_column, repair, controls)
    finally:
        _NORM_CACHE.clear()


def _assemble(tables, exposure_hints, count_column, repair, controls):
    hints = exposure_hints or {}
    inputs = {name: parse_dates(df) for name, df in tables.items()}
    frames = {name: split_composites(df) for name, df in inputs.items()}
    expo = {name: exposure_column(df, hints.get(name)) for name, df in frames.items()}
    with_expo = [n for n in frames if expo[n]]
    base = max(with_expo or list(frames), key=lambda n: frames[n].height)
    cur = frames[base]
    exposure_total = float(cur[expo[base]].sum()) if expo[base] else None

    plan = Plan(base=base, exposure=expo[base], steps=[], unresolved=[])
    known_cache: dict = {}
    remaining = [n for n in frames if n != base]
    while remaining:
        found: list[tuple[int, float, str, _Fit, KeyPair]] = []
        reasons_failed: dict[str, str] = {}
        for name in remaining:
            other = frames[name]
            # Dates place events in time (windows, below); as a first key
            # they only coincide.
            cands = detect_join_keys(_no_dates(cur), _no_dates(other)).candidates[:3]
            if not cands:
                reasons_failed[name] = "No column shares values with the table built so far."
                continue
            best = None
            for c, aim in ((c, aim) for c in cands for aim in ("lookup", "rollup", "group")):
                first = KeyPair(c.left_column, c.right_column, c.key_form, c.example)
                fit = _resolve(cur, other, first, aim)
                if fit is None or fit.ambiguity(aim) > _RESOLVED:
                    continue
                if aim == "lookup" and (fit.match_base < _LOOKUP_COVERAGE or _sum_columns(other, set())):
                    continue
                # Lookups first: they bring the columns (term dates) a later
                # rollup may need to place its events.
                rank = (_AIMS.index(aim), -fit.match_other)
                if best is None or rank < best[0]:
                    best = (rank, fit, first)
            if best is None:
                c = cands[0]
                reasons_failed[name] = (
                    f"{c.right_column} matches {c.left_column}, but rows still match several partners and no other "
                    "column or date narrows them to one.")
                continue
            (rank0, rank1), fit, first = best
            found.append((rank0, rank1, name, fit, first))
        if not found:
            plan.unresolved += [{"table": n, "reason": reasons_failed.get(n, "")} for n in remaining]
            break
        rank0, _, name, fit, first = min(found, key=lambda f: (f[0], f[1]))
        aim = _AIMS[rank0]
        other = frames[name]
        fixes: list = []
        stats: dict = {}
        if repair:
            tiers = REPAIR_TIERS if repair == "all" else REPAIR_DEFAULT
            fixes, suspects, stats = find_repairs(cur, other, fit.keys, fit.window, aim, base_name=base, other_name=name,
                                           evaluate=_evaluator(cur, other, fit.keys, fit.window, aim), apply_tiers=tiers,
                                           known=_known_ids(inputs, fit.keys[0].form, exclude=base if aim == "lookup" else name,
                                                            cache=known_cache))
            plan.suspects += [s.__dict__ for s in suspects]
            if fixes:
                mapping = {f.original: f.corrected for f in fixes}
                if aim == "lookup":
                    cur = apply_repairs(cur, fit.keys[0].base, mapping)
                else:
                    other = frames[name] = apply_repairs(other, fit.keys[0].other, mapping)
                fit = _fit(cur, other, fit.keys, fit.window) or fit
        reasons = _explain(cur, other, fit, "lookup" if aim == "lookup" else "rollup", _label(base), _label(name))
        if fixes:
            n = sum(f.rows for f in fixes)
            line = (f"{len(fixes)} mistyped key value{'s' if len(fixes) > 1 else ''} corrected first "
                    f"({n:,} row{'s' if n > 1 else ''}); each is listed in the findings.")
            if "expected_wrong" in stats:
                line += (f" Chance check: a key that is missing, not mistyped, finds a convincing near match "
                         f"{stats['chance_rate']:.1%} of the time on this book, so "
                         + ("none of these is expected to be wrong." if stats["expected_wrong"] < 0.05
                            else f"about {stats['expected_wrong']:.1f} of them could be."))
            reasons.append(line)
        elif stats.get("withheld"):
            reasons.append("Near matches found but not applied: on this book a missing key finds a convincing near match "
                           f"{stats['chance_rate']:.1%} of the time, too often to trust them. They are listed as likely typos.")
        if aim == "lookup":
            step = Step(table=name, kind="lookup", keys=fit.keys, reasons=reasons)
            cur = _attach_lookup(cur, other, step, _label(name))
        else:
            used = {k.other for k in fit.keys} | ({fit.window.date} if fit.window else set())
            if aim == "group":
                counter = f"{_label(name)}_count"
            else:
                # A second events table (convictions, say) gets its own count.
                counter = count_column if count_column not in cur.columns else f"{_label(name)}_count"
            step = Step(table=name, kind=aim, keys=fit.keys, window=fit.window,
                        count_of=_count_id(other, used), count_column=counter,
                        sums=_sum_columns(other, used), reasons=reasons)
            if step.count_of and other[step.count_of].n_unique() < other.height:
                step.reasons.append(
                    f"{step.count_of} repeats across rows ({other.height:,} rows for "
                    f"{other[step.count_of].n_unique():,} distinct): each is counted once.")
            if aim == "group":
                cur = _attach_group(cur, other, step, expo[base])
                g = step.totals.get("rows_per_group", 1)
                step.reasons.append(
                    f"Each row belongs to a group of {g:g} {_label(base)} rows on average (the rows sharing "
                    + " and ".join(k.base for k in fit.keys) + (f" and the {fit.window.start} to {fit.window.end} window"
                    if fit.window else "") + "), not to one row.")
                if step.count_column:
                    step.reasons.append(f"{step.count_column} is the group's count, repeated on each of its rows: "
                                        "an attribute, not something to add up.")
                if step.allocated:
                    step.reasons.append("Split across each group's rows by " + (expo[base] or "equal shares")
                                        + ", so the total still ties: " + ", ".join(step.allocated) + ".")
            else:
                cur = _attach_rollup(cur, other, step)
        step.repairs = [f.as_dict() for f in fixes]
        if repair:
            step.repair_stats = stats
        plan.steps.append(step)
        remaining.remove(name)

    cur = _visible(cur)
    plan.totals = {
        "rows": cur.height, "base_rows": frames[base].height,
        "exposure_in": exposure_total,
        "exposure_out": float(cur[expo[base]].sum()) if expo[base] else None,
    }
    from tieback.checks import run_checks
    plan.findings = run_checks(inputs, plan, cur, controls or {})
    return cur, plan


def _shares(cur: pl.DataFrame, bg: pl.DataFrame, exposure: str | None) -> pl.DataFrame:
    """Each base row's share of its group: by exposure where there is one,
    equal shares otherwise (or where the group's exposure is zero)."""
    e = (cur[exposure].cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0) if exposure
         else pl.Series([1.0] * cur.height))
    w = bg.join(pl.DataFrame({"__b": pl.int_range(cur.height, dtype=pl.UInt32, eager=True), "__e": e}), on="__b")
    return w.with_columns(
        pl.when(pl.col("__e").sum().over("__g") > 0)
        .then(pl.col("__e") / pl.col("__e").sum().over("__g"))
        .otherwise(1.0 / pl.len().over("__g")).alias("__share"))


def _attach_group(cur: pl.DataFrame, other: pl.DataFrame, step: Step, exposure: str | None) -> pl.DataFrame:
    """Events that belong to a group of base rows (convictions per policy
    term, premium per term) rather than to one row. Counts are the group's
    count, repeated on each of its rows (an attribute: do not add it up).
    Amounts are split across the group's rows by exposure, so they still add
    up to the file."""
    o, assigned, _ = _rollup_status(cur, other, step, group=True)
    unit = step.count_of
    ev = o.join(assigned.select("__o", "__g").unique(), on="__o", how="inner")
    bg = assigned.select("__b", "__g").unique()
    per_g = ev.group_by("__g").agg(
        [(pl.col(unit).n_unique() if unit else pl.len()).alias("__n")] + [pl.col(c).sum().alias(c) for c in step.sums])
    w = _shares(cur, bg, exposure).join(per_g, on="__g", how="left")
    cols = []
    one_each = per_g.height == 0 or int(per_g["__n"].max() or 0) <= 1
    if not (one_each and not unit):
        cols.append(pl.col("__n").alias(step.count_column))
    cols += [(pl.col(c) * pl.col("__share")).alias(f"{c}_sum") for c in step.sums]
    rows = w.select([pl.col("__b")] + cols)
    out = (cur.with_columns(pl.int_range(cur.height, dtype=pl.UInt32).alias("__b"))
              .join(rows, on="__b", how="left").drop("__b"))
    generated = [c for c in rows.columns if c != "__b"]
    out = out.with_columns([pl.col(c).fill_null(0) for c in generated])
    step.added_columns = generated
    step.allocated = list(step.sums)
    if not cols or step.count_column not in generated:
        step.count_column = None

    def units(frame: pl.DataFrame) -> int:
        return frame[unit].n_unique() if unit else frame.height

    by = {s: o.filter(pl.col("__status") == s) for s in ("attached", "no_key", "outside_window", "ambiguous")}
    step.totals = {
        "base_rows_before": cur.height, "base_rows_after": out.height,
        "other_rows": other.height, "unit": unit or "row", "grain": "group",
        "groups": per_g.height, "rows_per_group": round(bg.height / max(per_g.height, 1), 2),
        "units_in_file": units(o),
        **{f"units_{s}": units(f) for s, f in by.items()},
        "units_in_table": int(per_g["__n"].sum() or 0),
        "amounts": {
            c: {"in_file": float(o[c].sum() or 0.0),
                **{s: float(f[c].sum() or 0.0) for s, f in by.items()},
                "in_table": float(out[f"{c}_sum"].sum() or 0.0)}
            for c in step.sums
        },
    }
    return out


# ── Replaying a saved plan ────────────────────────────────────────────────────


def _step_from(d: dict) -> Step:
    keys = [KeyPair(k["base"], k["other"], k.get("form", "exact"),
                    tuple(k["example"]) if k.get("example") else None) for k in d["keys"]]
    w = d.get("window")
    return Step(table=d["table"], kind=d["kind"], keys=keys,
                window=Window(w["start"], w["end"], w["date"], w.get("share", 0.0)) if w else None,
                count_of=d.get("count_of"), count_column=d.get("count_column"), sums=list(d.get("sums") or []))


def _unique_on(df: pl.DataFrame, cols: list[str]) -> bool:
    return bool(cols) and all(c in df.columns for c in cols) and df.select(cols).is_unique().all()


def _grain(df: pl.DataFrame, keys: list[str], unit: str | None = None) -> str:
    if unit and unit in df.columns:
        n = df[unit].n_unique()
        return f"{df.height:,} rows for {n:,} {unit}s" if n < df.height else f"one row per {unit}"
    if _unique_on(df, keys):
        return "one row per " + " and ".join(keys)
    return f"{df.height:,} rows; {' and '.join(keys)} repeat" if keys else f"{df.height:,} rows"


def replay(tables: dict[str, pl.DataFrame], plan: dict) -> dict:
    """Run a saved plan again, as it was decided (no keys re-detected), and
    say which rows of each file did not make it into the table and why.
    Returns each file's grain and, per step, the missed rows of its file."""
    frames = {name: _prepare(df) for name, df in tables.items()}
    base = plan["base"]
    cur = frames[base]
    base_keys = list(dict.fromkeys(k["base"] for s in plan["steps"] for k in s["keys"] if k["base"] in cur.columns))
    grain = {base: _grain(cur, base_keys)}
    steps = []
    for d in plan["steps"]:
        other = frames[d["table"]]
        step = _step_from(d)
        for fix in d.get("repairs") or []:
            mapping = {fix["original"]: fix["corrected"]}
            if fix["table"] == d["table"]:
                other = apply_repairs(other, fix["column"], mapping)
            else:
                cur = apply_repairs(cur, fix["column"], mapping)
        first = step.keys[0]
        misses = []
        if step.kind == "lookup":
            grain[step.table] = _grain(other, [k.other for k in step.keys])
            b = _keyed(cur, step.keys, "base", [], "__b")
            o = _keyed(other, step.keys, "other", [], "__o")
            on = [f"__k{i}" for i in range(len(step.keys))]
            dup = o.group_by(on).len().filter(pl.col("len") > 1).select(on)
            used = o.join(dup, on=on, how="anti").join(b.select(on).unique(), on=on, how="semi")["__o"]
            first_only = _keyed(cur, [first], "base", [], "__b")
            o1 = _keyed(other, [first], "other", [], "__o")
            on_first = o1.join(first_only.select("__k0").unique(), on="__k0", how="semi")["__o"]
            in_dup = o.join(dup, on=on, how="semi")["__o"]
            idx = pl.int_range(other.height, dtype=pl.UInt32, eager=True)
            fate = pl.DataFrame({"__o": idx}).with_columns(
                pl.when(pl.col("__o").is_in(used.implode())).then(pl.lit("used"))
                .when(pl.col("__o").is_in(in_dup.implode())).then(pl.lit("listed_twice"))
                .when(pl.col("__o").is_in(on_first.implode())).then(pl.lit("second_key"))
                .otherwise(pl.lit("no_key")).alias("__why"))
            missed = other.with_columns(fate["__why"]).with_row_index("__o").filter(pl.col("__why") != "used")
            for why, frame in missed.group_by("__why", maintain_order=True):
                misses.append({"why": why[0], "rows": frame})
            cur = _attach_lookup(cur, other, step, _label(step.table))
        else:
            grain[step.table] = _grain(other, [k.other for k in step.keys], step.count_of)
            o, _, keyed_only = _rollup_status(cur, other, step, group=step.kind == "group")
            missed = o.filter(pl.col("__status") != "attached")
            w = step.window
            if w is not None and missed.filter(pl.col("__status") == "outside_window").height:
                # The nearest term each stray event missed, and by how many days.
                near = (keyed_only.join(missed.select("__o", w.date), on="__o")
                        .join(cur.select(w.start, w.end).with_row_index("__b"), on="__b")
                        .with_columns(
                            pl.when(pl.col(w.date) >= pl.col(w.end))
                            .then((pl.col(w.date) - pl.col(w.end)).dt.total_days())
                            .otherwise((pl.col(w.date) - pl.col(w.start)).dt.total_days()).alias("__days"))
                        .with_columns(pl.col("__days").abs().alias("__dist"))
                        .sort(["__o", "__dist"]).unique("__o", keep="first")
                        .select("__o", pl.col(w.start).alias("__term_start"), pl.col(w.end).alias("__term_end"), "__days"))
                missed = missed.join(near, on="__o", how="left")
            if len(step.keys) > 1:
                first_hit = _keyed(other, [first], "other", [], "__o").join(
                    _keyed(cur, [first], "base", [], "__b").select("__k0").unique(), on="__k0", how="semi")["__o"]
                missed = missed.with_columns(pl.col("__o").is_in(first_hit.implode()).alias("__first_hit"))
            def why(row: dict) -> str:
                st = row["__status"]
                if st == "outside_window":
                    return "before_start" if (row.get("__days") or 0) < 0 else "after_end"
                if st == "no_key" and row.get("__first_hit"):
                    return "second_key"
                return st
            reasons = [why(r) for r in missed.iter_rows(named=True)]
            missed = missed.with_columns(pl.Series("__why", reasons, dtype=pl.Utf8)).drop("__status")
            for why_, frame in missed.group_by("__why", maintain_order=True):
                misses.append({"why": why_[0], "rows": frame})
            cur = _attach_group(cur, other, step, plan.get("exposure")) if step.kind == "group" else _attach_rollup(cur, other, step)
        steps.append({"table": step.table, "kind": step.kind, "misses": misses})
    return {"grain": grain, "steps": steps}
