"""Mistyped keys: find them, and correct the ones the data can vouch for.

A claim filed against P0412430 when the policy is 412403 does not join, and
it drops out of the book as "no such policy". Most of those are typing
errors, of three kinds, tried in this order:

  case and spacing   'p0412403 ', 'P-0412403'      the same characters
  swapped neighbours 412430 for 412403              one adjacent swap
  one character      412408 for 412403, 41243       one change, insertion or deletion

A candidate is only a candidate. On a book with dense, sequential policy
numbers, a one-digit slip lands on dozens of real policies, and a claim on a
policy that is simply missing from the extract often sits one digit away from
a real one by chance. So a candidate must be vouched for by the rest of the
join: with the corrected number, the claim's vehicle and loss date fall in
exactly one term of exactly one candidate (with no other column to check, it
must be the only candidate at all).

Even then, only the two kinds of slip that chance rarely produces are applied
by default: case and spacing, and swapped neighbours. A single changed
character is reported as a likely correction with its evidence, and applied
only with repair="all". Nothing ambiguous is ever guessed.

Pure compute: frames in, corrected frame and the list of corrections out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

# Keys shorter than this (after normalising) are codes, not ids: one slip
# turns a code into another real code.
_MIN_LEN = 5
_TIERS = ("case and spacing", "swapped neighbours", "one character")


@dataclass
class Repair:
    table: str
    column: str
    original: str        # the raw value as it appears in the file
    corrected: str       # the raw value written in its place
    rows: int
    kind: str            # one of _TIERS
    evidence: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Suspect:
    """An unmatched key with near matches: either one the data vouches for
    but that is not applied by default (`likely`), or several that the data
    cannot tell apart."""
    table: str
    column: str
    value: str
    rows: int
    candidates: list[str]
    kind: str
    likely: str | None = None


def _loose(s: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", s.upper())


def _distance_one(a: str, b: str) -> str | None:
    """The tier of a one-edit difference between a and b, or None."""
    if a == b:
        return None
    la, lb = len(a), len(b)
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return _TIERS[2]
        if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
            return _TIERS[1]
        return None
    if abs(la - lb) != 1:
        return None
    short, long_ = (a, b) if la < lb else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return _TIERS[2] if short[i:] == long_[i + 1:] else None


class _Index:
    """Symmetric-delete index: values within one edit share a deletion."""

    def __init__(self, values: list[str]):
        self.by_loose: dict[str, set[str]] = {}
        self.dels: dict[str, set[str]] = {}
        for v in values:
            lv = _loose(v)
            self.by_loose.setdefault(lv, set()).add(v)
            for d in {lv} | {lv[:i] + lv[i + 1:] for i in range(len(lv))}:
                self.dels.setdefault(d, set()).add(lv)

    def near(self, value: str) -> dict[str, set[str]]:
        """Target values one tier away, by tier."""
        lv = _loose(value)
        out: dict[str, set[str]] = {}
        if lv in self.by_loose:
            out[_TIERS[0]] = set(self.by_loose[lv])
        seen: set[str] = set()
        for d in {lv} | {lv[:i] + lv[i + 1:] for i in range(len(lv))}:
            for cand in self.dels.get(d, ()):
                if cand in seen or cand == lv:
                    continue
                seen.add(cand)
                tier = _distance_one(lv, cand)
                if tier:
                    out.setdefault(tier, set()).update(self.by_loose[cand])
        return out


def rewrite(raw: str, form: str, target: str) -> str:
    """The raw value with its key replaced by `target`, keeping its own format
    ('P0412430' corrected to 412403 stays 'P0412403')."""
    if form == "digits":
        m = list(re.finditer(r"\d+", raw))
        if len(m) == 1:
            run = m[0]
            return raw[:run.start()] + target.zfill(len(run.group())) + raw[run.end():]
    return target


def _with_value(df: pl.DataFrame, col: str, rows: pl.Series, value: str) -> pl.DataFrame:
    """The given rows of df with `col` set to value, cast to the column's type."""
    sub = df[rows]
    dtype = df[col].dtype
    try:
        new = pl.Series(col, [value] * sub.height).cast(dtype, strict=True)
    except Exception:
        return sub.head(0)
    return sub.with_columns(new)


APPLY_DEFAULT = (_TIERS[0], _TIERS[1])
# Above this expected share of wrong corrections, none are applied.
MAX_WRONG_SHARE = 0.2


def chance_rate(src: pl.DataFrame, s_col: str, ns: pl.Series, targets: set[str], index: "_Index", first,
                evaluate, apply_tiers: tuple[str, ...], n: int = 200, seed: int = 7) -> float:
    """How often a key that is wrong but not mistyped (a policy missing from
    the extract) would still be 'corrected' on this book: real rows are given
    random keys that exist nowhere, and the share that find exactly one near
    key whose other columns and dates fit is the rate of coincidence."""
    import random

    rng = random.Random(seed)
    live = [i for i, v in enumerate(ns.to_list()) if v is not None and v in targets]
    if not live:
        return 0.0
    rows = rng.sample(live, min(n, len(live)))
    hits = 0
    for i in rows:
        v = ns[i]
        decoy = None
        for _ in range(20):
            cand = "".join(rng.choice("0123456789") if ch.isdigit() else ch for ch in v)
            if cand not in targets and cand.lstrip("0") == cand:
                decoy = cand
                break
        if decoy is None:
            continue
        near = index.near(decoy)
        cands = [c for tier in apply_tiers for c in near.get(tier, ())]
        if not cands:
            continue
        one = pl.Series([i], dtype=pl.UInt32)
        if sum(1 for c in cands if evaluate(src, s_col, one, c) == 1) == 1:
            hits += 1
    return hits / len(rows)


def find_repairs(base: pl.DataFrame, other: pl.DataFrame, keys, window, aim: str, *,
                 base_name: str, other_name: str, evaluate,
                 apply_tiers: tuple[str, ...] = APPLY_DEFAULT,
                 known: set[str] = frozenset()) -> tuple[list[Repair], list[Suspect], dict]:
    """Corrections for the first key of a join step, and the suspects that
    could not be corrected.

    For a lookup the base rows without a partner are the ones checked (a
    vehicle whose policy number was mistyped); for a roll-up, the events
    (a claim against a policy that does not exist). `evaluate(src_rows_frame,
    candidate)` is supplied by the plan: it says how many of those rows join
    with the candidate key in place, under the step's other keys and window.
    """
    from tieback.keys import normalize_key

    first = keys[0]
    src, tgt = (base, other) if aim == "lookup" else (other, base)
    s_col, t_col = (first.base, first.other) if aim == "lookup" else (first.other, first.base)
    s_name = base_name if aim == "lookup" else other_name
    ns = normalize_key(src[s_col], first.form) if src[s_col].dtype != pl.Date else None
    nt = normalize_key(tgt[t_col], first.form) if tgt[t_col].dtype != pl.Date else None
    if ns is None or nt is None:
        return [], [], {}
    targets = set(nt.drop_nulls().unique().to_list())
    orphans = [v for v in ns.drop_nulls().unique().to_list() if v not in targets and len(_loose(v)) >= _MIN_LEN]
    if not orphans:
        return [], [], {"unmatched": 0, "corrected": 0}
    index = _Index(sorted(targets))
    raw = src[s_col].cast(pl.Utf8)
    corroborate = len(keys) > 1 or window is not None
    repairs, suspects = [], []
    for v in orphans:
        if v in known:
            continue  # a real id elsewhere in the extracts: missing here, not mistyped
        rows = (ns == v).arg_true()
        tiers = index.near(v)
        if not tiers:
            continue
        original = str(raw[int(rows[0])])
        # Judge the slip on the value as written: 'P9016838' for 'P0916838' is
        # a swap, though without its leading zero it looks like an insertion.
        found: dict[str, str] = {}
        for tier in _TIERS:
            for c in tiers.get(tier, ()):
                as_written = rewrite(original, first.form, c)
                found[c] = (_TIERS[0] if _loose(as_written) == _loose(original)
                            else _distance_one(_loose(original), _loose(as_written)) or tier)
        cands = sorted(found, key=lambda c: (_TIERS.index(found[c]), c))
        if corroborate:
            ok = [c for c in cands if evaluate(src, s_col, rows, c) == len(rows)]
        else:
            ok = cands if len(cands) == 1 else []
        if len(ok) != 1:
            # Nothing fits, or several do (a swap and a changed digit can both
            # land on real policies whose vehicle and dates fit): never guess.
            shown = ok if len(ok) > 1 else cands
            suspects.append(Suspect(s_name, s_col, original, len(rows),
                                    [rewrite(original, first.form, c) for c in shown[:6]], found[shown[0]]))
            continue
        chosen = ok[0]
        kind = found[chosen]
        evidence = ("its other keys and dates match only this one" if corroborate
                    else "the only value one keystroke away")
        corrected = rewrite(original, first.form, chosen)
        if kind not in apply_tiers:
            suspects.append(Suspect(s_name, s_col, original, len(rows), [corrected], kind, likely=corrected))
            continue
        repairs.append(Repair(s_name, s_col, original, corrected, len(rows), kind, evidence))
    stats = {"unmatched": len([v for v in orphans if v not in known]), "corrected": len(repairs)}
    if repairs and corroborate:
        rate = chance_rate(src, s_col, ns, targets, index, first, evaluate, apply_tiers)
        expected = rate * stats["unmatched"]
        stats.update(chance_rate=round(rate, 4), expected_wrong=round(expected, 2))
        if expected > MAX_WRONG_SHARE * len(repairs):
            # Too many coincidences on this book to trust any one correction.
            suspects += [Suspect(r.table, r.column, r.original, r.rows, [r.corrected], r.kind, likely=r.corrected)
                         for r in repairs]
            repairs = []
            stats["withheld"] = True
    return repairs, suspects, stats


def apply(df: pl.DataFrame, column: str, mapping: dict[str, str]) -> pl.DataFrame:
    """Replace raw values of `column` (compared as text) by their corrections,
    keeping the column's type."""
    if not mapping:
        return df
    s = df[column]
    fixed = s.cast(pl.Utf8).replace(mapping)
    try:
        fixed = fixed.cast(s.dtype, strict=True)
    except Exception:
        return df
    return df.with_columns(fixed.alias(column))
