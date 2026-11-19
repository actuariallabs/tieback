"""Multi-dataset reconciliation — infer join keys from values, not names.

An actuary rarely gets one clean CSV. They get a policy extract, a claims
extract, and often a vehicle or coverage file, exported from different
systems by different teams. The policy identifier is `POLICY_NO` in one and
`POL_NUM` in the other; the claims file has fifteen rows whose policy no
longer exists in the in-force snapshot. Reconciling those by hand in Excel
is the work this module removes.

Detection is by **value overlap**, not column name. Two columns join if
their values actually intersect — `POLICY_NO`↔`POL_NUM` overlap at 96% and
share no substring worth matching on. Name similarity only breaks ties
between candidates that already overlap.

The three questions an actuary asks before trusting a join, each answered
here as a first-class diagnostic:

  1. **Did it match?**   `match_rate` — the share of right rows finding a
     left partner. A silent 60% match rate is how exposure goes missing.
  2. **What did I lose?** `orphan_*` counts, both directions. Orphan claims
     are the dangerous ones: losses with no exposure inflate every rate.
  3. **Did rows multiply?** `cardinality`. A 1:N join fans the left side
     out; without aggregation, exposure is double-counted and every
     frequency is wrong.

That last point is why `execute_join` aggregates by default on 1:N rather
than fanning out. A claims file joined naively onto policies produces one
row per claim, and summing `EARNED_CAR_YRS` over that is the single most
common way to get a wrong answer here. Aggregating to COUNT and SUM per
policy is what the modeling pipeline actually wants — it is exactly the
frequency/severity shape.

Pure compute: frames in, frames out. Nothing is written or sent anywhere.

Example::

    >>> from tieback.keys import detect_join_keys
    >>> result = detect_join_keys(policy_df, claims_df)
    >>> best = result.best
    >>> best.left_column, best.right_column
    ('POLICY_NO', 'POL_NUM')
    >>> best.cardinality, round(best.match_rate, 3)
    ('one_to_many', 0.964)
    >>> best.orphan_right_rows
    15
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

# ── Tunables ──────────────────────────────────────────────────────────────────

# A join key must be *selective*. A column with 3 distinct values across
# 50k rows is a rating factor, not an identifier — joining on it produces a
# cartesian explosion.
_MIN_KEY_UNIQUE_RATIO = 0.02
_MIN_KEY_UNIQUE_COUNT = 8

# Below this share of right-side rows finding a partner, the pair is not a
# real relationship — it is coincidental overlap (two small integer columns
# both containing 1..10).
_MIN_MATCH_RATE = 0.20

# Report-worthy match rate. Below this the join runs but the diagnostics
# lead with a warning: the actuary must decide whether the missing rows are
# expected (new business mid-term) or a broken extract.
_HEALTHY_MATCH_RATE = 0.90

# Sampling cap for the overlap pass. Join-key detection is a set-intersection
# question; 50k distinct values settle it, and this keeps a 5M-row upload
# from being materialised in full.
_SAMPLE_ROWS = 50_000

# How far a match on stripped ids ('ON-0482113' ~ 482113) ranks below one on
# ids as written.
_DIGITS_PENALTY = 0.15
# Digits (padding included) a value needs for its number to stand as an id.
_MIN_ID_DIGITS = 4

Cardinality = Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
JoinHow = Literal["left", "inner", "outer"]

# Identifier vocabulary — used ONLY to break ties between candidates that
# already pass the overlap test, never to propose a candidate.
_ID_VOCAB = re.compile(
    r"(^id$|_id$|^id_|_no$|_nbr$|_num$|number|policy|pol_|claim|clm_|"
    r"contract|cert|account|acct)",
    re.IGNORECASE,
)


def _norm_name(name: str) -> str:
    """Strip separators and common id suffixes so POLICY_NO and PolicyNumber
    compare equal-ish. Tie-break only."""
    s = re.sub(r"[^a-z0-9]", "", name.lower())
    for suffix in ("number", "num", "nbr", "no", "id", "code", "cd"):
        if s.endswith(suffix) and len(s) > len(suffix) + 2:
            s = s[: -len(suffix)]
            break
    return s


@dataclass(frozen=True)
class JoinDiagnostics:
    """What the actuary needs to see before trusting the join."""

    left_rows: int
    right_rows: int
    matched_right_rows: int
    match_rate: float
    orphan_left_rows: int
    orphan_right_rows: int
    result_rows: int

    @property
    def is_healthy(self) -> bool:
        return self.match_rate >= _HEALTHY_MATCH_RATE and self.orphan_right_rows == 0

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.match_rate < _HEALTHY_MATCH_RATE:
            out.append(
                f"Only {self.match_rate:.1%} of right-hand rows found a match — "
                f"{self.orphan_right_rows:,} rows have no counterpart. Confirm "
                "this is expected (mid-term new business) and not a truncated "
                "extract."
            )
        if self.orphan_right_rows > 0 and self.match_rate >= _HEALTHY_MATCH_RATE:
            out.append(
                f"{self.orphan_right_rows:,} right-hand rows are orphaned. If "
                "these are claims, they represent losses with no matching "
                "exposure and will bias every rate built on this join."
            )
        return out


@dataclass
class JoinCandidate:
    """One inferred relationship between two datasets."""

    left_column: str
    right_column: str
    cardinality: Cardinality
    match_rate: float
    jaccard: float
    confidence: float
    left_unique: int
    right_unique: int
    overlap_count: int
    orphan_left_rows: int
    orphan_right_rows: int
    reasons: list[str] = field(default_factory=list)
    # "digits" when the ids only match once prefixes and leading zeros are
    # dropped; `example` is one such pair of raw values, to show the actuary.
    key_form: str = "exact"
    example: tuple[str, str] | None = None

    @property
    def requires_aggregation(self) -> bool:
        """1:N fans the left side out. Aggregating instead of fanning is what
        keeps exposure from being double-counted."""
        return self.cardinality in ("one_to_many", "many_to_many")


@dataclass
class ReconciliationResult:
    candidates: list[JoinCandidate]

    @property
    def best(self) -> JoinCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def is_confident(self) -> bool:
        """A clear winner, well ahead of the runner-up. Two near-tied keys
        (policy_no and a 1:1 surrogate key) should be shown, not guessed
        between."""
        if not self.candidates:
            return False
        top = self.candidates[0]
        if top.confidence < 0.70:
            return False
        if len(self.candidates) > 1:
            return (top.confidence - self.candidates[1].confidence) >= 0.10
        return True


# ── Key detection ─────────────────────────────────────────────────────────────


KeyForm = Literal["exact", "digits"]


def normalize_key(s: pl.Series, form: str = "exact") -> pl.Series | None:
    """The column as key values, one per row (nulls kept), or None if the
    column cannot be a key in this form.

    - "exact": the value as text. An integer `POL_ID` matches a string
      "1234" on the other side; integer-valued floats (CSV inference types an
      id column as Float64 whenever it has nulls, as freMTPL2's IDpol
      arrives) normalise to their integer form.
    - "digits": the one run of digits in the value, without leading zeros,
      so the same policy written 'ON-0482113' by the policy system and
      'AU/482113' or 482113 by the claims system is one key. Only for
      columns where (nearly) every value holds exactly one run of digits:
      a date or 'CLM-2024-0013' has several and is not a number in disguise.
    """
    if len(s) == 0 or s.dtype == pl.Boolean:
        return None
    if s.dtype in (pl.Float32, pl.Float64):
        # Genuine floats are measurements, not identifiers: a float key is
        # usually coincidence (two exposure columns overlapping on 1.0).
        nn = s.drop_nulls()
        try:
            if len(nn) and float((nn - nn.round(0)).abs().max() or 0.0) > 1e-9:
                return None
            s = s.round(0).cast(pl.Int64, strict=False)
        except Exception:
            return None
    if s.dtype.is_integer():
        return s.cast(pl.Utf8)
    if form == "digits" and s.dtype in (pl.Date, pl.Datetime):
        return None
    try:
        t = s.cast(pl.Utf8, strict=False).str.strip_chars()
    except Exception:
        return None
    t = pl.Series(s.name, t).replace("", None)
    if form == "exact":
        return t
    nn = t.drop_nulls()
    if len(nn) == 0:
        return None
    runs = t.str.count_matches(r"\d+")
    if float((runs.drop_nulls() == 1).mean() or 0.0) < 0.95:
        return None
    raw = t.str.extract(r"(\d+)", 1)
    # Policy numbers are long; 'Z5' or 'Unit 3' is a label, and stripped to
    # 5 or 3 it would match any small integer column.
    if float((raw.drop_nulls().str.len_chars() >= _MIN_ID_DIGITS).mean() or 0.0) < 0.95:
        return None
    core = raw.str.strip_chars_start("0")
    core = pl.when(core == "").then(pl.lit("0")).otherwise(core)
    return pl.select(pl.when(runs == 1).then(core).otherwise(None).alias(s.name)).to_series()


def _key_values(df: pl.DataFrame, col: str, form: str = "exact") -> pl.Series | None:
    """Normalised, non-null key values for overlap testing."""
    v = normalize_key(df[col], form)
    return None if v is None else v.drop_nulls()


def _raw_for(df: pl.DataFrame, col: str, form: str, value: str) -> str:
    """A raw value of `col` whose key is `value`, for showing the match."""
    k = normalize_key(df[col], form)
    idx = (k == value).arg_true()
    return str(df[col][int(idx[0])]) if len(idx) else value


def _has_letters(s: pl.Series) -> bool:
    if s.dtype != pl.Utf8:
        return False
    nn = s.drop_nulls()
    return len(nn) > 0 and float(nn.str.contains(r"[A-Za-z]").mean() or 0.0) >= 0.5


def _has_digits_form(s: pl.Series) -> bool:
    """A text column whose digits form differs from its exact form."""
    return s.dtype == pl.Utf8


def _is_viable_key(values: pl.Series, row_count: int) -> bool:
    uniq = values.n_unique()
    if uniq < _MIN_KEY_UNIQUE_COUNT:
        return False
    return (uniq / row_count if row_count else 0.0) >= _MIN_KEY_UNIQUE_RATIO


def _cardinality(left_dup: bool, right_dup: bool) -> Cardinality:
    if not left_dup and not right_dup:
        return "one_to_one"
    if not left_dup and right_dup:
        return "one_to_many"
    if left_dup and not right_dup:
        return "many_to_one"
    return "many_to_many"


def detect_join_keys(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    sample_rows: int = _SAMPLE_ROWS,
) -> ReconciliationResult:
    """Rank every (left column, right column) pair by how well it joins.

    Detection is purely value-based: a pair scores on the share of right
    rows that find a left partner and on set overlap. Column names only
    adjust confidence once a pair already overlaps.
    """
    if left.height == 0 or right.height == 0:
        return ReconciliationResult(candidates=[])

    # Sampling asymmetry is deliberate. Match rate is "share of RIGHT rows
    # finding a LEFT partner", so the left key set must stay COMPLETE or the
    # rate is understated. Sampling both sides is what breaks: on freMTPL2
    # (sorted by IDpol) a true 100%-match join measured 7.1% and fell below
    # threshold, because each side retained a different ~1/14 of the keys.
    #
    # So: stride-sample only the right side (bounding the expensive
    # per-row is_in test), and always build the left key set from every row.
    # Left-side columns are cheap here — one pass to a Python set.
    r_scan = (
        right
        if right.height <= sample_rows
        else right.gather_every(math.ceil(right.height / sample_rows))
    )
    l_scan = left

    def _forms(frame: pl.DataFrame) -> dict[tuple[str, str], pl.Series]:
        out: dict[tuple[str, str], pl.Series] = {}
        for c in frame.columns:
            for form in ("exact", "digits") if _has_digits_form(frame[c]) else ("exact",):
                v = _key_values(frame, c, form)
                if v is not None and _is_viable_key(v, frame.height):
                    out[(c, form)] = v
        return out

    l_keys = _forms(l_scan)
    r_keys = _forms(r_scan)

    # Build each key set ONCE: a 40-column file against a 40-column file
    # would otherwise build 1,600 identical sets.
    l_sets: dict[tuple[str, str], set] = {k: set(v.to_list()) for k, v in l_keys.items()}
    r_sets: dict[tuple[str, str], set] = {k: set(v.to_list()) for k, v in r_keys.items()}

    # One candidate per column pair, in whichever form matches more rows.
    best_by_pair: dict[tuple[str, str], JoinCandidate] = {}
    for (lc, lform), lv in l_keys.items():
        l_set = l_sets[(lc, lform)]
        l_dup = lv.n_unique() < len(lv)
        for (rc, rform), rv in r_keys.items():
            # Digits only pair with digits (an integer column's exact form is
            # already its digits), so a text id is never compared raw against
            # its stripped twin.
            if "digits" in (lform, rform) and not (
                {lform, rform} == {"digits"}
                or (lform == "digits" and not _has_digits_form(r_scan[rc]))
                or (rform == "digits" and not _has_digits_form(l_scan[lc]))
            ):
                continue
            form = "digits" if "digits" in (lform, rform) else "exact"
            # Letters on both sides are two numbering schemes, not one id
            # written twice: 'A17' and 'B17' are different things. Stripping
            # is for a prefix or padding one system adds ('ON-0482113' and
            # 482113), so one side must be a bare number.
            if form == "digits" and _has_letters(l_scan[lc]) and _has_letters(r_scan[rc]):
                continue
            overlap = l_set & r_sets[(rc, rform)]
            if not overlap:
                continue

            # Match rate is per-row on the right, not per-distinct-value:
            # what matters is how many actual claim rows find a policy.
            matched_rows = int(rv.is_in(list(overlap)).sum())
            match_rate = matched_rows / len(rv)
            if match_rate < _MIN_MATCH_RATE:
                continue
            prior = best_by_pair.get((lc, rc))
            if prior is not None and prior.match_rate >= match_rate:
                continue

            union = len(l_set | r_sets[(rc, rform)])
            jaccard = len(overlap) / union if union else 0.0
            r_dup = rv.n_unique() < len(rv)
            card = _cardinality(l_dup, r_dup)

            reasons = [
                f"{match_rate:.1%} of right-hand rows match a left-hand key.",
                f"{len(overlap):,} shared key values (Jaccard {jaccard:.2f}).",
            ]
            example = None
            if form == "digits":
                v = next(iter(overlap))
                example = (_raw_for(l_scan, lc, lform, v), _raw_for(r_scan, rc, rform, v))
                reasons.append(
                    f"The ids match once prefixes and leading zeros are dropped: "
                    f"'{example[0]}' is '{example[1]}'."
                )
            # Match rate dominates; Jaccard is secondary because a small
            # claims file legitimately covers a fraction of the policy book.
            score = 0.70 * match_rate + 0.30 * jaccard

            if _ID_VOCAB.search(lc) and _ID_VOCAB.search(rc):
                score += 0.10
                reasons.append("Both column names read as identifiers.")
            if _norm_name(lc) == _norm_name(rc):
                score += 0.10
                reasons.append(f"Normalised names agree ('{_norm_name(lc)}').")

            if form == "digits":
                # Stripping prefixes makes any two sequences collide
                # ('CL00012' and 'PN000012' are both 12), so a match that
                # needs it ranks below one that holds as written.
                score -= _DIGITS_PENALTY

            card_label = {
                "one_to_one": "1:1 — each key appears once on both sides.",
                "one_to_many": "1:N — the right side has repeats; aggregation "
                "is required to avoid double-counting exposure.",
                "many_to_one": "N:1 — the left side has repeats.",
                "many_to_many": "N:M — both sides repeat; this fans rows out "
                "and is rarely the intended join.",
            }[card]
            reasons.append(card_label)
            if card == "many_to_many":
                score -= 0.20

            orphan_r = len(rv) - matched_rows
            orphan_l = int((~lv.is_in(list(overlap))).sum())

            best_by_pair[(lc, rc)] = JoinCandidate(
                left_column=lc,
                right_column=rc,
                cardinality=card,
                match_rate=match_rate,
                jaccard=jaccard,
                confidence=max(0.0, min(score, 1.0)),
                left_unique=lv.n_unique(),
                right_unique=rv.n_unique(),
                overlap_count=len(overlap),
                orphan_left_rows=orphan_l,
                orphan_right_rows=orphan_r,
                reasons=reasons,
                key_form=form,
                example=example,
            )

    candidates = list(best_by_pair.values())
    candidates.sort(key=lambda c: (-c.confidence, c.left_column, c.right_column))
    return ReconciliationResult(candidates=candidates)


# ── Join execution ────────────────────────────────────────────────────────────


@dataclass
class JoinSpec:
    """A reconciliation the actuary has confirmed."""

    left_column: str
    right_column: str
    how: JoinHow = "left"
    # On a 1:N join, aggregate the right side to one row per key instead of
    # fanning the left side out. Default True — see module docstring.
    aggregate: bool = True
    # Right-hand numeric columns to SUM per key (e.g. PAID_AMT). Others are
    # dropped when aggregating; a COUNT is always emitted.
    sum_columns: list[str] = field(default_factory=list)
    count_column_name: str = "claim_count"
    # "exact", "digits" (ids that match once prefixes and leading zeros are
    # dropped) or "auto": whichever matches more right-hand rows.
    key_form: str = "auto"


def _aggregate_right(
    right: pl.DataFrame, spec: JoinSpec, key: str
) -> pl.DataFrame:
    aggs = [pl.len().alias(spec.count_column_name)]
    for c in spec.sum_columns:
        if c in right.columns and c != key:
            aggs.append(pl.col(c).sum().alias(f"{c}_sum"))
    return right.group_by(key).agg(aggs)


def execute_join(
    left: pl.DataFrame, right: pl.DataFrame, spec: JoinSpec
) -> tuple[pl.DataFrame, JoinDiagnostics]:
    """Materialise the join and report what it did to the data.

    Both key columns are cast to Utf8 before joining so an integer key on
    one side matches a zero-padded string on the other.

    On a 1:N join with `aggregate=True` the right side is collapsed to one
    row per key (COUNT plus any requested SUMs) before joining, so the left
    row count is preserved and exposure is never double-counted.
    """
    if spec.left_column not in left.columns:
        raise ValueError(f"left_column {spec.left_column!r} not in left DataFrame")
    if spec.right_column not in right.columns:
        raise ValueError(f"right_column {spec.right_column!r} not in right DataFrame")

    key = "__join_key__"

    def _keys(form: str) -> tuple[pl.Series, pl.Series] | None:
        if form == "digits" and spec.key_form == "auto" and _has_letters(left[spec.left_column]) and _has_letters(right[spec.right_column]):
            return None
        lk = normalize_key(left[spec.left_column], form)
        rk = normalize_key(right[spec.right_column], form)
        if lk is None or rk is None:
            return None
        return lk.alias(key), rk.alias(key)

    forms = ["exact", "digits"] if spec.key_form == "auto" else [spec.key_form]
    options = [k for k in (_keys(f) for f in forms) if k is not None]
    if not options:
        raise ValueError(f"{spec.left_column!r} and {spec.right_column!r} cannot be compared as keys")
    # Auto: the form under which more right-hand rows find a partner, as in
    # detection, so a key detected on stripped ids also executes on them.
    lk, rk = max(options, key=lambda o: int(o[1].is_in(o[0].drop_nulls().unique().implode()).sum()))
    l = left.with_columns(lk)
    r = right.with_columns(rk)

    l_vals = set(l[key].drop_nulls().to_list())
    r_vals_series = r[key].drop_nulls()
    overlap = l_vals & set(r_vals_series.to_list())
    matched_right = int(r_vals_series.is_in(list(overlap)).sum()) if overlap else 0
    right_rows = len(r_vals_series)
    match_rate = matched_right / right_rows if right_rows else 0.0
    orphan_right = right_rows - matched_right
    orphan_left = int((~l[key].drop_nulls().is_in(list(overlap))).sum()) if overlap else l.height

    right_dup = r_vals_series.n_unique() < right_rows
    if spec.aggregate and right_dup:
        r_prepared = _aggregate_right(r.drop(spec.right_column), spec, key)
    else:
        r_prepared = r.drop(spec.right_column)

    out = l.join(r_prepared, on=key, how=spec.how).drop(key)

    # A left row with no match has ZERO claims, not UNKNOWN claims. Leaving
    # the aggregates null silently drops every claim-free policy from the
    # fit — the model then trains only on claimants, where frequency is 1 by
    # construction, and reports a perfect Gini on a useless model. This is
    # the most damaging thing a policy-to-claims join can do, so the
    # aggregate columns this join created are filled with 0.
    #
    # Only the columns WE generated are filled. A null in a right-hand
    # column we merely carried across is genuinely missing data and stays
    # null for the cleaner to report.
    if spec.how == "left":
        generated = [spec.count_column_name] + [
            f"{c}_sum" for c in spec.sum_columns if c != spec.right_column
        ]
        fills = [
            pl.col(c).fill_null(0).alias(c) for c in generated if c in out.columns
        ]
        if fills:
            out = out.with_columns(fills)

    diags = JoinDiagnostics(
        left_rows=left.height,
        right_rows=right.height,
        matched_right_rows=matched_right,
        match_rate=match_rate,
        orphan_left_rows=orphan_left,
        orphan_right_rows=orphan_right,
        result_rows=out.height,
    )
    return out, diags
